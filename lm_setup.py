import gc
import multiprocessing
import glob
import re
import logging
import os
import numpy as np

from collections import namedtuple

from base.utils import print_title, os_exec, verify_dir_exists
from lm_model import WrappedLm1bModel, encode_paragraphs, LM1B_MODEL_HIDDEN_DIM


LM1B_URL_PREFIX =  'http://download.tensorflow.org/models/LM_LSTM_CNN/'
LM1B_DATA_DIR = 'data/lm1b/'

LM1B_VOCAB_FILENAME = 'vocab-2016-09-10.txt'
LM1B_VOCAB_URL = LM1B_URL_PREFIX + LM1B_VOCAB_FILENAME
LM1B_VOCAB_PATH = LM1B_DATA_DIR + LM1B_VOCAB_FILENAME

LM1B_CKPT_BASE_URL = LM1B_URL_PREFIX + 'all_shards-2016-09-10/ckpt-base'
LM1B_CKPT_CHAR_EMB_URL = LM1B_URL_PREFIX + 'all_shards-2016-09-10/ckpt-char-embedding'
LM1B_CKPT_LSTM_URL = LM1B_URL_PREFIX + 'all_shards-2016-09-10/ckpt-lstm'

LM1B_GRAPH_DEF_FILENAME = 'altered-graph-2017-10-06.pbtxt'
LM1B_GRAPH_DEF_PATH = LM1B_DATA_DIR + LM1B_GRAPH_DEF_FILENAME


LmDataShardConfig = namedtuple('LmDataShardConfig', [
  'dataset',    # 'train' or 'dev'
  'sequences',  # 'contexts' or 'questions'
  'payload',    # 'EMB' or 'L1' or 'L2'
  'num_shards', # int
  'shard'       # int
])


LmDatasetEncodings = namedtuple('LmDatasetEncodings', [
  'h_starts',
  'lens',
  'hs'
])


LmData = namedtuple('LmData', [
  'trn_ctxs',   # LmDatasetEncodings
  'trn_qtns',   # LmDatasetEncodings
  'dev_ctxs',   # LmDatasetEncodings
  'dev_qtns',   # LmDatasetEncodings
  'tst_ctxs',   # LmDatasetEncodings
  'tst_qtns'    # LmDatasetEncodings
])



def _get_shard_seq_idxs(num_examples, num_shards, current_shard):
  assert 0 < current_shard and current_shard <= num_shards
  shard_size = num_examples // num_shards
  if num_examples % num_shards:
    shard_size += 1
  first_seq_idx = (current_shard - 1) * shard_size
  after_last_seq_idx = min(first_seq_idx + shard_size, num_examples)
  return first_seq_idx, after_last_seq_idx


def _write_lm_data_shard_process(lm_data_shard_cfg, device, seqs_originals, seqs_sent_lens):
  if device == 'cpu':
    cuda_visible_devices = ''
  else:
    assert device.startswith('gpu')
    cuda_visible_devices = device[3:]
  os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
  lm_model = WrappedLm1bModel(
    LM1B_GRAPH_DEF_PATH,
    LM1B_DATA_DIR + 'ckpt-*',
    LM1B_VOCAB_PATH)
  hs, lens = encode_paragraphs(lm_model, lm_data_shard_cfg, seqs_originals, seqs_sent_lens)
  shard_path = _get_shard_path(lm_data_shard_cfg)
  logging.getLogger().info('Saving to {:s} ...'.format(shard_path))
  verify_dir_exists(shard_path)
  np.savez(shard_path, hs=hs, lens=lens)


def _get_shard_path_prefix(payload, dataset, sequences):
  return LM1B_DATA_DIR + 'lm_encodings_{:s}_{:s}_{:s}'.format(
    payload, dataset, sequences)


def _get_shard_path(lm_data_shard_cfg):
  path_prefix = _get_shard_path_prefix(
    lm_data_shard_cfg.payload,
    lm_data_shard_cfg.dataset,
    lm_data_shard_cfg.sequences)
  shard_path = path_prefix + '_{:03d}_of_{:03d}.npz'.format(
    lm_data_shard_cfg.shard,
    lm_data_shard_cfg.num_shards)
  return shard_path


def _load_lm_dataset_encodings(dataset, sequences, payload):
  logger = logging.getLogger()
  path_prefix = _get_shard_path_prefix(payload, dataset, sequences)
  paths = glob.glob(path_prefix + '*')
  assert paths, 'Did not find expected LM data shards ' + path_prefix
  last_path = sorted(paths)[-1]
  p = re.compile('_of_([0-9]*?)\.npz$')
  num_shards = int(p.search(last_path).group(1))
  shard_paths = []
  lens_list = []
  for shard in range(1, num_shards+1):
    shard_cfg = LmDataShardConfig(
      dataset, sequences, payload, num_shards, shard)
    shard_path = _get_shard_path(shard_cfg)
    shard_paths.append(shard_path)
    shard_lens = np.load(shard_path)['lens']
    logger.info('Loaded shard lengths from {:<60s}: {:s} {:s}'.format(shard_path, shard_lens.dtype, shard_lens.shape))
    lens_list.append(shard_lens)
  lens = np.concatenate(lens_list, axis=0)

  h_starts = np.insert(lens[:-1], 0, [0]).cumsum(dtype=np.int32)
  num_vectors = lens.sum()
  hs = np.zeros((num_vectors, LM1B_MODEL_HIDDEN_DIM), dtype=np.float32)

  pos = 0
  for shard_path in shard_paths:
    shard_hs = np.load(shard_path)['hs']
    hs[pos:pos+shard_hs.shape[0], :] = shard_hs
    pos += shard_hs.shape[0]
    logger.info('Loaded shard hs      from {:<60s}: {:s} {:s}'.format(shard_path, shard_hs.dtype, shard_hs.shape))
    del shard_hs
    gc.collect()
  assert pos == num_vectors

  dataset_str = '{:s} {:s} {:s}'.format(payload, dataset, sequences)
  logger.info('Joined shard lengths for  {:<60s}: {:s} {:s}'.format(
    dataset_str, str(lens.dtype), str(lens.shape)))
  logger.info('Joined shard hs for       {:<60s}: {:s} {:s}'.format(
    dataset_str, str(hs.dtype), str(hs.shape)))

  gc.collect()
  return LmDatasetEncodings(h_starts, lens, hs)


##########################################
# Interface
##########################################

def download_lm_model():
  print_title('Downloading LM1B model')
  for url in [LM1B_CKPT_BASE_URL, LM1B_CKPT_CHAR_EMB_URL, LM1B_CKPT_LSTM_URL, LM1B_VOCAB_URL]:
    wget_cmd = 'wget {} -P {}'.format(url, LM1B_DATA_DIR)
    os_exec(wget_cmd)


def write_lm_data_shard(data, lm_data_shard_cfg, device):
  logger = logging.getLogger()
  logger.info('Writing LM data for lm_data_shard_cfg {:}'.format(lm_data_shard_cfg))

  ds_mapping = {'train': data.trn.tabular, 'dev': data.dev.tabular}
  tab_ds = ds_mapping[lm_data_shard_cfg.dataset]
  seqs_mapping = {'contexts': tab_ds.ctxs, 'questions': tab_ds.qtns}
  seqs = seqs_mapping[lm_data_shard_cfg.sequences]

  seqs_originals = [seq.tokenized.originals for seq in seqs]
  seqs_sent_lens = [seq.tokenized.sent_lens for seq in seqs]
  for originals, sent_lens in zip(seqs_originals, seqs_sent_lens):
    assert sum(sent_lens) == len(originals)

  first_seq_idx, after_last_seq_idx = _get_shard_seq_idxs(
    len(seqs_originals), lm_data_shard_cfg.num_shards, lm_data_shard_cfg.shard)
  logger.info('There are a total of {:d} {:s} in {:s} dataset, shard to be written covers indices {:08d}-{:08d}'.format(
    len(seqs_originals), lm_data_shard_cfg.sequences, lm_data_shard_cfg.dataset, first_seq_idx, after_last_seq_idx-1))

  seqs_originals = seqs_originals[first_seq_idx:after_last_seq_idx]
  seqs_sent_lens = seqs_sent_lens[first_seq_idx:after_last_seq_idx]

  job = multiprocessing.Process(
    target = _write_lm_data_shard_process,
    args = (lm_data_shard_cfg, device, seqs_originals, seqs_sent_lens))
  job.start()
  job.join()
  logger.info('Done writing LM data for lm_data_shard_cfg {:}'.format(lm_data_shard_cfg))


def get_lm_data(payload):
  trn_ctxs = _load_lm_dataset_encodings('train', 'contexts', payload)
  trn_qtns = _load_lm_dataset_encodings('train', 'questions', payload)
  dev_ctxs = _load_lm_dataset_encodings('dev', 'contexts', payload)
  dev_qtns = _load_lm_dataset_encodings('dev', 'questions', payload)
  return LmData(
    trn_ctxs, trn_qtns, dev_ctxs, dev_qtns, tst_ctxs=None, tst_qtns=None)

