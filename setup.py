import argparse
import os
import sys
import io
import logging
import json
import unicodedata
import numpy as np

from collections import Counter

from base.utils import set_up_logger, print_title, os_exec
from word_embeddings import (WordEmbData, write_word_emb_data, read_word_emb_data,
  get_word_emb_data_paths)
from reader import get_data
from lm_setup import LmDataShardConfig, download_lm_model, write_lm_data_shard, get_lm_data


TRN_JSON_PATH = 'data/train-v1.1.json'
DEV_JSON_PATH = 'data/dev-v1.1.json'

TOKENIZED_TRN_JSON_PATH = 'data/train-v1.1.tokenized.json'
TOKENIZED_DEV_JSON_PATH = 'data/dev-v1.1.tokenized.json'

GLOVE_MAX_WORDS = 2200000
GLOVE_EMB_DIM = 300
GLOVE_FIRST_KNOWN_WORD_IDX = 2     # reserve two first entries of word embedding matrix
GLOVE_NUM_UNK_EMBEDDINGS = 100000
GLOVE_ZIP_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
GLOVE_ZIP_PATH = 'data/glove.840B.300d.zip'
GLOVE_TXT_PATH = 'data/glove.840B.300d.txt'
GLOVE_STRS_PATH = 'data/glove.840B.300d_strs.txt'
GLOVE_PREPROC_PATH_PREFIX = 'data/preprocessed_glove'
GLOVE_PREPROC_WITH_UNKS_PATH_PREFIX = 'data/preprocessed_glove_with_unks'

CORENLP_ZIP_URL = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip'
CORENLP_ZIP_PATH = 'tokenizer/stanford-corenlp-full-2016-10-31.zip'
CORENLP_EXTRACT_PATH = 'tokenizer/'

MAX_ANS_LEN = 30
MAX_CTX_LEN = 700


#######################################
# GloVe:
#######################################

def _download_and_unzip_glove(glove_zip_url, glove_zip_path, glove_txt_path):
  logger = logging.getLogger()
  if os.path.isfile(glove_txt_path):
    logger.info('GloVe raw text found at {}'.format(glove_txt_path))
    return
  if not os.path.isfile(glove_zip_path):
    logger.info('Downloading GloVe')
    wget_cmd = 'wget {} -O {}'.format(glove_zip_url, glove_zip_path)
    os_exec(wget_cmd)
  logger.info('Unzipping GloVe')
  unzip_cmd = 'unzip {} -d {}'.format(glove_zip_path, os.path.dirname(glove_txt_path))
  os_exec(unzip_cmd)


def _write_glove_data(glove_txt_path, glove_max_words, glove_emb_dim,
  glove_first_known_word_idx, glove_strs_path, glove_preproc_path_prefix):
  logger = logging.getLogger()

  metadata_path, emb_path = get_word_emb_data_paths(glove_preproc_path_prefix)
  if all(os.path.isfile(path) for path in [metadata_path, emb_path, glove_strs_path]):
    paths_str = '\n'.join(
      '\t{}'.format(path) for path in [metadata_path, emb_path, glove_strs_path])
    logger.info('Preprocessed GloVe files found at\n{}'.format(paths_str))
    return read_word_emb_data(glove_preproc_path_prefix)

  glove_str_to_word = {}
  glove_word_emb = np.zeros((glove_max_words, glove_emb_dim), dtype=np.float32)
  dups = Counter()
  logger.info('Processing raw GloVe...')
  with io.open(glove_txt_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
      strs = line.rstrip().split(" ")
      word_str = strs[0]
      if word_str in glove_str_to_word:
        dups[word_str] += 1
      else:
        word_emb = map(float, strs[1:])
        if len(word_emb) != glove_emb_dim:
          raise AssertionError('Error. Line {}'.format(i))
        word_idx = glove_first_known_word_idx + len(glove_str_to_word)
        glove_str_to_word[word_str] = word_idx
        glove_word_emb[word_idx, :] = word_emb
      if i % 200000 == 0:
        logger.info('\tprocessed {:d} lines'.format(i))
  logger.info('\tprocessed {:d} lines in total'.format(i))
  num_glove_words = len(glove_str_to_word)
  glove_first_unknown_word_idx = glove_first_known_word_idx + num_glove_words
  glove_word_emb = glove_word_emb[:glove_first_unknown_word_idx]

  logger.info('Done. Number of GloVe words: {}'.format(num_glove_words))
  logger.info('Total number of duplicate word-types: {}'.format(len(dups)))

  glove_word_emb_data = WordEmbData(
    glove_word_emb, glove_str_to_word, glove_first_known_word_idx, glove_first_unknown_word_idx, None)
  write_word_emb_data(glove_preproc_path_prefix, glove_word_emb_data)

  # Write a txt file listing GloVe words, read by Java program which parses
  # the train / dev / test JSON files.
  with io.open(glove_strs_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(glove_str_to_word.keys()))
  logger.info('Written GloVe word strings to\n\t{}'.format(glove_strs_path))

  return glove_word_emb_data


#######################################
# Stanford CoreNLP:
#######################################

def _download_and_unzip_corenlp(corenlp_zip_url, corenlp_zip_path, corenlp_extract_path):
  logger = logging.getLogger()
  if not os.path.isfile(corenlp_zip_path):
    logger.info('Downloading Stanford CoreNLP')
    wget_cmd = 'wget {} -O {}'.format(corenlp_zip_url, corenlp_zip_path)
    os_exec(wget_cmd)
  logger.info('Unzipping Stanford CoreNLP')
  unzip_cmd = 'unzip {} -d {}'.format(corenlp_zip_path, corenlp_extract_path)
  os_exec(unzip_cmd)


#######################################
# Preprocess datasets:
#######################################


def _tokenize_json(json_path, tokenized_json_path, glove_strs_path, has_answers):
  cmd_pattern = 'java -cp {} SquadTokenizer {} {} --words_txt={}{}'
  class_path = '"tokenizer/:tokenizer/*:tokenizer/stanford-corenlp-full-2016-10-31/*"'
  has_answers_flag = ' --has_answers' if has_answers else ''
  cmd = cmd_pattern.format(
    class_path, json_path, tokenized_json_path, glove_strs_path, has_answers_flag)
  os_exec(cmd)


def _add_extra_embeddings(tokenized_json_paths, old_word_emb_data, num_unallocated):
  unknown_words = set()
  for tokenized_json_path in tokenized_json_paths:
    with io.open(tokenized_json_path, 'r', encoding='utf-8') as f:
      j = json.load(f)
      j_unknown_words_array = j['unknown_words']
      # tokenizer/SquadTokenizer.java produces a "?" token for untokenizable characters,
      # local work around:
      dups = [dup_word for dup_word, dup_count in \
        Counter(j_unknown_words_array).iteritems() if dup_count > 1]
      if dups:
        assert len(dups) == 1 and dups[0] == '?'
      j_unknown_words = set(j_unknown_words_array)
      unknown_words.update(j_unknown_words)

  old_word_emb, old_str_to_word, old_first_known_word, _, _ = old_word_emb_data

  unknown_words_to_add = unknown_words.difference(old_str_to_word.keys())
  extra_word_embs = _get_random_embeddings(
    len(unknown_words_to_add) + num_unallocated, old_word_emb_data)
  word_emb = np.concatenate([old_word_emb, extra_word_embs], axis=0)

  first_unknown_word = len(old_word_emb)
  str_to_word_to_add = {s: w for w, s in enumerate(unknown_words_to_add, first_unknown_word)}
  old_str_to_word.update(str_to_word_to_add)    # inplace

  first_unallocated_word = first_unknown_word + len(unknown_words_to_add)
  return WordEmbData(
    word_emb, old_str_to_word, old_first_known_word, first_unknown_word, first_unallocated_word)
  

def _get_random_embeddings(num_embeddings, word_emb_data):
  known_word_emb = word_emb_data.word_emb[
    word_emb_data.first_known_word:word_emb_data.first_unknown_word]
  np_rng = np.random.RandomState(123)
  rnd_idxs = np_rng.permutation(len(known_word_emb))[:100000]
  known_subset = known_word_emb[rnd_idxs]
  known_mean, known_cov = np.mean(known_subset, axis=0), np.cov(known_subset, rowvar=0)
  unknown_word_embs = np_rng.multivariate_normal(
    mean=known_mean, cov=known_cov, size=num_embeddings).astype(np.float32)
  return unknown_word_embs



#######################################
# LM:
#######################################

def _write_lm_encodings(lm_data_shard_cfg, device):
  data = get_data(
    word_emb_data_path_prefix=GLOVE_PREPROC_WITH_UNKS_PATH_PREFIX,
    tokenized_trn_json_path=TOKENIZED_TRN_JSON_PATH,
    tokenized_dev_json_path=TOKENIZED_DEV_JSON_PATH,
    max_ans_len=MAX_ANS_LEN,
    max_ctx_len=MAX_CTX_LEN)
  write_lm_data_shard(data, lm_data_shard_cfg, device)


#######################################
# Program:
#######################################

def _prepare_squad():

  print_title('Preprocessing GloVe')
  _download_and_unzip_glove(GLOVE_ZIP_URL, GLOVE_ZIP_PATH, GLOVE_TXT_PATH)
  glove_word_emb_data = _write_glove_data(
    GLOVE_TXT_PATH, GLOVE_MAX_WORDS, GLOVE_EMB_DIM,
    GLOVE_FIRST_KNOWN_WORD_IDX, GLOVE_STRS_PATH, GLOVE_PREPROC_PATH_PREFIX)

  print_title('Setting up Stanford CoreNLP')
  _download_and_unzip_corenlp(CORENLP_ZIP_URL, CORENLP_ZIP_PATH, CORENLP_EXTRACT_PATH)

  print_title('Tokenizing JSON')
  _tokenize_json(TRN_JSON_PATH, TOKENIZED_TRN_JSON_PATH, GLOVE_STRS_PATH, has_answers=True)
  _tokenize_json(DEV_JSON_PATH, TOKENIZED_DEV_JSON_PATH, GLOVE_STRS_PATH, has_answers=True)

  print_title('Adding random embeddings for unknown words')
  glove_with_unks_word_emb_data = _add_extra_embeddings(
    [TOKENIZED_TRN_JSON_PATH, TOKENIZED_DEV_JSON_PATH],
    glove_word_emb_data, GLOVE_NUM_UNK_EMBEDDINGS)

  write_word_emb_data(GLOVE_PREPROC_WITH_UNKS_PATH_PREFIX, glove_with_unks_word_emb_data)



if __name__ == '__main__':
  logger = set_up_logger(log_filename='logs/setup.log')
  logger.info('Setup started.')

  parser = argparse.ArgumentParser()
  parser.add_argument('command', nargs='?', help='"prepare-squad", "prepare-lm" or "lm-encode"')
  parser.add_argument('--device', help='device e.g. cpu, gpu0, gpu1, ...', default='cpu')
  parser.add_argument('--dataset', help='"train" or "dev"', default=None)
  parser.add_argument('--sequences', help='"contexts" or "questions"', default=None)
  parser.add_argument('--layer', help='"L1" or "L2" or "EMB"', default=None)
  parser.add_argument('--num_shards', help='number of shards', type=int, default=None)
  parser.add_argument('--shard', help='shard to encode, count starting from 1', type=int,
    default=None)

  args = parser.parse_args()
  args_str = '\n'.join('\t{:20s} : {}'.format(arg, getattr(args, arg)) for arg in vars(args))
  logger.info('Setup arguments:\n' + args_str)

  if args.command == 'prepare-squad':
    _prepare_squad()
  elif args.command == 'prepare-lm':
    download_lm_model()
  elif args.command == 'lm-encode':
    lm_data_shard_cfg = LmDataShardConfig(
      args.dataset, args.sequences, args.layer, args.num_shards, args.shard)
    _write_lm_encodings(lm_data_shard_cfg, args.device)

  logger.info('Setup done.')

