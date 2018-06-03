import argparse
import logging
import sys
import time
import numpy as np

from base.utils import set_up_logger
from evaluate11 import metric_max_over_ground_truths, exact_match_score, f1_score
from utils import EpochResult, format_epoch_results, plot_epoch_results
from reader import get_data, construct_answer_hat, write_test_predictions
from lm_setup import get_lm_data
from setup import (GLOVE_PREPROC_WITH_UNKS_PATH_PREFIX,
  TOKENIZED_TRN_JSON_PATH, TOKENIZED_DEV_JSON_PATH,
  MAX_ANS_LEN, MAX_CTX_LEN)


class Config(object):
  def __init__(self, compared=[], **kwargs):
    self.name = None
    self.desc = ''
    self.device = None                      # 'cpu' / 'gpu<index>'
    self.plot = True                        # whether to plot training graphs
    self.save_freq = None                   # how often to save model (in epochs); None for only after best EM/F1 epochs

    self.seed = np.random.random_integers(1e6, 1e9)
    self.max_ans_len = 30                   # maximal answer length, answers of longer length are discarded
    self.emb_dim = 300                      # dimension of word embeddings

    self.init_scale = 5e-3                  # uniformly random weights are initialized in [-init_scale, +init_scale]
    self.learning_rate = 1e-3
    self.lr_decay = 0.95
    self.lr_decay_freq = 5000               # frequency with which to decay learning rate, measured in updates
    self.max_grad_norm = 5                  # gradient clipping
    self.max_num_epochs = 250               # max number of epochs to train for
    self.ff_dims = [100]                    # dimensions of hidden FF layers
    self.ff_drop_x = 0.2                    # dropout rate of FF layers
    self.batch_size = 80
    self.num_bilstm_layers = 2              # number of BiLSTM layers, where BiLSTM is applied
    self.hidden_dim = 200                   # dimension of hidden state of each uni-directional LSTM
    self.lstm_drop_h = 0.1                  # dropout rate for recurrent hidden state of LSTM
    self.lstm_drop_x = 0.6                  # dropout rate for inputs of LSTM
    self.lstm_couple_i_and_f = True         # customizable LSTM configuration, see base/model.py
    self.lstm_learn_initial_state = False
    self.lstm_tie_x_dropout = True
    self.lstm_sep_x_dropout = False
    self.lstm_sep_h_dropout = False
    self.lstm_w_init = 'uniform'
    self.lstm_u_init = 'uniform'
    self.lstm_forget_bias_init = 'uniform'
    self.default_bias_init = 'uniform'
    self.q_aln_ff_tie = True                # whether to tie the weights of the FF over question and the FF over passage
    self.sep_stt_end_drop = True            # whether to have separate dropout masks for span start and
                                            # span end representations
    self.adam_beta1 = 0.9                   # see base/optimizer.py
    self.adam_beta2 = 0.999
    self.adam_eps = 1e-8
    self.objective = 'span_multinomial'     # 'span_multinomial': multinomial distribution over all spans
                                            # 'span_binary':      logistic distribution per span
                                            # 'span_endpoints':   two multinomial distributions, over span start and end
    self.max_ctx_len = None                 # max context length, training samples with longer context are ignored
    self.loss_min_prob = 1e-8               # if not None, training samples for which gold prob is less than this are discarded from training batch
    self.wdp_drop = 0.15                    # word-dropout: replacement probability for each word-type

    self.char_dim = 8                       # dim of char embeddings
    self.char_win_sizes = [5]               # list with an element for each char window size
    self.char_feats = 100                   # number of output features of char conv per window
    self.char_drop = 0                      # dropout rate to apply over input to cha conv

    self.mode = None                        # Re-embedding mode: 'TR', 'TR_MLP', or 'LM'
    self.wn_tied = True                     # whether re-embedder weights are tied for ctx re-embeddings and for qtn re-embedding
    self.wn_num_lyrs = 2                    # for mode='TR': number of re-embedder LSTM layers
    self.wn_ff_dims = [400, 865, 865, 400]  # For mode='TR_MLP': dimensions of MLP
    self.lm_layer = None                    # For mode='LM': 'EMB' or 'L1' or 'L2'
    self.lm_dim = 200                       # For mode='LM': dim of result of FF over lm hidden states
    self.lm_drop = 0.5                      # For mode='LM': dropout applied over lm hidden states when calcig FF
    self.lm_cache_num_batch = 5             # For mode='LM': size of cache holding LM encodings, in batches

    assert all(k in self.__dict__ for k in kwargs), 'invalid kwargs: ' + str(list(k for k in kwargs if k not in self.__dict__))
    assert all(k in self.__dict__ for k in compared), 'invalid compared: ' + str(list(k for k in compared if k not in self.__dict__))
    self.__dict__.update(kwargs)
    self._compared = compared

  def __repr__(self):
    ks = sorted(k for k in self.__dict__ if k not in ['name', 'desc', '_compared'])
    return '\n'.join('{:<30s}{:<s}'.format(k, str(self.__dict__[k])) for k in ks)

  def format_compared(self):
    ks = sorted(self._compared)
    lines = []
    while ks:
      line = ''.join('{:12s} '.format(k[:12]) for k in ks) + \
        ''.join('{:12s} '.format(str(self.__dict__[k])[:12]) for k in ks)
      lines.append(line)
      ks = ks[10:]
    return '\n'.join(ks)


def _get_lm_cache(idxs, lm_data_part):
  h_starts = lm_data_part.h_starts[idxs]
  lens = lm_data_part.lens[idxs]
  max_len = lens.max()
  # cache is padded with zeros, so that when model access cache[start:start+max_p_len] we won't be out of bounds
  cache_num_vectors = lens.sum() + max_len
  cache_val = np.zeros((cache_num_vectors, 1024), dtype=np.float32)
  cache_h_starts = np.insert(lens[:-1], 0, [0]).cumsum(dtype=np.int32)
  for h_start, _len, cache_h_start in zip(h_starts, lens, cache_h_starts):
    cache_val[cache_h_start:cache_h_start+_len, :] = lm_data_part.hs[h_start:h_start+_len, :]
  return cache_val, cache_h_starts


def _trn_epoch(config, model, data, lm_data, epoch, np_rng):
  logger = logging.getLogger()
  num_all_samples = data.trn.vectorized.qtn_ans_inds.size
  # indices of questions which have a valid answer
  valid_qtn_idxs = np.flatnonzero(data.trn.vectorized.qtn_ans_inds).astype(np.int32)
  np_rng.shuffle(valid_qtn_idxs)
  num_valid_samples = valid_qtn_idxs.size

  batch_sizes = []
  losses = []
  accs = []
  grad_norms = []
  num_unsafe_samples = 0

  samples_per_sec = []
  ss = range(0, num_valid_samples, config.batch_size)

  cache_time = 0
  valid_cache_h_starts_trn_qtn = np.zeros(num_valid_samples, dtype=np.int32)
  valid_cache_h_starts_trn_ctx = np.zeros(num_valid_samples, dtype=np.int32)

  for b, s in enumerate(ss, 1):

    if config.mode == 'LM' and (b-1) % config.lm_cache_num_batch == 0:
      start_time = time.time()
      slice_start_idx = s
      slice_after_end_idx = min(s + config.lm_cache_num_batch * config.batch_size, num_valid_samples)
      cache_qtn_idxs = valid_qtn_idxs[slice_start_idx:slice_after_end_idx]
      cache_ctx_idxs = data.trn.vectorized.qtn_ctx_idxs[cache_qtn_idxs]
      cache_val_trn_qtn, cache_h_starts_trn_qtn = _get_lm_cache(cache_qtn_idxs, lm_data.trn_qtns)
      cache_val_trn_ctx, cache_h_starts_trn_ctx = _get_lm_cache(cache_ctx_idxs, lm_data.trn_ctxs)
      valid_cache_h_starts_trn_qtn[slice_start_idx:slice_after_end_idx] = cache_h_starts_trn_qtn
      valid_cache_h_starts_trn_ctx[slice_start_idx:slice_after_end_idx] = cache_h_starts_trn_ctx
      model.set_lm_qtn_cache(cache_val_trn_qtn)
      model.set_lm_ctx_cache(cache_val_trn_ctx)
      cache_time += (time.time() - start_time)

    batch_slice_start_idx = s
    batch_slice_after_end_idx = min(s+config.batch_size, num_valid_samples)
    batch_idxs = valid_qtn_idxs[batch_slice_start_idx:batch_slice_after_end_idx]
    batch_size = len(batch_idxs)
    batch_sizes.append(batch_size)

    batch_cache_h_starts_qtn = valid_cache_h_starts_trn_qtn[batch_slice_start_idx:batch_slice_after_end_idx]
    batch_cache_h_starts_ctx = valid_cache_h_starts_trn_ctx[batch_slice_start_idx:batch_slice_after_end_idx]

    start_time = time.time()
    try:
      loss, acc, global_grad_norm, batch_num_unsafe_samples = model.train(batch_idxs, batch_cache_h_starts_qtn, batch_cache_h_starts_ctx)
    except:
      max_qtn_len = data.trn.vectorized.qtn_lens[batch_idxs].max()
      max_ctx_len = data.trn.vectorized.ctx_lens[data.trn.vectorized.qtn_ctx_idxs[batch_idxs]].max()
      logger.info('\n\nmax_qtn_len={:d} max_ctx_len={:d}\n\n'.format(max_qtn_len, max_ctx_len))
      raise
    samples_per_sec.append(batch_size / (time.time() - start_time))

    losses.append(loss)
    accs.append(acc)
    grad_norms.append(global_grad_norm)
    num_unsafe_samples += batch_num_unsafe_samples

    if b % 200 == 0 or b == len(ss):
      logger.info(
        '{:<8s} {:<15s} lr={:<8.7f} : train loss={:<8.5f}\tacc={:<8.5f}\tgrad={:<8.5f}\tsamples/sec={:<.1f}   cache_t={:<.0f}s'.format(
        config.device, 'e'+str(epoch)+'b'+str(b)+'\\'+str(len(ss)), float(model.get_lr_value()),
        float(loss), float(acc), float(global_grad_norm), float(samples_per_sec[b-1]), float(cache_time)))

  trn_loss = np.average(losses, weights=batch_sizes)
  trn_acc = np.average(accs, weights=batch_sizes)
  trn_samples_per_sec = np.average(samples_per_sec, weights=batch_sizes)

  trn_mean_grad_norm = np.average(grad_norms, weights=batch_sizes)
  trn_max_grad_norm = max(grad_norms)
  trn_min_grad_norm = min(grad_norms)

  if not config.loss_min_prob:
    num_unsafe_samples = -1

  return (trn_loss, trn_acc, trn_samples_per_sec, num_all_samples, num_valid_samples,
    trn_mean_grad_norm, trn_max_grad_norm, trn_min_grad_norm, num_unsafe_samples)


def _dev_epoch(config, model, data, lm_data):
  logger = logging.getLogger()
  num_all_samples = data.dev.vectorized.qtn_ans_inds.size
  ans_hat_starts = np.zeros(num_all_samples, dtype=np.int32)
  ans_hat_ends = np.zeros(num_all_samples, dtype=np.int32)

  # indices of questions which have a valid answer
  valid_qtn_idxs = np.flatnonzero(data.dev.vectorized.qtn_ans_inds).astype(np.int32)
  num_valid_samples = valid_qtn_idxs.size
  batch_sizes = []
  losses = []
  accs = []
  ss = range(0, num_valid_samples, config.batch_size)

  valid_cache_h_starts_dev_qtn = np.zeros(num_valid_samples, dtype=np.int32)
  valid_cache_h_starts_dev_ctx = np.zeros(num_valid_samples, dtype=np.int32)

  for b, s in enumerate(ss, 1):

    if config.mode == 'LM' and (b-1) % config.lm_cache_num_batch == 0:
      slice_start_idx = s
      slice_after_end_idx = min(s + config.lm_cache_num_batch * config.batch_size, num_valid_samples)
      cache_qtn_idxs = valid_qtn_idxs[slice_start_idx:slice_after_end_idx]
      cache_ctx_idxs = data.dev.vectorized.qtn_ctx_idxs[cache_qtn_idxs]
      cache_val_dev_qtn, cache_h_starts_dev_qtn = _get_lm_cache(cache_qtn_idxs, lm_data.dev_qtns)
      cache_val_dev_ctx, cache_h_starts_dev_ctx = _get_lm_cache(cache_ctx_idxs, lm_data.dev_ctxs)
      valid_cache_h_starts_dev_qtn[slice_start_idx:slice_after_end_idx] = cache_h_starts_dev_qtn
      valid_cache_h_starts_dev_ctx[slice_start_idx:slice_after_end_idx] = cache_h_starts_dev_ctx
      model.set_lm_qtn_cache(cache_val_dev_qtn)
      model.set_lm_ctx_cache(cache_val_dev_ctx)

    batch_slice_start_idx = s
    batch_slice_after_end_idx = min(s+config.batch_size, num_valid_samples)
    batch_idxs = valid_qtn_idxs[batch_slice_start_idx:batch_slice_after_end_idx]
    batch_sizes.append(len(batch_idxs))

    batch_cache_h_starts_qtn = valid_cache_h_starts_dev_qtn[batch_slice_start_idx:batch_slice_after_end_idx]
    batch_cache_h_starts_ctx = valid_cache_h_starts_dev_ctx[batch_slice_start_idx:batch_slice_after_end_idx]

    try:
      loss, acc, ans_hat_start_word_idxs, ans_hat_end_word_idxs = model.eval_dev(batch_idxs, batch_cache_h_starts_qtn, batch_cache_h_starts_ctx)
    except:
      max_qtn_len = data.dev.vectorized.qtn_lens[batch_idxs].max()
      max_ctx_len = data.dev.vectorized.ctx_lens[data.dev.vectorized.qtn_ctx_idxs[batch_idxs]].max()
      logger.info('\n\nmax_qtn_len={:d} max_ctx_len={:d}\n\n'.format(max_qtn_len, max_ctx_len))
      raise

    losses.append(loss)
    accs.append(acc)
    ans_hat_starts[batch_idxs] = ans_hat_start_word_idxs
    ans_hat_ends[batch_idxs] = ans_hat_end_word_idxs
    if b % 100 == 0 or b == len(ss):
      logger.info('{:<8s} {:<15s} : dev valid'.format(config.device, 'b'+str(b)+'\\'+str(len(ss))))
  dev_loss = np.average(losses, weights=batch_sizes)
  dev_acc = np.average(accs, weights=batch_sizes)

  # indices of questions which have an invalid answer
  invalid_qtn_idxs = np.setdiff1d(np.arange(num_all_samples), valid_qtn_idxs).astype(np.int32)
  num_invalid_samples = invalid_qtn_idxs.size
  ss = range(0, num_invalid_samples, config.batch_size)

  invalid_cache_h_starts_dev_qtn = np.zeros(num_invalid_samples, dtype=np.int32)
  invalid_cache_h_starts_dev_ctx = np.zeros(num_invalid_samples, dtype=np.int32)

  for b, s in enumerate(ss, 1):

    if config.mode == 'LM' and (b-1) % config.lm_cache_num_batch == 0:
      slice_start_idx = s
      slice_after_end_idx = min(s + config.lm_cache_num_batch * config.batch_size, num_invalid_samples)
      cache_qtn_idxs = invalid_qtn_idxs[slice_start_idx:slice_after_end_idx]
      cache_ctx_idxs = data.dev.vectorized.qtn_ctx_idxs[cache_qtn_idxs]
      cache_val_dev_qtn, cache_h_starts_dev_qtn = _get_lm_cache(cache_qtn_idxs, lm_data.dev_qtns)
      cache_val_dev_ctx, cache_h_starts_dev_ctx = _get_lm_cache(cache_ctx_idxs, lm_data.dev_ctxs)
      invalid_cache_h_starts_dev_qtn[slice_start_idx:slice_after_end_idx] = cache_h_starts_dev_qtn
      invalid_cache_h_starts_dev_ctx[slice_start_idx:slice_after_end_idx] = cache_h_starts_dev_ctx
      model.set_lm_qtn_cache(cache_val_dev_qtn)
      model.set_lm_ctx_cache(cache_val_dev_ctx)

    batch_slice_start_idx = s
    batch_slice_after_end_idx = min(s+config.batch_size, num_invalid_samples)
    batch_idxs = invalid_qtn_idxs[batch_slice_start_idx:batch_slice_after_end_idx]

    batch_cache_h_starts_qtn = invalid_cache_h_starts_dev_qtn[batch_slice_start_idx:batch_slice_after_end_idx]
    batch_cache_h_starts_ctx = invalid_cache_h_starts_dev_ctx[batch_slice_start_idx:batch_slice_after_end_idx]

    _, _, ans_hat_start_word_idxs, ans_hat_end_word_idxs = model.eval_dev(batch_idxs, batch_cache_h_starts_qtn, batch_cache_h_starts_ctx)

    ans_hat_starts[batch_idxs] = ans_hat_start_word_idxs
    ans_hat_ends[batch_idxs] = ans_hat_end_word_idxs
    if b % 100 == 0 or b == len(ss):
      logger.info('{:<8s} {:<15s} : dev invalid'.format(config.device, 'b'+str(b)+'\\'+str(len(ss))))

  # calculate EM, F1
  ems = []
  f1s = []
  for qtn_idx, (ans_hat_start_word_idx, ans_hat_end_word_idx) in enumerate(zip(ans_hat_starts, ans_hat_ends)):
    qtn = data.dev.tabular.qtns[qtn_idx]
    ctx = data.dev.tabular.ctxs[qtn.ctx_idx]
    ans_hat_str = construct_answer_hat(ctx, ans_hat_start_word_idx, ans_hat_end_word_idx)
    ans_strs = qtn.ans_texts
    ems.append(metric_max_over_ground_truths(exact_match_score, ans_hat_str, ans_strs))
    f1s.append(metric_max_over_ground_truths(f1_score, ans_hat_str, ans_strs))
  dev_em = np.mean(ems)
  dev_f1 = np.mean(f1s)
  return dev_loss, dev_acc, dev_em, dev_f1, num_all_samples, num_valid_samples


def _get_config(name, device, mode, lm_layer):
  compared = [
    'ff_dims', 'ff_drop_x',
    'hidden_dim',
    'lstm_drop_x', 'num_bilstm_layers',
    'wn_num_lyrs',
    'char_dim', 'char_win_sizes', 'char_feats', 'char_drop',
    'wdp_drop',
    'lm_layer',
  ]
  return Config(compared,
    name=name,
    device=device,
    mode=mode,
    lm_layer=lm_layer)

  
def _main(config):
  base_filename = config.name
  logger_filename = 'logs/' + base_filename + '.log'
  logger = set_up_logger(logger_filename)
  title = '{}: {} ({})'.format(__file__, config.name, config.desc)
  logger.info('START ' + title + '\n\n{}\n'.format(config))

  data = get_data(
    word_emb_data_path_prefix=GLOVE_PREPROC_WITH_UNKS_PATH_PREFIX,
    tokenized_trn_json_path=TOKENIZED_TRN_JSON_PATH,
    tokenized_dev_json_path=TOKENIZED_DEV_JSON_PATH,
    max_ans_len=MAX_ANS_LEN,
    max_ctx_len=MAX_CTX_LEN)

  if config.device != 'cpu':
    assert 'theano' not in sys.modules 
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(config.device)

  from model  import get_model
  model = get_model(config, data)

  lm_data = get_lm_data(config.lm_layer) if config.mode == 'LM' else None

  # Training loop
  epoch_results = []
  max_em = -np.inf
  max_f1 = -np.inf
  np_rng = np.random.RandomState(config.seed // 2)
  for epoch in range(1, config.max_num_epochs+1):
    trn_loss, trn_acc, trn_samples_per_sec, trn_num_all_samples, trn_num_valid_samples, \
      trn_mean_grad_norm, trn_max_grad_norm, trn_min_grad_norm, trn_num_unsafe_samples = \
        _trn_epoch(config, model, data, lm_data, epoch, np_rng)
    dev_loss, dev_acc, dev_em, dev_f1, dev_num_all_samples, dev_num_valid_samples = \
      _dev_epoch(config, model, data, lm_data)

    best_filename = base_filename
    if dev_em > max_em:
      model.save('models/' + best_filename + '_best_em.pkl')
      max_em = dev_em
    if dev_f1 > max_f1:
      model.save('models/' + best_filename + '_best_f1.pkl')
      max_f1 = dev_f1
    if config.save_freq and epoch % config.save_freq == 0:
      model.save('models/' + base_filename + '_e{:03d}.pkl'.format(epoch))

    epoch_results.append(
      EpochResult(trn_loss, trn_acc, dev_loss, dev_acc, dev_em, dev_f1))
    if config.plot:
      plot_epoch_results(epoch_results, 'logs/' + base_filename + '.png')
    logger.info((
      '\n\nEpc {} {}: (smp/sec: {:<.1f})'
      ' (trn: {}/{}) (dev: {}/{})'
      ' (grad: avg:{} max:{} min:{}) (low probability predictions:{})'
      '\n{}\n\nResults:\n{}\n\n').format(
      epoch, config.name, trn_samples_per_sec,
      trn_num_valid_samples, trn_num_all_samples, dev_num_valid_samples, dev_num_all_samples,
      trn_mean_grad_norm, trn_max_grad_norm, trn_min_grad_norm, trn_num_unsafe_samples,
      config.format_compared(), format_epoch_results(epoch_results)))

  logger.info('END ' + title)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', help='unique name given to experiment')
  parser.add_argument('--device', help='device e.g. cpu, gpu0, gpu1, ...', default='cpu')
  parser.add_argument('--mode', help='Re-embedding variant', choices=['TR', 'TR_MLP', 'LM'])
  parser.add_argument('--lm_layer', help='LM layer to utilize when mode=LM', choices=['L1', 'L2', 'EMB', None], default=None)
  args = parser.parse_args()
  config = _get_config(args.name, args.device, args.mode, args.lm_layer)
  _main(config)

