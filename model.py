import logging

import numpy as np
import theano
import theano.tensor as tt

from theano.ifelse import ifelse
from theano.compile.nanguardmode import NanGuardMode

from base.utils import namer
from base.theano_utils import (floatX, cast_floatX, get_shared_floatX, gpu_int32,
  softmax_columns_with_mask, softmax_depths_with_mask, argmax_with_mask)
from base.model import BaseModel
from base.optimizer import AdamOptimizer


def get_model(config, data):
  logger = logging.getLogger()
  logger.info('Building model...')
  model = Model(config, data)
  total_size, params_sizes = model.get_param_sizes()
  sorted_param_sizes = sorted(params_sizes.items())
  logger.info('Done building model. Total number of parameters: {}. Sizes:\n{}\n'.format(total_size,
    '\n'.join('\t{:<8d} {:s}'.format(p_size, p_name) for p_name, p_size in sorted_param_sizes)))
  return model


class Model(BaseModel):
  def __init__(self, config, data):
    self.init_start(config)
    # cuda optimized batched dot product
    batched_dot = tt.batched_dot if config.device == 'cpu' else theano.sandbox.cuda.blas.batched_dot
    ff_dim = config.ff_dims[-1]

    ###################################################
    # Load all data onto GPU
    ###################################################

    word_emb_val = data.word_emb_data.word_emb                                              # (voc size, emb_dim)
    first_known_word = data.word_emb_data.first_known_word
    first_unknown_word = data.word_emb_data.first_unknown_word

    assert config.emb_dim == word_emb_val.shape[1]
    assert first_known_word == 2
    from reader import PLACEHOLDER_IDX
    assert PLACEHOLDER_IDX == 1

    word_zero_emb = get_shared_floatX(np.zeros((2, config.emb_dim)), 'word_zero_emb')       # (2, emb_dim)
    word_other_emb = get_shared_floatX(word_emb_val[first_known_word:], 'word_other_emb')   # (voc size - 2, emb_dim)
    word_emb = tt.concatenate([word_zero_emb, word_other_emb], axis=0)                      # (voc size, emb_dim)

    num_chars = len(data.char_data.char_to_idx)
    assert max(data.char_data.char_to_idx.values()) == num_chars - 1
    char_zero_emb = get_shared_floatX(np.zeros((1, config.char_dim)), 'char_zero_emb')                # (1, char_dim)
    char_other_emb = self.make_param('char_other_emb', (num_chars - 1, config.char_dim), 'uniform')   # (num_chars - 1, char_dim)
    char_emb = tt.concatenate([char_zero_emb, char_other_emb], axis=0)                                # (num_chars, char_dim)

    original_chars = gpu_int32('original_chars', data.char_data.original_chars)                       # (num originals + 1, max original len)

    max_original_len = max(data.char_data.original_lens)
    assert max_original_len == data.char_data.original_chars.shape[1]

    original_masks_val = np.zeros_like(data.char_data.original_chars, dtype=floatX)
    for original_idx, original_len in enumerate(data.char_data.original_lens):
      original_masks_val[original_idx, :original_len] = 1
    original_masks = get_shared_floatX(original_masks_val, 'original_masks')              # (num originals + 1, max original len)

    trn_ctxs, trn_ctx_masks, trn_ctx_lens, trn_qtns, trn_qtn_masks, trn_qtn_lens, trn_qtn_ctx_idxs, \
      trn_anss, trn_ans_stts, trn_ans_ends, \
      trn_ctx_originals, trn_qtn_originals, \
      trn_ctx_wdp_seq_ids, trn_qtn_wdp_seq_ids = \
        _gpu_dataset('trn', data.trn, config)

    dev_ctxs, dev_ctx_masks, dev_ctx_lens, dev_qtns, dev_qtn_masks, dev_qtn_lens, dev_qtn_ctx_idxs, \
      dev_anss, dev_ans_stts, dev_ans_ends, \
      dev_ctx_originals, dev_qtn_originals, \
      dev_ctx_wdp_seq_ids, dev_qtn_wdp_seq_ids = \
        _gpu_dataset('dev', data.dev, config)

    tst_ctxs, tst_ctx_masks, tst_ctx_lens, tst_qtns, tst_qtn_masks, tst_qtn_lens, tst_qtn_ctx_idxs, \
      tst_anss, tst_ans_stts, tst_ans_ends, \
      tst_ctx_originals, tst_qtn_originals, \
      tst_ctx_wdp_seq_ids, tst_qtn_wdp_seq_ids = \
        _gpu_dataset('tst', data.tst, config)


    # will be of size (num hs in cache, 1024)
    lm_ctx_cache = get_shared_floatX(np.zeros((1,1)), 'lm_ctx_cache')
    self.set_lm_ctx_cache = lambda new_ctx_cache: lm_ctx_cache.set_value(new_ctx_cache)
      
    # will be of size (num hs in cache, 1024)
    lm_qtn_cache = get_shared_floatX(np.zeros((1,1)), 'lm_qtn_cache')
    self.set_lm_qtn_cache = lambda new_qtn_cache: lm_qtn_cache.set_value(new_qtn_cache)


    ###################################################
    # Map input given to interface functions to an actual mini batch
    ###################################################

    masks_func = tt.matrix

    qtn_idxs = tt.ivector('qtn_idxs')                           # (batch_bize,)
    batch_size = qtn_idxs.size

    dataset_ctxs = tt.imatrix('dataset_ctxs')                   # (num contexts in dataset, max_p_len of dataset)
    dataset_ctx_masks = masks_func('dataset_ctx_masks')         # (num contexts in dataset, max_p_len of dataset)
    dataset_ctx_lens = tt.ivector('dataset_ctx_lens')           # (num contexts in dataset,)
    dataset_qtns = tt.imatrix('dataset_qtns')                   # (num questions in dataset, max_q_len of dataset)
    dataset_qtn_masks = masks_func('dataset_qtn_masks')         # (num questions in dataset, max_q_len of dataset)
    dataset_qtn_lens = tt.ivector('dataset_qtn_lens')           # (num questions in dataset,)
    dataset_qtn_ctx_idxs = tt.ivector('dataset_qtn_ctx_idxs')   # (num questions in dataset,)
    dataset_anss = tt.ivector('dataset_anss')                   # (num questions in dataset,)

    dataset_ans_stts = tt.ivector('dataset_ans_stts')           # (num questions in dataset,)
    dataset_ans_ends = tt.ivector('dataset_ans_ends')           # (num questions in dataset,)

    dataset_ctx_originals = tt.imatrix('dataset_ctx_originals') # (num contexts in dataset, max_p_len of dataset)
    dataset_qtn_originals = tt.imatrix('dataset_qtn_originals') # (num questions in dataset, max_q_len of dataset)

    dataset_ctx_wdp_seq_ids = tt.imatrix('dataset_ctx_wdp_seq_ids')   # (num contexts in dataset, max_p_len of dataset)
    dataset_qtn_wdp_seq_ids = tt.imatrix('dataset_qtn_wdp_seq_ids')   # (num questions in dataset, max_q_len of dataset)


    ctx_idxs = dataset_qtn_ctx_idxs[qtn_idxs]                   # (batch_size,)
    p_lens = dataset_ctx_lens[ctx_idxs]                         # (batch_size,)
    max_p_len = p_lens.max()
    p = dataset_ctxs[ctx_idxs][:,:max_p_len].T                  # (max_p_len, batch_size)
    p_mask = dataset_ctx_masks[ctx_idxs][:,:max_p_len].T        # (max_p_len, batch_size)
    float_p_mask = p_mask                                       # (max_p_len, batch_size)

    q_lens = dataset_qtn_lens[qtn_idxs]                         # (batch_size,)
    max_q_len = q_lens.max()
    q = dataset_qtns[qtn_idxs][:,:max_q_len].T                  # (max_q_len, batch_size)
    q_mask = dataset_qtn_masks[qtn_idxs][:,:max_q_len].T        # (max_q_len, batch_size)
    float_q_mask = q_mask                                       # (max_q_len, batch_size)

    a = dataset_anss[qtn_idxs]                                  # (batch_size,)
    a_stt = dataset_ans_stts[qtn_idxs]                          # (batch_size,)
    a_end = dataset_ans_ends[qtn_idxs]                          # (batch_size,)

    p_wdp_seq_ids = dataset_ctx_wdp_seq_ids[ctx_idxs][:,:max_p_len].T   # (max_p_len, batch_size)
    q_wdp_seq_ids = dataset_qtn_wdp_seq_ids[qtn_idxs][:,:max_q_len].T   # (max_q_len, batch_size)

    lm_ctx_cache_idxs = tt.ivector('lm_ctx_cache_idxs')   # (batch_size,)
    lm_qtn_cache_idxs = tt.ivector('lm_qtn_cache_idxs')   # (batch_size,)

    if config.mode == 'LM':
      lm_ctx_hs_list = []
      lm_qtn_hs_list = []
      for sample_i in range(config.batch_size):
        sample_i = tt.minimum(sample_i, batch_size - 1)
        lm_ctx_hs_i = lm_ctx_cache[lm_ctx_cache_idxs[sample_i]:lm_ctx_cache_idxs[sample_i]+max_p_len]   # (max_p_len, 1024)
        lm_qtn_hs_i = lm_qtn_cache[lm_qtn_cache_idxs[sample_i]:lm_qtn_cache_idxs[sample_i]+max_q_len]   # (max_q_len, 1024)
        lm_ctx_hs_list.append(lm_ctx_hs_i)
        lm_qtn_hs_list.append(lm_qtn_hs_i)
      lm_ctx_hs = tt.stack(lm_ctx_hs_list, axis=1)      # (max_p_len, config.batch_size, 1024)
      lm_qtn_hs = tt.stack(lm_qtn_hs_list, axis=1)      # (max_q_len, config.batch_size, 1024)
      lm_ctx_hs = lm_ctx_hs[:, :batch_size, :]
      lm_qtn_hs = lm_qtn_hs[:, :batch_size, :]
      lm_ctx_hs *= tt.shape_padright(float_p_mask)      # (max_p_len, batch_size, 1024)
      lm_qtn_hs *= tt.shape_padright(float_q_mask)      # (max_q_len, batch_size, 1024)
      lm_p = self.ff('lm_ctx_hs_ff',                                    # (max_p_len, batch_size, lm_dim)
        lm_ctx_hs, [1024, config.lm_dim], 'tanh', config.lm_drop, bias_init=config.default_bias_init)
      lm_q = self.ff('lm_qtn_hs_ff',                                    # (max_q_len, batch_size, lm_dim)
        lm_qtn_hs, [1024, config.lm_dim], 'tanh', config.lm_drop, bias_init=config.default_bias_init)

    p_originals = dataset_ctx_originals[ctx_idxs][:,:max_p_len]                   # (batch_size, max_p_len)
    q_originals = dataset_qtn_originals[qtn_idxs][:,:max_q_len]                   # (batch_size, max_q_len)

    p_originals_flat = p_originals.flatten()                                      # (batch_size*max_p_len,)
    q_originals_flat = q_originals.flatten()                                      # (batch_size*max_q_len,)

    p_original_chars_flat = original_chars[p_originals_flat]                      # (batch_size*max_p_len, max original len)
    q_original_chars_flat = original_chars[q_originals_flat]                      # (batch_size*max_q_len, max original len)

    p_char_mask_flat = original_masks[p_originals_flat]                           # (batch_size*max_p_len, max original len)
    q_char_mask_flat = original_masks[q_originals_flat]                           # (batch_size*max_q_len, max original len)

    p_char_emb_flat = char_emb[p_original_chars_flat]                             # (batch_size*max_p_len, max original len, char_dim)
    q_char_emb_flat = char_emb[q_original_chars_flat]                             # (batch_size*max_q_len, max original len, char_dim)

    p_char_conv_name = q_char_conv_name = 'char_conv'

    # (batch_size*max_p_len, char_feats)
    p_char_conv_flat, p_char_feats = self._char_conv(config, p_char_conv_name, p_char_emb_flat, p_char_mask_flat, max_original_len)
    # (batch_size*max_q_len, char_feats)
    q_char_conv_flat, q_char_feats = self._char_conv(config, q_char_conv_name, q_char_emb_flat, q_char_mask_flat, max_original_len)
    assert p_char_feats == q_char_feats
    char_feats = p_char_feats

    p_char_conv = p_char_conv_flat.reshape((batch_size, max_p_len, char_feats)).dimshuffle((1,0,2))   # (max_p_len, batch_size, char_feats)
    q_char_conv = q_char_conv_flat.reshape((batch_size, max_q_len, char_feats)).dimshuffle((1,0,2))   # (max_q_len, batch_size, char_feats)

    p_char_conv *= tt.shape_padright(float_p_mask)    # (max_p_len, batch_size, char_feats)
    q_char_conv *= tt.shape_padright(float_q_mask)    # (max_q_len, batch_size, char_feats)


    ###################################################
    # RaSoR
    ###################################################

    ############ embed words

    p_word_emb = word_emb[p]              # (max_p_len, batch_size, emb_dim)
    q_word_emb = word_emb[q]              # (max_q_len, batch_size, emb_dim)

    ############ word dropout

    if config.wdp_drop:

      wdp_seq_ids_zero_emb = get_shared_floatX(np.zeros((1, config.emb_dim)), 'wdp_seq_ids_zero_emb')                           # (1, emb_dim)

      wdp_seq_ids_bank_emb_shape = (100, config.emb_dim)
      wdp_seq_ids_bank_emb_val = self.get_param_init(wdp_seq_ids_bank_emb_shape, 0)
      wdp_seq_ids_bank_emb = get_shared_floatX(wdp_seq_ids_bank_emb_val, 'wdp_seq_ids_bank_emb')

      wdp_seq_ids_emb = tt.concatenate([wdp_seq_ids_zero_emb, wdp_seq_ids_bank_emb], axis=0)                                   # (bank_size+1, emb_dim)

      # max_wdp_seq_id is an int32 saying maximal word seq id in batch
      max_wdp_seq_id = tt.maximum(p_wdp_seq_ids.max(), q_wdp_seq_ids.max())

      num_wdp_seq_ids = max_wdp_seq_id + 1

      wdp_drop_rate = config.wdp_drop

      drop_word_mask = self._theano_rng.binomial(size=(batch_size, max_wdp_seq_id), p=wdp_drop_rate, n=1, dtype='int32')
      drop_word_zeros = tt.zeros((batch_size, 1), dtype='int32')
      drop_word_mask = tt.concatenate([drop_word_zeros, drop_word_mask], axis=1)    # (batch_size, num_wdp_seq_ids)

      cumsummed = tt.extra_ops.cumsum(drop_word_mask, axis=1)                       # (batch_size, num_wdp_seq_ids)
      dropped_idxs = drop_word_mask * cumsummed                                     # (batch_size, num_wdp_seq_ids)

      replaced_dropped_idxs = dropped_idxs
      replaced_dropped_idxs_flat = replaced_dropped_idxs.flatten()                # (batch_size*num_wdp_seq_ids,)

      seq_ids_shift = tt.shape_padleft(tt.arange(0, batch_size*num_wdp_seq_ids, num_wdp_seq_ids))   # (1, batch_size)
      p_wdp_seq_ids_shifted = p_wdp_seq_ids + seq_ids_shift               # (max_p_len, batch_size)
      q_wdp_seq_ids_shifted = q_wdp_seq_ids + seq_ids_shift               # (max_q_len, batch_size)

      p_wdp_new_seq_ids = replaced_dropped_idxs_flat[p_wdp_seq_ids_shifted]              # (max_p_len, batch_size)
      q_wdp_new_seq_ids = replaced_dropped_idxs_flat[q_wdp_seq_ids_shifted]              # (max_q_len, batch_size)

      p_dropped_inds = tt.gt(p_wdp_new_seq_ids, 0)     # (max_p_len, batch_size)
      q_dropped_inds = tt.gt(q_wdp_new_seq_ids, 0)     # (max_q_len, batch_size)

      p_dropped_inds = tt.shape_padright(p_dropped_inds)  # (max_p_len, batch_size, 1)
      q_dropped_inds = tt.shape_padright(q_dropped_inds)  # (max_q_len, batch_size, 1)

      p_dropped_emb = wdp_seq_ids_emb[p_wdp_new_seq_ids]                      # (max_p_len, batch_size, emb_dim)
      q_dropped_emb = wdp_seq_ids_emb[q_wdp_new_seq_ids]                      # (max_q_len, batch_size, emb_dim)


      trn_p_emb = (1 - p_dropped_inds) * p_word_emb  # (max_p_len, batch_size, emb_dim)
      trn_q_emb = (1 - q_dropped_inds) * q_word_emb  # (max_q_len, batch_size, emb_dim)
      evl_p_emb = (1 - wdp_drop_rate) * p_word_emb  # (max_p_len, batch_size, emb_dim)
      evl_q_emb = (1 - wdp_drop_rate) * q_word_emb  # (max_q_len, batch_size, emb_dim)
      anonymized_p_word_emb = ifelse(self._is_training, trn_p_emb, evl_p_emb)             # (max_p_len, batch_size, emb_dim)
      anonymized_q_word_emb = ifelse(self._is_training, trn_q_emb, evl_q_emb)             # (max_q_len, batch_size, emb_dim)

      trn_p_char_conv = (1 - p_dropped_inds) * p_char_conv    # (max_p_len, batch_size, char_feats)
      trn_q_char_conv = (1 - q_dropped_inds) * q_char_conv    # (max_q_len, batch_size, char_feats)
      evl_p_char_conv = (1 - wdp_drop_rate) * p_char_conv     # (max_p_len, batch_size, char_feats)
      evl_q_char_conv = (1 - wdp_drop_rate) * q_char_conv     # (max_q_len, batch_size, char_feats)
      anonymized_p_char_conv = ifelse(self._is_training, trn_p_char_conv, evl_p_char_conv)  # (max_p_len, batch_size, char_feats)
      anonymized_q_char_conv = ifelse(self._is_training, trn_q_char_conv, evl_q_char_conv)  # (max_q_len, batch_size, char_feats)

      if config.mode == 'LM':
        trn_lm_p = (1 - p_dropped_inds) * lm_p                  # (max_p_len, batch_size, lm_dim)
        trn_lm_q = (1 - q_dropped_inds) * lm_q                  # (max_q_len, batch_size, lm_dim)
        evl_lm_p = (1 - wdp_drop_rate) * lm_p                 # (max_p_len, batch_size, lm_dim)
        evl_lm_q = (1 - wdp_drop_rate) * lm_q                 # (max_q_len, batch_size, lm_dim)
        anonymized_lm_p = ifelse(self._is_training, trn_lm_p, evl_lm_p)  # (max_p_len, batch_size, lm_dim)
        anonymized_lm_q = ifelse(self._is_training, trn_lm_q, evl_lm_q)  # (max_q_len, batch_size, lm_dim)

    else:

      anonymized_p_word_emb = p_word_emb            # (max_p_len, batch_size, emb_dim)
      anonymized_q_word_emb = q_word_emb            # (max_q_len, batch_size, emb_dim)
      
      anonymized_p_char_conv = p_char_conv          # (max_p_len, batch_size, char_feats)
      anonymized_q_char_conv = q_char_conv          # (max_q_len, batch_size, char_feats)

      if config.mode == 'LM':
        anonymized_lm_p = lm_p                        # (max_p_len, batch_size, lm_dim)
        anonymized_lm_q = lm_q                        # (max_q_len, batch_size, lm_dim)


    if config.wn_tied:
      wn_p_name = wn_q_name = 'wn'
    else:
      wn_p_name = 'wn_p'
      wn_q_name = 'wn_q'

    if config.mode == 'LM':
      p_new_emb, p_clc_dim, p_gate_mean, p_gate_std = \
        self._reembed_lm(wn_p_name, config, anonymized_p_word_emb, float_p_mask, anonymized_p_char_conv, char_feats, anonymized_lm_p)
      q_new_emb, q_clc_dim, q_gate_mean, q_gate_std = \
        self._reembed_lm(wn_q_name, config, anonymized_q_word_emb, float_q_mask, anonymized_q_char_conv, char_feats, anonymized_lm_q)

    elif config.mode == 'TR':
      p_new_emb, p_clc_dim, p_gate_mean, p_gate_std = \
        self._reembed_tr_lstm(wn_p_name, config, anonymized_p_word_emb, float_p_mask, anonymized_p_char_conv, char_feats)
      q_new_emb, q_clc_dim, q_gate_mean, q_gate_std = \
        self._reembed_tr_lstm(wn_q_name, config, anonymized_q_word_emb, float_q_mask, anonymized_q_char_conv, char_feats)

    elif config.mode == 'TR_MLP':
      p_new_emb, p_clc_dim, p_gate_mean, p_gate_std = \
        self._reembed_tr_mlp(wn_p_name, config, anonymized_p_word_emb, float_p_mask, anonymized_p_char_conv, char_feats)
      q_new_emb, q_clc_dim, q_gate_mean, q_gate_std = \
        self._reembed_tr_mlp(wn_q_name, config, anonymized_q_word_emb, float_q_mask, anonymized_q_char_conv, char_feats)

    assert p_clc_dim == q_clc_dim
    clc_dim = p_clc_dim


    ############ q indep

    # (max_q_len, batch_size, 2*hidden_dim)
    q_indep_h = self.stacked_bi_lstm('q_indep_lstm', q_new_emb, float_q_mask,
      config.num_bilstm_layers, clc_dim, config.hidden_dim,
      config.lstm_drop_x, config.lstm_drop_h,
      couple_i_and_f = config.lstm_couple_i_and_f,
      learn_initial_state = config.lstm_learn_initial_state,
      tie_x_dropout = config.lstm_tie_x_dropout,
      sep_x_dropout = config.lstm_sep_x_dropout,
      sep_h_dropout = config.lstm_sep_h_dropout,
      w_init = config.lstm_w_init,
      u_init = config.lstm_u_init,
      forget_bias_init = config.lstm_forget_bias_init,
      other_bias_init = config.default_bias_init)

    # (max_q_len, batch_size, ff_dim)     # contains junk where masked
    q_indep_ff = self.ff('q_indep_ff', q_indep_h, [2*config.hidden_dim] + config.ff_dims,
      'relu', config.ff_drop_x, bias_init=config.default_bias_init)
    w_q = self.make_param('w_q', (ff_dim,), 'uniform')
    q_indep_scores = tt.dot(q_indep_ff, w_q)                                    # (max_q_len, batch_size)
    q_indep_weights = softmax_columns_with_mask(q_indep_scores, float_q_mask)   # (max_q_len, batch_size)
    q_indep = tt.sum(tt.shape_padright(q_indep_weights) * q_indep_h, axis=0)    # (batch_size, 2*hidden_dim)
    
    ############ q aligned

    if config.q_aln_ff_tie:
      q_align_ff_p_name = q_align_ff_q_name = 'q_align_ff'
    else:
      q_align_ff_p_name = 'q_align_ff_p'
      q_align_ff_q_name = 'q_align_ff_q'
    # (max_p_len, batch_size, ff_dim)     # contains junk where masked
    q_align_ff_p = self.ff(q_align_ff_p_name, p_new_emb, [clc_dim] + config.ff_dims,
      'relu', config.ff_drop_x, bias_init=config.default_bias_init)
    # (max_q_len, batch_size, ff_dim)     # contains junk where masked
    q_align_ff_q = self.ff(q_align_ff_q_name, q_new_emb, [clc_dim] + config.ff_dims,
      'relu', config.ff_drop_x, bias_init=config.default_bias_init)

    # http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_dot
    # https://groups.google.com/d/msg/theano-users/yBh27AJGq2E/vweiLoXADQAJ
    q_align_ff_p_shuffled = q_align_ff_p.dimshuffle((1,0,2))                    # (batch_size, max_p_len, ff_dim)
    q_align_ff_q_shuffled = q_align_ff_q.dimshuffle((1,2,0))                    # (batch_size, ff_dim, max_q_len)
    q_align_scores = batched_dot(q_align_ff_p_shuffled, q_align_ff_q_shuffled)  # (batch_size, max_p_len, max_q_len)

    p_mask_shuffled = float_p_mask.dimshuffle((1,0,'x'))                        # (batch_size, max_p_len, 1)
    q_mask_shuffled = float_q_mask.dimshuffle((1,'x',0))                        # (batch_size, 1, max_q_len)
    pq_mask = p_mask_shuffled * q_mask_shuffled                                 # (batch_size, max_p_len, max_q_len)

    q_align_weights = softmax_depths_with_mask(q_align_scores, pq_mask)         # (batch_size, max_p_len, max_q_len)
    q_emb_shuffled = q_new_emb.dimshuffle((1,0,2))                              # (batch_size, max_q_len, clc_dim)
    q_align = batched_dot(q_align_weights, q_emb_shuffled)                      # (batch_size, max_p_len, clc_dim)
    
    ############ p star

    q_align_shuffled = q_align.dimshuffle((1,0,2))            # (max_p_len, batch_size, clc_dim)
    q_indep_repeated = tt.extra_ops.repeat(                   # (max_p_len, batch_size, 2*hidden_dim)
      tt.shape_padleft(q_indep), max_p_len, axis=0)

    p_star_comps = [p_new_emb, q_align_shuffled, q_indep_repeated]
    p_star_dim = 2*clc_dim+2*config.hidden_dim
    p_star = tt.concatenate(p_star_comps, axis=2)             # (max_p_len, batch_size, p_star_dim)
      

    ############ passage-level bi-lstm

    # (max_p_len, batch_size, 2*hidden_dim)
    p_level_h = self.stacked_bi_lstm('p_level_lstm', p_star, float_p_mask,
      config.num_bilstm_layers, p_star_dim, config.hidden_dim,
      config.lstm_drop_x, config.lstm_drop_h,
      couple_i_and_f = config.lstm_couple_i_and_f,
      learn_initial_state = config.lstm_learn_initial_state,
      tie_x_dropout = config.lstm_tie_x_dropout,
      sep_x_dropout = config.lstm_sep_x_dropout,
      sep_h_dropout = config.lstm_sep_h_dropout,
      w_init = config.lstm_w_init,
      u_init = config.lstm_u_init,
      forget_bias_init = config.lstm_forget_bias_init,
      other_bias_init = config.default_bias_init)

    if config.sep_stt_end_drop:
      p_level_h_for_stt = self.dropout(p_level_h, config.ff_drop_x)
      p_level_h_for_end = self.dropout(p_level_h, config.ff_drop_x)
    else:
      p_level_h_for_stt = p_level_h_for_end = self.dropout(p_level_h, config.ff_drop_x)

    # Having a single FF hidden layer allows to compute the FF over the concatenation
    # of span-start-hidden-state and span-end-hidden-state by operating the linear transformation
    # separately over each rather than over their concatenations.
    assert len(config.ff_dims) == 1

    if config.objective in ['span_multinomial', 'span_binary']:

      ############ scores

      p_stt_lin = self.linear(                              # (max_p_len, batch_size, ff_dim)
        'p_stt_lin', p_level_h_for_stt, 2*config.hidden_dim, ff_dim, bias_init=config.default_bias_init)
      p_end_lin = self.linear(                              # (max_p_len, batch_size, ff_dim)
        'p_end_lin', p_level_h_for_end, 2*config.hidden_dim, ff_dim, with_bias=False)

      # (batch_size, max_p_len*max_ans_len, ff_dim), (batch_size, max_p_len*max_ans_len)
      span_lin_reshaped, span_masks_reshaped = _span_sums(
        p_stt_lin, p_end_lin, p_lens, max_p_len, batch_size, ff_dim, config.max_ans_len)

      span_ff_reshaped = tt.nnet.relu(span_lin_reshaped)    # (batch_size, max_p_len*max_ans_len, ff_dim)
      w_a = self.make_param('w_a', (ff_dim,), 'uniform')
      span_scores_reshaped = tt.dot(span_ff_reshaped, w_a)  # (batch_size, max_p_len*max_ans_len)

      ############ classification

      classification_func = _span_multinomial_classification if config.objective == 'span_multinomial' else \
        _span_binary_classification

      # (batch_size,), (batch_size), (batch_size,), (batch_size, max_p_len*max_ans_len), (batch_size,)
      xents, accs, a_hats, probs_reshaped, ents = classification_func(span_scores_reshaped, span_masks_reshaped, a)

      if not config.loss_min_prob:
        loss = xents.mean()
        num_unsafe_samples = tt.as_tensor_variable(-1)
      else:
        gold_probs = tt.exp(-xents)                           # (batch_size,)
        safe_inds = tt.gt(gold_probs, config.loss_min_prob)   # (batch_size,)
        num_safe_samples = safe_inds.sum()
        loss = tt.sum(safe_inds * xents) / cast_floatX(num_safe_samples)
        num_unsafe_samples = batch_size - num_safe_samples

      acc = accs.mean()

      # (batch_size,), (batch_size)
      ans_hat_start_word_idxs, ans_hat_end_word_idxs = _tt_ans_idx_to_ans_word_idxs(a_hats, config.max_ans_len)
      probs = probs_reshaped.reshape((batch_size, max_p_len, config.max_ans_len))

      [p_emb_grads, q_emb_grads] = theano.grad(loss, [p_word_emb, q_word_emb])  # (max_p_len, batch_size, emb_dim)
      p_emb_grad_norms = tt.sqrt(tt.sum(p_emb_grads ** 2, axis=2)).T            # (batch_size, max_p_len)
      q_emb_grad_norms = tt.sqrt(tt.sum(q_emb_grads ** 2, axis=2)).T            # (batch_size, max_q_len)

    elif config.objective == 'span_endpoints':

      ############ scores

      # note that dropout was already applied when assigning to p_level_h_for_stt/end
      p_stt_ff = self.ff(                                                 # (max_p_len, batch_size, ff_dim)
        'p_stt_ff', p_level_h_for_stt, [2*config.hidden_dim] + [ff_dim],
        'relu', dropout_ps=None, bias_init=config.default_bias_init)
      p_end_ff = self.ff(                                                 # (max_p_len, batch_size, ff_dim)
        'p_end_ff', p_level_h_for_end, [2*config.hidden_dim] + [ff_dim],
        'relu', dropout_ps=None, bias_init=config.default_bias_init)

      w_a_stt = self.make_param('w_a_stt', (ff_dim,), 'uniform')
      w_a_end = self.make_param('w_a_end', (ff_dim,), 'uniform')
      word_stt_scores = tt.dot(p_stt_ff, w_a_stt)                         # (max_p_len, batch_size)
      word_end_scores = tt.dot(p_end_ff, w_a_end)                         # (max_p_len, batch_size)

      ############ classification

      stt_log_probs, stt_xents = _word_multinomial_classification(        # (batch_size, max_p_len), (batch_size,)
        word_stt_scores.T, float_p_mask.T, a_stt)
      end_log_probs, end_xents = _word_multinomial_classification(        # (batch_size, max_p_len), (batch_size,)
        word_end_scores.T, float_p_mask.T, a_end)
      xents = stt_xents + end_xents                                       # (batch_size,)
      loss = xents.mean()

      if not config.loss_min_prob:
        loss = xents.mean()
        num_unsafe_samples = tt.as_tensor_variable(-1)
      else:
        gold_probs = tt.exp(-xents)                           # (batch_size,)
        safe_inds = tt.gt(gold_probs, config.loss_min_prob)   # (batch_size,)
        num_safe_samples = safe_inds.sum()
        loss = tt.sum(safe_inds * xents) / cast_floatX(num_safe_samples)
        num_unsafe_samples = batch_size - num_safe_samples

      [p_emb_grads, q_emb_grads] = theano.grad(loss, [p_word_emb, q_word_emb])  # (max_p_len, batch_size, emb_dim)
      p_emb_grad_norms = tt.sqrt(tt.sum(p_emb_grads ** 2, axis=2)).T            # (batch_size, max_p_len)
      q_emb_grad_norms = tt.sqrt(tt.sum(q_emb_grads ** 2, axis=2)).T            # (batch_size, max_q_len)

      ############ finding highest P(span) = P(span start) * P(span end)

      end_log_probs = end_log_probs.dimshuffle((1,0,'x'))                 # (max_p_len, batch_size, 1)
      stt_log_probs = stt_log_probs.dimshuffle((1,0,'x'))                 # (max_p_len, batch_size, 1)
      # (batch_size, max_p_len*max_ans_len, 1), (batch_size, max_p_len*max_ans_len)
      span_log_probs_reshaped, span_masks_reshaped = _span_sums(
        stt_log_probs, end_log_probs, p_lens, max_p_len, batch_size, 1, config.max_ans_len)

      span_log_probs_reshaped = span_log_probs_reshaped.reshape(          # (batch_size, max_p_len*max_ans_len)
        (batch_size, max_p_len*config.max_ans_len))
      a_hats = argmax_with_mask(                                          # (batch_size,)
        span_log_probs_reshaped, span_masks_reshaped)
      accs = cast_floatX(tt.eq(a_hats, a))                                # (batch_size,)
      acc = accs.mean()

      # (batch_size,), (batch_size)
      ans_hat_start_word_idxs, ans_hat_end_word_idxs = _tt_ans_idx_to_ans_word_idxs(a_hats, config.max_ans_len)

    else:
      raise AssertionError('unsupported objective')

    ############ optimization

    opt = AdamOptimizer(config, loss, self._params.values())
    updates = opt.get_updates()
    global_grad_norm = opt.get_global_grad_norm()
    self.get_lr_value = lambda : opt.get_lr_value()

    ############ interface

    trn_givens = {
      self._is_training : np.int32(1), 
      dataset_ctxs: trn_ctxs,
      dataset_ctx_masks: trn_ctx_masks,
      dataset_ctx_lens: trn_ctx_lens,
      dataset_qtns: trn_qtns,
      dataset_qtn_masks: trn_qtn_masks,
      dataset_qtn_lens: trn_qtn_lens,
      dataset_qtn_ctx_idxs: trn_qtn_ctx_idxs,
      dataset_anss: trn_anss,
      dataset_ans_stts: trn_ans_stts,
      dataset_ans_ends: trn_ans_ends,
      dataset_ctx_originals: trn_ctx_originals,
      dataset_qtn_originals: trn_qtn_originals,
      dataset_ctx_wdp_seq_ids: trn_ctx_wdp_seq_ids,
      dataset_qtn_wdp_seq_ids: trn_qtn_wdp_seq_ids}

    eval_trn_givens = {
      self._is_training : np.int32(0), 
      dataset_ctxs: trn_ctxs,
      dataset_ctx_masks: trn_ctx_masks,
      dataset_ctx_lens: trn_ctx_lens,
      dataset_qtns: trn_qtns,
      dataset_qtn_masks: trn_qtn_masks,
      dataset_qtn_lens: trn_qtn_lens,
      dataset_qtn_ctx_idxs: trn_qtn_ctx_idxs,
      dataset_anss: trn_anss,
      dataset_ans_stts: trn_ans_stts,
      dataset_ans_ends: trn_ans_ends,
      dataset_ctx_originals: trn_ctx_originals,
      dataset_qtn_originals: trn_qtn_originals,
      dataset_ctx_wdp_seq_ids: trn_ctx_wdp_seq_ids,
      dataset_qtn_wdp_seq_ids: trn_qtn_wdp_seq_ids}

    dev_givens = {
      self._is_training : np.int32(0), 
      dataset_ctxs: dev_ctxs,
      dataset_ctx_masks: dev_ctx_masks,
      dataset_ctx_lens: dev_ctx_lens,
      dataset_qtns: dev_qtns,
      dataset_qtn_masks: dev_qtn_masks,
      dataset_qtn_lens: dev_qtn_lens,
      dataset_qtn_ctx_idxs: dev_qtn_ctx_idxs,
      dataset_anss: dev_anss,
      dataset_ans_stts: dev_ans_stts,
      dataset_ans_ends: dev_ans_ends,
      dataset_ctx_originals: dev_ctx_originals,
      dataset_qtn_originals: dev_qtn_originals,
      dataset_ctx_wdp_seq_ids: dev_ctx_wdp_seq_ids,
      dataset_qtn_wdp_seq_ids: dev_qtn_wdp_seq_ids}

    tst_givens = {
      self._is_training : np.int32(0), 
      dataset_ctxs: tst_ctxs,
      dataset_ctx_masks: tst_ctx_masks,
      dataset_ctx_lens: tst_ctx_lens,
      dataset_qtns: tst_qtns,
      dataset_qtn_masks: tst_qtn_masks,
      dataset_qtn_lens: tst_qtn_lens,
      dataset_qtn_ctx_idxs: tst_qtn_ctx_idxs,
      dataset_ctx_originals: tst_ctx_originals,
      dataset_qtn_originals: tst_qtn_originals,
      dataset_ctx_wdp_seq_ids: tst_ctx_wdp_seq_ids,
      dataset_qtn_wdp_seq_ids: tst_qtn_wdp_seq_ids}

    self.train = theano.function(
      [qtn_idxs, lm_qtn_cache_idxs, lm_ctx_cache_idxs],
      [loss, acc, global_grad_norm, num_unsafe_samples],
      givens = trn_givens,
      updates = updates,
      on_unused_input = 'ignore')

    self.eval_dev = theano.function(
      [qtn_idxs, lm_qtn_cache_idxs, lm_ctx_cache_idxs],
      [loss, acc, ans_hat_start_word_idxs, ans_hat_end_word_idxs],
      givens = dev_givens,
      updates = None,
      on_unused_input = 'ignore')

    self.eval_tst = theano.function(
      [qtn_idxs, lm_qtn_cache_idxs, lm_ctx_cache_idxs],
      [ans_hat_start_word_idxs, ans_hat_end_word_idxs],
      givens = tst_givens,
      updates = None,
      on_unused_input = 'ignore')

    # __init__ end


  def _char_conv(self, config, name, char_emb_flat, char_mask_flat, max_original_len):
    # char_emb_flat     (batch_size*seq_len, max original len, char_dim)
    # char_mask_flat    (batch_size*seq_len, max original len)

    inp = self.dropout(char_emb_flat, config.char_drop)
    inp = inp.dimshuffle((0,'x',2,1))       # (batch_size*seq_len, input channels = 1, input rows = char_dim, input columns = max original len)

    max_window_size = max(config.char_win_sizes)
    window_filts = []
    for window_size in config.char_win_sizes:
      assert window_size % 2 == 1
      window_filt = self.make_param(name + '_window_size_' + str(window_size), (config.char_feats, 1, config.char_dim, window_size), 'uniform')
      if window_size < max_window_size:
        window_num_pad_columns = (max_window_size - window_size) / 2
        window_zeros = tt.zeros((config.char_feats, 1, config.char_dim, window_num_pad_columns))
        window_filt = tt.concatenate((window_zeros, window_filt, window_zeros), axis=3) # (window_num_features, 1, char_dim, max_window_size)
      window_filts.append(window_filt)

    num_features = len(config.char_win_sizes) * config.char_feats
    filt = tt.concatenate(window_filts, axis=0)     # (output channels = num_features, input channels = 1, filter rows = char_dim, filter cols = max_window_size)
    filter_shape = (num_features, 1, config.char_dim, max_window_size)

    num_pad_columns = (max_window_size - 1) / 2
    # http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv2d
    # (batch_size*seq_len, output channels = num_features, output rows = 1, output columns = max original len)
    output = tt.nnet.conv2d(inp, filt,
      input_shape = (None, 1, config.char_dim, max_original_len),                   # specifying None for unknown at compile-time
      filter_shape = filter_shape,
      border_mode = (0, num_pad_columns),
      filter_flip = False)

    preact = tt.addbroadcast(output, 2)
    # using dimshuffle to (also) remove output rows singleton dimension
    preact = preact.dimshuffle((0,3,1))                         # (batch_size*seq_len, max original len, num_features)

    bias = self.make_param(name + '_b', (num_features), 'uniform')
    preact += bias

    act_func = tt.nnet.relu
    act = act_func(preact)                                      # (batch_size*seq_len, max original len, num_features)

    char_mask_flat_shuffled = tt.shape_padright(char_mask_flat)   # (batch_size*seq_len, max original len, 1)
    act = char_mask_flat_shuffled * act - 1e6 * (1 - char_mask_flat_shuffled)

    pooled = tt.max(act, axis=1)                                # (batch_size*seq_len, num_features)
    return pooled, num_features


  def _reembed_lm(self, name, config, seq_emb, seq_mask, char_conv, char_feats, lm_h):
    # seq_emb       (max_len, batch_size, emb_dim)
    # seq_mask      (max_len, batch_size)
    # char_conv     (max_len, batch_size, char_feats)
    # lm_h          (max_len, batch_size, lm_dim)
    wn_namer = namer(name)
    lstm_drop = config.lstm_drop_x

    lower_h_inp = tt.concatenate([seq_emb, char_conv], axis=2)    # (max_p_len, batch_size, emb_dim + char_feats)
    lower_h_inp_dim = config.emb_dim + char_feats

    gate_dim = config.emb_dim

    # (max_len, batch_size, 2*hidden_dim)
    lower_h = self.stacked_bi_lstm(wn_namer('lower_lstm'), lower_h_inp, seq_mask,
      config.wn_num_lyrs, lower_h_inp_dim, config.hidden_dim,
      lstm_drop, config.lstm_drop_h,
      couple_i_and_f = config.lstm_couple_i_and_f,
      learn_initial_state = config.lstm_learn_initial_state,
      tie_x_dropout = config.lstm_tie_x_dropout,
      sep_x_dropout = config.lstm_sep_x_dropout,
      sep_h_dropout = config.lstm_sep_h_dropout,
      w_init = config.lstm_w_init,
      u_init = config.lstm_u_init,
      forget_bias_init = config.lstm_forget_bias_init,
      other_bias_init = config.default_bias_init)

    inp = tt.concatenate([lower_h_inp, lower_h, lm_h], axis=2)  # (max_len, batch_size, lower_h_inp_dim+2*hidden_dim+lm_dim)

    gate = self.ff(wn_namer('lower_gate'),                      # (max_len, batch_size, gate_dim)
      inp, [lower_h_inp_dim+2*config.hidden_dim+config.lm_dim, gate_dim], 'sigmoid', config.ff_drop_x, bias_init=config.default_bias_init)

    cand = self.ff(wn_namer('lower_cand'),                      # (max_len, batch_size, gate_dim)
      inp, [lower_h_inp_dim+2*config.hidden_dim+config.lm_dim, gate_dim], 'tanh', config.ff_drop_x, bias_init=config.default_bias_init)

    new_seq = gate * seq_emb + (1-gate) * cand                  # (max_len, batch_size, emb_dim)
    out_dim = config.emb_dim

    gate_mean = tt.mean(gate, axis=2).T                   # (batch_size, max_len)
    gate_std = tt.std(gate, axis=2).T                     # (batch_size, max_len)

    return new_seq, out_dim, gate_mean, gate_std


  def _reembed_tr_lstm(self, name, config, seq_emb, seq_mask, char_conv, char_feats):
    # seq_emb       (max_len, batch_size, emb_dim)
    # seq_mask      (max_len, batch_size)
    # char_conv     (max_len, batch_size, char_feats)
    wn_namer = namer(name)
    lstm_drop = config.lstm_drop_x

    lower_h_inp = tt.concatenate([seq_emb, char_conv], axis=2)    # (max_p_len, batch_size, emb_dim + char_feats)
    lower_h_inp_dim = config.emb_dim + char_feats
    gate_dim = config.emb_dim

    # (max_len, batch_size, 2*hidden_dim)
    lower_h = self.stacked_bi_lstm(wn_namer('lower_lstm'), lower_h_inp, seq_mask,
      config.wn_num_lyrs, lower_h_inp_dim, config.hidden_dim,
      lstm_drop, config.lstm_drop_h,
      couple_i_and_f = config.lstm_couple_i_and_f,
      learn_initial_state = config.lstm_learn_initial_state,
      tie_x_dropout = config.lstm_tie_x_dropout,
      sep_x_dropout = config.lstm_sep_x_dropout,
      sep_h_dropout = config.lstm_sep_h_dropout,
      w_init = config.lstm_w_init,
      u_init = config.lstm_u_init,
      forget_bias_init = config.lstm_forget_bias_init,
      other_bias_init = config.default_bias_init)

    inp = tt.concatenate([lower_h_inp, lower_h], axis=2)  # (max_len, batch_size, lower_h_inp_dim+2*hidden_dim)

    gate = self.ff(wn_namer('lower_gate'),                # (max_len, batch_size, gate_dim)
      inp, [lower_h_inp_dim+2*config.hidden_dim, gate_dim], 'sigmoid', config.ff_drop_x, bias_init=config.default_bias_init)

    cand = self.ff(wn_namer('lower_cand'),                # (max_len, batch_size, gate_dim)
      inp, [lower_h_inp_dim+2*config.hidden_dim, gate_dim], 'tanh', config.ff_drop_x, bias_init=config.default_bias_init)

    new_seq = gate * seq_emb + (1-gate) * cand                  # (max_len, batch_size, emb_dim)
    out_dim = config.emb_dim

    gate_mean = tt.mean(gate, axis=2).T                   # (batch_size, max_len)
    gate_std = tt.std(gate, axis=2).T                     # (batch_size, max_len)

    return new_seq, out_dim, gate_mean, gate_std


  def _reembed_tr_mlp(self, name, config, seq_emb, seq_mask, char_conv, char_feats):
    # seq_emb       (max_len, batch_size, emb_dim)
    # seq_mask      (max_len, batch_size)
    # char_conv     (max_len, batch_size, char_feats)
    wn_namer = namer(name)

    ff_inp = tt.concatenate([seq_emb, char_conv], axis=2) # (max_p_len, batch_size, emb_dim + char_feats)
    ff_inp_dim = config.emb_dim + char_feats

    lower_h = self.ff(wn_namer('lower_h_ff'),             # (max_len, batch_size, 2*hidden_dim)
      ff_inp, config.wn_ff_dims, 'tanh', config.ff_drop_x, bias_init=config.default_bias_init)

    inp = tt.concatenate([ff_inp, lower_h], axis=2)       # (max_len, batch_size, ff_inp_dim+ff_inp_dim)

    gate_dim = out_dim = config.emb_dim
    gate = self.ff(wn_namer('lower_gate'),                # (max_len, batch_size, gate_dim)
      inp, [2*ff_inp_dim, gate_dim], 'sigmoid', config.ff_drop_x, bias_init=config.default_bias_init)

    cand = self.ff(wn_namer('lower_cand'),                # (max_len, batch_size, gate_dim)
      inp, [2*ff_inp_dim, gate_dim], 'tanh', config.ff_drop_x, bias_init=config.default_bias_init)

    new_seq = gate * seq_emb + (1-gate) * cand            # (max_len, batch_size, emb_dim)
    out_dim = config.emb_dim

    gate_mean = tt.mean(gate, axis=2).T                   # (batch_size, max_len)
    gate_std = tt.std(gate, axis=2).T                     # (batch_size, max_len)

    return new_seq, out_dim, gate_mean, gate_std

# Model end


def _span_sums(stt, end, p_lens, max_p_len, batch_size, dim, max_ans_len):
  # Sum of every start element and corresponding max_ans_len end elements.
  #
  # stt     (max_p_len, batch_size, dim)
  # end     (max_p_len, batch_size, dim)
  # p_lens  (batch_size,)
  max_ans_len_range = tt.shape_padleft(tt.arange(max_ans_len))          # (1, max_ans_len)
  offsets = tt.shape_padright(tt.arange(max_p_len))                     # (max_p_len, 1)
  end_idxs = max_ans_len_range + offsets                                # (max_p_len, max_ans_len)
  end_idxs_flat = end_idxs.flatten()                                    # (max_p_len*max_ans_len,)

  end_padded = tt.concatenate(                                          # (max_p_len+max_ans_len-1, batch_size, dim)
    [end, tt.zeros((max_ans_len-1, batch_size, dim))], axis=0)    
  end_structured = end_padded[end_idxs_flat]                            # (max_p_len*max_ans_len, batch_size, dim)
  end_structured = end_structured.reshape(                              # (max_p_len, max_ans_len, batch_size, dim)
    (max_p_len, max_ans_len, batch_size, dim))
  stt_shuffled = stt.dimshuffle((0,'x',1,2))                            # (max_p_len, 1, batch_size, dim)

  span_sums = stt_shuffled + end_structured                             # (max_p_len, max_ans_len, batch_size, dim)
  span_sums_reshaped = span_sums.dimshuffle((2,0,1,3)).reshape(         # (batch_size, max_p_len*max_ans_len, dim)
    (batch_size, max_p_len*max_ans_len, dim))

  p_lens_shuffled = tt.shape_padright(p_lens)                           # (batch_size, 1)
  end_idxs_flat_shuffled = tt.shape_padleft(end_idxs_flat)              # (1, max_p_len*max_ans_len)

  span_masks_reshaped = tt.lt(end_idxs_flat_shuffled, p_lens_shuffled)  # (batch_size, max_p_len*max_ans_len)
  span_masks_reshaped = cast_floatX(span_masks_reshaped)

  # (batch_size, max_p_len*max_ans_len, dim), (batch_size, max_p_len*max_ans_len)
  return span_sums_reshaped, span_masks_reshaped


###################################################
# Variable-length data to GPU matrices and masks
###################################################

def _gpu_dataset(name, dataset, config):
  if dataset:
    ds_vec = dataset.vectorized
    ctxs, ctx_masks, ctx_lens = _gpu_sequences(name + '_ctxs', ds_vec.ctxs, ds_vec.ctx_lens, config)
    qtns, qtn_masks, qtn_lens = _gpu_sequences(name + '_qtns', ds_vec.qtns, ds_vec.qtn_lens, config)

    qtn_ctx_idxs = gpu_int32(name + '_qtn_ctx_idxs', ds_vec.qtn_ctx_idxs)
    anss, ans_stts, ans_ends = _gpu_answers(name, ds_vec.anss, config.max_ans_len)

    ctx_originals = gpu_int32(name + '_ctx_originals', ds_vec.ctx_originals)
    qtn_originals = gpu_int32(name + '_qtn_originals', ds_vec.qtn_originals)

    ctx_wdp_seq_ids = gpu_int32(name + '_ctx_wdp_seq_ids', ds_vec.ctx_wdp_seq_ids)
    qtn_wdp_seq_ids = gpu_int32(name + '_qtn_wdp_seq_ids', ds_vec.qtn_wdp_seq_ids)

  else:
    empty_matrix_int = gpu_int32(name + '_empty_matrix_int', np.zeros((1,1), dtype=np.int32))
    empty_matrix_float = get_shared_floatX(np.zeros((1,1), dtype=np.int32), name + '_empty_matrix_float')
    empty_vector_int = gpu_int32(name + '_empty_vector_int', np.zeros(1, dtype=np.int32))
    empty_vector_float = get_shared_floatX(np.zeros(1, dtype=np.int32), name + '_empty_vector_float')

    ctxs = qtns = ctx_originals = qtn_originals = ctx_wdp_seq_ids = qtn_wdp_seq_ids = empty_matrix_int
    ctx_masks = qtn_masks = empty_matrix_float
    ctx_lens = qtn_lens = qtn_ctx_idxs = anss = ans_stts = ans_ends = \
      empty_vector_int

  return (ctxs, ctx_masks, ctx_lens, qtns, qtn_masks, qtn_lens, qtn_ctx_idxs, anss, ans_stts, ans_ends,
    ctx_originals, qtn_originals,
    ctx_wdp_seq_ids, qtn_wdp_seq_ids)


def _gpu_sequences(name, seqs_val, lens, config):
  # print name + ' seqs_val shape:' + str(seqs_val.shape)
  assert seqs_val.dtype == lens.dtype == np.int32
  num_samples, max_seq_len = seqs_val.shape
  assert len(lens) == num_samples
  assert max(lens) == max_seq_len
  gpu_seqs = gpu_int32(name, seqs_val)
  seq_masks_val = np.zeros((num_samples, max_seq_len), dtype=np.int32)
  for i, sample_len in enumerate(lens):
    seq_masks_val[i,:sample_len] = 1
    assert np.all(seqs_val[i,:sample_len] > 0)
    assert np.all(seqs_val[i,sample_len:] == 0)
  gpu_seq_masks = get_shared_floatX(seq_masks_val, name + '_masks')
  gpu_lens = gpu_int32(name + '_lens', lens)
  return gpu_seqs, gpu_seq_masks, gpu_lens


def _np_ans_word_idxs_to_ans_idx(ans_start_word_idx, ans_end_word_idx, max_ans_len):
  # all arguments are concrete ints
  assert ans_end_word_idx - ans_start_word_idx + 1 <= max_ans_len
  return ans_start_word_idx * max_ans_len + (ans_end_word_idx - ans_start_word_idx)


def _tt_ans_idx_to_ans_word_idxs(ans_idx, max_ans_len):
  # ans_idx theano int32 variable (batch_size,)
  # max_ans_len concrete int
  ans_start_word_idx = ans_idx // max_ans_len
  ans_end_word_idx = ans_start_word_idx + ans_idx % max_ans_len
  return ans_start_word_idx, ans_end_word_idx


def _gpu_answers(name, anss, max_ans_len):
  assert anss.dtype == np.int32
  assert anss.shape[1] == 2
  anss_val = np.array([_np_ans_word_idxs_to_ans_idx(ans_stt, ans_end, max_ans_len) for \
    ans_stt, ans_end in anss], dtype=np.int32)
  ans_stts_val = anss[:,0]
  ans_ends_val = anss[:,1]

  gpu_anss = gpu_int32(name + '_anss', anss_val)
  gpu_ans_stts = gpu_int32(name + '_ans_stts', ans_stts_val)
  gpu_ans_ends = gpu_int32(name + '_ans_ends', ans_ends_val)
  return gpu_anss, gpu_ans_stts, gpu_ans_ends


###################################################
# Classification
###################################################

def _span_multinomial_classification(x, x_mask, y):
  # x       float32 (batch_size, num_classes)   scores i.e. logits
  # x_mask  int32   (batch_size, num_classes)   score masks (each sample has a variable number of classes)
  # y       int32   (batch_size,)               target classes i.e. ground truth answers (given as class indices)
  assert x.ndim == x_mask.ndim == 2
  assert y.ndim == 1

  # substracting min needed since all non masked-out elements of a row may be negative.
  x *= x_mask
  x -= x.min(axis=1, keepdims=True)             # (batch_size, num_classes)
  x *= x_mask                                   # (batch_size, num_classes)
  y_hats = tt.argmax(x, axis=1)                 # (batch_size,)
  accs = cast_floatX(tt.eq(y_hats, y))          # (batch_size,)

  x -= x.max(axis=1, keepdims=True)             # (batch_size, num_classes)
  x *= x_mask                                   # (batch_size, num_classes)
  exp_x = tt.exp(x)                             # (batch_size, num_classes)
  exp_x *= x_mask                               # (batch_size, num_classes)

  sum_exp_x = exp_x.sum(axis=1)                 # (batch_size,)
  log_sum_exp_x = tt.log(sum_exp_x)             # (batch_size,)

  x_star = x[tt.arange(x.shape[0]), y]          # (batch_size,)
  xents = log_sum_exp_x - x_star                # (batch_size,)

  # probs and ents used only for analysis
  probs = exp_x / tt.shape_padright(sum_exp_x)      # (batch_size, num_classes)
  log_probs = x - tt.shape_padright(log_sum_exp_x)  # (batch_size, num_classes)
  ents = tt.sum(-probs * log_probs, axis=1)         # (batch_size,)

  return xents, accs, y_hats, probs, ents


def _span_binary_classification(x, x_mask, y):
  # x       float32 (batch_size, num_classes)   scores i.e. logits
  # x_mask  int32   (batch_size, num_classes)   score masks (each sample has a variable number of classes)
  # y       int32   (batch_size,)               target classes i.e. ground truth answers (given as class indices)
  assert x.ndim == x_mask.ndim == 2
  assert y.ndim == 1

  # placing min in masked-out elements needed since all non masked-out elements of a row may be negative.
  x_min = x.min(axis=1, keepdims=True)                  # (batch_size, 1)
  x = x_mask * x + (1 - x_mask) * x_min                 # (batch_size, num_classes)
  y_hats = tt.argmax(x, axis=1)                         # (batch_size,)
  accs = cast_floatX(tt.eq(y_hats, y))                  # (batch_size,)

  log_z = tt.log(1 + tt.exp(-x))                        # (batch_size, num_classes)
  xents_false = x + log_z                               # (batch_size, num_classes)
  xents_false *= x_mask                                 # (batch_size, num_classes)
  sum_xents_false = xents_false.sum(axis=1)             # (batch_size,)

  x_star = x[tt.arange(x.shape[0]), y]                  # (batch_size,)
  sum_xents = sum_xents_false - x_star                  # (batch_size,)
  #xents = sum_xents / x_mask.sum(axis=1, keepdims=True) # (batch_size,)
  xents = sum_xents

  return xents, accs, y_hats


def _word_multinomial_classification(x, x_mask, y):
  # x       float32 (batch_size, num_classes)   scores i.e. logits
  # x_mask  int32   (batch_size, num_classes)   score masks (each sample has a variable number of classes)
  # y       int32   (batch_size,)               target classes i.e. ground truth answers (given as class indices)
  assert x.ndim == x_mask.ndim == 2
  assert y.ndim == 1

  # substracting min needed since all non masked-out elements of a row may be negative.
  x *= x_mask
  x -= x.min(axis=1, keepdims=True)             # (batch_size, num_classes)
  x *= x_mask                                   # (batch_size, num_classes)
  x -= x.max(axis=1, keepdims=True)             # (batch_size, num_classes)
  x *= x_mask                                   # (batch_size, num_classes)
  exp_x = tt.exp(x)                             # (batch_size, num_classes)
  exp_x *= x_mask                               # (batch_size, num_classes)

  sum_exp_x = exp_x.sum(axis=1, keepdims=True)  # (batch_size, 1)
  log_sum_exp_x = tt.log(sum_exp_x)             # (batch_size, 1)

  log_probs = x - log_sum_exp_x                 # (batch_size, num_classes)
  log_probs *= x_mask

  x_star_log_probs = log_probs[tt.arange(x.shape[0]), y]  # (batch_size,)
  xents = -x_star_log_probs

  return log_probs, xents

