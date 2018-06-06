# Contents of this file are a modified version of code found at:
#   https://github.com/tensorflow/models/tree/master/research/lm_1b
# which is an open-source version of the model described in
#   Exploring the Limits of Language Modeling
#   Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, Yonghui Wu; 2016
#   https://arxiv.org/abs/1602.02410
# ==============================================================================

import sys
import logging

import numpy as np


##########################################
# TF model
##########################################

LM1B_MODEL_HIDDEN_DIM = 1024
LM1B_MODEL_MAX_WORD_LEN = 50


def _load_tf_model(gd_file, ckpt_file):
  from google.protobuf import text_format
  import tensorflow as tf

  with tf.Graph().as_default():
    sys.stderr.write('Recovering graph.\n')
    with tf.gfile.FastGFile(gd_file, 'r') as f:
      s = f.read().decode()
      gd = tf.GraphDef()
      text_format.Merge(s, gd)

    tf.logging.info('Recovering Graph %s', gd_file)
    t = {}
    [t['states_init'], t['lstm/lstm_0/control_dependency'],
     t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
     t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
     t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
     t['all_embs'], t['softmax_weights'], t['global_step']
    ] = tf.import_graph_def(gd, {}, ['states_init',
                                     'lstm/lstm_0/control_dependency:0',
                                     'lstm/lstm_1/control_dependency:0',
                                     'softmax_out:0',
                                     'class_ids_out:0',
                                     'class_weights_out:0',
                                     'log_perplexity_out:0',
                                     'inputs_in:0',
                                     'targets_in:0',
                                     'target_weights_in:0',
                                     'char_inputs_in:0',
                                     'all_embs_out:0',
                                     'Reshape_3:0',
                                     'global_step:0'], name='')

    sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run('save/restore_all', {'save/Const:0': ckpt_file})
    sess.run(t['states_init'])

  return sess, t


class WrappedLm1bModel(object):

  def __init__(self, graph_def_path, ckpt_path, vocab_path):
    sess, t = _load_tf_model(graph_def_path, ckpt_path)
    self._sess = sess
    self._t = t

    import lm_utils
    self._vocab = lm_utils.CharsVocabulary(vocab_path, LM1B_MODEL_MAX_WORD_LEN)

    self._targets = np.zeros([1, 1], np.int32)
    self._weights = np.ones([1, 1], np.float32)
    self._t_payload_mapping = {
      'EMB': 'all_embs',
      'L1': 'lstm/lstm_0/control_dependency',
      'L2': 'lstm/lstm_1/control_dependency'
    }


  def _reset_states(self):
    self._sess.run(self._t['states_init'])

  def get_h(self, originals, payload):
    h = np.zeros((len(originals), 1024), dtype=np.float32)
    t_key = self._t_payload_mapping[payload]
    originals = ['<S>'] + originals
    self._reset_states()
    for i, original in enumerate(originals):
      char_ids = self._vocab.word_to_char_ids(original)                   # int32 (LM1B_MODEL_MAX_WORD_LEN,)
      char_ids_inputs = char_ids.reshape((1,1,LM1B_MODEL_MAX_WORD_LEN))   # int32 (1,1,LM1B_MODEL_MAX_WORD_LEN)
      try:
        lstm_emb = self._sess.run(self._t[t_key], feed_dict={
          self._t['char_inputs_in']: char_ids_inputs,
          #self._t['inputs_in']: inputs,
          self._t['targets_in']: self._targets,
          self._t['target_weights_in']: self._weights})
      except:
        msg = u'\n\noriginals:\n{:s}\n\nfailed at step {:d}\n\noriginals[i]: {:s}\n\n'.format(
          unicode(originals), i, originals[i])
        logging.getLogger().info(msg)
        raise
      if i > 0: # skip the '<S>' token
        h[i-1] = lstm_emb.flatten()      # going from (1, 1024) to (1024,)
    return h


def _encode_paragraph(lm_model, originals, sent_lens, payload):
  sent_hs = []
  sent_start_idx = 0
  for sent_len in sent_lens:
    sent_after_end_idx = sent_start_idx + sent_len
    sent_originals = originals[sent_start_idx:sent_after_end_idx]
    sent_h = lm_model.get_h(sent_originals, payload)
    sent_hs.append(sent_h)
    sent_start_idx = sent_after_end_idx
  h = np.concatenate(sent_hs, axis=0)   # (total number of tokens, LM1B_MODEL_HIDDEN_DIM)
  return h


def encode_paragraphs(lm_model, lm_data_shard_cfg, paragraphs, paragraphs_sent_lens):
  logger = logging.getLogger()

  lens = [len(originals) for originals in paragraphs]
  num_total_tokens = sum(lens)

  logger.info('{:s} : there is a total of {:d} tokens'.format(lm_data_shard_cfg, num_total_tokens))

  hs = np.zeros((num_total_tokens, LM1B_MODEL_HIDDEN_DIM), dtype=np.float32)
  lens = np.array(lens, dtype=np.int32)

  pos = 0
  for i, (originals, sent_lens) in enumerate(zip(paragraphs, paragraphs_sent_lens)):
    h = _encode_paragraph(lm_model, originals, sent_lens, lm_data_shard_cfg.payload)
    assert h.shape == (len(originals), LM1B_MODEL_HIDDEN_DIM)
    hs[pos:pos+len(originals), :] = h
    pos += len(originals)
    if (i+1) % 50 == 0 or i == len(paragraphs) - 1:
      logger.info('{:s} : done {:d} / {:d} texts'.format(lm_data_shard_cfg, i+1, len(paragraphs)))
  return hs, lens

