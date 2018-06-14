# -*- coding: utf-8 -*-
import logging
import sys
import io
import json
import unicodedata
import numpy as np

from collections import namedtuple, Counter
from copy import deepcopy

from word_embeddings import WordEmbData, read_word_emb_data


PLACEHOLDER_STR = '<(ph________)>'
PLACEHOLDER_IDX = 1   # in setup.py we started indexing words at idx 2


#######################################
# Types:
#######################################

SquadData = namedtuple('SquadData', [
  'word_emb_data',      # WordEmbData
  'char_data',          # CharData
  'trn',                # SquadDataset
  'dev',                # SquadDataset
  'tst'                 # SquadDataset
])

SquadDataset = namedtuple('SquadDataset', [
  'tabular',            # SquadDatasetTabular
  'vectorized'          # SquadDatasetVectorized
])

SquadDatasetVectorized = namedtuple('SquadDatasetVectorized', [
  'ctxs',               # int32 (num contexts, max context length)
  'ctx_lens',           # int32 (num contexts,)
  'qtns',               # int32 (num questions, max question length)
  'qtn_lens',           # int32 (num questions,)
  'qtn_ctx_idxs',       # int32 (num questions,)      index of context of question
  'qtn_ans_inds',       # int32 (num questions,)      indicator of whether question has a valid answer
  'anss',               # int32 (num questions, 2)    we keep only first valid answer as (answer start word idx,
                        #                             answer end word idx), undefined for all invalid

  'ctx_originals',      # int32 (num contexts, max context length)
  'qtn_originals',      # int32 (num questions, max question length)

  'ctx_wdp_seq_ids',    # int32 (num contexts, max context length)      # each token's suquential word-type id int
  'qtn_wdp_seq_ids',    # int32 (num questions, max question length)    # each token's suquential word-type id int
])

TokenizedText = namedtuple('TokenizedText', [
  'tokens',             # list of parsed tokens
  'originals',          # list of original tokens (may differ from parsed ones)
  'whitespace_afters',  # list of whitespace strings, each appears after corresponding original token in original text
  'sent_lens',          # list of sentence lengths
  'ex_wdp_seq_ids'      # list int32: sequential word-type id
])

SquadArticle = namedtuple('SquadArticle', [
  'art_title_str'
])

SquadContext = namedtuple('SquadContext', [
  'art_idx',
  'tokenized',          # TokenizedText of context's text
  'original_idx',       # The sequential index of this context when reading raw input json.
])

SquadQuestion = namedtuple('SquadQuestion', [
  'ctx_idx',            # The index of the SquadContext object in SquadDatasetTabular to which this question relates.
  'original_idx',       # The sequential index of this question when reading raw input json.
  'qtn_id',
  'tokenized',          # TokenizedText of question's text
  'ans_texts',          # list of (possibly multiple) answer text strings
  'ans_word_idxs',      # list where each entry is either a (answer start word index, answer end word index) tuple
                        # or None for answers that we failed to parse
])

class SquadDatasetTabular(object):
  def __init__(self):
    self.arts = []    # SquadArticle objects
    self.ctxs = []    # SquadContext objects
    self.qtns = []    # SquadQuestion objects
  def new_article(self, art_title_str):
    self.arts.append(SquadArticle(art_title_str))
    return len(self.arts) - 1
  def new_context(self, art_idx, ctx_tokenized, original_ctx_idx):
    self.ctxs.append(SquadContext(art_idx, ctx_tokenized, original_ctx_idx))
    return len(self.ctxs) - 1
  def new_question(self, ctx_idx, original_qtn_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs):
    self.qtns.append(
      SquadQuestion(ctx_idx, original_qtn_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs))


CharData = namedtuple('CharData', [
  'char_to_idx',      # map from char string to embedding index, 0 is for nil, 1 is for unk
  'original_to_idx',  # map from original string to index in original_chars
  'original_chars',   # int32 (num originals + 1, max original len) char embedding indices of original; entry at index 0 reserved for nil
  'original_lens'     # int32 (num originals + 1,) length in chars of each original
])


#######################################
# Functionality:
#######################################


def _prepare_dataset_for_word_type_dropout(name, tab_ds):

  new_tab_ds = SquadDatasetTabular()
  new_tab_ds.arts = tab_ds.arts     # keep arts
  new_tab_ds.ctxs = [None] * len(tab_ds.ctxs)
  new_tab_ds.qtns = [None] * len(tab_ds.qtns)

  ctx_idx_to_qtn_idxs = {}
  for qtn_idx, qtn in enumerate(tab_ds.qtns):
    ctx_idx = qtn.ctx_idx
    ctx_idx_to_qtn_idxs.setdefault(ctx_idx, []).append(qtn_idx)

  for ctx_idx in range(len(tab_ds.ctxs)):

    qtn_idxs = ctx_idx_to_qtn_idxs[ctx_idx]
    ctx = tab_ds.ctxs[ctx_idx]

    ctx_str_to_wdp_seq_id = {'<dummy>': 0}
    ctx_ex_wdp_seq_ids = []

    for ctx_token_idx, ctx_token in enumerate(ctx.tokenized.tokens):
      ctx_token_lower = ctx_token.lower()
      ctx_wdp_seq_id = ctx_str_to_wdp_seq_id.setdefault(ctx_token_lower, len(ctx_str_to_wdp_seq_id))
      ctx_ex_wdp_seq_ids.append(ctx_wdp_seq_id)

    # add ctx
    new_ctx_tokenized = TokenizedText(
      ctx.tokenized.tokens, ctx.tokenized.originals, ctx.tokenized.whitespace_afters,
      ctx.tokenized.sent_lens, ctx_ex_wdp_seq_ids)
    new_ctx = SquadContext(ctx.art_idx, new_ctx_tokenized, ctx.original_idx)
    new_tab_ds.ctxs[ctx_idx] = new_ctx

    for qtn_idx in qtn_idxs:
      qtn = tab_ds.qtns[qtn_idx]

      copied_ctx_str_to_wdp_seq_id = ctx_str_to_wdp_seq_id.copy()   # shallow copy
      qtn_ex_wdp_seq_ids = []

      for qtn_token in qtn.tokenized.tokens:
        qtn_token_lower = qtn_token.lower()

        qtn_wdp_seq_id = copied_ctx_str_to_wdp_seq_id.setdefault(qtn_token_lower, len(copied_ctx_str_to_wdp_seq_id))
        qtn_ex_wdp_seq_ids.append(qtn_wdp_seq_id)

      # add qtn
      assert qtn.ctx_idx == ctx_idx
      new_qtn_tokenized = TokenizedText(
        qtn.tokenized.tokens, qtn.tokenized.originals, qtn.tokenized.whitespace_afters,
        qtn.tokenized.sent_lens, qtn_ex_wdp_seq_ids)
      new_qtn = SquadQuestion(qtn.ctx_idx, qtn.original_idx, qtn.qtn_id, new_qtn_tokenized, qtn.ans_texts, qtn.ans_word_idxs)
      new_tab_ds.qtns[qtn_idx] = new_qtn

  return new_tab_ds


def _count_originals(tab_ds):
  originals = Counter()
  ctx_num_qtns = Counter()
  for qtn in tab_ds.qtns:
    ctx_num_qtns[qtn.ctx_idx] += 1
    for qtn_original in qtn.tokenized.originals:
      assert type(qtn_original) == unicode
      originals[qtn_original] += 1
  for ctx_idx, ctx in enumerate(tab_ds.ctxs):
    for ctx_original in ctx.tokenized.originals:
      assert type(ctx_original) == unicode
      originals[ctx_original] += ctx_num_qtns[ctx_idx]
  return originals


def _get_char_data(trn_tab_ds, other_tab_ds):
  logger = logging.getLogger()
  logger.info('Preparing character-level data:')
  trn_originals = _count_originals(trn_tab_ds)
  chars = Counter()
  for original in trn_originals.elements():
    chars.update(original)
  char_to_idx = {'<nil>': 0, '<unk>': 1}

  for c_rank, (c, _) in enumerate(chars.most_common(), 1):
    if c_rank <= 100:
      char_to_idx[c] = len(char_to_idx)

  other_originals = _count_originals(other_tab_ds)
  originals = trn_originals + other_originals

  num_originals = len(originals.keys())
  original_lens = [len(original) for original in originals.elements()]
  max_original_len = max(original_lens)

  original_to_idx = {'<nil>': 0}
  original_chars = np.zeros((num_originals+1, max_original_len), dtype=np.int32)
  original_lens = np.zeros((num_originals+1), dtype=np.int32)
  unk_char_counter = Counter()
  for original_idx, (original, _) in enumerate(originals.most_common(), 1):   # starting from 1; 0 reserved for nil
    original_to_idx[original] = original_idx
    oc = []
    for c in original:
      if c in char_to_idx:
        char_idx = char_to_idx[c]
      else:
        char_idx = char_to_idx['<unk>']
        unk_char_counter[c] += 1
      oc.append(char_idx)
    original_chars[original_idx, :len(oc)] = oc
    original_lens[original_idx] = len(oc)

  num_unk_char_types = len(unk_char_counter)
  num_unk_char_tokens = sum(unk_char_counter.values())

  logger.info(('There are {:d} original word-types in training set, {:d} in dev set,'
    ' {:d} acknowledged char-types, {:d} ({:d}) unk char types (tokens)').format(
      len(trn_originals), len(other_originals),
      len(char_to_idx)-2, num_unk_char_types, num_unk_char_tokens))
  qs = [1., 2., 5.] + list(np.arange(10., 91., 10.)) + [95., 99., 100.]
  msg = 'Lengths of word-types in characters:\n' + '\n'.join([
    '\t{:<20s}{:s}'.format('percentile:', ''.join(['%-5d' % q for q in qs])),
    '\t{:<20s}{:s}'.format('original length:', ''.join(['%-5d' % original_p for original_p in np.percentile(original_lens, qs)]))])
  logger.info(msg)

  return CharData(char_to_idx, original_to_idx, original_chars, original_lens)


def get_data(
  word_emb_data_path_prefix,
  tokenized_trn_json_path,
  tokenized_dev_json_path,
  max_ans_len,
  max_ctx_len):

  word_emb_data = read_word_emb_data(word_emb_data_path_prefix)
  word_strs = set()

  trn_tab_ds = _make_tabular_dataset(
    tokenized_trn_json_path, word_strs, has_answers=True, max_ans_len=max_ans_len, max_ctx_len=max_ctx_len)
  dev_tab_ds = _make_tabular_dataset(
    tokenized_dev_json_path, word_strs, has_answers=True, max_ans_len=max_ans_len, max_ctx_len=max_ctx_len)

  trn_tab_ds = _prepare_dataset_for_word_type_dropout('TRN', trn_tab_ds)
  dev_tab_ds = _prepare_dataset_for_word_type_dropout('DEV', dev_tab_ds)
  
  word_emb_data = _contract_word_emb_data(word_emb_data, word_strs)

  assert word_emb_data.first_known_word > PLACEHOLDER_IDX
  word_emb_data.str_to_word[PLACEHOLDER_STR] = PLACEHOLDER_IDX

  char_data = _get_char_data(trn_tab_ds, dev_tab_ds)

  trn_vec_ds = _make_vectorized_dataset('train', trn_tab_ds, word_emb_data, char_data)
  dev_vec_ds = _make_vectorized_dataset('dev', dev_tab_ds, word_emb_data, char_data)

  trn_ds = SquadDataset(trn_tab_ds, trn_vec_ds)
  dev_ds = SquadDataset(dev_tab_ds, dev_vec_ds)
  return SquadData(word_emb_data, char_data, trn_ds, dev_ds, None)


def construct_answer_hat(ctx, ans_hat_start_word_idx, ans_hat_end_word_idx):
  ctx_originals = ctx.tokenized.originals
  ctx_whitespace_afters = ctx.tokenized.whitespace_afters
  ans_hat_str = ''
  for word_idx in range(ans_hat_start_word_idx, ans_hat_end_word_idx+1):
    ans_hat_str += ctx_originals[word_idx]
    if word_idx < ans_hat_end_word_idx:
      ans_hat_str += ctx_whitespace_afters[word_idx]
  return ans_hat_str


def write_test_predictions(ans_hats, tst_prd_json_path):
  logger = logging.getLogger()
  s = json.dumps(ans_hats, ensure_ascii=False)
  with io.open(tst_prd_json_path, 'w', encoding='utf-8') as f:
    f.write(s)
  logger.info('Written test predictions to {}'.format(tst_prd_json_path))


def _make_tabular_dataset(tokenized_json_path, word_strs, has_answers, max_ans_len=None, max_ctx_len=None):
  logger = logging.getLogger()
  tabular = SquadDatasetTabular()

  num_questions = 0
  num_answers = 0
  num_invalid_answers = 0
  num_long_answers = 0
  num_long_contexts = 0
  num_invalid_questions = 0

  answers_per_question_counter = Counter()
  with io.open(tokenized_json_path, 'r', encoding='utf-8') as f:
    j = json.load(f)
    #version = j['version']
    data = j['data']
    for article in data:
      art_title_str = article['title']
      art_idx = tabular.new_article(art_title_str)

      paragraphs = article['paragraphs']
      for paragraph in paragraphs:

        ctx_tokens = paragraph['tokens']
        ctx_originals = paragraph['originals']
        ctx_whitespace_afters = paragraph['whitespace_afters']

        ctx_sent_lens = paragraph['sentence_lengths']

        ctx_tokenized = TokenizedText(ctx_tokens, ctx_originals, ctx_whitespace_afters, ctx_sent_lens, None)
        ctx_idx = tabular.new_context(art_idx, ctx_tokenized, None)

        word_strs.update(ctx_tokens)

        is_long_context = True if (max_ctx_len and len(ctx_tokens) > max_ctx_len) else False
        num_long_contexts += is_long_context

        qas = paragraph['qas']
        for qa in qas:
          num_questions += 1

          qtn_id = qa['id']
          qtn_tokens = qa['tokens']
          qtn_originals = qa['originals']
          qtn_whitespace_afters = qa['whitespace_afters']

          # note: we forcibly regard qtn as having a single sentence
          qtn_sent_lens = [len(qtn_tokens)]
          qtn_tokenized = TokenizedText(qtn_tokens, qtn_originals, qtn_whitespace_afters, qtn_sent_lens, None)
          word_strs.update(qtn_tokens)

          ans_texts = []
          ans_word_idxs = []
          if has_answers:
            answers = qa['answers']
            assert answers
            for answer in answers:
              num_answers += 1
              ans_text = answer['text']
              assert ans_text
              ans_texts.append(ans_text)
              if not answer['valid']:
                ans_word_idxs.append(None)
                num_invalid_answers += 1
                continue
              if is_long_context:
                ans_word_idxs.append(None)
                continue
              ans_start_word_idx = answer['start_token_idx']
              ans_end_word_idx = answer['end_token_idx']
              if max_ans_len and ans_end_word_idx - ans_start_word_idx + 1 > max_ans_len:
                ans_word_idxs.append(None)
                num_long_answers += 1
              else:
                ans_word_idxs.append((ans_start_word_idx, ans_end_word_idx))
            answers_per_question_counter[len(ans_texts)] += 1   # this counts also invalid answers
            num_invalid_questions += 1 if all(ans is None for ans in ans_word_idxs) else 0

          tabular.new_question(ctx_idx, None, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs)

  logger.info('Processed {:s}:\n'
    '\ttotal {:d} questions, {:d} invalid questions, '
    'total {:d} answers, {:d} invalid answers, {:d} too long answers, {:d} too long contexts\n'
    '\t{{x: num of questions having x answers}}: {{{:s}}}'.format(
    tokenized_json_path,
    num_questions, num_invalid_questions,
    num_answers, num_invalid_answers, num_long_answers, num_long_contexts,
    ', '.join('{:d}: {:d}'.format(x, num_x) for x, num_x in sorted(answers_per_question_counter.iteritems()))))
  return tabular


def _contract_word_emb_data(old_word_emb_data, word_strs):
  logger = logging.getLogger()
  old_word_emb, old_str_to_word, old_first_known_word, old_first_unknown_word, old_first_unallocated_word = \
    old_word_emb_data

  known_word_strs = []
  unknown_word_strs = []
  for word_str in word_strs:
    if word_str in old_str_to_word and old_str_to_word[word_str] < old_first_unknown_word:
      known_word_strs.append(word_str)
    else:
      unknown_word_strs.append(word_str)

  str_to_word = {}
  emb_size = old_first_known_word + len(word_strs)
  word_emb = np.zeros((emb_size, old_word_emb.shape[1]), dtype=np.float32)

  for i, word_str in enumerate(known_word_strs):
    word = old_first_known_word + i
    str_to_word[word_str] = word
    word_emb[word, :] = old_word_emb[old_str_to_word[word_str]]

  first_unknown_word = old_first_known_word + len(known_word_strs)

  num_new_unks = 0
  for i, word_str in enumerate(unknown_word_strs):
    word = first_unknown_word + i
    str_to_word[word_str] = word
    if word_str in old_str_to_word:
      word_emb[word, :] = old_word_emb[old_str_to_word[word_str]]
    else:
      if old_first_unallocated_word + num_new_unks >= len(old_word_emb):
        logger.info('Error: too many unknown words, can increase number of alloted random embeddings in setup.py')
        sys.exit(1)
      word_emb[word, :] = old_word_emb[old_first_unallocated_word + num_new_unks]
      num_new_unks += 1
  logger.info('Contracted word embeddings:\n'
    '\t{} known word-types, {} pre-existing unknown word-types, {} new unknown word-types'.format(
      len(known_word_strs), len(unknown_word_strs) - num_new_unks, num_new_unks))
  return WordEmbData(
    word_emb, str_to_word, old_first_known_word, first_unknown_word, None)


def _make_vectorized_dataset(name, tabular, word_emb_data, char_data):
  num_ctxs = len(tabular.ctxs)
  num_qtns = len(tabular.qtns)
  max_ctx_len = max(len(ctx.tokenized.tokens) for ctx in tabular.ctxs)
  max_qtn_len = max(len(qtn.tokenized.tokens) for qtn in tabular.qtns)

  ctxs = np.zeros((num_ctxs, max_ctx_len), dtype=np.int32)
  ctx_lens = np.zeros(num_ctxs, dtype=np.int32)
  qtns = np.zeros((num_qtns, max_qtn_len), dtype=np.int32)
  qtn_lens = np.zeros(num_qtns, dtype=np.int32)
  qtn_ctx_idxs = np.zeros(num_qtns, dtype=np.int32)
  qtn_ans_inds = np.zeros(num_qtns, dtype=np.int32)
  anss = np.zeros((num_qtns, 2), dtype=np.int32)

  ctx_wdp_seq_ids = np.zeros((num_ctxs, max_ctx_len), dtype=np.int32)
  qtn_wdp_seq_ids = np.zeros((num_qtns, max_qtn_len), dtype=np.int32)

  ctx_originals = np.zeros((num_ctxs, max_ctx_len), dtype=np.int32)
  qtn_originals = np.zeros((num_qtns, max_qtn_len), dtype=np.int32)

  for ctx_idx, ctx in enumerate(tabular.ctxs):
    ctx_words = [word_emb_data.str_to_word[word_str] for word_str in ctx.tokenized.tokens]
    ctxs[ctx_idx, :len(ctx_words)] = ctx_words
    ctx_lens[ctx_idx] = len(ctx_words)

    ctx_original_words = [char_data.original_to_idx[original_str] for original_str in ctx.tokenized.originals]
    assert len(ctx_original_words) == len(ctx_words)
    ctx_originals[ctx_idx, :len(ctx_words)] = ctx_original_words

    ctx_wdp_seq_ids[ctx_idx, :len(ctx_words)] = ctx.tokenized.ex_wdp_seq_ids

  for qtn_idx, qtn in enumerate(tabular.qtns):
    qtn_words = [word_emb_data.str_to_word[word_str] for word_str in qtn.tokenized.tokens]
    qtns[qtn_idx, :len(qtn_words)] = qtn_words
    qtn_lens[qtn_idx] = len(qtn_words)
    qtn_ctx_idxs[qtn_idx] = qtn.ctx_idx

    qtn_original_words = [char_data.original_to_idx[original_str] for original_str in qtn.tokenized.originals]
    assert len(qtn_original_words) == len(qtn_words)
    qtn_originals[qtn_idx, :len(qtn_words)] = qtn_original_words

    qtn_wdp_seq_ids[qtn_idx, :len(qtn_words)] = qtn.tokenized.ex_wdp_seq_ids

    ans = next((ans for ans in qtn.ans_word_idxs if ans), None) if qtn.ans_word_idxs else None
    if ans:
      ans_start_word_idx, ans_end_word_idx = ans
      anss[qtn_idx] = [ans_start_word_idx, ans_end_word_idx]
      qtn_ans_inds[qtn_idx] = 1

  msg = 'Vectorized {} samples.'.format(name)
  msg += '\nctxs: {:d}, max ctx length: {:d}, sum lengths: {:d}'.format(
    num_ctxs, max_ctx_len, ctx_lens.sum())
  msg += '\nqtns: {:d}, max qtn length: {:d}, sum lengths: {:d}'.format(
    num_qtns, max_qtn_len, qtn_lens.sum())
  
  qs = [1., 2., 5.] + list(np.arange(10., 91., 10.)) + [95., 99., 100.]
  msg += '\nLengths:\n' + '\n'.join([
    '\t{:<15s}{:s}'.format('percentile:', ''.join(['%-5d' % q for q in qs])),
    '\t{:<15s}{:s}'.format('ctx length:', ''.join(['%-5d' % ctx_p for ctx_p in np.percentile(ctx_lens, qs)])),
    '\t{:<15s}{:s}'.format('qtn length:', ''.join(['%-5d' % qtn_p for qtn_p in np.percentile(qtn_lens, qs)]))])
  logging.getLogger().info(msg)

  return SquadDatasetVectorized(ctxs, ctx_lens, qtns, qtn_lens, qtn_ctx_idxs, qtn_ans_inds, anss,
    ctx_originals, qtn_originals,
    ctx_wdp_seq_ids, qtn_wdp_seq_ids)

