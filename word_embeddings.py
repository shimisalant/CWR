import logging
import cPickle
import numpy as np

from collections import namedtuple


#######################################
# Word embeddings:
#######################################

WordEmbData = namedtuple('WordEmbData', [
  'word_emb',                 # float32 (num words, emb dim)
  'str_to_word',              # map word string to word index
  'first_known_word',         # words found in GloVe are at positions [first_known_word, first_unknown_word)
  'first_unknown_word',       # words not found in GloVe are at positions [first_unknown_word, first_unallocated_word)
  'first_unallocated_word'    # extra random embeddings
])


def get_word_emb_data_paths(path_prefix):
  return path_prefix + '.metadata.pkl', path_prefix + '.emb.npy'


def write_word_emb_data(path_prefix, word_emb_data):
  metadata_path, emb_path = get_word_emb_data_paths(path_prefix)
  with open(metadata_path, 'wb') as f:
    cPickle.dump((word_emb_data.str_to_word, word_emb_data.first_known_word,
      word_emb_data.first_unknown_word, word_emb_data.first_unallocated_word),
      f, protocol=cPickle.HIGHEST_PROTOCOL)
  with open(emb_path, 'wb') as f:
    np.save(f, word_emb_data.word_emb)
  logging.getLogger().info('Written word embedding data:\n\t{}\n\t{}'.format(metadata_path, emb_path))


def read_word_emb_data(path_prefix):
  metadata_path, emb_path = get_word_emb_data_paths(path_prefix)
  with open(metadata_path, 'rb') as f:
    str_to_word, first_known_word, first_unknown_word, first_unallocated_word = cPickle.load(f)
  with open(emb_path, 'rb') as f:
    word_emb = np.load(f)
  word_emb_data = WordEmbData(
    word_emb, str_to_word, first_known_word, first_unknown_word, first_unallocated_word)
  logging.getLogger().info('Read word embedding data from:\n\t{}\n\t{}'.format(metadata_path, emb_path))
  return word_emb_data

