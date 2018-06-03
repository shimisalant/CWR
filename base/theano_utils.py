import numpy as np
import theano
import theano.tensor as tt


floatX = theano.config.floatX


def cast_floatX_np(n):
  return np.asarray(n, dtype=floatX)


def cast_floatX(x):
  return tt.cast(x, floatX)


def get_shared_floatX(value, name):
  return theano.shared(cast_floatX_np(value), name)


def gpu_int32(name, x_val, return_shared_var=False):
  # theano trick: stored as float32 on GPU (if we're using a GPU), and cast back to int32
  assert x_val.dtype == np.int32
  shared_var = get_shared_floatX(x_val, name)
  cast_shared_var = tt.cast(shared_var, 'int32')
  cast_shared_var.underlying_shared_var = shared_var
  return (cast_shared_var, shared_var) if return_shared_var else cast_shared_var


def clip_sqrt(x):
  # # DBG
  # XXX
  # #return tt.sqrt(tt.clip(x, 0.0, np.float32(1000000)))
  return tt.sqrt(tt.clip(x, 0.0, np.inf))


def softmax_columns_with_mask(x, mask, allow_none=False):
  #     assert x.ndim == 2
  #     assert mask.ndim == 2
  # for numerical stability
  x *= mask
  x -= x.min(axis=0, keepdims=True)
  x *= mask
  x -= x.max(axis=0, keepdims=True)
  e_x = mask * tt.exp(x)
  sums = e_x.sum(axis=0, keepdims=True)
  if allow_none:
    sums += tt.eq(sums, 0)
  y = e_x / sums      # every column must have at least one non-masked non-zero element
  return y


def softmax_rows_with_mask(x, mask):
  assert x.ndim == 2
  assert mask.ndim == 2
  # for numerical stability
  x *= mask
  x -= x.min(axis=1, keepdims=True)
  x *= mask
  x -= x.max(axis=1, keepdims=True)
  e_x = mask * tt.exp(x)
  sums = e_x.sum(axis=1, keepdims=True)
  y = e_x / (sums + tt.le(sums,0))
  y *= mask
  return y


def softmax_depths_with_mask(x, mask):
  assert x.ndim == 3
  assert mask.ndim == 3
  # for numerical stability
  x *= mask
  x -= x.min(axis=2, keepdims=True)
  x *= mask
  x -= x.max(axis=2, keepdims=True)
  e_x = mask * tt.exp(x)
  sums = e_x.sum(axis=2, keepdims=True)
  #y = e_x / (sums + tt.eq(sums,0))
  y = e_x / (sums + tt.le(sums,0))
  y *= mask
  return y


def softmax_depths(x):
  assert x.ndim == 3
  # for numerical stability
  x -= x.max(axis=2, keepdims=True)
  e_x = tt.exp(x)
  sums = e_x.sum(axis=2, keepdims=True)
  y = e_x / sums
  return y


def argmax_with_mask(x, mask):
  assert x.ndim == 2
  assert mask.ndim == 2
  x_min = x.min(axis=1, keepdims=True)
  x = mask * x + (1 - mask) * x_min
  return x.argmax(axis=1)


def log_softmax_rows_with_mask(x, mask):
  assert x.ndim == 2
  assert mask.ndim == 2
  # for numerical stability
  x *= mask
  x -= x.min(axis=1, keepdims=True)
  x *= mask
  x -= x.max(axis=1, keepdims=True)
  exp_x = mask * tt.exp(x)
  sum_exp_x = exp_x.sum(axis=1, keepdims=True)
  sum_exp_x += tt.le(sum_exp_x, 0)
  log_sum_exp_x = tt.log(sum_exp_x)
  log_probs = x - log_sum_exp_x
  probs = tt.exp(log_probs)
  log_probs *= mask
  probs *= mask
  return log_probs, probs


def log_softmax_depths_with_mask(x, mask):
  assert x.ndim == 3
  assert mask.ndim == 3
  # for numerical stability
  x *= mask
  x -= x.min(axis=2, keepdims=True)
  x *= mask
  x -= x.max(axis=2, keepdims=True)
  exp_x = mask * tt.exp(x)
  sum_exp_x = exp_x.sum(axis=2, keepdims=True)
  sum_exp_x += tt.le(sum_exp_x, 0)
  log_sum_exp_x = tt.log(sum_exp_x)
  log_probs = x - log_sum_exp_x
  probs = tt.exp(log_probs)
  log_probs *= mask
  probs *= mask
  return log_probs, probs








def max_and_argmax_k(x, k):   # NO MASK HERE
  # x         (batch_size, max_p_len*max_ans_len)   needs to be non-negative (otherwise the zero we put in to zero out a max can get picked out as the new max)
  # k         int
  maxs, argmaxs = [], []
  for _ in range(k):
    max_k, argmax_k = tt.max_and_argmax(x, axis=1, keepdims=True)     # (batch_size, 1), (batch_size, 1)
    maxs.append(max_k)
    argmaxs.append(tt.cast(argmax_k, 'int32'))
    x *= tt.lt(x, max_k)
  maxs = tt.concatenate(maxs, axis=1)           # (batch_size, k)
  argmaxs = tt.concatenate(argmaxs, axis=1)     # (batch_size, k)
  return maxs, argmaxs

# note that there need to exist unique k maxs
def max_and_argmax_k_with_mask(x, x_mask, k):
  # x         (batch_size, max_p_len*max_ans_len)   doesn't need to be non-negative where not masked
  # x_mask    (batch_size, max_p_len*max_ans_len)
  # k         int
  maxs, argmaxs = [], []
  mins = tt.min(x, axis=1, keepdims=True)       # (batch_size, 1)
  x = x_mask * x + (1 - x_mask) * mins
  for _ in range(k):
    max_k, argmax_k = tt.max_and_argmax(x, axis=1, keepdims=True)     # (batch_size, 1), (batch_size, 1)
    maxs.append(max_k)
    argmaxs.append(tt.cast(argmax_k, 'int32'))
    keep_mask = tt.lt(x, max_k)
    x = keep_mask * x + (1 - keep_mask) * mins
  maxs = tt.concatenate(maxs, axis=1)           # (batch_size, k)
  argmaxs = tt.concatenate(argmaxs, axis=1)     # (batch_size, k)
  return maxs, argmaxs


# note that there need to exist unique k maxs
def max_and_argmax_k_with_mask_for_non_negative(x, x_mask, k):
  # x         (batch_size, max_p_len*max_ans_len)   assumed to be non-negative where not masked
  # x_mask    (batch_size, max_p_len*max_ans_len)
  # k         int
  maxs, argmaxs = [], []
  x *= x_mask
  for _ in range(k):
    max_k, argmax_k = tt.max_and_argmax(x, axis=1, keepdims=True)     # (batch_size, 1), (batch_size, 1)
    maxs.append(max_k)
    argmaxs.append(tt.cast(argmax_k, 'int32'))
    keep_mask = tt.lt(x, max_k)
    x *= keep_mask
  maxs = tt.concatenate(maxs, axis=1)           # (batch_size, k)
  argmaxs = tt.concatenate(argmaxs, axis=1)     # (batch_size, k)
  return maxs, argmaxs






def batched_indexing(x, idxs):
  # x         float32 (num_samples, num_options [,x_dim])
  # idxs      int32   (num_samples, k)
  #
  # returns a tensor of shape (num_samples, k [,x_dim]), where row i consists of k entries of x,
  # these are x's entries at indices idxs[i,:].
  assert x.dtype == 'float32'
  assert idxs.dtype == 'int32'
  assert x.ndim == 2 or x.ndim == 3
  assert idxs.ndim == 2

  num_samples = x.shape[0]
  num_options = x.shape[1]
  k = idxs.shape[1]
  if x.ndim == 3:
    x_dim = x.shape[2]
    x_flat_shape = (num_samples * num_options, x_dim)
    choices_shape = (num_samples, k, x_dim)
  else:
    x_flat_shape = (num_samples * num_options,)
    choices_shape = (num_samples, k)

  shifts = tt.shape_padright(tt.arange(0, num_samples * num_options, num_options))    # (num_samples, 1)
  idxs_shifted = idxs + shifts                                                        # (num_samples, k)
  idxs_flat = idxs_shifted.flatten()                                                  # (num_samples * k,)

  x_flat = x.reshape(x_flat_shape)                    # (num_samples * num_options [,x_dim])
  flat_choices = x_flat[idxs_flat]                    # (num_samples * k [,x_dim])
  choices = flat_choices.reshape(choices_shape)       # (num_samples, k [,x_dim])
  return choices


