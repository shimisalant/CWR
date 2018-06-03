import numpy as np
import theano
import theano.tensor as tt

from theano.ifelse import ifelse

from base.theano_utils import cast_floatX_np, cast_floatX, get_shared_floatX, clip_sqrt


class AdamOptimizerWithEma(object):

  def __init__(self, config, loss, param_dict, param_shadow_dict):
    self._lr = get_shared_floatX(config.learning_rate, 'lr')
    self._t = get_shared_floatX(1, 't')
    self._all_m_tm1 = []
    self._all_v_tm1 = []
    self._updates = [(self._t, self._t + 1)]

    if config.lr_decay:
      lr_coef = tt.pow(config.lr_decay, (self._t - 1) // config.lr_decay_freq)
      self._updates.append((self._lr, lr_coef * config.learning_rate))

    param_names = sorted(param_dict.keys())
    if config.ema:
      assert param_names == sorted(param_shadow_dict.keys())
    params = [param_dict[name] for name in param_names]

    grads = theano.grad(loss, params)
    #grads = theano.grad(loss, params, disconnected_inputs='ignore')

    self._global_grad_norm = tt.sqrt(tt.sum(tt.stack([tt.sum(g**2.) for g in grads])))
    if config.max_grad_norm:
      global_clip_factor = ifelse(tt.lt(self._global_grad_norm, config.max_grad_norm),
        cast_floatX_np(1.),
        cast_floatX(config.max_grad_norm/self._global_grad_norm))
      # global_clip_factor = tt.minimum(cast_floatX(config.max_grad_norm/self._global_grad_norm), cast_floatX_np(1))
      grads = [global_clip_factor * g for g in grads]

    lr_t = self._lr * \
      clip_sqrt(1 - tt.pow(config.adam_beta2, self._t)) / (1 - tt.pow(config.adam_beta1, self._t))

    for p_name, p, g in zip(param_names, params, grads):
        m_tm1 = get_shared_floatX(np.zeros_like(p.get_value()), 'adam_m_' + p.name)
        v_tm1 = get_shared_floatX(np.zeros_like(p.get_value()), 'adam_v_' + p.name)
        self._all_m_tm1.append(m_tm1)
        self._all_v_tm1.append(v_tm1)
        m_t = config.adam_beta1 * m_tm1 + (1-config.adam_beta1) * g
        v_t = config.adam_beta2 * v_tm1 + (1-config.adam_beta2) * tt.sqr(g)
        delta_t = -lr_t * m_t / (clip_sqrt(v_t) + config.adam_eps)
        p_t = p + delta_t
        self._updates += [(m_tm1, m_t), (v_tm1, v_t), (p, p_t)]
        if config.ema:
          ps = param_shadow_dict[p_name]
          ps_t = config.ema * ps + (1 - config.ema) * p_t
          self._updates += [(ps, ps_t)]

  def get_updates(self):
    return self._updates

  def get_global_grad_norm(self):
    return self._global_grad_norm

  def get_lr_value(self):
    return self._lr.get_value()



class AdamOptimizer(object):

  def __init__(self, config, loss, params):
    self._lr = get_shared_floatX(config.learning_rate, 'lr')
    self._t = get_shared_floatX(1, 't')
    self._all_m_tm1 = []
    self._all_v_tm1 = []
    self._updates = [(self._t, self._t + 1)]

    if config.lr_decay:
      lr_coef = tt.pow(config.lr_decay, (self._t - 1) // config.lr_decay_freq)
      self._updates.append((self._lr, lr_coef * config.learning_rate))

    grads = theano.grad(loss, params)
    #grads = theano.grad(loss, params, disconnected_inputs='ignore')

    self._global_grad_norm = tt.sqrt(tt.sum(tt.stack([tt.sum(g**2.) for g in grads])))
    if config.max_grad_norm:
      global_clip_factor = ifelse(tt.lt(self._global_grad_norm, config.max_grad_norm),
        cast_floatX_np(1.),
        cast_floatX(config.max_grad_norm/self._global_grad_norm))
      # global_clip_factor = tt.minimum(cast_floatX(config.max_grad_norm/self._global_grad_norm), cast_floatX_np(1))
      grads = [global_clip_factor * g for g in grads]

    lr_t = self._lr * \
      clip_sqrt(1 - tt.pow(config.adam_beta2, self._t)) / (1 - tt.pow(config.adam_beta1, self._t))

    for p, g in zip(params, grads):
        m_tm1 = get_shared_floatX(np.zeros_like(p.get_value()), 'adam_m_' + p.name)
        v_tm1 = get_shared_floatX(np.zeros_like(p.get_value()), 'adam_v_' + p.name)
        self._all_m_tm1.append(m_tm1)
        self._all_v_tm1.append(v_tm1)
        m_t = config.adam_beta1 * m_tm1 + (1-config.adam_beta1) * g
        v_t = config.adam_beta2 * v_tm1 + (1-config.adam_beta2) * tt.sqr(g)
        delta_t = -lr_t * m_t / (clip_sqrt(v_t) + config.adam_eps)
        p_t = p + delta_t
        self._updates += [(m_tm1, m_t), (v_tm1, v_t), (p, p_t)]

  def get_updates(self):
    return self._updates

  def get_global_grad_norm(self):
    return self._global_grad_norm

  def get_lr_value(self):
    return self._lr.get_value()




class SgdOptimizer(object):

  def __init__(self, config, loss, params):
    self._lr = get_shared_floatX(config.learning_rate, 'lr')
    self._t = get_shared_floatX(1, 't')
    self._updates = [(self._t, self._t + 1)]

    if config.lr_decay:
      lr_coef = tt.pow(config.lr_decay, (self._t - 1) // config.lr_decay_freq)
      self._updates.append((self._lr, lr_coef * config.learning_rate))

    grads = theano.grad(loss, params)
    #grads = theano.grad(loss, params, disconnected_inputs='ignore')

    self._global_grad_norm = tt.sqrt(tt.sum(tt.stack([tt.sum(g**2.) for g in grads])))
    if config.max_grad_norm:
      global_clip_factor = ifelse(tt.lt(self._global_grad_norm, config.max_grad_norm),
        cast_floatX_np(1.),
        cast_floatX(config.max_grad_norm/self._global_grad_norm))
      # global_clip_factor = tt.minimum(cast_floatX(config.max_grad_norm/self._global_grad_norm), cast_floatX_np(1))
      grads = [global_clip_factor * g for g in grads]

    for p, g in zip(params, grads):
      delta_t = -self._lr * g
      p_t = p + delta_t
      self._updates += [(p, p_t)]

  def get_updates(self):
    return self._updates

  def get_global_grad_norm(self):
    return self._global_grad_norm

  def get_lr_value(self):
    return self._lr.get_value()


