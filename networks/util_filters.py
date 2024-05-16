import math
import torch


def tanh01(x):
  # return tf.tanh(x) * 0.5 + 0.5
  return torch.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):
  def get_activation(left, right, initial):
    def activation(x):
      if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left
    return activation
  return get_activation(l, r, initial)
