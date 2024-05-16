import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from networks.util_filters import tanh_range


class Filter(nn.Module):

  def __init__(self, net):
    super(Filter, self).__init__()

    # Specified in child classes
    self.num_filter_parameters = None
    self.short_name = None
    self.filter_parameters = None

  def get_short_name(self):
    assert self.short_name
    return self.short_name

  def get_num_filter_parameters(self):
    assert self.num_filter_parameters
    return self.num_filter_parameters

  def get_begin_filter_parameter(self):
    return self.begin_filter_parameter

  def extract_parameters(self, features):
    return features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())], \
           features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())]

  # Should be implemented in child classes
  def filter_param_regressor(self, features):
    assert False

  # Process the whole image, without masking
  # Should be implemented in child classes
  def process(self, img, param, defog, IcA):
    assert False

  def debug_info_batched(self):
    return False

  def no_high_res(self):
    return False

  # Apply the whole filter with masking
  def apply(self,
            img,
            img_features=None,
            defog_A=None,
            IcA=None,
            specified_parameter=None,
            high_res=None):
    assert (img_features is None) ^ (specified_parameter is None)
    if img_features is not None:
      # num_batchsize
      filter_features, mask_parameters = self.extract_parameters(img_features)
      filter_parameters = self.filter_param_regressor(filter_features)
    else:
      assert not self.use_masking()
      filter_parameters = specified_parameter

    if high_res is not None:
      # working on high res...
      pass
    debug_info = {}
    # We only debug the first image of this batch
    if self.debug_info_batched():
      debug_info['filter_parameters'] = filter_parameters
    else:
      debug_info['filter_parameters'] = filter_parameters[0]
    low_res_output = self.process(img, filter_parameters, defog_A, IcA)
                 
    return low_res_output, filter_parameters

  def use_masking(self):
    return True

  def get_num_mask_parameters(self):
    return 6


class ExposureFilter(Filter):
  # 图像曝光
  def __init__(self, net):
    Filter.__init__(self, net)
    self.short_name = 'E'
    self.begin_filter_parameter = 0
    self.num_filter_parameters = 1
    self.exposure_range = 3.5

  def filter_param_regressor(self, features):
    return tanh_range(
        -self.exposure_range, self.exposure_range, initial=0)(features)

  def process(self, img, param, defog, IcA):
    # import pdb; pdb.set_trace()
    # param:-2.3356
    return img * torch.exp(param * np.log(2))


class UsmFilter(Filter):
  """
  Usm_param is in [Defog_range]
  sharpen (4,1,96,96,96)
  需要构建一个三维的高斯滤波核（图像锐化）
  """
  def __init__(self, net):

    Filter.__init__(self, net)
    self.short_name = 'UF'
    self.begin_filter_parameter = 1
    self.num_filter_parameters = 1
    self.usm_range = (0.0, 5)

  def filter_param_regressor(self, features):
    return tanh_range(*self.usm_range)(features)

  def process(self, img, param, defog_A, IcA):

    # img.shape = (B, 1, W, H, D) kernel.shape = (1,1,5,5,5)
    # 需要构建一个三维的高斯滤波核
    B, _, W, H, D = img.shape
    kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
              [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
              [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # (1,1,5,5)
    # kernel = np.repeat(kernel, 3, axis=0)   # (3,1,5,5)

    kernel = kernel.to(img.device)
    img_reshape = img.permute(0, 4, 1, 2, 3).flatten(0, 1)   # (B*D, 1, W, H)
    output = F.conv2d(img_reshape, kernel, padding=2)
    output = output.view(B, D, 1, W, H).permute(0, 2, 3, 4, 1)  # (B, 1, W, H, D)
    # import pdb; pdb.set_trace()
    # param:0.0141
    img_out = (img - output) * param + img

    return img_out


class GammaFilter(Filter):  #gamma_param is in [1/gamma_range, gamma_range]

  def __init__(self, net):
    Filter.__init__(self, net)
    self.short_name = 'G'
    self.begin_filter_parameter = 2
    self.num_filter_parameters = 1
    self.gamma_range = 3

  def filter_param_regressor(self, features):
    log_gamma_range = np.log(self.gamma_range)
    return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

  def process(self, img, param, defog_A, IcA):
    # import pdb; pdb.set_trace()
    # param:2.396
    zero = torch.zeros_like(img) + 0.00001
    img = torch.where(img <= 0, zero, img)

    return torch.pow(img, param)


class ContrastFilter(Filter):

  def __init__(self, net):
    Filter.__init__(self, net)
    self.short_name = 'Ct'
    self.begin_filter_parameter = 3
    self.num_filter_parameters = 1
    self.cont_range = (0.0, 1.0)

  def filter_param_regressor(self, features):
    # return tf.sigmoid(features)
    # return torch.tanh(features)
    return tanh_range(*self.cont_range)(features)

  def process(self, img, param, defog, IcA):
    # print('param.shape:', param.shape)

    # luminance = torch.minimum(torch.maximum(rgb2lum(img), 0.0), 1.0)
    # img.shape = (B, 1, W, H, D)
    # luminance = rgb2lum(img)  3D医学数据只有1通道，这个操作没意义
    B, _, W, H, D = img.shape
    img = img.view(B, W, H, D, 1)
    luminance = img.view(B, W, H, D, 1)
    zero = torch.zeros_like(luminance)
    one = torch.ones_like(luminance)
    # 截断，限制像素值在(0,1)之间
    luminance = torch.where(luminance < 0, zero, luminance)
    luminance = torch.where(luminance > 1, one, luminance)

    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    img_out = lerp(img, contrast_image, param)
    img_out = img_out.view(B, 1, W, H, D)
    return img_out
    # return lerp(img, contrast_image, torch.tensor(0.015).cuda())


def rgb2lum(image):
  image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :, 1] + 0.06 * image[:, :, :, 2]
  return image[:, :, :, None]


def lerp(a, b, l):
  return (1 - l) * a + l * b