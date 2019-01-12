
import numpy as np
import math
import copy

from common.math_func import my_round

class Histogram1DConfigParams:

  def __init__(self, *, x_min, x_max, \
               reso, pred_sigma, meas_sigma, \
               gt_landmark, init_pose=0.0):
    self.x_min = x_min
    self.x_max = x_max
    self.reso = reso
    self.init_pose = init_pose
    self.x_tick_num = my_round((x_max-x_min)/reso) + 1
    self.pred_sigma = pred_sigma
    self.meas_sigma = meas_sigma
    self.gt_landmark = gt_landmark

class HistogramFilter1D():

  def __init__(self, *, histogram_config):
    self._config = histogram_config

    self._last_prob = np.array(\
      [0.0 for i in range(self._config.x_tick_num)])
    self._new_prob = np.array(\
      [0.0 for i in range(self._config.x_tick_num)])

    st_pos = my_round((self._config.init_pose - \
              self._config.x_min) / self._config.reso)
    self._last_prob[st_pos] = 1.0 / self._config.reso
    self._new_prob[st_pos] = 1.0 / self._config.reso

    delta = self._config.reso * 0.00000001
    self._x_ticks = np.mgrid[\
      slice(self._config.x_min-self._config.reso/2.0, \
      self._config.x_max+self._config.reso/2.0+delta, \
      self._config.reso)]

  def predict(self, u):

    self._last_prob = copy.deepcopy(self._new_prob)
    self._new_prob[:] = 0
    for i in range(self._config.x_tick_num):
      new_x = i * self._config.reso + self._config.x_min
      for j in range(self._config.x_tick_num):
        last_x = j * self._config.reso + self._config.x_min
        dx = new_x - last_x
        transprob = self._trans_prob(dx, u)
        self._new_prob[i] = \
          self._new_prob[i] + transprob * self._last_prob[j]

    # Normalization
    total = np.sum(self._new_prob)
    self._new_prob = self._new_prob / total

  def _trans_prob(self, dx, u):
    scaled_sigma = self._config.pred_sigma * u
    prob = 1 / math.sqrt(2 * math.pi * scaled_sigma**2) \
      * math.exp(-(dx - u)**2 / (2 * scaled_sigma**2))

    return prob

  def meas_update(self, meas):

    self._last_prob = copy.deepcopy(self._new_prob)
    self._new_prob[:] = 0
    for i in range(self._config.x_tick_num):
      x = i * self._config.reso + self._config.x_min
      measprob = self._meas_prob(meas, x)
      self._new_prob[i] = measprob * self._last_prob[i]

    # Normalization
    total = np.sum(self._new_prob)
    self._new_prob = self._new_prob / total

  def _meas_prob(self, meas, x):
    exp_meas = self._config.gt_landmark - x
    prob = 1 / math.sqrt(2 * math.pi * self._config.meas_sigma**2) \
      * math.exp(-(exp_meas - meas)**2 / (2 * self._config.meas_sigma**2))
    return prob
