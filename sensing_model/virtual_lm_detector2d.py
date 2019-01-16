

import math

# Common Module
from common.math_func import normalize_angle_pi_2_pi
from common.transform import conv_pnt_global_2_local
from common.container import LandmarkMeas


class VirtualLMDetectorConfigParams:

  def __init__(self, range_max, fov):
    self.range_max = range_max
    self.fov = fov

class VirtualLMDetector():

  def __init__(self, *, config, lm_coord_list):
    self._conf = config
    self._lm_coord_list = lm_coord_list

  def detect(self, *, pose):

    idx_list = []
    meas_list = []

    for index, lm_coord in enumerate(self._lm_coord_list):
      
      lm_loc_coord = conv_pnt_global_2_local(pose, lm_coord)  
      phi = math.atan2(lm_loc_coord.y, lm_loc_coord.x)
      phi = normalize_angle_pi_2_pi(phi)
      
      r = math.sqrt(lm_loc_coord.x**2 + lm_loc_coord.y**2)
      if (-self._conf.fov / 2.0 <= phi and \
          phi <= self._conf.fov / 2.0 and \
          r < self._conf.range_max):
        idx_list.append(index)
        lm_meas = LandmarkMeas(r=r, phi=phi)
        meas_list.append(lm_meas)

    return idx_list, meas_list


