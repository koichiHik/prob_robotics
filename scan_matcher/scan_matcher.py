
import sys
import os
sys.path.append(os.pardir)

import numpy as np
import math

# Interface Module
from interface.interface import IScanMatcher

# Sensing Model Module
from sensing_model.virtual_lidar2d import VirtualLidar2D
from sensing_model.virtual_lidar2d import VirtualLidar2DConfigParams

# Common Module
from common.container import Scans

class ScanMatcherConfigParams():
  
  def __init__(self, *, zhit, zshort, zmax, zmax_width, zrand, lambda_short):
    self._zhit = zhit
    self._zshort = zshort
    self._zmax = zmax
    self._zmax_width = zmax_width
    self._zrand = zrand
    self._lambda_short = lambda_short

class ScanMatcher(IScanMatcher):

  def __init__(self, *, lidar_config, scan_match_config):

    self.__range_max = lidar_config.range_max
    self.__sigma = lidar_config.sigma
    self.__lidar_scan_gen = \
        VirtualLidar2D(lidar_config=lidar_config)
    self.__scan_match_cfg = scan_match_config

    bin = 0.001
    normalization = 0
    
    for i in range(int(self.__range_max/bin)):
      normalization = normalization + (1 / math.sqrt(2*math.pi*self.__sigma**2) \
      * math.exp(-(self.__range_max/2.0 - bin*i)**2 / (2*self.__sigma**2)))
    print("Integral Result @ Likelihood Generator : {0}".format(str(normalization)))

  def update_pose_via_scan_match(self, *, cur_pose, scans, map2d):
    pass

  def calc_likelihood(self, *, pose, scans, map2d):

    scans_wrt_pose = self.__lidar_scan_gen.generate_scans(pose, map2d)
    prob_array_hit = np.exp(-(scans.ranges - scans_wrt_pose.ranges)**2 \
                        / (2*self.__sigma**2)) * self.__scan_match_cfg._zhit
    #prob_array_short 
    prob_array_max = (self.__range_max - self.__scan_match_cfg._zmax_width < scans.ranges) \
                        * self.__scan_match_cfg._zmax
    prob_array_rand = np.zeros(prob_array_hit.shape[0]) + 1 / self.__range_max \
                        * self.__scan_match_cfg._zrand

    p = 0
    for i in range(prob_array_hit.shape[0]):
      pz = prob_array_hit[i] + prob_array_max[i] + prob_array_rand[i]
      p = p + pz**10
      #p = p * (prob_array_hit[i] + prob_array_max[i] + prob_array_rand[i])

    return p

if __name__ == "__main__":

  from grid_map.grid_map_2d import GridMap2DConfigParams
  from grid_map.grid_map_2d import GridMap2D
  from common.container import Pose2D

  # Map Property
  grid_conf = GridMap2DConfigParams(
                  x_min=-50.0, y_min=-25.0, \
                  x_max=50.0, y_max=25.0, \
                  reso=0.2, init_val=0.0)
  map2d_src = GridMap2D(gridmap_config=grid_conf)

  for i in range(int(50/0.2)):
    y_coor = i * grid_conf.reso -25.0
    if (map2d_src.is_valid_global_coord(10.0, y_coor)):
      map2d_src.update_val_via_global_coord(x=10.0,y=y_coor,value=1.0)

  # Lidar Property
  lidar_config = VirtualLidar2DConfigParams(range_max=10.0, \
                                   min_angle=-math.pi/2.0, \
                                   max_angle=math.pi/2.0, \
                                   angle_res=math.pi/360.0, \
                                   sigma=2.0)
  fakeScanGen = VirtualLidar2D(lidar_config=lidar_config)

  true_pose = Pose2D(x=5.0,y=0.0,theta=45.0/180.0*math.pi)
  diff_pose = Pose2D(x=4.3,y=0.0,theta=42.0/180.0*math.pi)

  scans = fakeScanGen.generate_scans(true_pose, map2d_src)

  scan_match_cfg = ScanMatcherConfigParams(
                          zhit=0.8, \
                          zshort=0.0,
                          zmax=0.1,
                          zmax_width=0.2,
                          zrand=0.1,
                          lambda_short=0.0)
  likelihoodGen = ScanMatcher(lidar_config=lidar_config, scan_match_config=scan_match_cfg)

  prob = likelihoodGen.calc_likelihood(pose=diff_pose, scans=scans, map2d=map2d_src)

  print(prob)