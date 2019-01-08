
import copy

# Common Module
from common.container import ParticleMCL

# Sampler Module
from stats.sampler import AbstractSampler
from stats.sampler import LowVarianceSampler

# Motion Model Module
from motion_model.motion_model import MotionErrorModel2D

# Sensing Model Module
from scan_matcher.scan_matcher import ScanMatcherConfigParams
from scan_matcher.scan_matcher import ScanMatcher

# Interface Module
from interface.interface import ILocalizer

class MonteCarloLocalizer(ILocalizer):

  def __init__(self, *, 
                map2d, particle_num=10, \
                init_pose, \
                sampler=LowVarianceSampler(), \
                err_model, \
                scan_matcher):
    self.__particles = [ParticleMCL() for i in range(particle_num)]
    self.__map2d = map2d
    self.__sampler = sampler
    self.__err_model = err_model
    self.__scan_matcher = scan_matcher
    self.__heaviest_particle = ParticleMCL()

    M = len(self.__particles)
    for particle in self.__particles:
      particle.update_pose(copy.deepcopy(init_pose))
      particle.update_weight(float(1/M))

  def localize(self, *, cur_odom, last_odom, scans):

    cur_pose_list = self.__sample_motion(cur_odom, last_odom)

    self.__weighting_by_measurement(scans, cur_pose_list)

    self.__heaviest_particle = self.__resapmle()

    return self.__heaviest_particle.get_pose()

  def get_current_pose(self, all_pose=False):

    if (not all_pose):
      return self.__heaviest_particle.get_pose()
    else:
      return [p.get_pose() for p in self.__particles]

  def __sample_motion(self, cur_odom, last_odom):

    cur_pose_list = [particle.get_pose() for particle in self.__particles]

    cur_pose_list = \
        self.__err_model.sample_motion(cur_odom=cur_odom, \
                                           last_odom=last_odom, \
                                           last_particle_poses=cur_pose_list)

    return cur_pose_list

  def __weighting_by_measurement(self, scans, cur_pose_list):

    for index, particle in enumerate(self.__particles):
      particle.update_pose(cur_pose_list[index])
      weight = self.__scan_matcher.calc_likelihood( \
                          pose=particle.get_pose(), scans=scans, map2d=self.__map2d)
      particle.update_weight(weight)

  def __resapmle(self):
    weight_list = [particle.get_weight() for particle in self.__particles]
    sampled_idx = self.__sampler.sample(weights_list=weight_list)

    for index, par_idx in enumerate(sampled_idx):

      if (index == par_idx):
        continue;

      self.__particles[index] = copy.deepcopy(self.__particles[par_idx])

    heaviest_particle = copy.copy(max(self.__particles, key=lambda x: x.get_weight()))

    M = len(weight_list)
    for index, particle in enumerate(self.__particles):
      particle.update_weight(float(1/M))

    return copy.copy(heaviest_particle)
