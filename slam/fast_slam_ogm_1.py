
import copy

# Slam Interface
from slam.abstract_slam import AbstractSLAM

# Sampler Module
from stats.sampler import LowVarianceSampler

# Common Module
from common.container import Pose2D

# Grid Map Module
from grid_map.grid_map_2d import OccupancyGridMap2D

class ParticleSLAM():
  
  def __init__(self, *, pose=Pose2D(), weight=1.0, map):
    self._pose_list = [pose]
    self._weight = weight
    self._map = map
  
  def set_weight(self, weight):
    self._weight = weight

  def set_pose(self, pose):
    self._pose_list.append(pose)

  def get_weight(self):
    return self._weight

  def get_pose_list(self):
    return self._pose_list

  def get_map(self):
    return self._map

class FastSLAMOGM_Ver1(AbstractSLAM):

  def __init__(self, *,
               particle_num=10,
               init_pose=Pose2D(),
               sampler=LowVarianceSampler(),
               err_model,
               likelihood_generator,
               grid_map_conf,
               inv_sens_model):

    self._particle_num = particle_num
    self._init_pose = init_pose
    self._sampler = sampler
    self._err_model = err_model
    self._likelihood_generator = likelihood_generator

    # Particle Vector Generation
    map_elem = OccupancyGridMap2D(conf=grid_map_conf, \
                        inv_sens_model=inv_sens_model)
    particle = ParticleSLAM(pose=self._init_pose, weight=float(1/self._particle_num), map=map_elem)

    self._particles = [copy.deepcopy(particle) for i in range(self._particle_num)]

  def update(self, *, cur_odom, last_odom, meas):
    
    # 1. Sample Motion.
    cur_pose_list = self.__sample_motion(cur_odom, last_odom)

    # 2. Weighting Calculation
    self.__weighting_by_measurment(meas)

    # 3. Update Occupancy Grid
    self.__update_occupancy_grid(meas)

    # 4. Resample
    self.__resample()

  def get_current_pose(self, num=1):

    length = len(self._particles[0].get_pose_list())
    if (num==1):
      return self._particles[0].get_pose_list()[length - 1]
    else:
      pose_list = []
      for idx in range(min(self._particle_num, num)):
        pose_list.append(self._particles[idx].get_pose_list()[length - 1])
      return pose_list

  def get_current_map(self, num=1):

    if (num==1):
      return self._particles[0].get_map()
    else:
      map_list = []
      for idx in range(min(self._particle_num, num)):
        map_list.append(self._particles[idx].get_map())
      return map_list

  def __sample_motion(self, cur_odom, last_odom):

    length = len(self._particles[0].get_pose_list())
    cur_pose_list = [particle.get_pose_list()[length-1] for particle in self._particles]

    cur_pose_list = \
        self._err_model.sample_motion(cur_odom=cur_odom, \
                                           last_odom=last_odom, \
                                           last_particle_poses=cur_pose_list)

    # Pose Update
    for index, particle in enumerate(self._particles):
      particle.set_pose(cur_pose_list[index])

    return cur_pose_list

  def __weighting_by_measurment(self, scans):

    length = len(self._particles[0].get_pose_list())
    for particle in self._particles:
      weight = self._likelihood_generator.calc_likelihood( \
                          pose=particle.get_pose_list()[length-1], scans=scans)
      particle.set_weight(weight)

  def __update_occupancy_grid(self, scans):

    length = len(self._particles[0].get_pose_list())
    for particle in self._particles:
      pose = particle.get_pose_list()[length-1]
      particle.get_map().register_scan(pose=pose, scans=scans)

  def __resample(self):

    weight_list = [particle.get_weight() for particle in self._particles]
    sampled_idx = self._sampler.sample(weights_list=weight_list)

    for index, par_idx in enumerate(sampled_idx):

      if (index == par_idx):
        continue;

      self._particles[index] = copy.deepcopy(self._particles[par_idx])

    # Upon Finishing Resample, Sort.
    self._particles.sort(key=lambda x:x.get_weight(),reverse=True) 

    # Resetting Weight for Next Iteration.
    M = len(weight_list)
    for index, particle in enumerate(self._particles):
      particle.set_weight(float(1/M))



