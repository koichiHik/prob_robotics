
from abc import ABCMeta, abstractmethod

class IMotionModel(metaclass=ABCMeta):

  @abstractmethod
  def sample_motion(self,*, cur_odom, last_odom, last_particle_poses):
    pass

class IScanMatcher(metaclass=ABCMeta):

  @abstractmethod
  def update_pose_via_scan_match(self, *, cur_pose, scans, map2d):
    pass

  @abstractmethod
  def calc_likelihood(self, *, pose, scan, map2d):
    pass

class ILocalizer(metaclass=ABCMeta):

  @abstractmethod
  def localize(self, *, cur_odom, last_odom, meas):
    pass

  @abstractmethod
  def get_current_pose(self, all_pose=False):
    pass

class ISlam(metaclass=ABCMeta):
  
  @abstractmethod
  def update(self, *, cur_odom, last_odom, meas):
    pass

  @abstractmethod
  def get_current_pose(self, all_pose=False):
    pass

  @abstractmethod
  def get_current_map(self, all_map=False):
    pass