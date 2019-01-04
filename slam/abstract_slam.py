
from abc import ABCMeta, abstractmethod

class AbstractSLAM(metaclass=ABCMeta):

  def __init__(self):
    pass

  @abstractmethod
  def update(self, *, cur_odom, last_odom, meas):
    pass

  @abstractmethod
  def get_current_pose(self, num=1):
    pass

  @abstractmethod
  def get_current_map(self, num=1):
    pass