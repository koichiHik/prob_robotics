
import numpy as np

from common.math_func import my_round

class Pose2D():

  def __init__(self, x=0.0, y=0.0, theta=0.0):
    self.x = x
    self.y = y
    self.theta = theta

  def __str__(self):
    return "(x=" + str(self.x) + ", y=" \
          + str(self.y) + ", theta=" \
          + str(self.theta) + ")"

  def numpy_array(self):
    return np.array([self.x, self.y, self.theta])

class Control():

  def __init__(self, *, v=0.0, omg=0.0):
    self.v = v
    self.omg = omg

  def numpy_array(self):
    return np.array([self.v, self.omg])

class MapIntCoord2D():

  def __init__(self, *, x=0, y=0):
    self.x = int(x)
    self.y = int(y)

  def __str__(self):
    return "(x=" + str(self.x) + ", y=" + str(self.y) + ")"

class Coord2D():

  def __init__(self, *, x=0.0, y=0.0):
    self.x = x
    self.y = y

class Size():
  
  def __init__(self, *, x=0.0, y=0.0):
    self.x = x
    self.y = y

class Scans():

  def __init__(self, *, max_range, min_angle, max_angle, angle_res):
    self.max_range = max_range
    self.min_angle = min_angle
    self.max_angle = max_angle
    self.angle_res = angle_res
    self.ray_cnt = my_round((max_angle - min_angle) / angle_res) + 1
    self.ranges = np.zeros(self.ray_cnt)

class LandmarkMeas():

  def __init__(self, *, r, phi, feat=1.0):
    self.r = r
    self.phi = phi
    self.feat = feat

  def numpy_array(self):
    return np.array([self.r, self.phi, self.feat])

class ParticleMCL():

  def __init__(self, *, pose=Pose2D(), weight=1.0):
    self.__pose = pose
    self.__weight = weight

  def get_weight(self):
    return self.__weight

  def get_pose(self):
    return self.__pose

  def update_pose(self, pose):
    self.__pose = pose

  def update_weight(self, weight):
    self.__weight = weight
