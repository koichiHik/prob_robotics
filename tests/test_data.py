import math
import numpy as np

from common.container import Pose2D
from common.container import Coord2D
from common.container import Control
from common.math_func import normalize_angle_pi_2_pi

from grid_map.grid_map_2d import GridMap2D
from grid_map.grid_map_2d import GridMap2DDrawer

def create_sample_room(map2d):
  
  conf = map2d.get_map_config()

  GridMap2DDrawer.fill_area(map2d, conf.x_min, -15, -40, conf.y_max, 1)
  GridMap2DDrawer.fill_area(map2d, -35, -15, -20, 15, 1)
  GridMap2DDrawer.fill_area(map2d, -10, -15, 0, 10, 1)
  GridMap2DDrawer.fill_area(map2d, 30, -15, conf.x_max, 10, 1)
  GridMap2DDrawer.fill_area(map2d, 15, -9, 17, -7, 1)
  GridMap2DDrawer.fill_area(map2d, 15, -5, 17, -3, 1)
  GridMap2DDrawer.fill_area(map2d, 15, -1, 17, 1, 1)
  GridMap2DDrawer.fill_area(map2d, 15, 3, 17, 5, 1)
  GridMap2DDrawer.fill_area(map2d, 15, 7, 17, 9, 1)
  GridMap2DDrawer.fill_area(map2d, conf.x_min, conf.y_min, conf.x_max, -20, 1)
  GridMap2DDrawer.fill_area(map2d, conf.x_min, 20, conf.x_max, conf.y_max, 1)

def create_sample_1d_path():

  path = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

  return path

def create_robocup_field(map2d):

  poles = [Coord2D(x=2.2, y=1.45),\
           Coord2D(x=0, y=1.45), \
           Coord2D(x=-2.2, y=1.45), \
           Coord2D(x=-2.2, y=-1.45), \
           Coord2D(x=0, y=-1.45), \
           Coord2D(x=2.2, y=-1.45)]

  for pole in poles:
    GridMap2DDrawer.draw_point(map2d, pole.x, pole.y, 0.10, 1.0)

  return poles

def create_robocup_ellipse_path(dT):

  a = 1.5
  b = 0.9
  omg = 2 * math.pi / 60.0
  start = -3.0 / 4.0 * math.pi
  end = 3.0 / 4.0 * math.pi

  path = []
  controls = []
  for angle in np.arange(start, end, omg * dT):
    x = a * math.cos(angle)
    y = b * math.sin(angle)
    dx_dt = -a * math.sin(angle) * omg
    dy_dt = b * math.cos(angle) * omg
    v = math.sqrt(dx_dt**2 + dy_dt**2)
    
    theta = normalize_angle_pi_2_pi(math.atan2(dy_dt, dx_dt))
    path.append(Pose2D(x=x, y=y, theta=theta))
    controls.append(Control(v=v, omg=omg))
  
  return path, controls

def draw_path(path, map2d):
  for pose in path:
    GridMap2DDrawer.draw_point(map2d, pose.x, pose.y, 0.04, 1.0)

def create_sample_robot_path(map2d, trans_reso=1.0, angle_reso=math.pi/2.0):

  path = []
  for i in range(35 * 4):
    path.append(Pose2D(-49+i*0.25, -17.5, 0))

  path.append(Pose2D(-15, -17.5, 0.0))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0 * 0.1))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0 * 0.2))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0 * 0.3))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0 * 0.4))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0 * 0.5))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0 * 0.6))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0 * 0.7))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0 * 0.8))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0 * 0.9))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0))

  #for i in range(35):
  #  path.append(Pose2D(-15, -17.5+i, math.pi / 2.0))
  for i in range(35 * 4):
    path.append(Pose2D(-15, -17.5+i*0.25, math.pi / 2.0))
  path.append(Pose2D(-15, 17.5, math.pi))

  for i in range(23):
    path.append(Pose2D(-15-i, 17.5, math.pi))
  path.append(Pose2D(-37, 17.5, math.pi * 3.0 / 2.0))

  for i in range(35):
    path.append(Pose2D(-37, 17.5-i, math.pi * 3.0 / 2.0))
  path.append(Pose2D(-37, -17.5, 0.0))

  for i in range(61):
    path.append(Pose2D(-37+i, -17.5, 0))
  path.append(Pose2D(24, -17.5, math.pi / 2.0))

  for i in range(35):
    path.append(Pose2D(24, -17.5+i, math.pi / 2.0))
  path.append(Pose2D(24, 17.5, math.pi))

  for i in range(20):
    path.append(Pose2D(24-i, 17.5, math.pi))
  path.append(Pose2D(4, 17.5, math.pi * 3.0 / 2.0))

  for i in range(32):
    path.append(Pose2D(4, 17.5-i, math.pi * 3.0 / 2.0))
  path.append(Pose2D(4, -14.5, 0))

  for i in range(18):
    path.append(Pose2D(4+i, -14.5, 0))
  path.append(Pose2D(22, -14.5, math.pi / 2.0))

  for i in range(31):
    path.append(Pose2D(22, -14.5+i, math.pi / 2.0))  
  path.append(Pose2D(22, 15.5, 0))

  for i in range(28):
    path.append(Pose2D(22+i, 15.5, 0))  

  for pose in path:
    GridMap2DDrawer.draw_point(map2d, pose.x, pose.y, 0.25, 1.0)

  return path