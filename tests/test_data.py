import math

from common.container import Pose2D

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

def create_sample_robot_path(map2d):

  path = []
  for i in range(35):
    path.append(Pose2D(-49+i, -17.5, 0))
  path.append(Pose2D(-15, -17.5, math.pi / 2.0))

  for i in range(35):
    path.append(Pose2D(-15, -17.5+i, math.pi / 2.0))
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