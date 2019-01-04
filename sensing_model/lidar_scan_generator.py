
import sys
import os
sys.path.append(os.pardir)

import math

# Common Module
from common.container import Scans
from common.container import Pose2D
from common.math_func import my_round
from common.container import MapIntCoord2D

from ray_tracing.ray_tracing_2d import RayTracing2D

# Grid Map 2D Module
from grid_map.grid_map_2d import GridMap2D
from grid_map.grid_map_2d import GridMap2DConfigParams

class LidarConfigParams:

  def __init__(self, *, range_max, min_angle, max_angle, angle_res, sigma=0.0):
    self.range_max = range_max
    self.min_angle = min_angle
    self.max_angle = max_angle
    self.angle_res = angle_res
    self.sigma = sigma
    self.ray_cnt = int(my_round((max_angle - min_angle) / angle_res) + 1)   

class LidarScanGenerator2D():

  def __init__(self, *, lidar_config):
    self._lidar_config = lidar_config

  def generate_scans(self, pose, map2d):

    scans = Scans(max_range = self._lidar_config.range_max,
                  min_angle = self._lidar_config.min_angle, \
                  max_angle = self._lidar_config.max_angle, \
                  angle_res = self._lidar_config.angle_res)

    conf = map2d.get_map_config()

    min_map_coord = MapIntCoord2D(x=0,y=0)
    max_map_coord = MapIntCoord2D(x=conf.x_tick_num,y=conf.y_tick_num)

    x_ray_src_map, y_ray_src_map = \
      GridMap2D.convert_global_2_map(pose.x, pose.y, conf.x_min, conf.y_min, conf.reso)
    ray_src_map_coord = MapIntCoord2D(x=x_ray_src_map, y=y_ray_src_map)

    for i in range(self._lidar_config.ray_cnt):
      beam_angle = pose.theta + self._lidar_config.min_angle + self._lidar_config.angle_res * i
      x_hit = self._lidar_config.range_max * math.cos(beam_angle) + pose.x
      y_hit = self._lidar_config.range_max * math.sin(beam_angle) + pose.y
      
      x_hit_map, y_hit_map = GridMap2D.convert_global_2_map(x_hit, y_hit, conf.x_min, conf.y_min, conf.reso)
      ray_hit_map_coord = MapIntCoord2D(x=x_hit_map,y=y_hit_map)

      # 1. Generate path along this specific ray in map coords.
      path = RayTracing2D.ray_tracing(ray_src_map_coord, \
                                      ray_hit_map_coord, \
                                      min_map_coord, \
                                      max_map_coord)

      # 2. Search area along point list.
      for index, coord in enumerate(path):

        x_map = int(coord.x)
        y_map = int(coord.y)

        # If this coord is outside or reaches last element of the path, set range_max and break.
        if (not map2d.is_valid_map_coord(x_map, y_map) or index == len(path) - 1):
          scans.ranges[i] = self._lidar_config.range_max
          break

        # If the considered cell is occupied, set range and break.
        if (map2d.get_val_via_map_coord(x_map=x_map, y_map=y_map) > 0.7):
          scans.ranges[i] = \
            math.sqrt((x_map - x_ray_src_map)**2 + (y_map - y_ray_src_map)**2) * conf.reso
          break
      
    return scans

if __name__ == "__main__":

  import matplotlib.pyplot as plt


  config_src = GridMap2DConfigParams(\
              x_min=-50.0, y_min=-25.0, \
              x_max=50.0, y_max=25.0, \
              reso=0.2, init_val=0.0)
  map2d_src = GridMap2D(gridmap_config=config_src)

  config_dst = GridMap2DConfigParams(\
              x_min=-50.0, y_min=-25.0, \
              x_max=50.0, y_max=25.0, \
              reso=0.2, init_val=0.5)
  map2d_dst = GridMap2D(gridmap_config=config_dst)

  for i in range(int(50/0.2)):
    y_coor = i * config_src.reso - 25.0
  
    if (map2d_src.is_valid_global_coord(10.0, y_coor)):
      map2d_src.update_val_via_global_coord(x=10.0,y=y_coor,value=1.0)

  # Lidar Property
  lidar_config = LidarConfigParams(range_max=10.0, \
                                   min_angle=-math.pi/2.0, \
                                   max_angle=math.pi/2.0, \
                                   angle_res=math.pi/360.0, \
                                   sigma=2.0)
  fakeScanGen = LidarScanGenerator2D(lidar_config=lidar_config)

  pose = Pose2D(x=5.0,y=0.0,theta=45.0/180.0*math.pi)
  scans = fakeScanGen.generate_scans(pose, map2d_src)

  fig, ax = plt.subplots(1,1)

  map2d_dst.register_scan(pose=pose, scans=scans)
  map2d_dst.show_heatmap(ax)
  plt.show()