
import sys
import os
sys.path.append(os.pardir)

import math

# Common Module
from common.math_func import my_round
from common.container import Coord2D
from common.container import MapIntCoord2D

# Ray Tracing Module
from ray_tracing.ray_tracing_2d import RayTracing2D

# Grid Map 2D Module
from grid_map.grid_map_2d import GridMap2D

class InverseRangeSensorModelConfigParams:

  def __init__(self, *, range_max, l0, locc, lfree, alpha, beta):

    self.range_max = range_max
    self.l0 = l0
    self.locc = locc
    self.lfree = lfree
    self.alpha = alpha
    self.beta = beta

class InverseRangeSensorModel():

  def __init__(self, conf):
    
    self.__range_max = conf.range_max
    self.__l0 = conf.l0
    self.__locc = conf.locc
    self.__lfree = conf.lfree
    self.__alpha = conf.alpha
    self.__beta = conf.beta

  def log_likelihood(self, *, pose, tgt, scanned_range):
    
    dist = math.sqrt((tgt.x - pose.x)**2 + (tgt.y - pose.y)**2)
    phi = math.atan2(tgt.y-pose.y, tgt.x-pose.x) - pose.theta

    val = self.__l0
    if ((self.__range_max < dist) or \
        (scanned_range + self.__alpha/2.0 < dist)):
      val = self.__l0
    elif ((scanned_range < self.__range_max) and \
          (abs(scanned_range-dist) < self.__alpha/2.0)):
      val = self.__locc
    elif (dist < scanned_range):
      val = self.__lfree
    else:
      val = self.__l0

    return val

  def register_scan(self, *, pose, scans, map2d):
    
    conf = map2d.get_map_config()
    # Convert coordinate from global to map.
    x_map_scan_st, y_map_scan_st = \
      GridMap2D.convert_global_2_map(pose.x, pose.y, conf.x_min, conf.y_min, conf.reso)
    map_scan_st = MapIntCoord2D(x=x_map_scan_st, y=y_map_scan_st)
    map_scan_ends_world, map_scan_ends_map = self._convert_scan_hit_pnt_2_world_and_map_pnt(pose, scans, conf)

    min_pnt = MapIntCoord2D(x=0,y=0)
    max_pnt = MapIntCoord2D(x=conf.x_tick_num - 1,y=conf.y_tick_num - 1)

    for scan_idx, end_coord in enumerate(map_scan_ends_map):
  
      if (map_scan_st.x == end_coord.x and map_scan_st.y == end_coord.y):
        continue

      path = RayTracing2D.ray_tracing(map_scan_st, end_coord, min_pnt, max_pnt)

      for coord in path:          
        x_world, y_world = \
          GridMap2D.convert_map_2_global(\
            coord.x, coord.y, conf.x_min, conf.y_min, conf.reso)
        
        tgt_world = Coord2D(x=x_world, y=y_world)
        add_log = self.log_likelihood(\
                          pose=pose, \
                          tgt=tgt_world, \
                          scanned_range=scans.ranges[scan_idx])
               
        val = map2d.get_val_via_map_coord(x_map=coord.x, y_map=coord.y) \
                  + add_log - conf.init_val
        map2d.update_val_via_map_coord(x_map=coord.x, y_map=coord.y, value=val)

  def register_fixed_scan(self, *, pose, scans, map2d):
  
    conf = map2d.get_map_config()
    
    # Convert coordinate from global to map.
    x_map_scan_st, y_map_scan_st = \
      GridMap2D.convert_global_2_map(pose.x, pose.y, conf.x_min, conf.y_min, conf.reso)
    map_scan_st = MapIntCoord2D(x=x_map_scan_st, y=y_map_scan_st)
    map_scan_ends_world, map_scan_ends_map = self._convert_scan_hit_pnt_2_world_and_map_pnt(pose, scans, conf)

    min_pnt = MapIntCoord2D(x=0,y=0)
    max_pnt = MapIntCoord2D(x=conf.x_tick_num - 1,y=conf.y_tick_num - 1)

    for scan_idx, end_coord in enumerate(map_scan_ends_map):
  
      if (map_scan_st.x == end_coord.x and map_scan_st.y == end_coord.y):
        continue

      path = RayTracing2D.ray_tracing(map_scan_st, end_coord, min_pnt, max_pnt)

      for index, coord in enumerate(path):

        # When reacnes last element, break.
        if (index == len(path) - 1):
          break

        x_map = my_round(coord.x)
        y_map = my_round(coord.y)
        if (not map2d.is_valid_map_coord(x_map, y_map)):
          break

        val = map2d.get_val_via_map_coord(x_map=x_map, y_map=y_map)
        if (val != 1.0):
          map2d.update_val_via_map_coord(x_map=x_map, y_map=y_map, value=0.0)

      x_hit_map = my_round(path[len(path)-1].x)
      y_hit_map = my_round(path[len(path)-1].y)

      if (map2d.is_valid_map_coord(x_hit_map, y_hit_map) and scans.ranges[scan_idx] < scans.max_range):
        map2d.update_val_via_map_coord(x_map=x_hit_map, y_map=y_hit_map, value=1.0)

  def _convert_scan_hit_pnt_2_world_and_map_pnt(self, pose, scans, map_conf):
    
    x_map, y_map = \
      GridMap2D.convert_global_2_map(pose.x, pose.y, map_conf.x_min, map_conf.y_min, map_conf.reso)

    end_pnts_world = [Coord2D() for i in range(scans.ray_cnt)]
    end_pnts_map = [MapIntCoord2D() for i in range(scans.ray_cnt)]

    for index in range(scans.ray_cnt):
      beam_angle = pose.theta + scans.min_angle + scans.angle_res * index

      coord_world = end_pnts_world[index]
      coord_map = end_pnts_map[index]

      # If scan is negative, invalid.
      if (scans.ranges[index] > 0):
        x = scans.ranges[index] * math.cos(beam_angle) + pose.x
        y = scans.ranges[index] * math.sin(beam_angle) + pose.y

        coord_world.x = x
        coord_world.y = y
        coord_map.x, coord_map.y = \
            GridMap2D.convert_global_2_map(x, y, map_conf.x_min, map_conf.y_min, map_conf.reso)
      else:
        coord_world.x = pose.x
        coord_world.y = pose.y
        coord_map.x = x_map
        coord_map.y = y_map

    return end_pnts_world, end_pnts_map


if __name__ == "__main__":
  print(__file__ + "Started!")