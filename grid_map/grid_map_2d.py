
import sys
import os
sys.path.append(os.pardir)

import math
import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow
from pylab import figure
from abc import ABCMeta, abstractmethod

from ray_tracing.ray_tracing_2d import RayTracing2D

# Common Module
from common.math_func import my_round
from common.container import Pose2D
from common.container import Scans
from common.container import Size
from common.container import MapIntCoord2D
from common.container import Coord2D

# Sensing Model Module
from sensing_model.inverse_model import InverseRangeSensorModel

class GridMap2DConfigParams:

  def __init__(self, *, x_min, y_min, x_max, y_max, \
                        reso, init_val=0.0):
    self.x_min=x_min
    self.y_min=y_min
    self.x_max=x_max
    self.y_max=y_max
    self.reso=reso
    self.init_val=init_val
    self.x_tick_num=my_round((x_max-x_min)/reso)+1
    self.y_tick_num=my_round((y_max-y_min)/reso)+1

class GridMap2D(object):

  __metaclass__ = ABCMeta

  def __init__(self, *, gridmap_config):
    self.config=gridmap_config

    self._data = np.array(\
      [[self.config.init_val for i in range(self.config.y_tick_num)] for j in range(self.config.x_tick_num)])

    # Hack for rounding error.
    delta = self.config.reso * 0.00000001
    self._x_ticks, self._y_ticks = \
          np.mgrid[slice(self.config.x_min-self.config.reso/2.0, \
                         self.config.x_max+self.config.reso/2.0+delta, self.config.reso),
                   slice(self.config.y_min-self.config.reso/2.0, \
                         self.config.y_max+self.config.reso/2.0+delta, self.config.reso)]

  def update_val_via_global_coord(self, *, x, y, value):

    if (self.is_valid_global_coord(x, y)):
      assert(False)

    x_map, y_map = \
      GridMap2D.convert_global_2_map(x, y, self.config.x_min, self.config.y_min, self.config.reso)
    self._data[x_map, y_map] = value

  def get_val_via_global_coord(self, *, x, y):

    if (not self.is_valid_global_coord(x, y)):
      assert(False)

    x_map, y_map = \
      GridMap2D.convert_global_2_map(x, y, self.config.x_min, self.config.y_min, self.config.reso)
    return self._data[x_map, y_map]

  def update_val_via_map_coord(self, *, x_map, y_map, value):
    if (self.is_valid_map_coord(x_map, y_map)):
      self._data[x_map, y_map] = value

  def fill_val_via_map_coord(self, *, x_map_st, y_map_st, x_map_en, y_map_en, value):
    x_st = int(max(0, x_map_st))
    y_st = int(max(0, y_map_st))
    x_en = int(min(x_map_en+1, self.config.x_tick_num - 1))
    y_en = int(min(y_map_en+1, self.config.y_tick_num - 1))
    self._data[min(x_st,x_en):max(x_st,x_en), min(y_st,y_en):max(y_st,y_en)] = value

  def get_val_via_map_coord(self, *, x_map, y_map):

    if (not self.is_valid_map_coord(x_map, y_map)):
      assert(False)

    return self._data[x_map, y_map]

  @abstractmethod
  def register_scan(self, *, pose, scans):
    
    # Convert coordinate from global to map.
    x_map_scan_st, y_map_scan_st = \
      GridMap2D.convert_global_2_map(pose.x, pose.y, self.config.x_min, self.config.y_min, self.config.reso)
    map_scan_st = MapIntCoord2D(x=x_map_scan_st, y=y_map_scan_st)
    map_scan_ends_world, map_scan_ends_map = self._convert_scan_hit_pnt_2_world_and_map_pnt(pose, scans)

    min_pnt = MapIntCoord2D(x=0,y=0)
    max_pnt = MapIntCoord2D(x=self.config.x_tick_num - 1,y=self.config.y_tick_num - 1)

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
        if (not self.is_valid_map_coord(x_map, y_map)):
          break

        if (self._data[x_map,y_map] != 1.0):
          self._data[x_map,y_map] = 0.0

      x_hit_map = my_round(path[len(path)-1].x)
      y_hit_map = my_round(path[len(path)-1].y)

      if (self.is_valid_map_coord(x_hit_map, y_hit_map) and scans.ranges[scan_idx] < scans.max_range):
        self._data[x_hit_map, y_hit_map] = 1.0

  def get_map_config(self):
    return self.config

  def get_map_data(self):
    return self._data

  def is_valid_global_coord(self, x, y):
    if (x < self.config.x_min or self.config.x_max < x or \
        y < self.config.y_min or self.config.y_max < y):
      return False
    else:
      return True

  def is_valid_map_coord(self, x_map, y_map):
    if (x_map < 0 or y_map < 0 or \
        self._data.shape[0] <= x_map or self._data.shape[1] <= y_map):
      return False
    else:
      return True

  def _convert_scan_hit_pnt_2_world_and_map_pnt(self, pose, scans):

    x_map, y_map = \
      GridMap2D.convert_global_2_map(pose.x, pose.y, self.config.x_min, self.config.y_min, self.config.reso)

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
            GridMap2D.convert_global_2_map(x, y, self.config.x_min, self.config.y_min, self.config.reso)
      else:
        coord_world.x = pose.x
        coord_world.y = pose.y
        coord_map.x = x_map
        coord_map.y = y_map

    return end_pnts_world, end_pnts_map

  @staticmethod
  def convert_map_2_global(x_map, y_map, min_x, min_y, reso):
    x = x_map * reso + min_x
    y = y_map * reso + min_y
    return x, y

  @staticmethod
  def convert_global_2_map(x, y, min_x, min_y, reso):
    x_map = my_round((x - min_x)/reso)
    y_map = my_round((y - min_y)/reso)
    return x_map, y_map

  @abstractmethod
  def show_heatmap(self, subplot):
    subplot.imshow(X=self._data.T, cmap='gray_r', interpolation='none', origin='lower', \
                  extent=[self.config.x_min, self.config.x_max, self.config.y_min, self.config.y_max])

class OccupancyGridMap2D(GridMap2D):

  def __init__(self, *, conf, inv_sens_model):

    super(OccupancyGridMap2D, self).__init__(gridmap_config=conf)
    self._inv_sens_model = inv_sens_model

  def register_scan(self, *, pose, scans):

    # Convert coordinate from global to map.
    x_map_scan_st, y_map_scan_st = \
      GridMap2D.convert_global_2_map(pose.x, pose.y, self.config.x_min, self.config.y_min, self.config.reso)
    map_scan_st = MapIntCoord2D(x=x_map_scan_st, y=y_map_scan_st)
    map_scan_ends_world, map_scan_ends_map = self._convert_scan_hit_pnt_2_world_and_map_pnt(pose, scans)

    min_pnt = MapIntCoord2D(x=0,y=0)
    max_pnt = MapIntCoord2D(x=self.config.x_tick_num - 1,y=self.config.y_tick_num - 1)

    for scan_idx, end_coord in enumerate(map_scan_ends_map):
  
      if (map_scan_st.x == end_coord.x and map_scan_st.y == end_coord.y):
        continue

      path = RayTracing2D.ray_tracing(map_scan_st, end_coord, min_pnt, max_pnt)

      for coord in path:          
        x_world, y_world = \
          GridMap2D.convert_map_2_global(\
            coord.x, coord.y, self.config.x_min, self.config.y_min, self.config.reso)
        
        tgt_world = Coord2D(x=x_world, y=y_world)
        add_log = self._inv_sens_model.log_likelihood(\
                          pose=pose, \
                          tgt=tgt_world, \
                          scanned_range=scans.ranges[scan_idx])

        self._data[coord.x, coord.y] = \
            self._data[coord.x, coord.y] + add_log - self.config.init_val           

  def show_heatmap(self, subplot):
    vis_data = 1 - 1 / (1 + np.exp(self._data))
    subplot.imshow(X=vis_data.T, vmin=0.0, vmax=1.0, cmap='gray_r', \
                   interpolation='none', origin='lower', \
                  extent=[self.config.x_min, self.config.x_max, self.config.y_min, self.config.y_max])  


class GridMap2DDrawer():

  @staticmethod
  def draw_line(map2d, x_st, y_st, x_en, y_en, value):

    conf = map2d.get_map_config()

    x_st_map, y_st_map = GridMap2D.convert_global_2_map(x_st, y_st, conf.x_min, conf.y_min, conf.reso)
    x_en_map, y_en_map = GridMap2D.convert_global_2_map(x_en, y_en, conf.x_min, conf.y_min, conf.reso)

    st_pnt_map = MapIntCoord2D(x=x_st_map, y=y_st_map)
    end_pnt_map = MapIntCoord2D(x=x_en_map, y=y_en_map)
    min_pnt_map = MapIntCoord2D(x=0, y=0)
    max_pnt_map = MapIntCoord2D(x=conf.x_tick_num - 1, y=conf.y_tick_num - 1)
    
    path = RayTracing2D.ray_tracing(st_pnt_map, end_pnt_map, min_pnt_map, max_pnt_map)

    for coord in path:
      map2d.update_val_via_map_coord(x_map=coord.x, y_map=coord.y, value=value)

  @staticmethod
  def fill_area(map2d, x_st, y_st, x_en, y_en, value):

    conf = map2d.get_map_config()

    x_st_map, y_st_map = GridMap2D.convert_global_2_map(x_st, y_st, conf.x_min, conf.y_min, conf.reso)
    x_en_map, y_en_map = GridMap2D.convert_global_2_map(x_en, y_en, conf.x_min, conf.y_min, conf.reso)

    map2d.fill_val_via_map_coord(\
        x_map_st=x_st_map, \
        y_map_st=y_st_map, \
        x_map_en=x_en_map, \
        y_map_en=y_en_map, \
        value=value)

  @staticmethod
  def draw_point(map2d, x, y, width, value):

    conf = map2d.get_map_config()
    x_map, y_map = GridMap2D.convert_global_2_map(x, y, conf.x_min, conf.y_min, conf.reso)

    half_width_map = my_round(width / conf.reso / 2.0)
    map2d.fill_val_via_map_coord(
        x_map_st=x_map-half_width_map,
        y_map_st=y_map-half_width_map,
        x_map_en=x_map+half_width_map,
        y_map_en=y_map+half_width_map,
        value=value)
