
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
#from sensing_model.inverse_model import InverseRangeSensorModel

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

    if (not self.is_valid_global_coord(x, y)):
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
    subplot.imshow(X=self._data.T, vmin=0.0, vmax=1.0, cmap='gray_r', \
                   interpolation='none', origin='lower', \
                  extent=[self.config.x_min, self.config.x_max, self.config.y_min, self.config.y_max])

class OccupancyGridMap2D(GridMap2D):

  def __init__(self, *, conf):
    super(OccupancyGridMap2D, self).__init__(gridmap_config=conf)

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
