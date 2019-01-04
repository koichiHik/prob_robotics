
import sys
import os
sys.path.append(os.pardir)

import math
import numpy as np

import common.math_func as my_math
from common.container import MapIntCoord2D


class RayTracing2D():

  @staticmethod
  def __calc_t_dist_to_next_border(x0, x1):
  
    dist = int(math.fabs(my_math.my_round(x0) - my_math.my_round(x1)))

    if (x0 < x1):
      dt_dx = 1 / (x1 - x0)
      dx = ((math.floor(x0) + 1) - x0)
      proc_dir = 1
      next_t_dist = dt_dx * dx
    elif(x1 < x0):
      dt_dx = 1 / (x0 - x1)
      dx = (x0 - math.floor(x0))
      proc_dir = -1
      next_t_dist = dt_dx * dx
    else:
      dt_dx = float('inf')
      dx = 0.0
      proc_dir = 0
      next_t_dist = float('inf')
    
    return next_t_dist, dt_dx, proc_dir, dist

  @staticmethod
  def ray_tracing(st_pnt, end_pnt, min_pnt, max_pnt):

    next_t_dist_ver, dt_dx, proc_dir_x, dist_x =\
        RayTracing2D.__calc_t_dist_to_next_border(st_pnt.x, end_pnt.x)
    next_t_dist_hor, dt_dy, proc_dir_y, dist_y =\
        RayTracing2D.__calc_t_dist_to_next_border(st_pnt.y, end_pnt.y)
    
    total_dist = dist_x + dist_y + 1    

    x = my_math.my_round(st_pnt.x)
    y = my_math.my_round(st_pnt.y)

    path = np.zeros((2, total_dist), dtype=int)
    path = [MapIntCoord2D() for i in range(total_dist)]

    path[0].x = x
    path[0].y = y

    cnt = 1
    for i in range(total_dist - 1, 0, -1):

      if (next_t_dist_ver < next_t_dist_hor):
        x = x + proc_dir_x
        next_t_dist_ver = next_t_dist_ver + dt_dx
      else:
        y = y + proc_dir_y
        next_t_dist_hor = next_t_dist_hor + dt_dy

      if ((x < min_pnt.x or max_pnt.x < x) or \
          (y < min_pnt.y or max_pnt.y < y)):
        break

      path[cnt].x = x
      path[cnt].y = y
      cnt = cnt + 1

    return path[0:cnt]
    
if __name__ == "__main__":

  st_pnt = MapIntCoord2D(x=0, y=0)
  end_pnt = MapIntCoord2D(x=20, y=5)

  min_pnt = MapIntCoord2D(x=0, y=0)
  max_pnt = MapIntCoord2D(x=10, y=10)
  path = RayTracing2D.ray_tracing(st_pnt, end_pnt, min_pnt, max_pnt)

  for item in path:
    print(item)