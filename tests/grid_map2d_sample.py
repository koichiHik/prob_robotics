
import sys
import os
sys.path.append(os.pardir)

import matplotlib.pyplot as plt

# Grid Map 2D Module
from grid_map.grid_map_2d import GridMap2DConfigParams
from grid_map.grid_map_2d import GridMap2D

# Sample Data
from tests.test_data import create_sample_robot_path
from tests.test_data import create_sample_room

if __name__ == "__main__":
  
  grid_conf = GridMap2DConfigParams(x_min=-50.0, y_min=-25.0, x_max=50.0, y_max=25.0, reso=0.2, init_val=0.0)
  grid2d = GridMap2D(gridmap_config=grid_conf)

  create_sample_room(grid2d)
  create_sample_robot_path(grid2d)

  fig1, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(14, 9),dpi=100)
  fig2, ax2 = plt.subplots(nrows=1,ncols=1,figsize=(14, 9),dpi=100)

  grid2d.show_heatmap(ax1)
  grid2d.show_heatmap(ax2)
  plt.show()
