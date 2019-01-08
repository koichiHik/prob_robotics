
import sys
import os
sys.path.append(os.pardir)

import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# Common Module
from common.container import Pose2D

# Grid Map Module
from grid_map.grid_map_2d import GridMap2DConfigParams
from grid_map.grid_map_2d import GridMap2D
from grid_map.grid_map_2d import GridMap2DDrawer

# Motion Model Module
from motion_model.motion_model import MotionErrorModel2dConfigParams
from motion_model.motion_model import MotionErrorModel2D

# Test Data Module
from tests.test_data import create_sample_robot_path
from tests.test_data import create_sample_room

if __name__ == "__main__":

  # Motion Model Configuration
  particle_num=100
  err_conf = MotionErrorModel2dConfigParams(std_rot_per_rot=0.01,\
                                            std_rot_per_trans=0.02,\
                                            std_trans_per_trans=0.05,\
                                            std_trans_per_rot=0.01)
  err_model = MotionErrorModel2D(conf=err_conf)

  # Grid Map Configuration
  grid_conf = GridMap2DConfigParams(\
                    x_min=-50.0, y_min=-25.0, \
                    x_max=50.0, y_max=25.0, reso=0.2, init_val=0.0)
  grid2d_gt = GridMap2D(gridmap_config=grid_conf)
  grid2d_odom_disp = GridMap2D(gridmap_config=grid_conf)
  grid2d_pose = GridMap2D(gridmap_config=grid_conf)

  # Create Testing Configuration.
  create_sample_room(grid2d_gt)
  create_sample_room(grid2d_odom_disp)
  create_sample_room(grid2d_pose)
  path = create_sample_robot_path(grid2d_odom_disp)

  # Visualization Preparation
  fig, ax_main = plt.subplots(nrows=1,ncols=1,figsize=(13, 9),dpi=100)
  fig, (ax00, ax10) = plt.subplots(nrows=2,ncols=1,figsize=(4, 6),dpi=100)
  grid2d_gt.show_heatmap(ax00)
  grid2d_odom_disp.show_heatmap(ax10)

  cur_particle_poses = [Pose2D() for i in range(particle_num)]
  last_particle_poses = [Pose2D() for i in range(particle_num)]

  last_pose = Pose2D()
  for index, cur_pose in enumerate(path):  
    print("Loop {0}".format(str(index)))

    if index == 0:
      last_pose = cur_pose
      for i in range(particle_num):
        cur_particle_poses[i] = cur_pose
      continue

    # Sample Artificial Odometry Error 
    cur_particle_poses = \
          err_model.sample_motion(cur_odom=cur_pose, \
                                  last_odom=last_pose, \
                                  last_particle_poses=cur_particle_poses)   
    last_pose = cur_pose

    # Update Odometry Map
    for i in range(particle_num):
      GridMap2DDrawer.draw_point(grid2d_pose, \
                                cur_particle_poses[i].x, \
                                cur_particle_poses[i].y, \
                                0.0, 1.0)

    grid2d_pose.show_heatmap(ax_main)

    # Pause for Visualization Update
    if (index % 30 == 0):
      plt.pause(0.1)
