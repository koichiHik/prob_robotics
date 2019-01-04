
import sys
import os
sys.path.append(os.pardir)

import math
import time
import matplotlib.pyplot as plt

# Common Module
from common.container import Pose2D

# Sensing Model Module
from sensing_model.lidar_scan_generator import LidarConfigParams
from sensing_model.lidar_scan_generator import LidarScanGenerator2D

# Grid Map Module
from grid_map.grid_map_2d import GridMap2DConfigParams
from grid_map.grid_map_2d import GridMap2D
from grid_map.grid_map_2d import GridMap2DDrawer

# Sample Data
from tests.test_data import create_sample_robot_path
from tests.test_data import create_sample_room

if __name__ == "__main__":

  print(__file__ + "Started!")

  # Grid Map Configuration
  grid_conf_src = GridMap2DConfigParams( \
                              x_min=-50.0, y_min=-25.0, \
                              x_max=50.0, y_max=25.0, \
                              reso=0.2, init_val=0.0)
  grid2d_gt = GridMap2D(gridmap_config=grid_conf_src)
  grid2d_odom_disp = GridMap2D(gridmap_config=grid_conf_src)
  grid2d_pose = GridMap2D(gridmap_config=grid_conf_src)

  # Grid Map Configuration
  grid_conf_dst = GridMap2DConfigParams( \
                              x_min=-50.0, y_min=-25.0, \
                              x_max=50.0, y_max=25.0, \
                              reso=0.2, init_val=0.5)
  grid2d_scanmap = GridMap2D(gridmap_config=grid_conf_dst)

  create_sample_room(grid2d_gt)
  create_sample_room(grid2d_odom_disp)
  create_sample_room(grid2d_pose)
  path = create_sample_robot_path(grid2d_odom_disp)

  # Lidar Property
  lidar_config = LidarConfigParams(range_max=10.0, \
                                   min_angle=-math.pi/2.0, \
                                   max_angle=math.pi/2.0, \
                                   angle_res=math.pi/360.0, \
                                   sigma=2.0)  
  fakeScanGen = LidarScanGenerator2D(lidar_config=lidar_config)

  fig_scanmap, ax_scanmap = plt.subplots(nrows=1,ncols=1,figsize=(13, 9),dpi=100)
  fig, (ax00, ax10, ax20) = plt.subplots(nrows=3,ncols=1,figsize=(4, 6),dpi=100)

  grid2d_gt.show_heatmap(ax00)
  grid2d_odom_disp.show_heatmap(ax10)

  for index, pose in enumerate(path):
    print("Loop : {0}".format(str(index)))

    # Generate Artificial Scan
    start_time = time.time()
    scans = fakeScanGen.generate_scans(pose, grid2d_gt)
    fake_scan_time = time.time() - start_time

    # Register Scans
    start_time = time.time()
    grid2d_scanmap.register_scan(pose=pose, scans=scans)
    scan_registration_time = time.time() - start_time

    # Visualization.
    grid2d_scanmap.show_heatmap(ax_scanmap)
    GridMap2DDrawer.draw_point(grid2d_pose, pose.x, pose.y, 0.2, 1.0)
    grid2d_pose.show_heatmap(ax20)

    plt.pause(0.1)

    print("fake_scan_time : {0}".format(fake_scan_time) + "[sec]")
    print("registration_time : {0}".format(scan_registration_time) + "[sec]")
