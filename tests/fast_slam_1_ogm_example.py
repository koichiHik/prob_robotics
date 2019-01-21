
import sys
import os
sys.path.append(os.pardir)

import time
import math
import matplotlib.pyplot as plt

# Common Module
from common.container import Pose2D

# Motion Model Module
from motion_model.motion_model import MotionErrorModel2dConfigParams
from motion_model.motion_model import MotionErrorModel2D

# Sensing Model Module
from sensing_model.virtual_lidar2d import VirtualLidar2DConfigParams
from sensing_model.virtual_lidar2d import VirtualLidar2D
from sensing_model.inverse_model import InverseRangeSensorModelConfigParams
from sensing_model.inverse_model import InverseRangeSensorModel

# Scan Matching Module
from scan_matcher.scan_matcher import ScanMatcherConfigParams
from scan_matcher.scan_matcher import ScanMatcher

# Grid Map Module
from grid_map.grid_map_2d import GridMap2DConfigParams
from grid_map.grid_map_2d import GridMap2D
from grid_map.grid_map_2d import GridMap2DDrawer

# SLAM Module
from slam.fast_slam_ogm_1 import FastSLAMOGM_Ver1

# Test Data
from tests.test_data import create_sample_robot_path
from tests.test_data import create_sample_room

if __name__ == '__main__':
  
  print(__file__ + " Started!")

  particle_num=50
  
  # Error Config
  err_conf = MotionErrorModel2dConfigParams(
          std_rot_per_rot=0.01, std_rot_per_trans=0.01,
          std_trans_per_trans=0.01, std_trans_per_rot=0.01)
  err_model = MotionErrorModel2D(conf=err_conf)

  # Grid Map Configuration
  grid_conf_src = GridMap2DConfigParams(
                            x_min=-50.0,y_min=-25.0, \
                            x_max=50.0, y_max=25.0, \
                            reso=0.2, init_val=0.0)

  # Grid Generation
  grid2d_gt = GridMap2D(gridmap_config=grid_conf_src)
  grid2d_odom_disp = GridMap2D(gridmap_config=grid_conf_src)
  grid2d_pose = GridMap2D(gridmap_config=grid_conf_src)

  create_sample_room(grid2d_gt)
  create_sample_room(grid2d_odom_disp)
  create_sample_room(grid2d_pose)
  path = create_sample_robot_path(grid2d_odom_disp)

  # Lidar Property
  lidar_config = VirtualLidar2DConfigParams(range_max=15.0, \
                                   min_angle=-math.pi * 3.0 / 4.0, \
                                   max_angle=math.pi * 3.0 / 4.0, \
                                   angle_res=math.pi/360.0, \
                                   sigma=0.05)

  # Inverse Sensor Model
  inv_lidar_conf = InverseRangeSensorModelConfigParams(\
                        range_max=lidar_config.range_max,
                        l0 = 0.0, locc = 0.3, lfree = -0.3, \
                        alpha = 0.8, beta = 0.0)
  inv_lidar_model= InverseRangeSensorModel(conf=inv_lidar_conf)

  # Grid Map Configuration

  # Scan Generator Instantiation
  v_lidar2d = VirtualLidar2D(lidar_config=lidar_config)

  # Scan Likelihood Generator Instantiation
  scan_match_cfg = ScanMatcherConfigParams(
                        zhit=0.8, \
                        zshort=0.0,
                        zmax=0.1,
                        zmax_width=0.2,
                        zrand=0.1,
                        lambda_short=0.0)
  scan_matcher = ScanMatcher(lidar_config=lidar_config, \
                             scan_match_config=scan_match_cfg)

  grid_conf_ogm = GridMap2DConfigParams(
                            x_min=-50.0,y_min=-25.0, \
                            x_max=50.0, y_max=25.0, \
                            reso=0.2, init_val=inv_lidar_conf.l0)
  # SLAM Instantiation
  fast_slam_ogm = FastSLAMOGM_Ver1(particle_num=particle_num, \
                                   init_pose=path[0], \
                                   err_model=err_model,\
                                   likelihood_generator=scan_matcher,\
                                   grid_map_conf=grid_conf_ogm, \
                                   inv_sens_model=inv_lidar_model)

  #fig_scanmap, ax_scanmap = plt.subplots(nrows=1,ncols=1,figsize=(13, 9),dpi=100)
  #fig_est_pose, ax_est_pose = plt.subplots(nrows=1,ncols=1,figsize=(13, 9),dpi=100)
  fig, (ax_est_pose, ax_scanmap) = plt.subplots(nrows=2,ncols=1,figsize=(10, 9),dpi=100)
  #fig, (ax00, ax10) = plt.subplots(nrows=2,ncols=1,figsize=(4, 6),dpi=100)

  #grid2d_gt.show_heatmap(ax00)
  #grid2d_odom_disp.show_heatmap(ax10)

  last_true_pose = path[0]
  cur_noised_pose = [last_true_pose]
  last_noised_pose = [last_true_pose]
  for index, cur_true_pose in enumerate(path):
    print("Loop {0}".format(str(index)))

    if (index==0):
      continue

    # Create Noised Odometry
    cur_noised_pose = err_model.sample_motion(cur_odom=cur_true_pose, last_odom=last_true_pose, last_particle_poses=last_noised_pose)

    # Generate Artificial Scan
    # This odometry has to be true pose.
    start_time = time.time()
    scans = v_lidar2d.generate_scans(cur_true_pose, grid2d_gt)
    last_true_pose = cur_true_pose
    fake_scan_time = time.time() - start_time

    # Generate Localized Pose
    # This odometry has to be noised pose.
    start_time = time.time()
    fast_slam_ogm.update(cur_odom=cur_noised_pose[0], last_odom=last_noised_pose[0], meas=scans)
    last_noised_pose = cur_noised_pose
    slam_time = time.time() - start_time

    # Data Aquisition
    slam_pose = fast_slam_ogm.get_current_pose()
    slam_map = fast_slam_ogm.get_current_map()

    # Visualization.
    slam_map.show_heatmap(ax_scanmap)
    GridMap2DDrawer.draw_point(grid2d_pose, slam_pose.x, slam_pose.y, 0.2, 1.0)
    #GridMap2DDrawer.draw_point(grid2d_pose, cur_noised_pose[0].x, cur_noised_pose[0].y, 0.2, 1.0)
    grid2d_pose.show_heatmap(ax_est_pose)
    print(slam_pose.theta * 180.0 / math.pi)

    if (index % 10 == 0):
      plt.pause(0.1)

    print("fake_scan_time : {0}".format(fake_scan_time) + "[sec]")
    print("fast_slam_time : {0}".format(str(slam_time)) + "[sec]")
    