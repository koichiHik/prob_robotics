
import sys
import os
sys.path.append(os.pardir)

import math
import numpy as np
import matplotlib.pyplot as plt

# Common Module
from common.container import Control

# Grid Map Module
from grid_map.grid_map_2d import GridMap2DConfigParams
from grid_map.grid_map_2d import GridMap2D
from grid_map.grid_map_2d import GridMap2DDrawer

# Sensing Module
from sensing_model.virtual_lm_detector2d import VirtualLMDetectorConfigParams
from sensing_model.virtual_lm_detector2d import VirtualLMDetector

# Test Data Module
from tests.test_data import draw_path
from tests.test_data import create_robocup_field
from tests.test_data import create_robocup_ellipse_path

# EKF Localizer Module
from localizer.ekf_localizer import EKFLocalizerConfigParams
from localizer.ekf_localizer import EKFKCMeas
from localizer.ekf_localizer import EKFLocalizerKC

def generate_noised_control(ctrl, err_v_v, err_v_omg, err_omg_v, err_omg_omg):
  
  sig_v = err_v_v * abs(ctrl.v) + err_v_omg * abs(ctrl.omg)
  sig_omg = err_omg_v * abs(ctrl.v) +  err_omg_omg * abs(ctrl.omg)
  v_noised = ctrl.v + np.random.normal(0, sig_v)
  omg_noised = ctrl.omg + np.random.normal(0, sig_omg)
  return Control(v=v_noised, omg=omg_noised)

def draw_result(plt, true_pose_list, mm_pose_list, ekf_pose_list, poles, meas_idx):

  plt.xlim(-2.6, 2.6)
  plt.ylim(-1.85, 1.85)

  for idx, pose in enumerate(ekf_pose_list):
    if (idx == 0):
      plt.scatter(pose.x, pose.y, s=20, c='r', label='EKF Localization')
    else:
      plt.scatter(pose.x, pose.y, s=20, c='r')

  for idx, pose in enumerate(mm_pose_list):
    if (idx == 0):
      plt.scatter(pose.x, pose.y, s=10, c='blue', label='Motion Model')
    else:
      plt.scatter(pose.x, pose.y, s=10, c='blue')

  for idx, pose in enumerate(true_pose_list):
    if (idx == 0):
      plt.scatter(pose.x, pose.y, s=10, c='k', label='Ground Truth')
    else:
      plt.scatter(pose.x, pose.y, s=10, c='k')

  for idx, pole in enumerate(poles):
    print(meas_idx)
    if (idx in meas_idx):
      plt.scatter(pole.x, pole.y, s=160, c='y')
    else:
      plt.scatter(pole.x, pole.y, s=40, c='k')

  plt.legend(bbox_to_anchor=(0, 0.6), loc='upper left')


if __name__ == "__main__":
  print(__file__ + "Started!")

  # GridMap Preparation
  grid_conf_fld = GridMap2DConfigParams(
          x_min=-2.70,y_min=-1.95,x_max=2.7,y_max=1.95,reso=0.02, init_val=0.0)
  grid_conf_loc = GridMap2DConfigParams(
          x_min=-2.70,y_min=-1.95,x_max=2.7,y_max=1.95,reso=0.02, init_val=0.0)
  grid2d_gt = GridMap2D(gridmap_config=grid_conf_fld)
  grid2d_loc = GridMap2D(gridmap_config=grid_conf_fld)

  dT = 0.05
  poles = create_robocup_field(grid2d_gt)
  poles = create_robocup_field(grid2d_loc)
  ellipse_path, controls = create_robocup_ellipse_path(dT)
  draw_path(ellipse_path, grid2d_gt)

  # Virutual Detector
  vir_lm_det_conf = VirtualLMDetectorConfigParams(\
                                range_max=2.0,
                                fov=math.pi)
  vir_lm_det = VirtualLMDetector(config=vir_lm_det_conf, lm_coord_list=poles)

  # EKF Localizer
  err_v_v = 0.2
  err_v_omg = 0.1
  err_omg_v = 0.1
  err_omg_omg = 0.1
  ekf_conf = EKFLocalizerConfigParams(
                dT=dT, \
                err_v_v=err_v_v, err_v_omg=err_v_omg,
                err_omg_v=err_omg_v, err_omg_omg=err_omg_omg,
                err_r=0.01, err_phi=0.05, err_feat=1.0)
  ekf_localizer = EKFLocalizerKC(config=ekf_conf, landmarks=poles, ini_pose=ellipse_path[0])

  mm_pose_list = []
  true_pose_list = []
  ekf_pose_list = []
  for index, true_pose in enumerate(ellipse_path):
    
    ctrl = controls[index-1]
    noised_ctrl = generate_noised_control(ctrl, err_v_v, err_v_omg, err_omg_v, err_omg_omg)
    # Generate Detection
    idx_list, meas_list = vir_lm_det.detect(pose=true_pose)
    meas = EKFKCMeas(corresp_list=idx_list, meas_list=meas_list) 

    # Localize by EKF
    ekf_pose, sig, likelihood = ekf_localizer.localize(control=noised_ctrl, meas=meas)
    mm_pose = ekf_localizer.get_motion_model_pose()

    if (index % 25 == 0):
      GridMap2DDrawer.draw_point(grid2d_loc, ekf_pose.x, ekf_pose.y, 0.04, 1.0)
      mm_pose_list.append(mm_pose)
      true_pose_list.append(true_pose)
      ekf_pose_list.append(ekf_pose)
      plt.cla()
      draw_result(plt, true_pose_list, mm_pose_list, ekf_pose_list, poles, idx_list)
      plt.pause(0.1)

  plt.show()

