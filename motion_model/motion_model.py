
import sys
import os
sys.path.append(os.pardir)

import math
import numpy as np

from interface.interface import IMotionModel

# Common Module
from common.container import Pose2D

class MotionErrorModel2dConfigParams:

  def __init__(self, *, std_rot_per_rot, std_rot_per_trans,\
                        std_trans_per_trans, std_trans_per_rot):
    self._std_rot_per_rot = std_rot_per_rot
    self._std_rot_per_trans = std_rot_per_trans
    self._std_trans_per_trans = std_trans_per_trans
    self._std_trans_per_rot = std_trans_per_rot

class MotionErrorModel2D(IMotionModel):

  def __init__(self, *, conf):
    self.conf = conf
  
  def update_err_config(self, *, conf):
    self._conf = conf

  def sample_motion(self, *, cur_odom, last_odom, last_particle_poses):

    nor_cur_odom = cur_odom.numpy_array()
    nor_last_odom = last_odom.numpy_array()
    nor_cur_odom[2] = self.__normalize_angle_between_npi_to_ppi(nor_cur_odom[2])
    nor_last_odom[2] = self.__normalize_angle_between_npi_to_ppi(nor_last_odom[2])

    rot1, trans, rot2 = self.__decompose_odom_diff(nor_cur_odom, nor_last_odom)

    np_last_particle_poses = self.__convert_list_poses_2_numpy_list(last_particle_poses)
    particle_num = np_last_particle_poses.shape[1]

    odom_diff_mat = self.__generate_true_odom_diff_array(rot1, trans, rot2, np_last_particle_poses, particle_num)

    np_cur_particle_poses = np_last_particle_poses + odom_diff_mat
    cur_particle_poses = self.__convert_numpy_list_2_list_poses(np_cur_particle_poses)

    return cur_particle_poses

  def __calc_sigma(self, rot1, trans, rot2):

    sigma_rot1 = self.conf._std_rot_per_rot * rot1 + self.conf._std_rot_per_trans * trans
    sigma_trans = self.conf._std_trans_per_trans * trans + self.conf._std_trans_per_rot * rot2
    sigma_rot2 = self.conf._std_rot_per_rot * rot2 + self.conf._std_rot_per_trans * trans

    return sigma_rot1, sigma_trans, sigma_rot2

  def __decompose_odom_diff(self, cur_odom, last_odom):
    u = cur_odom - last_odom
    rot1 = math.atan2(u[1], u[0]) - last_odom[2]
    rot1 = self.__normalize_angle_between_npi_to_ppi(rot1)
    trans = math.sqrt(u[0]**2 + u[1]**2)
    u[2] = self.__normalize_angle_between_npi_to_ppi(u[2])
    rot2 = u[2] - rot1
    rot2 = self.__normalize_angle_between_npi_to_ppi(rot2)
    return rot1, trans, rot2

  def __normalize_angle_between_npi_to_ppi(self, val1):
    value = val1
    if (math.pi < val1):
      value = val1 - 2 * math.pi
    elif (val1 < -math.pi):
      value = val1 + 2 * math.pi
    return value

  def __convert_numpy_list_2_list_poses(self, np_list_poses):

    list_poses = [Pose2D() for i in range(np_list_poses.shape[1])]
    for i in range(np_list_poses.shape[1]):
      list_poses[i].x = np_list_poses[0, i]
      list_poses[i].y = np_list_poses[1, i]
      list_poses[i].theta = np_list_poses[2, i]
    return list_poses

  def __convert_list_poses_2_numpy_list(self, list_poses):

    np_poses = np.zeros((3, len(list_poses)))
    for index, pose in enumerate(list_poses):
      np_poses[:,index] = pose.numpy_array()
    return np_poses

  def __generate_true_odom_diff_array(self, rot1, trans, rot2, last_particle_poses, particle_num):

    rot1_arr = np.zeros(particle_num) + rot1
    trans_arr = np.zeros(particle_num) + trans
    rot2_arr = np.zeros(particle_num) + rot2

    sig_r1, sig_t, sig_r2 = self.__calc_sigma(rot1, trans, rot2)

    t_rot1_arr = rot1_arr - np.random.normal(0, abs(sig_r1), particle_num)
    t_trans_arr = trans_arr - np.random.normal(0, abs(sig_t), particle_num)
    t_rot2_arr = rot2_arr - np.random.normal(0, abs(sig_r2), particle_num)

    d_x_arr = np.array(t_trans_arr) * np.array(np.cos(last_particle_poses[2,:] + t_rot1_arr))
    d_y_arr = np.array(t_trans_arr) * np.array(np.sin(last_particle_poses[2,:] + t_rot1_arr))
    d_theta_arr = t_rot1_arr + t_rot2_arr

    return np.concatenate(([d_x_arr], [d_y_arr], [d_theta_arr]), axis=0)
