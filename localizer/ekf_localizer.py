
import sys
import os
sys.path.append(os.pardir)

import copy
import numpy as np
import math

# Common Module
from common.container import LandmarkMeas
from common.container import Pose2D
from common.container import Control
from common.transform import conv_pnt_global_2_local
from common.math_func import normalize_angle_pi_2_pi

# Interface Module
from interface.interface import ILocalizer

class EKFKCMeas():

  def __init__(self, *, corresp_list, meas_list):
    self._corresp = corresp_list
    self._measurements = meas_list

class EKFLocalizerConfigParams():

  def __init__(self, *, dT, \
                  err_v_v, err_v_omg, \
                  err_omg_v, err_omg_omg, 
                  err_r, err_phi, err_feat):
    self.dT = dT
    self.err_v_v = err_v_v
    self.err_v_omg = err_v_omg
    self.err_omg_v = err_omg_v
    self.err_omg_omg = err_omg_omg
    self.err_r = err_r
    self.err_phi = err_phi
    self.err_feat = err_feat

class EKFLocalizerKC():

  def __init__(self, *, config, landmarks, ini_pose=Pose2D()):

    self._conf = config
    self._landmarks = landmarks
    self._last_avg = ini_pose
    self._mm_pose = ini_pose
    self._last_sigma = np.matrix(np.zeros((3,3)))
    self._Q = np.matrix(np.eye(3))
    self._Q[0,0] = self._conf.err_r**2
    self._Q[1,1] = self._conf.err_phi**2
    self._Q[2,2] = self._conf.err_feat**2

  def localize(self, *, control, meas):

    avg, sig = self._predict(control, self._last_avg)
    self._mm_pose, mm_sig = self._predict(control, self._mm_pose)

    likelihood = 0.0
    if (len(meas._measurements) != 0):
      avg, sig, likelihood = self._correct(avg, sig, meas)

    self._last_avg = avg
    self._last_sig = sig
    return self._last_avg, self._last_sig, likelihood

  def get_motion_model_pose(self):
    return self._mm_pose

  def get_current_pose(self, all_pose=False):
    pass

  def _predict(self, control, last_pose):
    
    v = control.v
    omg = control.omg
    theta = last_pose.theta
    dT = self._conf.dT

    pred_pose = Pose2D()
    pred_pose.x = last_pose.x + v * dT * math.cos(theta - omg * dT / 2.0)
    pred_pose.y = last_pose.y + v * dT * math.sin(theta - omg * dT / 2.0)
    pred_pose.theta = normalize_angle_pi_2_pi(theta + omg * dT)

    G = self._calc_jacobi_G(control, last_pose)
    V, M = self.__calc_control_transmat_V_M(control, last_pose)

    pred_sigma = G * self._last_sigma * G.T + V * M * V.T

    return pred_pose, pred_sigma

  def _calc_jacobi_G(self, control, last_pose):
    v = control.v
    omg = control.omg
    theta = last_pose.theta
    dT = self._conf.dT
    G = np.matrix(np.eye(3))
    G[0,2] = - v * dT * math.sin(theta)
    G[1,2] = v * dT * math.cos(theta)
    return G

  def __calc_control_transmat_V_M(self, control, last_pose):
    v = control.v
    omg = control.omg
    theta = last_pose.theta
    dT = self._conf.dT 
    
    V = np.matrix(np.zeros((3,2)))
    V[0,0] = dT * math.cos(theta)
    V[1,0] = dT * math.sin(theta)
    V[2,0] = 0
    V[0,1] = -v * dT * math.sin(theta) * dT
    V[1,1] = v * dT * math.cos(theta) * dT
    V[2,1] = dT

    M = np.matrix(np.zeros((2,2)))
    #M[0,0] = self._conf.err_v_v * v**2 + self._conf.err_v_omg * omg**2
    #M[1,1] = self._conf.err_omg_v * v**2 + self._conf.err_omg_omg * omg**2
    M[0,0] = self._conf.err_v_v * abs(v) + self._conf.err_v_omg * abs(omg)
    M[1,1] = self._conf.err_omg_v * abs(v) + self._conf.err_omg_omg * abs(omg)

    return V, M

  def _correct(self, pred_pose, pred_sigma, ekf_kc_meas):
    
    pose_mat = np.matrix(pred_pose.numpy_array()).T
    sigma = copy.deepcopy(pred_sigma)
    pred_meas_list = []
    S_list = []
    z_diff_list = []
    for index, meas in enumerate(ekf_kc_meas._measurements):
      # Measurement Update for each landmark.
      lm_idx = ekf_kc_meas._corresp[index]
      lm_glob = self._landmarks[lm_idx]
      calc_meas = self._calc_measurement(pose_mat, lm_glob)
      pred_meas_list.append(calc_meas)
      H = self._calc_jacobi_H(pose_mat, lm_glob)
      S = H * sigma * H.T + self._Q
      S_list.append(copy.deepcopy(S))
      K = sigma * H.T * np.linalg.inv(S)
      z_diff_mat = np.matrix(meas.numpy_array() - calc_meas.numpy_array()).T
      z_diff_mat[1,0] = normalize_angle_pi_2_pi(z_diff_mat[1,0])

      if (abs(z_diff_mat[0,0]) > 5 or abs(z_diff_mat[1,0]) > 0.5):
        sys.exit()

      z_diff_list.append(copy.deepcopy(z_diff_mat))
      pose_mat = pose_mat + K * (z_diff_mat)
      sigma = (np.matrix(np.eye(3)) - K * H) * sigma

    likelihood = 1.0
    for index, meas in enumerate(ekf_kc_meas._measurements):
      S_inv = np.linalg.inv(S_list[index])
      z_diff = z_diff_list[index]
      likelihood = likelihood \
          * 1 / math.sqrt(2 * math.pi) \
          * math.exp(-1/2*(z_diff).T * S_inv * (z_diff))
    
    normalized_theta = normalize_angle_pi_2_pi(pose_mat[2,0])
    updated_pose = Pose2D(x=pose_mat[0,0],y=pose_mat[1,0],theta=normalized_theta)
    return updated_pose, sigma, likelihood

  def _calc_measurement(self, pose_mat, lm_glob):
    pose = Pose2D(x=pose_mat[0,0],y=pose_mat[1,0],theta=pose_mat[2,0])
    lm_loc = conv_pnt_global_2_local(pose, lm_glob)
    r = math.sqrt(lm_loc.x**2 + lm_loc.y**2)
    phi = normalize_angle_pi_2_pi(math.atan2(lm_loc.y, lm_loc.x))
    return LandmarkMeas(r=r, phi=phi)

  def _calc_jacobi_H(self, pose_mat, lm_glob):

    pose = Pose2D(x=pose_mat[0,0],y=pose_mat[1,0],theta=pose_mat[2,0])
    r = (lm_glob.x - pose.x)**2 + (lm_glob.y - pose.y)**2 

    H = np.matrix(np.zeros((3,3)))
    H[0,0] = -(lm_glob.x - pose.x)/math.sqrt(r)
    H[1,0] =  (lm_glob.y - pose.y)/r
    H[0,1] = -(lm_glob.y - pose.y)/math.sqrt(r)
    H[1,1] = -(lm_glob.x - pose.x)/r
    H[1,2] = -1

    return H

if __name__ == "__main__":
  print(__file__ + "Started!")
