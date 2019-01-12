
import sys
import os
sys.path.append(os.pardir)

import matplotlib.pyplot as plt
import numpy as np

# Histogram Filter Module
from histogram_filter.histogram_filter import Histogram1DConfigParams
from histogram_filter.histogram_filter import HistogramFilter1D

# Test Data Module
from tests.test_data import create_sample_1d_path

def generate_noised_measurement(gt_loc, gt_pose, sigma):
  return gt_loc - gt_pose + np.random.normal(0.0, sigma)

def generate_noised_u(cur_pose, last_pose, sigma):
  diff = cur_pose - last_pose 
  return diff + diff * np.random.normal(0.0, sigma)

if __name__ == "__main__":
  print(__file__ + "Started!")

  landmark_ground_truth = 80.0
  meas_sigma = 2.0
  pred_sigma = 0.3

  hist_config =  Histogram1DConfigParams( \
                  x_min=-10.0, x_max=90.0, \
                  reso=0.1, init_pose=0.0, \
                  pred_sigma=pred_sigma, meas_sigma=meas_sigma,
                  gt_landmark=landmark_ground_truth)
  x_ticks = [hist_config.reso * i + hist_config.x_min
              for i in range(hist_config.x_tick_num)]

  hist_filter1 = HistogramFilter1D(histogram_config=hist_config)
  hist_filter2 = HistogramFilter1D(histogram_config=hist_config)
  hist_filter3 = HistogramFilter1D(histogram_config=hist_config)

  path = create_sample_1d_path()
  last_pose = path[0]
  cur_pose = path[0]

  fig, (ax1, ax2) = plt.subplots(2, 1)
  plt.subplots_adjust(hspace=0.4)

  ax1.set_title("Only Prediction")
  ax2.set_title("Bayes Filter used for merging Prediction and Measurement")
  ax1.set_xlim([-10, 90])
  ax1.set_ylim([0, 0.1])
  ax2.set_xlim([-10, 90])
  ax2.set_ylim([0, 0.1])
  ax1.vlines(np.linspace(-10, 90, 11),  ymin=0, ymax=1, linestyle='dashed', linewidth=0.2)
  ax2.vlines(np.linspace(-10, 90, 11),  ymin=0, ymax=1, linestyle='dashed', linewidth=0.2)

  for index, x_pose in enumerate(path):

    cur_pose = x_pose
    if (cur_pose - last_pose == 0):
      continue

    ax1.vlines([x_pose], ymin=0, ymax=1, linestyle='solid', linewidth=1.0, color='C'+str(index%10))
    ax2.vlines([x_pose], ymin=0, ymax=1, linestyle='solid', linewidth=1.0, color='C'+str(index%10))

    # Prediction
    noised_u = generate_noised_u(cur_pose, last_pose, pred_sigma)
    last_pose = cur_pose
    hist_filter1.predict(noised_u)
    ax1.plot(x_ticks, hist_filter1._new_prob, color='C'+str(index%10), linestyle='dashed')
    hist_filter3.predict(noised_u)
    ax2.plot(x_ticks, hist_filter3._new_prob, color='C'+str(index%10), linestyle='dashed')   
    plt.pause(1)
    
    # Measurement Update
    noised_meas = generate_noised_measurement(\
                      landmark_ground_truth, cur_pose, meas_sigma)
    hist_filter3.meas_update(noised_meas)
    ax2.plot(x_ticks, hist_filter3._new_prob, color='C'+str(index%10), linestyle='solid')
    plt.pause(1)

  plt.show()
    
  