
import matplotlib.pyplot as plt
import numpy as np
import random
import math

class BoxMuller():

  @staticmethod
  def sample_norm_dist(avg=0.0, sigma=1.0):
    x1 = random.random()
    x2 = random.random()
    y1 = math.sqrt(-2*math.log(x1)) * math.cos(2 * math.pi * x2)
    y2 = math.sqrt(-2*math.log(x1)) * math.sin(2 * math.pi * x2)
    return y1, y2

  @staticmethod
  def sample_norm_dist_fast(avg=0.0, sigma=1.0):
    x1 = random.random()
    x2 = random.random()

    while(True):
      x1 = 2.0 * random.random() - 1.0
      x2 = 2.0 * random.random() - 1.0
      r = x1**2 + x2**2

      if (0.0 < r and r < 1.0):
        break

    fac = math.sqrt(-2.0 * math.log(r) / r)

    return fac * x1, fac * x2


if __name__ == '__main__':
  print(__file__)

  gen_num = 10001
  bin_num = 1001
  draw_range = (-5, 5)
  bin_width = float(((draw_range[1] - draw_range[0]) / float(bin_num)))

  y1 = []
  y2 = []
  y3 = []
  y4 = []
  for i in range(gen_num):
    sample1, sample2 = BoxMuller.sample_norm_dist()
    sample3, sample4 = BoxMuller.sample_norm_dist_fast()
    y1.append(sample1)
    y2.append(sample2)
    y3.append(sample3)
    y4.append(sample4)

  # Histogram Generation
  hist1 = np.histogram(y1, bins=bin_num, range=draw_range)
  hist2 = np.histogram(y2, bins=bin_num, range=draw_range)
  hist3 = np.histogram(y3, bins=bin_num, range=draw_range)
  hist4 = np.histogram(y4, bins=bin_num, range=draw_range)

  histelem1 = np.array([hist1[0][i] / bin_width for i in range(bin_num)])
  histelem2 = np.array([hist2[0][i] / bin_width for i in range(bin_num)])
  histelem3 = np.array([hist3[0][i] / bin_width for i in range(bin_num)])
  histelem4 = np.array([hist4[0][i] / bin_width for i in range(bin_num)])

  exp_x = np.array([bin_width * i for i in range(-bin_num/2, bin_num/2)])
  exp_vars = 1 / math.sqrt(2 * math.pi) * np.exp(-np.power(exp_x, 2) / 2.0)

  plt.subplot(2, 1, 1)
  plt.plot(exp_x, histelem1)
  plt.plot(exp_x, histelem2)
  plt.plot(exp_x, exp_vars * gen_num)

  plt.subplot(2, 1, 2)
  plt.plot(exp_x, histelem3)
  plt.plot(exp_x, histelem4)
  plt.plot(exp_x, exp_vars * gen_num)

  plt.show()

    