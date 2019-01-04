
import sys
import os
sys.path.append(os.pardir)

import math

class InverseRangeSensorModelConfigParams:

  def __init__(self, *, range_max, l0, locc, lfree, alpha, beta):

    self.range_max = range_max
    self.l0 = l0
    self.locc = locc
    self.lfree = lfree
    self.alpha = alpha
    self.beta = beta

class InverseRangeSensorModel():

  def __init__(self, conf):
    
    self.__range_max = conf.range_max
    self.__l0 = conf.l0
    self.__locc = conf.locc
    self.__lfree = conf.lfree
    self.__alpha = conf.alpha
    self.__beta = conf.beta

  def log_likelihood(self, *, pose, tgt, scanned_range):
    
    dist = math.sqrt((tgt.x - pose.x)**2 + (tgt.y - pose.y)**2)
    phi = math.atan2(tgt.y-pose.y, tgt.x-pose.x) - pose.theta

    val = self.__l0
    if ((self.__range_max < dist) or \
        (scanned_range + self.__alpha/2.0 < dist)):
      val = self.__l0
    elif ((scanned_range < self.__range_max) and \
          (abs(scanned_range-dist) < self.__alpha/2.0)):
      val = self.__locc
    elif (dist < scanned_range):
      val = self.__lfree
    else:
      val = self.__l0

    return val

if __name__ == "__main__":
  print(__file__ + "Started!")