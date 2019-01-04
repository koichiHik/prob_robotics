
import random
import numpy
import math

from abc import ABCMeta, abstractstaticmethod

class AbstractSampler(metaclass=ABCMeta):
  
  @abstractstaticmethod
  def sample(*, weights_list):
    pass

class LowVarianceSampler(AbstractSampler):

  @staticmethod
  def __normalize_weights(weights_list):
    total = sum(weights_list)
    return weights_list / total

  @staticmethod
  def sample(*, weights_list):
    sampled_index = []

    normalized_weights = \
        LowVarianceSampler.__normalize_weights(weights_list)

    M = len(normalized_weights)
    r = random.uniform(0, 1/M)
    c = normalized_weights[0]
    i = 0

    for index in range(M):
      U = r + index * float(1 / M)
      #print(U)

      while (U > c):
        i = i + 1
        #print(i)
        c = c + normalized_weights[i]

      sampled_index.append(i)

    return sampled_index

if __name__ == "__main__":
  
  LowVarianceSampler.sample(weights_list=[1])
