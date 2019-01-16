
import sys
import os
sys.path.append(os.pardir)

import math
import unittest


def normalize_angle_0_2_2pi(value):
  
  while (True):
    if (value < 0):
      value = value + 2 * math.pi
    elif (2 * math.pi <= value):
      value = value - 2 * math.pi
    else:
      break
  
  return value

def normalize_angle_pi_2_pi(value):

  while (True):
    if (value < -math.pi):
      value = value + 2 * math.pi
    elif (math.pi <= value):
      value = value - 2 * math.pi
    else:
      break

  return value

def my_round(value):
  
  rounded = 0
  if (value >= 0):
    rounded = int(value + 0.5)
  else:
    rounded = -int(-value + 0.5)
  return rounded

class TestMyRound(unittest.TestCase):

  def test_my_round1(self):
    # Positive Value
    self.assertEqual(my_round(2.49999), 2)
    self.assertEqual(my_round(2.5), 3)

  def test_my_round2(self):
    # Negative Value Rounding
    self.assertEqual(my_round(-2.5), -3)
    self.assertEqual(my_round(-2.49999), -2)

if __name__ == '__main__':
  unittest.main()      