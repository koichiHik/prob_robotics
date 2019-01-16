
import sys
import os
sys.path.append(os.pardir)

import math
import unittest

# Common Module
from common.container import Pose2D
from common.container import Coord2D

def conv_pnt_global_2_local(local_fr, glob_pnt):
  dx = glob_pnt.x - local_fr.x
  dy = glob_pnt.y - local_fr.y
  phi = math.atan2(dy, dx) - local_fr.theta

  r = math.sqrt(dx**2 + dy**2)
  x = r * math.cos(phi)
  y = r * math.sin(phi)
  return Coord2D(x=x, y=y)

def conv_pnt_local_2_global(local_fr, loc_pnt):
  theta = math.atan2(loc_pnt.y, loc_pnt.x) + local_fr.theta

  r = math.sqrt(loc_pnt.x**2 + loc_pnt.y**2)
  dx = r * math.cos(theta)
  dy = r * math.sin(theta)
  return Coord2D(x=local_fr.x + dx, y=local_fr.y + dy)

class TestConvPntGlobal2Local(unittest.TestCase):

  def test_conv_pnt_global_2_local1(self):

    local_fr = Pose2D(1, 1, 45.0/180.0 * math.pi)
    glo_pnt = Coord2D(x=3.0, y=1.0)
    loc_pnt = conv_pnt_global_2_local(local_fr, glo_pnt)
    self.assertAlmostEqual(loc_pnt.x, math.sqrt(2), places=5)
    self.assertAlmostEqual(loc_pnt.y, -math.sqrt(2), places=5)

  def test_conv_pnt_local_2_global(self):

    local_fr = Pose2D(1, 1, 45.0/180.0 * math.pi)
    loc_pnt = Coord2D(x=math.sqrt(2), y=-math.sqrt(2))
    glo_pnt = conv_pnt_local_2_global(local_fr, loc_pnt)
    self.assertAlmostEqual(glo_pnt.x, 3.0, places=5)
    self.assertAlmostEqual(glo_pnt.y, 1.0, places=5)


if __name__ == "__main__":
  unittest.main()