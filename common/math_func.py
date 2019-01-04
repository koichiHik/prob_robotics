
import unittest

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