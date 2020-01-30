import unittest
import numpy as np
from gwdt import gwdt

strel_4_connected = np.array(
    [[False, True, False],
     [True, True, True],
     [False, True, False]])

strel_6_connected = np.array(
    [
        [
            [False, False, False],
            [False, True, False],
            [False, False, False]
        ],
        [
            [False, True, False],
            [True, True, True],
            [False, True, False]
        ],
        [
            [False, False, False],
            [False, True, False],
            [False, False, False]
        ]
    ]
)
class TestGWDT(unittest.TestCase):
    def test_background_2d(self):
        output = gwdt(np.zeros((3, 3)), strel_4_connected)
        np.testing.assert_equal(output, 0.0)

    def test_background_3d(self):
        output = gwdt(np.zeros((3, 3, 3)), strel_6_connected)
        np.testing.assert_equal(output, 0.0)

    def test_one_2d(self):
        input = np.zeros((3, 3), np.float32)
        input[1, 1] = 4.3
        output = gwdt(input, strel_4_connected)
        np.testing.assert_equal(output, input)

    def test_one_3d(self):
        input = np.zeros((3, 3, 3), np.float32)
        input[1, 1, 1] = 4.3
        output = gwdt(input, strel_6_connected)
        np.testing.assert_equal(output, input)

    def test_many_2d(self):
        input = np.array([
            [0, 1.0, 2.0],
            [1.4, 2.0, 3.0],
            [1.4, 1.0, 3.5]
        ])
        # Path through is
        # .   .   .
        # v
        # .   .   .
        # v
        # . > . > . = 1.4 + 1.4 + 1.0 + 3.5 = 7.3
        output = gwdt(input, strel_4_connected)
        self.assertAlmostEqual(output[2, 2], 7.3, delta=.05)


if __name__ == '__main__':
    unittest.main()
