import unittest
import numpy as np
from tools.benchmark_tracked_collection import _check_arrays


class DirectComparisonTests(unittest.TestCase):
    def test_equivalent_noncontiguous_arrays(self):
        array = np.arange(12, dtype=np.float32).reshape(3, 4).T
        _check_arrays((array,), (array.copy(),), 0)

    def test_changed_value_dtype_shape_or_signed_zero_fails(self):
        expected = np.array([0., 1.], np.float32)
        for actual in (np.array([-0., 1.], np.float32), np.array([0., 2.], np.float32),
                       expected.astype(np.float64), expected.reshape(1, 2)):
            with self.subTest(actual=actual):
                with self.assertRaises(AssertionError):
                    _check_arrays((expected,), (actual,), 1)


if __name__ == '__main__':
    unittest.main()
