import pytest
import numpy as np
import main

def test_finite_differences():
    im = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                   [[1, 3, 7], [1, 4, 7], [2, 7, 2]]])

    fwd0 = main.forward_diff(im, 0)
    fwd1 = main.forward_diff(im, 1)
    fwd2 = main.forward_diff(im, 2)

    bwd0 = main.backward_diff(im, 0)
    bwd1 = main.backward_diff(im, 1)
    bwd1 = main.backward_diff(im, 2)
    
