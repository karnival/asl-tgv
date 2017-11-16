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

    # Checks for a few values at boundaries and middle.
    assert fwd2[0,0,0] == 1
    assert fwd2[0,0,2] == 0
    assert fwd2[2,0,1] == 4

    bwd0 = main.backward_diff(im, 0)
    bwd1 = main.backward_diff(im, 1)
    bwd2 = main.backward_diff(im, 2)

    # Checks for a few values at boundaries and middle.
    assert bwd0[0,0,0] == 1
    assert bwd0[1,0,0] == 8
    assert bwd0[2,0,0] == -9

    assert bwd1[2,0,0] == 1
    assert bwd1[2,0,1] == 3
    assert bwd1[2,0,2] == 7
