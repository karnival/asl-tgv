import pytest
import numpy as np
from main import *

im = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
               [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
               [[1, 3, 7], [1, 4, 7], [2, 7, 2]]])


def test_finite_differences():
    fwd0 = forward_diff(im, 0)
    fwd1 = forward_diff(im, 1)
    fwd2 = forward_diff(im, 2)

    # Checks for a few values at boundaries and middle.
    assert fwd2[0,0,0] == 1
    assert fwd2[0,0,2] == 0
    assert fwd2[2,0,1] == 4

    bwd0 = backward_diff(im, 0)
    bwd1 = backward_diff(im, 1)
    bwd2 = backward_diff(im, 2)

    # Checks for a few values at boundaries and middle.
    assert bwd0[0,0,0] == 1
    assert bwd0[1,0,0] == 8
    assert bwd0[2,0,0] == -9

    assert bwd1[2,0,0] == 1
    assert bwd1[2,0,1] == 3
    assert bwd1[2,0,2] == 7

def test_finite_differences_4d():
    tmp = np.empty([3,3,3,3])
    tmp[:,:,:,0] = im
    tmp[:,:,:,1] = im
    tmp[:,:,:,2] = im

    F = forward_diff(tmp, 0)
    assert np.all(F[:,:,:,0] == F[:,:,:,1])

    B = backward_diff(tmp, 0)
    assert np.all(B[:,:,:,0] == B[:,:,:,1])


def test_grad():
    # Check shapes are right.
    assert np.shape(grad(im)) == (3, 3, 3, 3)
    assert np.shape(grad(np.hstack([im, im]))) == (3, 6, 3, 3)
    assert np.shape(grad(np.vstack([im, im]))) == (6, 3, 3, 3)

def test_epsilon():
    tmp = grad(im)

    assert np.shape(epsilon(tmp)) == (3, 3, 3, 3, 3)

def test_div():
    tmp_vec = grad(im)
    assert np.shape(div(tmp_vec)) == (3, 3, 3)

    tmp_mat = np.random.rand(3, 3, 3, 3, 3)
    assert np.shape(div(tmp_mat)) == (3, 3, 3, 3)


def test_stack():
    out = stack(im, 10)
    assert out.shape == (im.shape + (10,))
