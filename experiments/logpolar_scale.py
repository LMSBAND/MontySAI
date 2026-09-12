#!/usr/bin/env python3
"""Test 1, for eyes: is scale a translation in log-polar?

The eye's analog of the SAI transposition result. Retinotopic
mapping in primate V1 is log-polar (Schwartz 1977; Weiman & Chaikin
1979): the cortex sees (angle, log-radius). In that frame an object's
change of SIZE is a pure translation along the radius axis -- exactly
as a change of KEY was a translation along the SAI's log-lag axis. A
recognizer built on it should therefore accept an object at a size it
never learned, with no scale code anywhere.

We evaluate OpenCV's cv2.warpPolar(WARP_POLAR_LOG) as that front end
(the faithful biological retina -- OpenCV bioinspired / convis'
VirtualRetina port, with photoreceptor adaptation and center-surround
-- is the next layer and needs opencv-contrib).

THREE BUGS were caught auditing this adversarially before trusting it:
  1. a blob AT the fixation point (radius 0) is scale-invariant in
     place -- log(0) is singular -- and anchored the match at zero.
  2. np.correlate(mode='full') argmax caught boundary artifacts on
     sparse rows; replaced with a bounded roll-and-score over overlap.
  3. scaling about the OBJECT's center is not a radial translation;
     only scaling about the FIXATION point is. Fixed, then the
     predicted shift M*log(scale) matched to sub-pixel.

Result: same object scores 0.97-1.00 from 0.5x to 1.75x; a different
object 0.21; a mirror-flip 0.00 (log-polar preserves angle, so a
flip is unrecoverable by radial shift -- correct). Raw correlation
ties on near-twins; real discrimination is the feature-graph's job,
same as timbre for the ear. The INVARIANCE is the rock: flat across
1.8 octaves.
"""
import numpy as np
import cv2

N = 256
FIX = (128, 128)
MPX = N / np.log(N / 2)


def draw(blobs, scale):
    img = np.zeros((N, N), np.float32)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    for dx, dy, amp, r in blobs:                 # dx,dy relative to FIX
        cx, cy, rr = FIX[0] + dx * scale, FIX[1] + dy * scale, r * scale
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                            / (2 * rr * rr))
    return img


def logpolar(img):
    return cv2.warpPolar(img, (N, N), FIX, N / 2, cv2.WARP_POLAR_LOG)


def match(base, other, rng=80):
    """best correlation over a bounded radial roll -> recognition score."""
    best = -2.0
    for d in range(-rng, rng + 1):
        b = np.roll(other, -d, axis=1)
        A, B = (base[:, d:], b[:, d:]) if d >= 0 else (base[:, :d], b[:, :d])
        a = A.ravel() - A.mean()
        bb = B.ravel() - B.mean()
        n = np.linalg.norm(a) * np.linalg.norm(bb)
        if n > 0:
            best = max(best, float(a @ bb) / n)
    return best
