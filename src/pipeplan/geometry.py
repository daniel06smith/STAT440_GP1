"""Small computational-geometry helpers: convex hull and minimum enclosing circle.

A work unit is a circle. The *cheapest* circle that covers a group of pipes is
(almost always) the smallest circle containing all their end points, because
the area term of the cost grows with radius.
"""
from __future__ import annotations

import math
import random

import numpy as np


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull vertices of an (n, 2) point array (Andrew's monotone chain).

    The minimum enclosing circle of a point set is determined by its hull
    vertices alone, so clusters only need to remember their hull.
    """
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) <= 2:
        return pts
    P = [tuple(p) for p in pts]  # already sorted lexicographically by np.unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _circle_from_2(a, b):
    cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    return cx, cy, math.hypot(a[0] - b[0], a[1] - b[1]) / 2


def _circle_from_3(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:  # collinear: use the widest pair
        best = max(
            (_circle_from_2(a, b), _circle_from_2(a, c), _circle_from_2(b, c)),
            key=lambda t: t[2],
        )
        return best
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    return ux, uy, math.hypot(ux - ax, uy - ay)


def min_enclosing_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Smallest circle containing all points: returns (cx, cy, r).

    Welzl's algorithm (iterative, expected O(n)). Intended for hull vertices.
    """
    P = [tuple(p) for p in np.asarray(points, dtype=float).reshape(-1, 2)]
    if not P:
        raise ValueError("no points")
    random.Random(12345).shuffle(P)  # fixed seed -> deterministic output
    tol = 1e-9

    def inside(p, c):
        return math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] + tol

    c = (P[0][0], P[0][1], 0.0)
    for i in range(1, len(P)):
        if inside(P[i], c):
            continue
        c = (P[i][0], P[i][1], 0.0)
        for j in range(i):
            if inside(P[j], c):
                continue
            c = _circle_from_2(P[i], P[j])
            for k in range(j):
                if not inside(P[k], c):
                    c = _circle_from_3(P[i], P[j], P[k])
    return c
