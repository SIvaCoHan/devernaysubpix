#! /usr/bin/env python3.7
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import filters, convolve


def image_gradient(image, sigma):
    image = np.asfarray(image)
    gx = filters.gaussian_filter(image, sigma, order=[0, 1])
    gy = filters.gaussian_filter(image, sigma, order=[1, 0])
    return gx, gy


# CurvePoints = (x, y, gx, gy, valid)
def compute_edge_points(grads, min_magnitude=0):
    gx, gy = grads
    rows, cols = gx.shape
    edges = []

    mag = (gx**2 + gy**2)**0.5
    mag_valid = mag > min_magnitude
    mag_direction = np.absolute(gx) - np.absolute(gy) > 0

    mag_left = mag - np.roll(mag, 1, axis=1) > 0
    mag_right = mag - np.roll(mag, -1, axis=1) > 0

    mag_top = mag - np.roll(mag, 1, axis=0) > 0
    mag_bottom = mag - np.roll(mag, -1, axis=0) > 0

    theta_x = mag_left & mag_right & mag_direction
    theta_y = mag_top & mag_bottom & ~mag_direction

    pts = mag_valid & (theta_x | theta_y)

    # TODO edge point取对了，但是上和左的mag取错了，需要仔细检查一下坐标
    # TODO 为啥x, y 是反的?
    idx_b = np.transpose(pts.nonzero())
    y = theta_x[idx_b[:, 0], idx_b[:, 1]]
    x = theta_y[idx_b[:, 0], idx_b[:, 1]]
    idx_a = np.column_stack((idx_b[:, 0] - x, idx_b[:, 1] - y))
    idx_c = np.column_stack((idx_b[:, 0] + x, idx_b[:, 1] + y))

    mag_a = mag[idx_a[:, 0], idx_a[:, 1]]
    mag_b = mag[idx_b[:, 0], idx_b[:, 1]]
    mag_c = mag[idx_c[:, 0], idx_c[:, 1]]
    lamda = (mag_a - mag_c) / (2 * (mag_a - 2 * mag_b + mag_c))

    pts_x = idx_b[:, 0] - lamda * x
    pts_y = idx_b[:, 1] - lamda * y
    pts = np.column_stack((pts_x, pts_y))
    # TODO 数据拿到了，想办法构建KDTree
    return pts


if __name__ == '__main__':
    import cv2

    pad = 20
    circle = cv2.imread("./2018112101.jpg", 0)
    I = np.zeros((circle.shape[0] + 2 * pad, circle.shape[1] + 2 * pad), dtype=np.uint8) + 255
    I[pad:circle.shape[0] + pad, pad:circle.shape[1] + pad] = circle
    I = I.astype(np.float32)

    grads = image_gradient(I, 2.0)
    edges = compute_edge_points(grads)
    # links = chain_edge_points(edges, grads)
    # print(len(links))
    # chains = thresholds_with_hysteresis(edges, links, grads, 1, 0.1)
    # print(len(chains))
    # print(chains[0])
