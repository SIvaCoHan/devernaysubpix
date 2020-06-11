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

    # TODO 这里的坐标我有可能弄反了, 稍后找一个非对称图形来看看
    # TODO 目标，我已经取得了全部B的坐标，而后再取得A，C的offset，就饿可以得到A，C的坐标
    # TODO 然后从mag里面分别提取 A， B， C，形成三个np.array 即可算出全部的lambda
    idx_b = np.transpose(pts.nonzero())

    # idx_a = idx_b - np.transpose(theta_x[pts], theta_y[pts])
    # idx_c = idx_b + np.transpose(theta_x[pts], theta_y[pts])
    # print(idx_b)
    # idx_c = 1



    # 使用3*3的window 检查center mag > min_mag
    # 上 下 左 右 分别 移动一个位置然后去减，得到四个bool 矩阵
    # 上下 / 左右 分别判断，得到m
    # gx - gy 还得算一次，计算方向性
    print(np.count_nonzero(pts))
    return pts


if __name__ == '__main__':
    import cv2

    pad = 20
    circle = cv2.imread("./kreis.png", 0)
    I = np.zeros((circle.shape[0] + 2 * pad, circle.shape[1] + 2 * pad), dtype=np.uint8) + 255
    I[pad-5:circle.shape[0] + pad -5, pad-5:circle.shape[1] + pad -5] = circle
    I = I.astype(np.float32)

    grads = image_gradient(I, 2.0)
    edges = compute_edge_points(grads)
    # links = chain_edge_points(edges, grads)
    # print(len(links))
    # chains = thresholds_with_hysteresis(edges, links, grads, 1, 0.1)
    # print(len(chains))
    # print(chains[0])
