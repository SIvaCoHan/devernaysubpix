#! /usr/bin/env python3.7
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from bidict import bidict
from scipy.ndimage import filters
from scipy.spatial import cKDTree as KDTree


class Points(object):
    def __init__(self, sp: tuple, g: tuple):
        self.sp_coords = sp
        self.x, self.y = sp
        self.g = g
        self.valid = False

    def __len__(self):
        return len(self.sp_coords)

    def __getitem__(self, i):
        return self.sp_coords[i]

    def __repr__(self):
        return 'Item({}, {})'.format(self.sp_coords[0], self.sp_coords[1])


def sharpness_detect(image):
    sharpness = cv2.Laplacian(image, cv2.CV_16U)
    return cv2.mean(sharpness)[0]


def image_gradient(image, sigma):
    image = np.asfarray(image)
    gx = filters.gaussian_filter(image, sigma, order=[0, 1])
    gy = filters.gaussian_filter(image, sigma, order=[1, 0])
    return gx, gy


# CurvePoints = (x, y, gx, gy, valid)
def compute_edge_points(grads, min_magnitude=0):
    gx, gy = grads

    # shape: height, width, color
    # 计算梯度
    mag = (gx ** 2 + gy ** 2) ** 0.5
    mag_valid = mag > min_magnitude
    mag_direction = np.absolute(gx) - np.absolute(gy) > 0
    mag_left = mag - np.roll(mag, 1, axis=1) > 0
    mag_right = mag - np.roll(mag, -1, axis=1) > 0
    mag_top = mag - np.roll(mag, 1, axis=0) > 0
    mag_bottom = mag - np.roll(mag, -1, axis=0) > 0

    # 确定目标点偏移量
    theta_x = mag_left & mag_right & mag_direction
    theta_y = mag_top & mag_bottom & ~mag_direction

    pts = mag_valid & (theta_x | theta_y)
    idx_b = np.transpose(pts.nonzero())

    x = theta_x[idx_b[:, 0], idx_b[:, 1]]
    y = theta_y[idx_b[:, 0], idx_b[:, 1]]

    idx_a = np.column_stack((idx_b[:, 0] - y, idx_b[:, 1] - x))
    idx_c = np.column_stack((idx_b[:, 0] + y, idx_b[:, 1] + x))

    mag_a = mag[idx_a[:, 0], idx_a[:, 1]]
    mag_b = mag[idx_b[:, 0], idx_b[:, 1]]
    mag_c = mag[idx_c[:, 0], idx_c[:, 1]]
    lamda = (mag_a - mag_c) / (2 * (mag_a - 2 * mag_b + mag_c))

    pts_x = idx_b[:, 1] + lamda * x
    pts_y = idx_b[:, 0] + lamda * y
    pts = np.column_stack((pts_x, pts_y))

    # 准备数据点
    gx = gx[idx_b[:, 0], idx_b[:, 1]].tolist()
    gy = gy[idx_b[:, 0], idx_b[:, 1]].tolist()
    pts = pts.tolist()

    ret = []
    for i in range(len(pts)):
        ret.append(Points(pts[i], (gx[i], gy[i])))

    return ret


def chain_edge_points(pts):
    tree = KDTree(np.array([(pt[0], pt[1]) for pt in pts]))

    def dot(p0, p1):
        return p0[0] * p1[0] + p0[1] * p1[1]

    def envec(p0, p1):
        return p1[0] - p0[0], p1[1] - p0[1]

    def perp(p0):
        return p0[1], -p0[0]

    def dist(p0, p1):
        return ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5

    links = bidict()
    for e in pts:
        # TODO 循环内可以优化
        # nhood = [n for n in tree.search_nn_dist(e, 4) if dot(e.g, n.g) > 0]
        idx_nhood = [n for n in tree.query_ball_point(np.array([e[0], e[1]]), r=2)]
        nhood = [pts[idx] for idx in idx_nhood]
        nf = [n for n in nhood if dot(envec(e, n), perp(e.g)) > 0]
        nb = [n for n in nhood if dot(envec(e, n), perp(e.g)) < 0]

        if nf:
            nf_dist = [dist(e, n) for n in nf]
            f = nf[nf_dist.index(min(nf_dist))]
            if f not in links.inv or dist(e, f) < dist(links.inv[f], f):
                if f in links.inv:
                    del links.inv[f]
                if e in links:
                    del links[e]
                links[e] = f

        if nb:
            nb_dist = [dist(e, n) for n in nb]
            b = nb[nb_dist.index(min(nb_dist))]
            if b not in links or dist(b, e) < dist(b, links[b]):
                if b in links:
                    del links[b]
                if e in links.inv:
                    del links.inv[e]
                links[b] = e
    return links


def thresholds_with_hysteresis(edges, links, high_threshold, low_threshold):
    def mag(p):
        return np.hypot(p.g[0], p.g[1])

    chains = []
    for e in edges:
        if not e.valid and mag(e) >= high_threshold:
            forward = []
            backward = []
            e.valid = True
            f = e
            while f in links and not links[f].valid and mag(links[f]) >= low_threshold:
                n = links[f]
                n.valid = True
                f = n
                forward.append(f)
            b = e
            while b in links.inv and not links.inv[b].valid and mag(links.inv[f]) >= low_threshold:
                n = links.inv[b]
                n.valid = True
                b = n
                backward.insert(0, b)
            chain = backward + [e] + forward
            # 检查曲线是否闭合
            if links.inv.get(chain[0]) == chain[-1]:
                chain.append(chain[0])
            chains.append(np.asarray([(c.x, c.y) for c in chain]))
    return chains
