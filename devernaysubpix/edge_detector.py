import numpy as np
from scipy.ndimage import filters
from bidict import bidict


def image_gradient(image, sigma):
    image = np.asfarray(image)
    gx = filters.gaussian_filter(image, sigma, order=[0, 1])
    gy = filters.gaussian_filter(image, sigma, order=[1, 0])
    return gx, gy


class CurvePoint(object):
    __slots__ = ['x', 'y', 'valid']

    def __init__(self, x, y, valid):
        self.x = x
        self.y = y
        self.valid = valid

    def __hash__(self):
        return hash((self.x, self.y))


def compute_edge_points(partial_gradients, min_magnitude=0):
    gx, gy = partial_gradients
    rows, cols = gx.shape
    edges = []

    def mag(y, x):
        return np.hypot(gx[y, x], gy[y, x])

    ret_list = []
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):

            center_mag = mag(y, x)
            if center_mag < min_magnitude:
                continue

            left_mag = mag(y, x - 1)
            right_mag = mag(y, x + 1)
            top_mag = mag(y - 1, x)
            bottom_mag = mag(y + 1, x)

            theta_x, theta_y = 0, 0
            if (left_mag < center_mag >= right_mag) and abs(gx[y, x]) >= abs(gy[y, x]):
                theta_x = 1
            elif (top_mag < center_mag >= bottom_mag) and abs(gx[y, x]) <= abs(gy[y, x]):
                theta_y = 1
            if theta_x != 0 or theta_y != 0:
                a = mag(y - theta_y, x - theta_x)
                b = mag(y, x)
                c = mag(y + theta_y, x + theta_x)
                lamda = (a - c) / (2 * (a - 2 * b + c))
                ex = x + lamda * theta_x
                ey = y + lamda * theta_y
                ret_list.append((x, y))
                edges.append(CurvePoint(ex, ey, valid=False))
    # print(ret_list)
    # print([(e.x, e.y) for e in edges])
    return np.asarray(edges)


# TODO 需要准备 gx gy 梯度矩阵， edge 边缘点bool矩阵

def chain_edge_points(edges, g):
    gx, gy = g

    # TODO 优化 找neightbor不用每次都全部遍历，进行分区效果会更好。
    def neighborhood(p, max_dist):
        px, py = p.x, p.y
        for e in edges:
            ex, ey = e.x, e.y
            # if abs(ex - px) <= max_dist and abs(ey - py) <= max_dist:
            if ((ex - px) ** 2 + (ey - py) ** 2) <= max_dist ** 2:
                yield e

    def gval(p):
        px, py = int(p.x), int(p.y)
        return [gx[py, px], gy[py, px]]

    def envec(e, n):
        return np.asanyarray([n.x, n.y]) - np.asanyarray([e.x, e.y])

    def perp(v):
        x, y = gval(e)
        return np.asanyarray([y, -x])

    def dist(a, b):
        a = [a.x, a.y]
        b = [b.x, b.y]
        return np.hypot(*(np.subtract(b, a)))

    links = bidict()
    for e in edges:
        nhood = [n for n in neighborhood(e, 2) if np.dot(gval(e), gval(n)) > 0]
        nf = [n for n in nhood if np.dot(envec(e, n), perp(gval(e))) > 0]
        nb = [n for n in nhood if np.dot(envec(e, n), perp(gval(e))) < 0]

        if nf:
            f_idx = np.argmin([dist(e, n) for n in nf])
            f = nf[f_idx]
            if f not in links.inv or dist(e, f) < dist(links.inv[f], f):
                if f in links.inv: del links.inv[f]
                if e in links: del links[e]
                links[e] = f

        if nb:
            b_idx = np.argmin([dist(e, n) for n in nb])
            b = nb[b_idx]
            if b not in links or dist(b, e) < dist(b, links[b]):
                if b in links: del links[b]
                if e in links.inv: del links.inv[e]
                links[b] = e
    return links


def thresholds_with_hysteresis(edges, links, grads, high_threshold, low_threshold):
    gx, gy = grads

    def mag(p):
        x, y = int(p.x), int(p.y)
        return np.hypot(gx[y, x], gy[y, x])

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
            chains.append(np.asarray([(c.x, c.y) for c in chain]))
    return chains


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt

    pad = 20
    circle = cv2.imread("./kreis.png", 0)
    I = np.zeros((circle.shape[0] + 2 * pad, circle.shape[1] + 2 * pad), dtype=np.uint8) + 255
    I[pad:circle.shape[0] + pad, pad:circle.shape[1] + pad] = circle
    I = I.astype(np.float32)

    grads = image_gradient(I, 2.0)
    edges = compute_edge_points(grads)


    grads = image_gradient(I, 2.0)
    edgels = compute_edge_points(grads)
    xx = [e.x for e in edgels]
    yy = [e.y for e in edgels]
    plt.figure(figsize=(10, 10))
    plt.imshow(I, cmap='gray')
    plt.plot(yy, xx, 'or', ms=3, mew=3)
    plt.show()
