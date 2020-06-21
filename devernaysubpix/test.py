import os
import cv2
from matplotlib import pyplot as plt
import new_detector


def main():
    # 清晰度检测
    # data_path = './data_sharpness'
    # files = os.listdir(data_path)
    # for f in files:
    #     p = os.path.join(data_path, f)
    #     print('filepath: ', p)
    #     sharpness = new_detector.sharpness_detect(cv2.imread(p, 0))
    #     print('sharpness: ', sharpness)

    # 亚像素检测
    # I = cv2.imread("./2018112101.jpg", 0)
    I = cv2.imread("./real_eyes.png", 0)
    grads = new_detector.image_gradient(I, 2.0)
    edges = new_detector.compute_edge_points(grads)
    links = new_detector.chain_edge_points(edges)
    chains = new_detector.thresholds_with_hysteresis(edges, links, 1, 0.1)

    # 通过figure调整输出图像大小
    plt.figure(figsize=(64, 48))
    for chain in chains:
        plt.plot(chain[:, 0], chain[:, 1], '-')

    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.savefig('test.png')


if __name__ == '__main__':
    main()
