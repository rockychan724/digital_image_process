# -*- coding: utf-8 -*-

from cv2 import cv2
import os
import math
import numpy as np
import tensorflow as tf


def MSE(raw_img, new_img):
    height = raw_img.shape[0]
    width = raw_img.shape[1]
    sum = 0.0
    for i in range(0, height):
        for j in range(0, width):
            a = int(raw_img[i, j])
            b = int(new_img[i, j])
            sum = sum + (a - b)**2
    return sum / (height * width)


def SNR(raw_img, new_img):
    height = raw_img.shape[0]
    width = raw_img.shape[1]
    sum1 = 0.0
    sum2 = 0.0
    for i in range(0, height):
        for j in range(0, width):
            a = int(raw_img[i, j])
            b = int(new_img[i, j])
            sum1 = sum1 + a**2
            sum2 = sum2 + (a - b)**2
    return 10 * math.log10(sum1 / sum2)


def main():
    path = '/home/chenlei/Pictures/dip/'

    print('********* arithmetic_mean_filter')
    for i in range(2):
        raw_img = cv2.imread(os.path.join(path, 'org' + str(i + 1) + '.jpg'),
                             0)
        new_img = cv2.imread(
            os.path.join(
                path, 'output/test' + str(i + 1) + '_out_arithmetic_mean.jpg'),
            0)
        print('org{}.jpg: MSE = {}, SNR = {}'.format(i + 1,
                                                     MSE(raw_img, new_img),
                                                     SNR(raw_img, new_img)))

    print('********* geometric_mean_filter')
    for i in range(2):
        raw_img = cv2.imread(os.path.join(path, 'org' + str(i + 1) + '.jpg'),
                             0)
        new_img = cv2.imread(
            os.path.join(
                path, 'output/test' + str(i + 1) + '_out_geometric_mean.jpg'),
            0)
        print('org{}.jpg: MSE = {}, SNR = {}'.format(i + 1,
                                                     MSE(raw_img, new_img),
                                                     SNR(raw_img, new_img)))

    print('********* harmonic_mean_filter')
    for i in range(2):
        raw_img = cv2.imread(os.path.join(path, 'org' + str(i + 1) + '.jpg'),
                             0)
        new_img = cv2.imread(
            os.path.join(path, 'output/test' + str(i + 1) +
                         '_out_harmonic_mean.jpg'), 0)
        print('org{}.jpg: MSE = {}, SNR = {}'.format(i + 1,
                                                     MSE(raw_img, new_img),
                                                     SNR(raw_img, new_img)))

    print('********* median_filter')
    for i in range(2):
        raw_img = cv2.imread(os.path.join(path, 'org' + str(i + 1) + '.jpg'),
                             0)
        new_img = cv2.imread(
            os.path.join(path, 'output/test' + str(i + 1) + '_out_median.jpg'),
            0)
        print('org{}.jpg: MSE = {}, SNR = {}'.format(i + 1,
                                                     MSE(raw_img, new_img),
                                                     SNR(raw_img, new_img)))


if __name__ == "__main__":
    # main()
    a = tf.constant([1.2, 3.4], dtype=tf.float32, name='a')
    b = tf.constant([3.4, 5.6], dtype=tf.float32, name='b')
    result = a - b
    with tf.Session() as sess:
        print(sess.run(result))