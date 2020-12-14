# -*- coding: utf-8 -*-

import numpy as np


class Denoise:
    def __init__(self, img):
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.raw_img = img
        self.pad_img = np.pad(img, ((1, 1), (1, 1)), 'constant') # 全0填充

    # 算数均值滤波
    def arithmetic_mean_filter(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_arithmetic_mean(i, j)
        return new_img

    # 几何均值滤波
    def geometric_mean_filter(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_geometric_mean(i, j)
        return new_img

    # 谐波均值滤波
    def harmonic_mean_filter(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_harmonic_mean(i, j)
        return new_img

    # 中值滤波
    def median_filter(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_median(i, j)
        return new_img

    def _get_arithmetic_mean(self, i, j):
        pad_i = i + 1
        pad_j = j + 1
        sum = 0
        for m in range(-1, 2):
            for n in range(-1, 2):
                sum += int(self.pad_img[pad_i + m, pad_j + n])
        return sum / 9

    def _get_geometric_mean(self, i, j):
        pad_i = i + 1
        pad_j = j + 1
        product = 1
        for m in range(-1, 2):
            for n in range(-1, 2):
                product *= int(self.pad_img[pad_i + m, pad_j + n])
        return int(pow(product, 1/9))

    def _get_harmonic_mean(self, i, j):
        pad_i = i + 1
        pad_j = j + 1
        sum = 0.0
        for m in range(-1, 2):
            for n in range(-1, 2):
                if self.pad_img[pad_i + m, pad_j + n] > 0:
                    sum += 1 / int(self.pad_img[pad_i + m, pad_j + n])
        return int(9 / sum)

    def _get_median(self, i, j):
        pad_i = i + 1
        pad_j = j + 1
        k = []
        for m in range(-1, 2):
            for n in range(-1, 2):
                k.append(self.pad_img[pad_i + m, pad_j + n])
        k.sort()
        return k[4]
