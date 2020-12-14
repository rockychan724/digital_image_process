# -*- coding: utf-8 -*-

import numpy as np


class ImageSegmentation:
    def __init__(self, img):
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.raw_img = img
        self.pad_img = np.pad(img, ((1, 1), (1, 1)), 'constant') # 全0填充一层
        self.pad_img2 = np.pad(img, ((2, 2), (2, 2)), 'constant') # 全0填充两层

    # 阈值分割
    def threshold_seg(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                if self.raw_img[i, j] < 55:
                    new_img[i, j] = 0
                elif self.raw_img[i, j] > 160:
                    new_img[i, j] = 255
                else:
                    new_img[i, j] = 100
        return new_img

    # Roberts边缘检测
    def roberts_edge_detect(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_roberts_operator(i, j)
        return new_img

    # Prewitt边缘检测
    def prewitt_edge_detect(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_other_operator(i, j, 'prewitt')
        return new_img

    # Sobel边缘检测
    def sobel_edge_detect(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_other_operator(i, j, 'sobel')
        return new_img
    
    # 4邻域Laplacian边缘检测
    def laplacian4_edge_detect(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_other_operator(i, j, 'laplacian4')
        return new_img
    
    # 8邻域Laplacian边缘检测
    def laplacian8_edge_detect(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_other_operator(i, j, 'laplacian8')
        return new_img

    # LoG边缘检测
    def log_edge_detect(self):
        new_img = self.raw_img
        for i in range(0, self.height):
            for j in range(0, self.width):
                new_img[i, j] = self._get_log_operator(i, j)
        return new_img

    def _get_roberts_operator(self, i, j):
        pad_i = i + 1
        pad_j = j + 1
        k = []
        for m in range(0, 2):
            for n in range(0, 2):
                k.append(int(self.pad_img[pad_i + m, pad_j + n]))
        return abs(k[3] - k[0]) + abs(k[2] - k[1])
    
    def _get_log_operator(self, i, j):
        pad2_i = i + 2
        pad2_j = j + 2
        k = [[0 for a in range(5)] for b in range(5)]
        for m in range(-2, 3):
            for n in range(-2, 3):
                k[m + 2][n + 2] = int(self.pad_img2[pad2_i + m, pad2_j + n])
        return (16*k[2][2] - 2*(k[1][2] + k[2][1] + k[3][2] + k[2][3]) - (k[0][2] + k[1][1] + k[2][0] + k[3][1] + k[4][2] + k[3][3] + k[2][4] + k[1][3]))

    def _get_other_operator(self, i, j, mode):
        pad_i = i + 1
        pad_j = j + 1
        k = [[0 for a in range(3)] for b in range(3)]
        for m in range(-1, 2):
            for n in range(-1, 2):
                k[m + 1][n + 1] = int(self.pad_img[pad_i + m, pad_j + n])
        if mode == 'prewitt':
            return abs(k[0][2] + k[1][2] + k[2][2] - k[0][0] - k[1][0] - k[2][0]) + abs(k[2][0] + k[2][1] + k[2][2] - k[0][0] - k[0][1] - k[0][2])
        elif mode == 'sobel':
            return abs(k[0][2] + 2*k[1][2] + k[2][2] - k[0][0] - 2*k[1][0] - k[2][0]) + abs(k[2][0] + 2*k[2][1] + k[2][2] - k[0][0] - 2*k[0][1] - k[0][2])
        elif mode == 'laplacian4':
            return (k[0][1] + k[1][0] + k[2][1] + k[1][2] - 4*k[1][1])
        elif mode == 'laplacian8':
            return (k[0][0] + k[0][1] + k[0][2] + k[1][0] + k[1][2] + k[2][0] + k[2][1] + k[2][2] - 8*k[1][1])
        else:
            return 0
