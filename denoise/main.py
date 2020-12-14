# -*- coding: utf-8 -*-

import cv2
import os
import time

import Denoise


def main():
    path = '/home/chenlei/Pictures/dip/'
    raw_img = cv2.imread(os.path.join(path, 'test1.jpg'), 0)
    cv2.imshow('origin image', raw_img)

    denoise = Denoise.Denoise(raw_img)
    start = time.time()
    # new_img = denoise.arithmetic_mean_filter()
    # new_img = denoise.geometric_mean_filter()
    # new_img = denoise.harmonic_mean_filter()
    new_img = denoise.median_filter()
    end = time.time()
    print('cost {}ms.'.format(int((end - start) * 1000)))

    cv2.imshow('output', new_img)
    cv2.imwrite(os.path.join(path, 'output/test1_out_median.jpg'), new_img)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
