# -*- coding: utf-8 -*-

from cv2 import cv2
import os
import time

import ImageSegmentation


def main():
    path = '/home/chenlei/Pictures/dip/'
    raw_img = cv2.imread(os.path.join(path, 'test4.jpg'), 0)
    cv2.imshow('origin image', raw_img)

    imageSeg = ImageSegmentation.ImageSegmentation(raw_img)
    start = time.time()
    # new_img = imageSeg.threshold_seg()
    # new_img = imageSeg.roberts_edge_detect()
    # new_img = imageSeg.prewitt_edge_detect()
    # new_img = imageSeg.sobel_edge_detect()
    # new_img = imageSeg.laplacian4_edge_detect()
    # new_img = imageSeg.laplacian8_edge_detect()
    new_img = imageSeg.log_edge_detect()
    end = time.time()
    print('cost {}ms.'.format(int((end - start) * 1000)))

    cv2.imshow('output', new_img)
    cv2.imwrite(os.path.join(path, 'output/test4_out_log_edge_detect.jpg'), new_img)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()