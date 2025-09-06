# -*-coding: utf-8 -*-

import cv2
import numpy as np
import os



def getRectifyTransform(height, width, config):
    """
    获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
    config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
    :param height:
    :param width:
    :param config:
    :return:
    """
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


def rectify_image(imgL, imgR, map1x, map1y, map2x, map2y):
    """
    畸变校正和立体校正
    :param imgL:
    :param imgR:
    :param map1x:
    :param map1y:
    :param map2x:
    :param map2y:
    :return:
    """
    rectifyed_img1 = cv2.remap(imgL, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(imgR, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2


def get_rectify_image(imgL, imgR):
    """
    畸变校正和立体校正
    根据更正map对图片进行重构
    获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    :param imgL:
    :param imgR:
    :return:
    """
    rectifiedL = cv2.remap(imgL, camera_config["left_map_x"], camera_config["left_map_y"],
                           interpolation=cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
    rectifiedR = cv2.remap(imgR, camera_config["right_map_x"], camera_config["right_map_y"],
                           interpolation=cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
    return rectifiedL, rectifiedR


def draw_line_rectify_image(imgL, imgR, interval=50, color=(0, 255, 0), show=True, namewin='rect'):
    """
    绘制等间距平行线，检查立体校正的效果
    :param namewin:
    :param imgL: 畸变校正和立体校正后的左视图
    :param imgR：畸变校正和立体校正后的右视图
    :param interval:直线间隔
    :param show:是否显示
    :return:
    """
    # xl, yl = imgL.shape[0], imgL.shape[1]  # 读取图片尺寸（像素）
    # x_s = 640  # 定义缩小后的标准宽度
    # y_s = int(yl * x_s / xl)  # 基于标准宽度计算缩小后的高度
    # imgL = imgL.resize(x_s, y_s)  # 改变尺寸，保持图片高品质
    # xr, yr = imgR.shape[0], imgR.shape[1]  # 读取图片尺寸（像素）
    # x_s = 640  # 定义缩小后的标准宽度
    # y_s = int(y2 * x_s / x2)  # 基于标准宽度计算缩小后的高度
    # imgR = imgR.resize(x_s, y_s) # 改变尺寸，保持图片高品质

    h, w = imgL.shape[:2]
    # 缩放比例k，>1表示放大，<1表示缩小
    k = 2 / 3
    # 元组参数，为宽，高k
    imgL = cv2.resize(imgL, (int(w * k), int(h * k)), interpolation=cv2.INTER_LINEAR)

    h, w = imgR.shape[:2]
    # 缩放比例k，>1表示放大，<1表示缩小
    k = 2 / 3
    # 元组参数，为宽，高k
    imgR = cv2.resize(imgR, (int(w * k), int(h * k)), interpolation=cv2.INTER_LINEAR)

    height = max(imgL.shape[0], imgR.shape[0])
    width = imgL.shape[1] + imgR.shape[1]

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[0:imgL.shape[0], 0:imgL.shape[1]] = imgL
    img[0:imgR.shape[0], imgL.shape[1]:] = imgR
    # 绘制等间距平行线
    for k in range(height // interval):
        cv2.line(img, (0, interval * (k + 1)), (2 * width, interval * (k + 1)), color, 2, lineType=cv2.LINE_AA)

    if show:
        cv2.imshow(namewin, img)
        cv2.waitKey(1)
    return img


def save_rect_image(args, rectifiedL, rectifiedR):
    path1 = os.path.abspath(os.path.join(args.left_file, '../..'))
    save_rect_pathL = os.path.join(path1, 'rect')
    path2 = os.path.abspath(os.path.join(args.right_file, '../..'))
    save_rect_pathR = os.path.join(path2, 'rect')
    if not os.path.exists(save_rect_pathL):
        os.makedirs(save_rect_pathL)
    if not os.path.exists(save_rect_pathR):
        os.makedirs(save_rect_pathR)
    cv2.imwrite(save_rect_pathL + '/rectifiedL.png', rectifiedL)
    cv2.imwrite(save_rect_pathR + '/rectifiedR.png', rectifiedR)
