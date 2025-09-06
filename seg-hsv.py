import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
# 使用perf_counter
start = time.perf_counter()  # 记录开始时间

def log_memory_usage(step_name):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"{step_name}: {mem_mb:.2f} MB")

def select_roi(image):

      # 手动选择感兴趣区域
    roii = cv2.selectROI('Select cropping area', image, showCrosshair=False, fromCenter=False)
    print(roii)
    x, y, width, height = roii

    #   设定感兴趣区域
    # left
    # x, y, width, height = 475, 302, 587, 550  # 植物

    # x, y, width, height = 413, 294, 867, 691

    # right
    # x, y, width, height = 200, 177, 587, 550
    # x, y, width, height = 189, 225, 894, 763

    # roii = x, y, width, height
    # Create a mask of the same size as the image and initialize it to zeros
    mask = np.zeros_like(image, dtype=np.uint8)

    # Set the pixels inside the ROI to the values of the original image
    roi = image[y:y + height, x:x + width]
    mask[y:y + height, x:x + width] = roi

    # Display the result
    # cv2.imshow("Masked Image", mask)
    # cv2.waitKey(10)
    # cv2.destroyAllWindows()
    return roii

# 创建一个函数来处理鼠标点击事件
def getHSV(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 当鼠标左键点击时，获取当前像素点的BGR颜色
        bgr_color = image[y, x]
        # 将BGR颜色转换为HSV
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        # 打印HSV颜色值
        print("HSV color: ", hsv_color)
        # 将HSV颜色转换回BGR用于显示
        display_bgr = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
        # 在原始图像上绘制一个绿色的圆圈来表示选中的颜色
        # cv2.circle(image, (x, y), 10, display_bgr, -1)

def mouse(image):
    # 创建窗口
    cv2.namedWindow('Image')

    # 绑定鼠标事件
    cv2.setMouseCallback('Image', getHSV)

    while True:
        # 显示图像
        cv2.imshow('Image', image)
        # 等待键盘事件，按'q'退出
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # 释放窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    log_memory_usage("初始内存")  # <-- 添加检查点

    path = 'data/left/3_to_1/color_L.png'
    # set save path
    base_path=os.path.abspath(os.path.join(path,'../..'))
    save_basepath=os.path.join(base_path,'seg\hsv')
    os.makedirs(save_basepath,exist_ok=True)

    save_path =os.path.join(save_basepath,'L.png')
    save_color_path = os.path.join(save_basepath,'color_L.png')
    save_mask_path = os.path.join(save_basepath,'mask_L.png')
    # read image
    image = cv2.imread(path)
    gray_image=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # set roi
    roi = 475, 302, 587, 500  # 植物
    # roi = 485, 176, 609, 473 # plant2
    # roi=select_roi(image)
    roi_img = np.zeros_like(image, dtype=np.uint8)
    x, y, width, height = roi
    # Set the pixels inside the ROI to the values of the original image
    roiii = image[y:y + height, x:x + width]
    roi_img[y:y + height, x:x + width] = roiii
    image=roi_img
    # convert to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define Set thresholds
    lower_green = (1, 70, 1)
    upper_green = (255, 255, 255)

    # set mask
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # show image
    seg_color_image=cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    seg_gray_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    # cv2.imshow('seg',seg_gray_image)
    # cv2.waitKey(0)
    # write image
    cv2.imwrite(save_mask_path,mask)
    cv2.imwrite(save_path,seg_gray_image)
    cv2.imwrite(save_color_path,seg_color_image)


    # read image
    path = r'data\right\3_to_1\color_R.png'
    base_path = os.path.abspath(os.path.join(path, '../..'))
    save_basepath = os.path.join(base_path, 'seg\hsv')
    os.makedirs(save_basepath, exist_ok=True)
    save_path =os.path.join(save_basepath,'R.png')
    save_color_path = os.path.join(save_basepath,'color_R.png')
    save_mask_path = os.path.join(save_basepath,'mask_R.png')
    print(save_color_path)
    image = cv2.imread(path)
    gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    roi = 218, 255, 552, 492
    # roi =350, 176, 596, 506
    # roi=select_roi(image)
    roi_img = np.zeros_like(image, dtype=np.uint8)
    x, y, width, height = roi
    # Set the pixels inside the ROI to the values of the original image
    roiii = image[y:y + height, x:x + width]
    roi_img[y:y + height, x:x + width] = roiii
    image=roi_img
    # convert to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mouse(image) #鼠标选取

    # define threshold
    lower_green = (1, 70, 1)
    upper_green = (255, 255, 255)

    # set mask
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # show image
    seg_color_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    seg_gray_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)


    # cv2.imshow('seg',seg_gray_image)
    # cv2.waitKey(0)

    # write image
    cv2.imwrite(save_path, seg_gray_image)
    cv2.imwrite(save_color_path, seg_color_image)

    end = time.perf_counter()  # 记录结束时间
    run_time = end - start  # 计算运行时间
    print(f"运行时间：{run_time}秒")
    log_memory_usage("处理数据后")  # <-- 添加检查点
