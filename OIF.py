import os
import numpy as np
from scipy.stats import entropy
from PIL import Image
import itertools
import time
import psutil
from sklearn.metrics import mutual_info_score
import cv2

def log_memory_usage(step_name):
    """记录并打印当前内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"{step_name}: {mem_mb:.2f} MB")


def compute_image_features(img_array):
    """计算图像特征：亮度差、均值、标准差、信息熵"""
    prob = np.histogram(img_array, bins=256, range=(0, 255))[0] / img_array.size
    prob = prob[prob > 0]  # 移除零概率项
    return [
        np.ptp(img_array),
        np.mean(img_array),
        np.std(img_array),
        entropy(prob)
    ]


def correlation_matrix(stacked_arrays):
    """计算波段间相关系数矩阵"""
    n, h, w = stacked_arrays.shape
    band_data = stacked_arrays.reshape((n, h * w))
    return np.corrcoef(band_data)


def calculate_oif(entropies, corr_matrix):
    """计算最佳波段组合的OIF值"""
    all_combos = list(itertools.combinations(range(len(entropies)), 3))
    oif_sums = []

    for combo in all_combos:
        S = [entropies[i] for i in combo]
        R = corr_matrix[np.ix_(combo, combo)].sum(axis=0)
        oif_sums.append(sum(S) / sum(R))

    return all_combos[np.argmax(oif_sums)]


def merge(arr, best_bands,path):
    # Read three spectral images (replace with your image paths)
    a,b,c=best_bands
    image1 = arr[a]
    image2 = arr[b]
    image3 = arr[c]

    # Optional: Normalize images to a common range
    image1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX)
    image2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX)
    image3 = cv2.normalize(image3, None, 0, 255, cv2.NORM_MINMAX)

    # Stack the spectral bands horizontally
    merged_image = np.stack((image1, image2, image3), axis=2)
    gray_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2GRAY)

    path2 = os.path.abspath(os.path.join(path, ".."))

    save_path = os.path.join(path2, '3_to_1')
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(os.path.join(save_path, 'color.png'))
    cv2.imwrite(os.path.join(save_path, 'color.png'), merged_image)
    cv2.imwrite(os.path.join(save_path,  'gray.png'), gray_image)


if __name__ == '__main__':
    # 启动内存和时间监控
    start = time.perf_counter()
    log_memory_usage("初始内存")

    file_path = r'F:\data\2023\1024\right\cube_20231024_151302\png'

    # 加载并处理所有图像
    arrays, features = [], []
    for fname in os.listdir(file_path):
        img = np.array(Image.open(os.path.join(file_path, fname)).convert('L'))

        arrays.append(img)
        features.append(compute_image_features(img))
    log_memory_usage("加载所有图像后")

    # 堆叠数组并计算相关系数
    stacked = np.stack(arrays)
    corr_mat = correlation_matrix(stacked)
    log_memory_usage("计算相关系数矩阵后")

    # 计算最佳波段组合
    best_bands = calculate_oif([f[3] for f in features], corr_mat)
    log_memory_usage("计算最佳波段组合后")

    print(f"最佳波段组合: {best_bands}")


    merge(arrays,best_bands,file_path)
    log_memory_usage("merge后")
    print(f"运行时间: {time.perf_counter() - start:.2f}秒")