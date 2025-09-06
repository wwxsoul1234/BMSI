import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import core.camera_params
import os
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False

def plot_derivative(band_data,title):
    # plt.clf()
    wavelength = ['713', '736', '759', '782', '805', '828', '851', '874', '897', '920']
    # Method 1
    # order = 1
    # delta = 1
    # for n in range(order):
    #     spectra = np.gradient(band_data, delta, axis=0)
    # plt.plot(wavelength, spectra, marker='o', label='1st Derivative')
    # plt.title(title)
    # plt.xlabel('Band Index')
    # plt.ylabel('Derivative Value')
    # plt.ylim(-0.2, 0.2)  # 设置y轴显示范围
    # plt.legend()
    # plt.show()

    # Method 2
    # band_data_extended = band_data + [band_data[0]]  # 添加第一个数据到列表末尾
    # # 计算一阶导数（相邻元素之差）
    # first_derivative11 = [band_data_extended[i + 1] - band_data_extended[i] for i in range(len(band_data))]
    # plt.plot(wavelength, first_derivative11, marker='o')
    # plt.title(title)
    # plt.xlabel('waveBand Index')
    # plt.ylabel('Derivative Value')
    # plt.ylim(-0.2, 0.2)  # 设置y轴显示范围
    # plt.legend()
    # plt.show()

    # Method 3
    # 计算一阶导数
    first_derivative = np.diff(band_data) / (np.diff(np.arange(len(band_data))) + 1)
    # 由于一阶导数数组长度比原始数据短1，我们需要创建一个新的x轴数组来匹配
    x_axis_first_derivative = np.arange(1, len(band_data))
    # 绘制一阶导数的图
    plt.plot(x_axis_first_derivative, first_derivative, marker='o', label='1st Derivative')
    plt.title('First Derivative of Band Data')
    plt.xlabel('Band Index')
    plt.ylabel('Derivative Value')
    plt.ylim(-0.1, 0.1)  # 设置y轴显示范围
    plt.legend()
    plt.show()


def L_TP_Reflect(oldpath, x, y, left_map1, left_map2):
    # 通过相对路径返回上一级目录
    # path = os.path.abspath(os.path.join(oldpath, ".."))
    path = oldpath
    ref = []
    images_left = os.listdir(path)
    for fname in images_left:
        iml = cv2.imread(os.path.join(path, fname))
        imgl_rectified = cv2.remap(iml, left_map1, left_map2, cv2.INTER_LINEAR)
        iml_gray = cv2.cvtColor(imgl_rectified, cv2.COLOR_BGR2GRAY)
        # sum = 0
        # for i in range(i - 1, i + 1):
        #     for j in range(j - 1, j + 1):
        #         sum = sum + iml_gray[i, j]
        # ave = sum / len(sum)
        ok = iml_gray[x - 1:x + 1, y - 1:y + 1]
        ave = np.sum(ok) / ok.size / 255

        ref.append(ave)
    return ref


def plot_Ref(ref, title):
    plt.clf()
    wavelength = ['713', '736', '759', '782', '805', '828', '851', '874', '897', '920']
    plt.plot(wavelength, ref,label='reflection')
    plt.ylim((0, 1))
    plt.title(title)
    plt.legend()
    plt.show()


def plt_REF_DER(ref,title):
    plt.close("all")  # 关闭所有已打开的图形窗口
    wavelength = ['713', '736', '759', '782', '805', '828', '851', '874', '897', '920']
    data1=[i*100 for i in ref]
    band_data = ref
    first_derivative = np.diff(band_data) / (np.diff(np.arange(len(band_data))) + 1)

    data2 =first_derivative
    x = np.arange(1, len(band_data))
    fig, ax1 = plt.subplots()
    # Plot data1 with its own y-axis scale and label
    color = 'tab:blue'
    plt.title(title)
    ax1.set_xlabel('Wavelength/nm')
    ax1.set_ylabel('Reflectance/%', color=color)
    ax1.plot(wavelength, data1, color=color, label='Reflectance')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    # Create a twin y-axis for data2 and plot it
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('First Derivative', color=color)
    ax2.plot(x, data2, color=color, label='First Derivative')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.1, 0.15)

    fig.tight_layout()  # Adjust subplot spacing

    plt.show()


if __name__ == '__main__':
    oldpath = r"F:\data\2024\stone\stone\20240322\2\2-far\cube_20240322_125922\pos_1\png"
    # current_dir = os.path.abspath(oldpath)
    # # 通过相对路径返回上一级目录
    # path= os.path.abspath(os.path.join(oldpath, ".."))

    camera_config = core.camera_params.get_stereo_coefficients("../configs/lenacv-camera/stereo_cam.yml")
    left_map1, left_map2 = camera_config["left_map_x"], camera_config["left_map_y"]

    x, y = (939, 779)
    refer = L_TP_Reflect(oldpath, x, y, left_map1, left_map2)
    # plot_Ref(refer, 'ok')
    # plt.show()
    # plot_derivative(refer,'ok')
    plt_REF_DER(refer,'ok')
