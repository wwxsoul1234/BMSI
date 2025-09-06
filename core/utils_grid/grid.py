# -*- coding: utf-8 -*-

import matplotlib  # 导入 matplotlib 库，主要用于绘图
import numpy as np  # 导入 numpy 库，主要用于处理数组
import open3d as o3d  # 导入 Open3D 库，用于处理点云数据
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于创建图像和画图
import cv2
import os


def read_points(path, img_path):
    try:
        # 使用Open3D的read_point_cloud函数读取点云
        pcd = o3d.io.read_point_cloud(path)

        # 检查点云是否成功读取
        if pcd is not None:
            print(f"Number of points in the read point cloud: {len(pcd.points)}")
            print("Point cloud successfully loaded.")
        else:
            print("Failed to load the point cloud.")
    except Exception as e:
        print(f"Error occurred while reading: {e}")
    # 使用 Open3D 读取点云数据
    print(pcd)  # 输出点云的个数
    points = np.array(list(pcd.points))

    ######

    image = cv2.imread(img_path,)
    height, width = image.shape[0:2]
    size = height * width
    colors_ = image.reshape(height * width, 3).astype(np.int64)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)
    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)
    # 将坐标+颜色叠加为点云数组
    points = np.hstack((points, rgb)).astype(np.float32)
    #####

    # points = select_point(points)
    return points


def mathlib_show(points):
    """

    @param points:挑选之后的点云
    """
    fig = plt.figure(figsize=(16, 10))  # 创建一个新的图形窗口，设置其大小为8x4
    ax1 = fig.add_subplot(121, projection='3d')  # 在图形窗口中添加一个3D绘图区域
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='g', s=0.01,
                alpha=0.5)  # 在这个区域中绘制点云数据的散点图，设置颜色为绿色，点的大小为0.01，透明度为0.5
    ax2 = fig.add_subplot(122)  # 在图形窗口中添加一个2D绘图区域
    # 1行2列的图形布局，其中该子图是第2个子图
    ax2.scatter(points[:, 1], points[:, 2], c='g', s=0.01, alpha=0.5)  # 在这个区域中绘制点云数据的散点图，设置颜色为绿色，点的大小为0.01，透明度为0.5
    ax1.set_title('3D')
    ax2.set_title('2D')
    plt.show()  # 显示图形窗口中的所有内容
    # plt.savefig(save_path)


def plot_two(points):
    # 将点云数据转化为 numpy 数组
    # print(points.shape)  # 输出数组的形状（行列数）
    fig = plt.figure(figsize=(16, 10))  # 创建一个新的图形窗口，设置其大小为8x4
    ax1 = fig.add_subplot(121, projection='3d')  # 在图形窗口中添加一个3D绘图区域
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], s=0.01,
                alpha=0.5)  # 在这个区域中绘制点云数据的散点图，设置颜色为绿色，点的大小为0.01，透明度为0.5
    ax2 = fig.add_subplot(122)  # 在图形窗口中添加一个2D绘图区域
    # 1行2列的图形布局，其中该子图是第2个子图
    ax2.scatter(points[:, 1], points[:, 2], c=points[:, 3], s=0.01,
                alpha=0.5)  # 在这个区域中绘制点云数据的散点图，设置颜色为绿色，点的大小为0.01，透明度为0.5
    ax1.set_title('3D')
    ax2.set_title('2D')
    # plt.show()  # 显示图形窗口中的所有内容
    # plt.savefig(save_path)
    # 将图形窗口中的内容保存到指定的路径
    # fig1 = plt.figure()  # 创建一个新的图形窗口，设置其大小为8x4
    # ax3 = fig1.add_subplot(111, projection='3d')  # 在图形窗口中添加一个3D绘图区域
    # ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], s=0.01,
    #             alpha=0.5)
    plt.show()


def select_point_plant(pointcloud):
    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]

    # 下面参数是经验性取值，需要根据实际情况调整
    idx1 = np.where(Z <=0 )  # 1
    # idx2 = np.where(Z > 1500000)
    idx2 = np.where(Z >0.8 )  # 5 2.8
    idx3 = np.where(X > 100)
    idx4 = np.where(X < -100)
    idx5 = np.where(Y > 100)
    idx6 = np.where(Y < -100)
    idx7 = np.where(abs(X) == float('inf'))
    idx8 = np.where(abs(Y) == float('inf'))
    idx9 = np.where(abs(Z) == float('inf'))
    # idx = np.hstack((idx1[0], idx2[0], idx3[0], idx4[0], idx5[0], idx6[0]))
    idx = np.hstack((idx1[0], idx2[0], idx3[0], idx4[0], idx5[0], idx6[0], idx7[0], idx8[0], idx9[0]))
    dst_pointcloud = np.delete(pointcloud, idx, 0)
    return dst_pointcloud


def select_point_heritage(pointcloud):
    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]

    # 下面参数是经验性取值，需要根据实际情况调整
    idx1 = np.where(Z <= 1)  # 0.8
    # idx2 = np.where(Z > 1500000)
    idx2 = np.where(Z > 2.8)  # 0.8、0.9
    idx3 = np.where(X > 0.65)
    idx4 = np.where(X < -0.4)
    idx5 = np.where(Y > 0.55)
    idx6 = np.where(Y < -0.5)
    idx7 = np.where(abs(X) == float('inf'))
    idx8 = np.where(abs(Y) == float('inf'))
    idx9 = np.where(abs(Z) == float('inf'))
    # idx = np.hstack((idx1[0], idx2[0], idx3[0], idx4[0], idx5[0], idx6[0]))
    idx = np.hstack((idx1[0], idx2[0], idx3[0], idx4[0], idx5[0], idx6[0], idx7[0], idx8[0], idx9[0]))
    dst_pointcloud = np.delete(pointcloud, idx, 0)
    return dst_pointcloud


def show_pointcloud(point_cloud):

    # # 设置显示角度
    # view_elevation = 0  # 观察者相对于XY平面的仰角（向上为正）
    # view_azimuth = 0  # 观察者相对于X轴的方位角（逆时针方向为正）

    # 创建3D坐标轴
    fig = plt.figure()  # figsize=(8, 6)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_cloud[:, 3], s=3, alpha=0.5)

    # # 设置观察角度
    # ax.view_init(elev=view_elevation, azim=view_azimuth)

    # 添加坐标轴标签及设置其他样式（可选）
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('Point Cloud at Specific Angle')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(3, 4)

    plt.show()


def view_pointcloud(point_cloud,view_elevation,view_azimuth):

    # # 设置显示角度
    # view_elevation = 0  # 观察者相对于XY平面的仰角（向上为正）
    # view_azimuth = 0  # 观察者相对于X轴的方位角（逆时针方向为正）

    # 创建3D坐标轴
    fig = plt.figure(figsize=(8, 8))  #
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    sc=ax.scatter(point_cloud[:, 0], point_cloud[:, 1] , point_cloud[:, 2],c=point_cloud[:, 3], s=3, alpha=0.5)

    # 设置观察角度
    ax.view_init(elev=view_elevation, azim=view_azimuth)

    # 添加坐标轴标签及设置其他样式（可选）
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title='Point Cloud at  '+str(view_elevation)+','+str(view_azimuth)
    ax.set_title(title)
    # 添加颜色条来表示颜色图例  植物
    cbar = fig.colorbar(sc, ax=ax, shrink=0.25, aspect=8)
    cbar.set_label('NDRE')
    # 添加颜色条
    # cbar = fig.colorbar(sc, ax=ax)
    # cbar.ax.tick_params(labelsize=8)  # 调整颜色条标签的字体大小
    # cbar.set_label('NDRE', fontsize=12)  # 设置颜色条标题及字体大小
    # cbar.ax.yaxis.set_ticks_position('left')  # 将颜色条放置在图形左侧
    # cbar.ax.set_aspect(8)# 调整颜色条的大小


    # 显示图例
    ax.legend(*sc.legend_elements(), loc="best")
    ax.add_artist(ax.legend())
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(3, 4)

    plt.show()

    # file=r'F:\数据保存\论文在写\植物\配图\hsv\灰度'
    # name = file+'//' + str(view_elevation) + str(view_azimuth) + '.png'
    # fig.savefig(name, dpi=600)
    # name = file + '//' + str(view_elevation) + str(view_azimuth) + '.tif'
    # fig.savefig(name, dpi=600)

    # 十个波段
    # file=r'F:\数据保存\论文在写\植物\配图\hsv\波段\920'
    # file=r'F:\数据保存\论文在写\植物\配图\3D'
    # name = file +'/'+save_name+ '.png'
    # fig.savefig(name,bbox_inches='tight', pad_inches=0.01,dpi=600)
    # name = file +'/'+save_name+ '.tif'
    # fig.savefig(name,bbox_inches='tight', pad_inches=0.01, dpi=600)

if __name__ == "__main__":
    # kmeans
    # point_path = r'F:\data\2023\1024\left\cube_20231024_151131\pcd-1/CWL_713.pcd'
    # point_path = r'F:\data\2024\stone\stone\20240322\2\2-far\cube_20240322_125922\pcd-1\R.pcd'
    # # img_path = r'F:\Python\Camera-Calibration-Reconstruct\Camera-Calibration-Reconstruct\pre\paper_pic/plant/NDRE-713-851.png'
    # img_path = r'F:\data\2024\stone\stone\20240322\2\2-far\cube_20240322_125922\pos_3\rectfied\CWL_713.png'
    # 分割之后
    point_path = r'F:\data\2023\1024\left\cube_20231024_151131\pcd-1\hsv-Cwl_713.pcd'
    # img_path = r'F:\Python\Camera-Calibration-Reconstruct\Camera-Calibration-Reconstruct\pre\paper_pic/plant/NDRE-713-851.png'
    img_path = r'F:\data\2023\1024\left\cube_20231024_151131\rectfied\show-Cwl_713.png'
    # img_path =r'rect-NDRE-713-851.png'
    # img_path = r'F:\data\2023\1024\left\cube_20231024_151131\rectfied\show-Cwl_713.png'
    rd_points = read_points(point_path, img_path)
    # pppp = select_point_heritage(rd_points)
    pppp = select_point_plant(rd_points)
    view_elevation, view_azimuth =-80,-90   # -90,-9030,-90,,,=80,90,  -90, -90
    view_pointcloud(pppp, view_elevation, view_azimuth)

    # # 不同波段
    # spec_path=r'F:\data\2023\1024\left\cube_20231024_151131\rectfied\no-seg'
    # spec_img=os.listdir(spec_path)
    # for i in spec_img:
    #     img_path=os.path.join(spec_path,i)
    #     save_name, file_extension = os.path.splitext(i)
    #     # img_path = r'F:\data\2023\1024\left\cube_20231024_151131\rectfied\no-seg\Cwl_920.png'
    #
    #
    #     rd_points = read_points(point_path, img_path)
    #     # pppp = select_point_heritage(rd_points)
    #     pppp = select_point_plant(rd_points)
    #     # plot_two(pppp)
    #     # mathlib_show(pppp)
    #     # 可以拖动的显示
    #     # show_pointcloud(pppp)
    #
    #     # view_elevation = int(input("请输入视角1: "))
    #     # view_azimuth =int(input("请输入视角2: "))
    #     # view_elevation = 0  # 观察者相对于XY平面的仰角（向上为正）
    #     # view_azimuth = 0  # 观察者相对于X轴的方位角（逆时针方向为正）
    #     # 佛像的几个角度（）
    #
    #     view_elevation,  view_azimuth=-80,-90#-90,-9030,-90,,,=80,90
    #     view_pointcloud(pppp,view_elevation,view_azimuth)


    # point_path = r'F:\data\2024\stone\stone\20240322\2\2-far\cube_20240322_125922\pcd-1/R.pcd'
    # img_path = r'F:\data\2024\stone\stone\20240322\2\2-far\cube_20240322_125922\pos_3\rectfied/L.png'
    # now=os.getcwd()
    # print(now)
    # print(os.path.abspath(os.path.join(os.getcwd(),'../..')))