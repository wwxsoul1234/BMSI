
# Project Introduction 
The BMSI project is a python language based project developed to implement binocular multispectral 3D imaging.  
This paper consists of three main parts: 
 - [x] Optimum Band Selection  
 - [x] Image segmentation   
 - [x] 4D data fusion 
  
 # Installation  
environments：  
```
python==3.6  
```  
Install dependency packages:
- config==0.5.1  
- h5py==3.14.0  
- imageio==2.16.0  
- matplotlib==3.3.4  
- modules==1.0.0  
- numpy==1.19.5  
- open3d==0.13.0  
- opencv_python==4.5.1.48  
- pandas==1.1.5  
- pcl==0.0.0.post1  
- pclpy==0.11.0  
- Pillow==11.3.0  
- psutil==5.9.5  
- python_pcl==0.3  
- requests==2.27.1  
- scikit_learn==1.7.1  
- scipy==1.5.4  
- spacepy==0.7.0  
- tools==1.0.5  
  
  
  
# File structure  
  
```  
·
├── configs          # Documentation of calibration parameters for binocular multispectral cameras  
├── core             # Core Algorithm Package  
├── data             # Image data from binocular multispectral camera 
│   ├── left         # Documentation for the left camera  
│      ├── png       # multispectral image  
│      ├── 3_to_1    # Spectral images with optimum band fusion  
│      └── seg       # Segmented spectral image    
│   ├── right        # Documentation for camera right (same as camera left)  
│      ├── ……  
│   └── rect         # Rectified binocular camera spectral image
├── pictures         # Illustrations for the description document  
├── BMSI.py          # Main program  
├── OIF.py           # Optimum band selection program 
├── seg-hsv.py       # Image Segmentation Program  
├── requirements.txt        # dependency package  
└── README.md          
  ```  
  
# usage instructions  
  
### -OIF 
Run OIF.py   
 1.Optimum band selection for multispectral images in ten bands to print band combinations. 
 2.Fusion of spectral images from the three best bands to form a pseudo-coloured spectral image.  

![输入图片说明](pictures\1.png)
  
  
### -Image segmentation  
 Run seg-hsv.py, Image segmentation based on fused images and HSV colour space.

 ![输入图片说明](pictures\2.png)
### -4D data 
Run BMSI.py   
1.3D reconstruction based on binocular disparity and calibration parameters of the camera.  
2.The 3D data and spectral data were fused to obtain 4D data. 

![输入图片说明](pictures\3.png)

3.after closing “figure 1", Clicking on the leaf in the "left" window displays the 3D coordinates and spectral distribution of the point on the leaf. 

![输入图片说明](pictures\4.png)
  









# 项目介绍  
BMSI项目是一个基于python语言开发的项目，用于实现双目多光谱三维成像。  
本文主要由三个部分组成：  
 - [x] 最佳波段选择  
 - [x] 图像分割  
 - [x] 4D数据融合  
  
 # 安装  
编译环境：  
```
python==3.6  
```  
安装依赖项  
- config==0.5.1  
- h5py==3.14.0  
- imageio==2.16.0  
- matplotlib==3.3.4  
- modules==1.0.0  
- numpy==1.19.5  
- open3d==0.13.0  
- opencv_python==4.5.1.48  
- pandas==1.1.5  
- pcl==0.0.0.post1  
- pclpy==0.11.0  
- Pillow==11.3.0  
- psutil==5.9.5  
- python_pcl==0.3  
- requests==2.27.1  
- scikit_learn==1.7.1  
- scipy==1.5.4  
- spacepy==0.7.0  
- tools==1.0.5  
  
  
  
# 项目结构  
  
```  
·
├── configs          # 双目多光谱相机标定参数文件  
├── core             # 核心算法包  
├── data             # 双目多光谱相机的图像数据 
│   ├── left         # 左相机的文件  
│      ├── png       # 多光谱图像  
│      ├── 3_to_1    # 最佳波段融合的光谱图像  
│      └── seg       # 分割后的光谱图像  
│   ├── right        # 右相机的文件（和左相机相同）  
│      ├── ……  
│   └── rect         # 立体矫正的双目相机光谱图像
├── pictures         # 说明文档的配图  
├── BMSI.py          # 主程序  
├── OIF.py           # 最佳波段选择程序  
├── seg-hsv.py       # 图像分割程序  
├── requirements.txt        # 依赖包  
└── README.md        #说明文件  
  ```  
  
# 使用说明  
  
### -最佳波段选择  
运行 OIF.py 程序  
 1.对十个波段的多光谱图像进行最佳波段选择，得到波段组合。  
 2.将三个最佳波段的光谱图像融合形成一个伪彩的光谱图像。  
![输入图片说明](pictures\1.png)
  
  
### -图像分割  
 运行 seg-hsv.py  程序，基于融合图像和HSV 颜色空间进行图像分割。  
 ![输入图片说明](pictures\2.png)
### -4D数据  
运行BMSI.py   
1.基于双目视差和相机的标定参数进行三维重建。  
2.将三维数据和光谱数据进行融合得到4D数据。
![输入图片说明](pictures\3.png)
3.关闭”figure 1“窗口,鼠标点击"left"图窗内叶片的某个位置，显示该点的三维坐标和光谱分布。  
![输入图片说明](pictures\4.png)
  

