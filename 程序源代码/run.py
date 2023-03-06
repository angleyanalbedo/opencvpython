# pyuic5 -o system.py system.ui

import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from system import Ui_MainWindow

import numpy as np
import cv2 # 调用该库中的数字图像处理方法，与自己手写的方法处理同一张图像，并比比对处理效果
import matplotlib.pyplot as plt # 调用该库用来绘制所导入的图像和经过处理后的图像
import time # 调用该库用来计算一次图像处理操作所用的时间

tmp_path="D:\\result.png"
demo_path="D:\\demo.png"
save_path="D:\\save.png"

class Demo(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Demo, self).__init__()
        self.setupUi(self)    

        # 采用示例按钮
        self.demo_button.clicked.connect(self.use_demo_file)

        # 文件菜单
        self.open.triggered.connect(self.read_file)
        self.save.triggered.connect(self.save_file)    

        # 开始菜单
        self.grey.triggered.connect(self.grey_img)
        self.resiz.triggered.connect(self.resize_img)

        # 灰度变换
        self.thres.triggered.connect(self.thres_func)
        self.nega.triggered.connect(self.nega_func)
        self.logtrans.triggered.connect(self.logtrans_func)
        self.gammatrans.triggered.connect(self.gammatrans_func)
        self.bitwise.triggered.connect(self.bitwise_func)

        # 直方图处理
        self.equal.triggered.connect(self.equal_func)

        # 空间滤波
        self.mean.triggered.connect(self.mean_func)
        self.median.triggered.connect(self.median_func)
        self.gaussfilt.triggered.connect(self.gaussfilt_func)
        self.bilateral.triggered.connect(self.bilateral_func)
        self.sharpen.triggered.connect(self.sharpen_func)

        # 频域滤波
        self.lowpass.triggered.connect(self.lowpass_func)
        self.highpass.triggered.connect(self.highpass_func)

        # 噪声添加
        self.gaussnoi.triggered.connect(self.gaussnoi_func)

# #################################  文件菜单  ################################# 

    def use_demo_file(self):
        # 打开示例图像

        self.path.setText(demo_path) # 显示示例图像的路径        

        img=cv2.imread(demo_path) # 用cv库打开文件
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
            # 这里虽然只是打开图像而没有进行处理，但是也会拷贝一份作为缓存，供后续的图像处理函数读取
        self.img_src.setPixmap(QPixmap(tmp_path)) # 读取原始图像并显示

        self.description.setText("示例图像导入成功！下面开始你的数字图像处理之旅吧") # 功能描述
    
    def read_file(self):
        # 打开原始图像

        filename, filetype =QFileDialog.getOpenFileName(self, 
            "打开文件", "D:/", "All Files(*);;Text Files(*.png)")
        if filetype=='': # 若路径无效或打开文件失败，则什么也不做，让用户回到主界面
            return

        self.path.setText(filename) # 显示原始图像的路径
        
        img=cv2.imread(filename) # 用cv库打开文件
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
            # 这里虽然只是打开图像而没有进行处理，但是也会拷贝一份作为缓存，供后续的图像处理函数读取
        self.img_src.setPixmap(QPixmap(tmp_path)) # 读取原始图像并显示

        self.description.setText("图像打开成功！下面让我们开始数字图像处理之旅吧") # 功能描述
        

    def save_file(self):
        #保存当前图像
        
        img = cv2.imread(tmp_path)
        cv2.imwrite(save_path,img) # 将处理后的图像保存在临时路径中

        self.description.setText("图像保存成功！请在默认路径中查看刚才保存的图像") # 功能描述

# #################################  开始菜单  #################################

    def grey_img(self):
        # 灰度化图像

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像，用cv库进行灰度化处理
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("图像灰度化处理成功！") # 功能描述

    def resize_img(self):
        # resize图像

        img=cv2.imread(tmp_path) # 读取临时路径中的图像
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            p2=int(self.para2.toPlainText())
            img = cv2.resize(img, (p1,p2))
        except Exception: # 否则，以默认的方式处理图像
            img = cv2.resize(img, (256,256))
        
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("图像缩放处理成功！"+"\n"+
            "你还可以选择在参数1和参数2里输入需要缩放的尺寸，默认缩放尺寸为128*128。快去试试吧") # 功能描述

# #################################  灰度变换  #################################
    def thres_func(self):
        # 二极阈值化图像

        img=cv2.imread(tmp_path) # 读取临时路径中的图像
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            if p1<0 or p1>255:
                raise ValueError
            img[img > p1]=255
            img[img <= p1]=0
        except Exception: # 否则，以默认的方式处理图像
            img[img > 128]=255
            img[img <= 128]=0
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("图像阈值化处理成功！"+"\n"+
            "你还可以选择在参数1里输入所需要的阈值（0到255之间），默认阈值为128。快去试试吧") # 功能描述


    def nega_func(self):
        # 反转图像

        img=cv2.imread(tmp_path) # 读取临时路径中的图像
        img=255-img
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("图像反转处理成功！") # 功能描述
    
    def logtrans_func(self):
        # 对数变换

        img=cv2.imread(tmp_path) # 读取临时路径中的图像

        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            c=p1
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            c=42 # 如果不能成功识别到参数，那么对数变换的参数c默认为1

        des = c * np.log(1.0 + img)
        des = np.uint8(des + 0.5) # 四舍五入转成uint8的像素数据格式类型
        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("对数变换处理成功！"+"\n"+
            "你还可以选择在参数1里输入所需要的对数变换参数c，默认的c值为42。快去试试吧") # 功能描述

    def gammatrans_func(self):
        # 幂律变换

        img=cv2.imread(tmp_path) # 读取临时路径中的图像
        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=float(self.para1.toPlainText())
            c=p1
            p2=float(self.para2.toPlainText())
            lamda=p2
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            c=1 # 如果不能成功识别到参数，那么幂律变换的参数c默认为1
            lamda=0.9 # 如果不能成功识别到参数，那么幂律变换的参数lamda默认为0.9，以暴露更多暗处细节

        img = img/255**lamda*c*255
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像
        self.description.setText("幂律变换处理成功！"+"\n"+
            "你还可以选择在参数1和2里输入所需要的幂律变换参数c和lamda，默认的c值为1，lamda为0.9。快去试试吧") # 功能描述

    def bitwise_func(self):
        # 位图切割

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            layer=p1
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            layer=7 # 如果不能成功识别到参数，那么默认切第7个比特位
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                tmpvalue=img[i][j]
                bits = bin(tmpvalue)[2:].rjust(8, '0')
                fill = int(bits[-layer - 1])
                img[i][j] = 255 if fill else 0
        cv2.imwrite(tmp_path,img) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像
        self.description.setText("位图切割成功！"+"\n"+
            "你还可以选择在参数1里输入所需要切割的比特位layer（0到7之间），默认layer为7。快去试试吧") # 功能描述




# #################################  直方图处理  #################################
    def equal_func(self):
        # 直方图均衡

        img=cv2.imread(tmp_path) # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)
        x = np.zeros([256]) # 8位数字图像一共有256个像素
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                x[src[i][j]] += 1
        x = x / src.size

        sum_x = np.zeros([256])
        for i, _ in enumerate(x):
            sum_x[i] = sum(x[:i])

        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                des[i][j] = 255 * sum_x[src[i][j]]

        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("直方图均衡处理成功！") # 功能描述

# #################################  空间滤波  #################################
    def mean_func(self):
        # 均值滤波

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)

        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            if p1<1 or p1>5:
                raise ValueError
            r=p1
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            r=1 # 如果不能成功识别到参数，那么默认模板的r为1，即核的边长为3

        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                count=0
                local_sum=0
                for ii in range(-r,r+1):
                    for jj in range(-r,r+1):
                        if (i+ii>0 and j+jj>0 and i+ii<src.shape[0] and j+jj<src.shape[1]):
                            local_sum+=src[i+ii][j+jj]
                            count+=1              
                des[i][j]=local_sum//count

        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("均值滤波处理成功！"+"\n"+
            "你还可以选择在参数1里输入x（x在1到5之间），以指定滤波时模板长度为2*x-1，默认x为1。快去试试吧") # 功能描述

    def median_func(self):
        # 中值滤波

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)

        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            if p1<1 or p1>5:
                raise ValueError
            r=p1
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            r=1 # 如果不能成功识别到参数，那么默认模板的r为1，即核的边长为3

        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                tmparray=[]
                for ii in range(-r,r+1):
                    for jj in range(-r,r+1):
                        if (i+ii>0 and j+jj>0 and i+ii<src.shape[0] and j+jj<src.shape[1]):
                            tmparray.append(src[i+ii][j+jj])           
                des[i][j]=tmparray[(len(tmparray)+1)//2-1]

        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("中值滤波处理成功！"+"\n"+
            "你还可以选择在参数1里输入x（x在1到5之间），以指定滤波时模板长度为2*x-1，默认x为1。快去试试吧") # 功能描述

    def gaussfilt_func(self):
        # 高斯滤波

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)

        weights=np.array([[0.09474,0.11832,0.09474],[0.11832,0.14776,0.11832],[0.09474,0.11832,0.09474]])
        r=1
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                count=0
                local_sum=0
                for ii in range(-r,r+1):
                    for jj in range(-r,r+1):
                        if (i+ii>0 and j+jj>0 and i+ii<src.shape[0] and j+jj<src.shape[1]):
                            local_sum+=weights[1+ii][1+jj]*src[i+ii][j+jj]
                            count+=1              
                des[i][j]=local_sum

        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("高斯滤波处理成功！") # 功能描述

    def bilateral_func(self):
        # 双边滤波

        self.description.setText("双边滤波的执行时间较长（128*128规格图片在r=1时需要执行一分钟），请耐心等待")

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        src=img.copy()

        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            p2=int(self.para2.toPlainText())
            p3=int(self.para3.toPlainText())
            r=p1
            sigmaColor=p2
            sigmaSpace=p3
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False
        if not isParaReady:
            r=2 # 如果不能成功识别到参数，那么默认模板的r为2，即核的边长为5
            sigmaColor=20
            sigmaSpace=20

    # def my_bilateralFilter(src, d, sigmaColor, sigmaSpace):
        dst= src.copy()
        H,W = src.shape
        for i in range(r, H - r): # 不考虑padding，遍历原图像的非边界部分的每一个像素
            for j in range(r, W - r):
                weight_sum = 0.0 # 权重和，用于归一化
                pixel_sum = 0.0
                for x in range(-r, r + 1):
                    for y in range(-r, r + 1):                        
                        spatial_weight = -(x ** 2 + y ** 2) / (2 * (sigmaSpace ** 2)) 
                        # 空间域权重。该计算可以也提出到x和y循环的外面做预处理以提高效率，但是不改变时间复杂度
                        color_weight = -(int(src[i][j]) - int(src[i + x][j + y])) ** 2 / (2 * (sigmaColor ** 2)) 
                        # 颜色域权重。这里通过将灰度像素变成整数保证权值为正
                        weight = np.exp(spatial_weight + color_weight) # 通过指数乘积得到对于一个像素的总权重                        
                        weight_sum += weight
                        pixel_sum += (weight * src[i + x][j + y])
                        dst[i][j] = pixel_sum / weight_sum # 将像素值归一化后放到结果里面
        dst=dst.astype(np.uint8) # 将输出图像转化成灰度值在0-255之间的标准灰度图像

        cv2.imwrite(tmp_path,dst) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("双边滤波处理成功！"+"\n"+
            "你还可以选择在参数1、参数2、参数3里分别制定参数r、sigmaColor和sigmaSpace，默认r为2，sigmaColor和sigmaSpace都为20。快去试试吧") # 功能描述

    def sharpen_func(self):
        # 锐化滤波

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)

        weights=np.array([[0,1,0],[1,-4,1],[0,1,0]])
        r=1
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                count=0
                local_sum=0
                for ii in range(-r,r+1):
                    for jj in range(-r,r+1):
                        if (i+ii>0 and j+jj>0 and i+ii<src.shape[0] and j+jj<src.shape[1]):
                            local_sum+=weights[1+ii][1+jj]*src[i+ii][j+jj]
                            count+=1              
                des[i][j]=local_sum
        des=des.astype(np.uint8)
        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("锐化滤波处理成功！") # 功能描述

# #################################  频域滤波  #################################
    def lowpass_func(self):
        # 低通滤波器

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)
        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            d0=p1
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False

        if not isParaReady:
            d0=100 # 如果不能成功识别到参数，那么截止频率默认为100

        r_ext = np.zeros((src.shape[0] * 2, src.shape[1] * 2))
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                r_ext[i][j] = src[i][j]

        # 频域变换相关操作
        r_ext_fu = np.fft.fft2(r_ext)
        r_ext_fu = np.fft.fftshift(r_ext_fu)
        # 频率域中心坐标
        center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
        h = np.empty(r_ext_fu.shape)
        # 绘制滤波器 H(u, v)
        for u in range(h.shape[0]):
            for v in range(h.shape[1]):
                duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
                h[u][v] = duv < d0

        s_ext_fu = r_ext_fu * h
        s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
        s_ext = np.abs(s_ext)
        des = s_ext[0:src.shape[0], 0:src.shape[1]]

        for i in range(des.shape[0]):
            for j in range(des.shape[1]):
                des[i][j] = min(max(des[i][j], 0), 255)
        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("低通滤波处理成功！"+"\n"+
            "你还可以选择在参数1里输入要求的截止频率，默认为100。快去试试吧") # 功能描述


    def highpass_func(self): 
        # 高通滤波器

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)
        isParaReady=True
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=int(self.para1.toPlainText())
            d0=p1
        except Exception: # 否则，以默认的方式处理图像
            isParaReady=False

        if not isParaReady:
            d0=100 # 如果不能成功识别到参数，那么截止频率默认为100

        r_ext = np.zeros((src.shape[0] * 2, src.shape[1] * 2))
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                r_ext[i][j] = src[i][j]

        # 频域变换相关操作
        r_ext_fu = np.fft.fft2(r_ext)
        r_ext_fu = np.fft.fftshift(r_ext_fu)
        # 频率域中心坐标
        center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
        h = np.empty(r_ext_fu.shape)
        # 绘制滤波器 H(u, v)
        for u in range(h.shape[0]):
            for v in range(h.shape[1]):
                duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
                h[u][v] = np.e ** (-duv**2 / d0 ** 2)

        s_ext_fu = r_ext_fu * h
        s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
        s_ext = np.abs(s_ext)
        des = s_ext[0:src.shape[0], 0:src.shape[1]]

        for i in range(des.shape[0]):
            for j in range(des.shape[1]):
                des[i][j] = min(max(des[i][j], 0), 255)
        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("高通滤波处理成功！"+"\n"+
            "你还可以选择在参数1里输入要求的截止频率，默认为100。快去试试吧") # 功能描述
# #################################  噪声添加  #################################
    def gaussnoi_func(self):

        img=cv2.imread(tmp_path,0) # 读取临时路径中的图像
        src=img.copy()
        des = np.empty(src.shape, dtype=np.uint8)
        try: # 如果输入的参数有效，则以该参数处理图像
            p1=float(self.para1.toPlainText())
            p2=float(self.para2.toPlainText())
            src = img.copy()
            src = np.array(src/255, dtype=float)
            noise = np.random.normal(p1, p2 ** 0.5, src.shape)
            des = src + noise
            if des.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            des = np.clip(des, low_clip, 1.0)
            des = np.uint8(des*255)
        except Exception: # 否则，以默认的方式处理图像
            src = img.copy()
            src = np.array(src/255, dtype=float)
            noise = np.random.normal(0, 1 ** 0.5, src.shape)
            des = src + noise
            if des.min() < 0:
                low_clip = -1.
            else:
                low_clip = 0.
            des = np.clip(des, low_clip, 1.0)
            des = np.uint8(des*255)
        cv2.imwrite(tmp_path,des) # 将处理后的图像保存在临时路径中
        self.img_des.setPixmap(QPixmap(tmp_path)) # 显示处理后的图像

        self.description.setText("高斯噪声添加成功！"+"\n"+
            "你还可以选择在参数1和参数2里指定均值和方差。快去试试吧") # 功能描述




if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec_())