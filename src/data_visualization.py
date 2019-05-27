#!/bin/env python 3.6
# -*- coding: utf-8 -*-
#==============================================================================
# Author:  胡伟锋
# Created: 2019-05-27
# Version: 1.0.0
# E-mail:  674649741@qq.com
# Purpose: 对温度场数据进行可视化分析。
#==============================================================================
import os
import sys
sys.path.append('src\\')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#自己写的模块,用于原始数据读取
from temperature_prediction import getData

#重构原始df中指定行数据的空间位置：即根据真实空间位置将指定行转换为一个16X8的电池包温度矩阵
def generatingTMatrix(data,row):#指定原始采集的数据，指定的行
    data_row=data.iloc[row,:]
    #print(data_row['CH001'])
    #根据真实空间位置生成列索引顺序
    index1=['CH0'+str(i) for i in range(33,100)]
    index2=['CH'+str(i) for i in range(100,129)]
    index3=['CH0'+str(i) for i in range(17,25)]
    index4=['CH00'+str(i) for i in range(9,10)]
    index5=['CH0'+str(i) for i in range(10,17)]
    index6=['CH0'+str(i) for i in range(25,33)]
    index7=['CH00'+str(i) for i in range(1,9)]
    indexs=index1+index2+index3+index4+index5+index6+index7
    #print(indexs)
    #根据列索引顺序重新排列指定行数据，成为16X8的矩阵（电池包形状）
    indexs=np.array(indexs).reshape(16,8)
    matrix=np.empty([16,8])#用于存储生成的矩阵
    for i,row in enumerate(indexs):#i行数
        for j,ele in enumerate(row):#j列数
            matrix[i,j]=data_row[ele]
    return matrix

#接收温度矩阵，将其绘制成2D热力图
def draw2DHeatmap(matrix,name='ThermodynamicChart'):
    plt.figure(name)
    #设置标注前后左右距离
#    plt.subplots_adjust(left=0.2,right=0.95,bottom=0.15,top=0.95)
    plt.imshow(matrix,interpolation='nearest',cmap=plt.cm.hot,)
#    plt.xlabel('横坐标标签')
#    plt.ylabel('纵坐标标签')
    plt.colorbar()
#    plt.xticks()
    if name !='ThermodynamicChart':#有给name重新赋值时(即调用saveImgs)，才保存图片，否则默认不保存
        plt.savefig('output/picture/'+str(name)+'.jpg')
    plt.show()

#绘制成3D热力图
def draw3DHeatmap(matrix):
    Z=matrix
    Y=range(0,Z.shape[0],1)
    X=range(0,Z.shape[1],1)
    print(X,Y)
    X,Y=np.meshgrid(X,Y)
    fig=plt.figure('3D热力图')
    ax=Axes3D(fig,)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature(°)')
    surf=ax.plot_surface(X,Y,Z,cmap=plt.cm.hot,)
    fig.colorbar(surf,shrink=0.5,aspect=20)
    #通过设置轴限制来抵消自动缩放
    ax.set_xlim(0,16)
    ax.set_ylim(0,16)
    plt.show()

##绘制3D插值热力图
#def draw3DHeatmapInterpolate(matrix):
#    from PIL import Image
#    img=Image.fromarray(matrix)
#    img=img.resize((80,160))
#    matrix=np.asarray(img)
#    draw3DHeatmap(matrix)

#将温度场图片批量存入文件夹
def saveImgs():
    data=getData()
    for i in range(2000):
        matrix=generatingTMatrix(data,row=i)
        draw2DHeatmap(matrix,name=i)

#将文件夹内的图片合成视频
def writeVideo(path='output/picture/'):
    filelist=os.listdir(path)
    filelist.sort(key=lambda x : int(x[:-4]))#将图片重新正确排序
    fps=60
    size=(432,288)#图片实际大小，若不符则视频无法播放
#    import cv2.cv2 as cv2
    import cv2
    video=cv2.VideoWriter('output/video/test.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),fps,size)
    for item in filelist:
        img=cv2.imread(path+item)
#        print(path+item)
        video.write(img)
    video.release()
    cv2.destroyAllWindows()

#做单点温度分析,指定单点或多点名称，画出其所有时间内的温度
def draw_singleData_multiPoint(data,points='CH001'):
    x=range(0,data.shape[0],)
    if isinstance(points,list):#判断是否为list，若是，则说明有多个输入  
        plt.figure('PointAnalysis')
        for point in points:
            y=data[point]
            plt.plot(x,y,linewidth=1,label=point)
    else:#否则只画出一条曲线
        y=data[points]
        plt.plot(x,y,linewidth=1,label=points) 
    plt.legend()
    plt.show()
    
#将指定点的真实值和预测值画在同一张图上
def draw_multiData_singlePoint(dataTrue,dataPred,point='CH001'):
    print('当前评估点为：'+point+'.')
    #直接画出
    plt.figure('truePred_'+point)
    x=range(dataTrue.shape[0],)
    y1,y2=dataTrue[point],dataPred[point]
    plt.plot(x,y1,linewidth=1,label='y_true')
    plt.plot(x,y2,linewidth=1,label='y_pred')
    #绘制两者的差值：
#    d=dataTrue[point]-dataPred[point]
#    plt.figure('真实值与预测值之差')
#    plt.plot(x,d)
    #将其排序后再画出来
    plt.figure('truePred_sorted_'+point)
    y1_,y2_=sorted(y1),sorted(y2)
    plt.plot(x,y1_,linewidth=1,label='y_true_sorted')
    plt.plot(x,y2_,linewidth=1,label='y_pred_sorted')
    #两者差值
#    y3=list(map(lambda x:x[0]-x[1],zip(y1_,y2_)))
#    plt.plot(x,y3,linewidth=1,label='(y_true-y_pred)_sorted')
    plt.legend()
    plt.show()
    
    
if __name__=='__main__':
    data=getData()

    matrix=generatingTMatrix(data,row=7000)
    draw2DHeatmap(matrix,)
#    draw3DHeatmap(matrix)
#    draw3DHeatmapInterpolate(matrix)
    
#    saveImgs()
#    writeVideo()
    
    draw_singleData_multiPoint(data,points=['CH092','CH093','CH033'])

