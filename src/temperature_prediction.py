#!/bin/env python 3.6
# -*- coding: utf-8 -*-
#==============================================================================
# Author:  胡伟锋
# Created: 2019-05-27
# Version: 1.0.0
# E-mail:  674649741@qq.com
# Purpose: 使用多种算法对锂电池包的温度场进行稀疏重建，电池包由16X8个电池单体组成。
#==============================================================================
import sys 
sys.path.append('./src')
import numpy as np
import pandas as pd
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense,Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge,ElasticNet,LinearRegression,SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score,explained_variance_score
from sklearn.metrics import mean_squared_error,mean_absolute_error


"""1.数据读入、数据XY划分、归一化操作"""
#读入两次实验数据，并合并成单个csv,并做初步处理
def getData():
    print(10*'==','数据相关信息',10*'==')
    #读入csv
    df1='data/AT4532/AUTO_0001_2018-07-09.csv'
    df2='data/AT4532/AUTO_0002_2018-07-13.csv'
    data_1=pd.read_csv(df1,sep=',')
    data_2=pd.read_csv(df2,sep=',')
#    print(data_1.shape,data_2.shape)
    #合并两个csv
    data=pd.concat([data_1,data_2],axis=0,ignore_index=True)
    print('两次实验原始数据总量：',data.shape)
    #丢掉无意义的特征列,仅保留剩余128列，即128个温度点
    useless_cols=['NO.','ELAPSED TIME','DATE TIME']
    data.drop(useless_cols,axis=1,inplace=True)
    print('删除无用特征后的数据总量：',data.shape)
    #返回data
    return data
#    print(d['CH001'])
#    print(d.columns)

#原始数据做初步处理，包括归一化等操作
#此时就进行归一化不合逻辑，弃用该函数
def dataScale(data):
    #使用np进行归一化
#    data_norm=data.apply(lambda x: (x-np.min(x))/(np.max(x)-np.min(x)))
#    print(data_norm)
    #使用pandas进行归一化
#    data_norm=(data-data.min())/(data.max()-data.min())
#    print(data_norm)
    #使用sklearn进行归一化，好处是可以直接进行反归一化
    cols=data.columns
    mms=MinMaxScaler(feature_range=(0,1))
    data_norm=mms.fit_transform(data)
    data_norm=pd.DataFrame(data_norm,columns=cols)#重新赋给columns，便于索引
    #print(cols)
    return data_norm
      
#设置多种压缩率模式，即设置输入输出点数量，及排布方式，据此来分割data
    #采用flag编程模式
input_cols=output_cols=[]#建立全局列名称，便于后续使用
def setDataMode(data,flag='1-127'):
    data_cols=data.columns
#    print(data_cols)
    if flag == '1-127' or flag == '1':#简写为1也可
        input_cols=['CH092',]
    if flag == '2-126' or flag == '2':
        input_cols=['CH060','CH021',]
    if flag == '4-124' or flag == '4':
        input_cols=['CH059','CH062','CH019','CH022',]
    if flag == '8-120' or flag == '8':
        input_cols=['CH051','CH054','CH075','CH078','CH115','CH118','CH011','CH014',]
    if flag == '16-112' or flag == '16':
        input_cols=['CH035','CH046','CH051','CH062','CH067','CH078','CH083','CH094',
                    'CH099','CH110','CH115','CH126','CH019','CH014','CH027','CH006',]
    if flag == '32-96' or flag == '32':
        input_cols=['CH035','CH039','CH042','CH046','CH051','CH055','CH058','CH062',
                    'CH067','CH071','CH074','CH078','CH083','CH087','CH090','CH094',
                    'CH099','CH103','CH106','CH110','CH115','CH119','CH122','CH126',
                    'CH019','CH023','CH010','CH014','CH027','CH031','CH002','CH006',]
    if flag == '64-64' or flag == '64':
        input_cols=['CH033','CH035','CH037','CH039','CH042','CH044','CH046','CH048',
                    'CH049','CH051','CH053','CH055','CH058','CH060','CH062','CH064',
                    'CH065','CH067','CH069','CH071','CH074','CH076','CH078','CH080',
                    'CH081','CH083','CH085','CH087','CH090','CH092','CH094','CH096',
                    'CH097','CH099','CH101','CH103','CH106','CH108','CH110','CH112',
                    'CH113','CH115','CH117','CH119','CH122','CH124','CH126','CH128',
                    'CH017','CH019','CH021','CH023','CH010','CH012','CH014','CH016',
                    'CH025','CH027','CH029','CH031','CH002','CH004','CH006','CH008',]
    #导出输入特征点和输出特征点
    output_cols=[col for col in data_cols if col not in input_cols]
    X=data[input_cols]
    Y=data[output_cols]
    print('输入数据维度:',X.shape,';输出数据维度:',Y.shape)
    return X,Y

mmsX=MinMaxScaler(feature_range=(0,1))#建立全局归一化，便于后续进行逆归一化
mmsY=MinMaxScaler(feature_range=(0,1))
#输入X、Y，划分训练集和测试集，并进行归一化
def data_Split_Scale(X,Y):
    #划分训练集和测试集
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,shuffle=True,
                                                   random_state=2019)
    X_cols,Y_cols=X_train.columns,Y_train.columns
    #分别对训练集的X、Y进行归一化，再transform到测试集
    X_train=mmsX.fit_transform(X_train)
    X_test=mmsX.transform(X_test)
    Y_train=mmsY.fit_transform(Y_train)
    Y_test=mmsY.transform(Y_test)
    #重新赋予列名：
    X_train=pd.DataFrame(X_train,columns=X_cols)
    X_test=pd.DataFrame(X_test,columns=X_cols)
    Y_train=pd.DataFrame(Y_train,columns=Y_cols)
    Y_test=pd.DataFrame(Y_test,columns=Y_cols)
    return X_train,X_test,Y_train,Y_test


"""2.建立各类机器学习、深度学习预测模型"""
#线性回归模型：贝叶斯回归
def BayesianRidgeModel(X_train,Y_train):
    model=BayesianRidge(alpha_1=1e-06,alpha_2=1e-06,
                        compute_score=False,lambda_1=1e-06,lambda_2=1e-06,
                        n_iter=300,normalize=False,tol=0.01,verbose=False)
    model=MultiOutputRegressor(model)
    model.fit(X_train,Y_train)
    return model

#弹性网络回归,是一种使用L1和L2先验作为正则化矩阵的线性回归模型
def ElasticNetModel(X_train,Y_train):
    model=ElasticNet(alpha=0.1, l1_ratio=0.1,max_iter=1000, selection='cyclic', tol=0.0001)
    model.fit(X_train,Y_train)
    return model

#线性回归  
def LinearRegressionModel(X_train,Y_train):
    model=LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    model.fit(X_train,Y_train,)
    return model

#SGD回归
def SGDModel(X_train,Y_train):
    model=SGDRegressor(loss='squared_loss',max_iter=1000,tol=1e-3)
    model=MultiOutputRegressor(model)
    model.fit(X_train,Y_train,)
    return model

#SVM模型，并使用MultiOutputRegressor完成多输出回归
def SVRModel(X_train,Y_train):
    model=SVR(C=1.0,kernel='rbf',tol=1e-3,epsilon=0.1,verbose=False)
    model=MultiOutputRegressor(model)
    model.fit(X_train,Y_train)
    return model

#k近邻模型
def KNNModel(X_train,Y_train):
    model=KNeighborsRegressor(n_neighbors=5,weights='uniform',algorithm='auto',
                              leaf_size=30,p=2,metric='minkowski',metric_params=None)
    model.fit(X_train,Y_train)
    return model

#决策树模型
def DecisionTreeModel(X_train,Y_train):
    model=DecisionTreeRegressor(criterion='mse',max_depth=None,min_samples_split=2,
                                min_samples_leaf=1,min_weight_fraction_leaf=0.0,)
    model.fit(X_train,Y_train)
    return model

#通过Keras搭建深度神经网络模型
def DNNModel(X_train,y_train):
    input_shape=X_train.shape[1]
    output_shape=y_train.shape[1]
    #print('输入X输出：',input_shape,'X',output_shape)
    #建立深度神经网络模型
    input_layer=Input(shape=(input_shape,),)
    y=Dense(64,activation='relu')(input_layer)
    y=Dense(64,activation='relu')(y)
    output_layer=Dense(output_shape,activation='sigmoid')(y)
    model=Model(inputs=input_layer,outputs=output_layer)
    model.compile(loss='mse',optimizer='adam')
    earlystopping=EarlyStopping(monitor='loss',patience=10,verbose=2)
    print('开始训练',6*'.')
    model.summary() 
    model.fit(X_train,y_train,batch_size=32,epochs=200,callbacks=[earlystopping,],verbose=1)
    plot_model(model=model,to_file='output/DNNStructure.png',show_shapes=True)
    return model

#通过sklearn搭建深度神经网络模型
def NNModel(X_train,Y_train):
    input_shape=X_train.shape[1]
    output_shape=Y_train.shape[1]
    #print('输入X输出：',input_shape,'X',output_shape)
    h1,h2=32,64
    print('神经网络-输入层X隐藏层X输出层:'+str(input_shape)+'X'+str(h1)+'X'
          +str(h2)+'X'+str(output_shape)+'.')
    mlp=MLPRegressor(hidden_layer_sizes=(h1,h2),activation='relu',solver='adam',
                     alpha=0.0001,batch_size=32,learning_rate='constant',
                     max_iter=500,shuffle=True,random_state=2019,tol=1e-5,
                     verbose=100,early_stopping=True,beta_1=0.9,beta_2=0.999)
    mlp.fit(X_train,Y_train)
    return mlp

"""3.对多种模型进行交叉验证评估、测试集评估（常用指标）"""
def modelCrossValidation(model,X_train,Y_train,
                         cv=5,scoring=['r2','explained_variance',
                                       'neg_mean_absolute_error',
                                       'neg_mean_squared_error']):
    print(10*'==','交叉验证评估',10*'==')
    #输出交叉验证结果
    def CV(scoring):
        cv_score=cross_val_score(model,X_train,Y_train,cv=cv,scoring=scoring)
        print('交叉验证折数CV='+str(cv),';','评估指标:'+scoring+';',)
        print('各折结果：',cv_score)
        print('平均结果：',cv_score.mean())
    #同时对多个指标进行交叉验证
    if isinstance(scoring,list):
        for s in scoring:
            print(50*'-')
            CV(s)
    else:
        CV(scoring)

    
#模型初步评估
def modelEvaluation(model,X_test,y_test):
    y_pred=model.predict(X_test)#计算结果
    #将结果逆归一化成原始数据
    y_pred_inverse=mmsY.inverse_transform(y_pred)
    y_test_inverse=mmsY.inverse_transform(y_test)
#    print(y_pred)
    #开始评估：
    #针对y_test_inverse、y_pred_inverse
    print(10*'==','模型测试集评估',10*'==')
    #==========================================================================
    print('测试集维度:',y_pred_inverse.shape,)
    '''1.平均mse'''
    mse=mean_squared_error(y_test_inverse,y_pred_inverse)
    print('1.MSE:',mse)
    #-------------------------------------------------------------------------
    '''2.平均绝对误差mae'''
    mae=mean_absolute_error(y_test_inverse,y_pred_inverse)
    print('2.MAE:',mae)
    #-------------------------------------------------------------------------
    '''3.R2决定系数(拟合优度)，值越接近1越好'''
    r2=r2_score(y_test_inverse,y_pred_inverse)
    print('3.R2_score:',r2)
    #-------------------------------------------------------------------------
    '''4.可解释方差'''
    ev=explained_variance_score(y_test_inverse,y_pred_inverse)
    print('4.ev_score:',ev)
    #-------------------------------------------------------------------------
    '''5.全局最大温度误差（所有待预测点的所有数据）'''
    max_error=max((y_test_inverse-y_pred_inverse).reshape(-1))
    print('5.max_error:',max_error)
    #-------------------------------------------------------------------------
    '''6.其他可视化分析方法'''
    print(10*'==','可视化分析(见图)',10*'==')
    #1.将y_test、y_pred保存为csv文件：
    Y_cols=y_test.columns
    yt=pd.DataFrame(y_test_inverse,columns=Y_cols)
    yt.to_csv('output/y_true.csv')
    yp=pd.DataFrame(y_pred_inverse,columns=Y_cols)
    yp.to_csv('output/y_pred.csv')
    #2.将指定通道的真实值和预测值画出：
    from data_visualization import draw_multiData_singlePoint
    draw_multiData_singlePoint(yt,yp,point='CH001')


if __name__=='__main__':
    '''1.读入数据，进行基本处理（设定X和Y、划分训练集和测试集、归一化）'''
    data=getData()
    X,Y=setDataMode(data=data,flag='8')
    X_train,X_test,Y_train,Y_test=data_Split_Scale(X,Y)
    '''2.使用模型进行训练'''
    #1.线性回归模型：
#    model=BayesianRidgeModel(X_train,Y_train)
#    model=ElasticNetModel(X_train,Y_train)
#    model=LinearRegressionModel(X_train,Y_train)
#    model=SGDModel(X_train,Y_train)
    #2.机器学习模型：
#    model=KNNModel(X_train,Y_train)
#    model=SVRModel(X_train,Y_train)
#    model=DecisionTreeModel(X_train,Y_train)
    #3.神经网络、深度学习模型：
#    model=DNNModel(X_train,Y_train)
    model=NNModel(X_train,Y_train)
    '''3.使用交叉验证和基本指标评估模型性能'''
#    modelCrossValidation(model,X_train,Y_train)
    modelEvaluation(model,X_test,Y_test)
    '''4.将模型评估结果进行可视化'''