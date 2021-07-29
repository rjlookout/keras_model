# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:51:30 2021

@author: RJ
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import models,layers

'''
读取数据
'''
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

'''
绘图展示
'''
def plot1():#展示死亡幸存的柱状图
    ax = df_train['Survived'].value_counts().plot(kind = 'bar',
                    figsize = (12,8),fontsize=15,rot = 0)
    ax.set_ylabel('Counts',fontsize = 15)
    ax.set_xlabel('Survived',fontsize = 15)
    plt.show()

def plot2():#展示年龄分布图
    ax = df_train['Age'].plot(kind = 'hist',bins = 20,color= 'purple',
                        figsize = (12,8),fontsize=15)

    ax.set_ylabel('Frequency',fontsize = 15)
    ax.set_xlabel('Age',fontsize = 15)
    plt.show()
    
def plot3():#年龄跟存活情况的密度分布
    ax = df_train.query('Survived == 0')['Age'].plot(kind = 'density',
                          figsize = (12,8),fontsize=15)
    df_train.query('Survived == 1')['Age'].plot(kind = 'density',
                          figsize = (12,8),fontsize=15)
    ax.legend(['Survived==0','Survived==1'],fontsize = 12)
    ax.set_ylabel('Density',fontsize = 15)
    ax.set_xlabel('Age',fontsize = 15)
    plt.show()
    
'''
数据预处理
'''
def preprocessing(dfdata):

    df_result= pd.DataFrame()

    #Pclass
    df_Pclass = pd.get_dummies(dfdata['Pclass'])
    df_Pclass.columns = ['Pclass_' +str(x) for x in df_Pclass.columns ]
    df_result = pd.concat([df_result,df_Pclass],axis = 1)

    #Sex
    df_Sex = pd.get_dummies(dfdata['Sex'])
    df_result = pd.concat([df_result,df_Sex],axis = 1)

    #Age
    df_result['Age'] = dfdata['Age'].fillna(0)
    df_result['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    df_result['SibSp'] = dfdata['SibSp']
    df_result['Parch'] = dfdata['Parch']
    df_result['Fare'] = dfdata['Fare']

    #Carbin
    df_result['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    df_Embarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    df_Embarked.columns = ['Embarked_' + str(x) for x in df_Embarked.columns]
    df_result = pd.concat([df_result,df_Embarked],axis = 1)

    return(df_result)
    
x_train = preprocessing(df_train)
y_train = df_train['Survived'].values

x_test = preprocessing(df_test)
y_test = df_test['Survived'].values

#x_train=pd.concat([x_train,x_test])
#y_train=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_test)])

'''
定义模型
'''
tf.keras.backend.clear_session()#清空原有的计算图
mymodel=tf.keras.Sequential()
mymodel.add(layers.Dense(24,activation='relu',input_shape=(15,)))
mymodel.add(layers.Dense(6,activation='relu'))
mymodel.add(layers.Dense(1,activation='sigmoid'))

mymodel.summary()

'''
训练
'''
mymodel.compile(optimizer='adam',#优化器
            loss='binary_crossentropy',#损失函数
            metrics=['accuracy'])#评价指标/AUC/Accuracy

history = mymodel.fit(x_train,y_train,
                    batch_size= 64,
                    epochs= 1000,
                    validation_data=(x_test,y_test) #分割一部分训练数据用于验证
                   )

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(history,"loss")
plot_metric(history,"accuracy")

#mymodel.evaluate(x = x_test,y = y_test)


