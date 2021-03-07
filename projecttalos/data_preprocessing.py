import numpy as np
import os
import cv2
import random 

def ImagePreprocess(path2img,train_size=0.7,size=(80,80)):
    x_test=[]
    y_test=[[]]
    x_train=[]
    y_train=[[]]
    for path in path2img:
        numberoffiles=len(os.listdir(path))
        counter=0
        for img in os.listdir(path):
            if counter<(numberoffiles*train_size):
                pic = cv2.cvtColor(cv2.imread(os.path.join(path,img)),cv2.COLOR_BGR2RGB)
                pic = cv2.imread(os.path.join(path,img))
                pic = cv2.resize(pic,size)
                x_train.append([pic])
                y_train[0].append(path2img.index(path))
                counter+=1
            else:
                pic = cv2.cvtColor(cv2.imread(os.path.join(path,img)),cv2.COLOR_BGR2RGB)
                pic = cv2.imread(os.path.join(path,img))
                pic = cv2.resize(pic,size)
                x_test.append([pic])
                y_test[0].append(path2img.index(path))
                counter+=1

    temp0=list(zip(x_train,y_train[0]))
    random.shuffle(temp0)
    x_train,y_train[0]=zip(*temp0)
    temp1=list(zip(x_test,y_test[0]))
    random.shuffle(temp1)
    x_test,y_test[0]=zip(*temp1)

    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    y_test=np.array(y_test)


    x_train=x_train.reshape(x_train.shape[0],-1).T
    x_test=x_test.reshape(x_test.shape[0],-1).T
    x_train=x_train/255.
    x_test=x_test/255.

    return x_train,x_test,y_train,y_test
x_train,x_test,y_train,y_test=ImagePreprocess(["Q:\Projects\ProjectTalos\projecttalos\\test0","Q:\Projects\ProjectTalos\projecttalos\\test1"])
print(x_test.shape)
print("---------")
print(y_test)