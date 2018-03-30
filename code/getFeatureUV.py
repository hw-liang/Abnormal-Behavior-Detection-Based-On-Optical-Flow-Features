import cv2
import scipy.io
import numpy as np
from weight_matrix import *
from split import *
font=cv2.FONT_HERSHEY_COMPLEX
data = scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')
u_seq_abnormal = data['u_seq_abnormal']
data = scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')
v_seq_abnormal = data['v_seq_abnormal']
weight = Weight_matrix().get_weight_matrix()

def getFeaturesUV(realPos,u,v):
    n = realPos.shape[0]
    if realPos.max()==0:
        data = np.zeros((1,2))
    else:
        data = np.zeros((n,2))
        for i in range(n):
            if realPos[i][0] == realPos[i][1]:
                realPos[i][0]+=1;
            if realPos[i][2] == realPos[i][3]:
                realPos[i][2]+=1;
            data[i][0]=u[int(realPos[i][1]):int(realPos[i][0]),int(realPos[i][3]):int(realPos[i][2])].mean()
            data[i][1]=v[int(realPos[i][1]):int(realPos[i][0]),int(realPos[i][3]):int(realPos[i][2])].mean()
    return data

def main_test():
    weight = Weight_matrix().get_weight_matrix()
    ab_img=cv2.imread('../ref_data/ab_fg_pics/105.bmp')
    ab_img = cv2.cvtColor(ab_img, cv2.COLOR_BGR2GRAY)
    origin=cv2.imread('../ref_data/original_pics/105.tif')
    origin= cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    #print(((origin)==origin).all())
    mask=cv2.bitwise_and(origin,origin,mask=ab_img)
    cv2.imshow('masked',mask)
    cv2.imshow('ab_img',ab_img)
    cv2.imshow('u_img105_original', u_seq_abnormal[:, :, 105])
    cv2.imshow('v_img105_original', v_seq_abnormal[:, :, 105])
    cv2.imshow('u_img105_after_weightMat',u_seq_abnormal[:,:,105]*weight.reshape(-1,1))
    cv2.imshow('v_img105_after_weightMat',v_seq_abnormal[:,:,105]*weight.reshape(-1,1))
    cv2.imshow('original',origin)
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_test()
