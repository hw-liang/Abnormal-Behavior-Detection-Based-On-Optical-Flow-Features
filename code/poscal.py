# coding:utf8
import cv2
import numpy as np
from skimage import measure
from weight_matrix import *
from split import *
font=cv2.FONT_HERSHEY_COMPLEX

def poscal(img):
    img = img[:,:,0]
    kernel = np.ones((6,1),np.uint8)
    im = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)    # open operation
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)    # close operation
    im_labels = measure.label(im,connectivity=1,neighbors=8)       #connected components

    num = im_labels.max()     # num of connected components

    if num==0:
        im_s = np.zeros((1,5))   # all black img
    else:
        im_s = np.zeros((num,5))
        for i in range(num):
            temp = np.copy(im_labels)
            temp[temp != (i+1)]=0
            index = np.where(temp ==(i+1))
            im_s[i,0]= max(index[0]) #person's foot y_val
            im_s[i,1]= min(index[0]) #person's head y_val
            im_s[i,2]= max(index[1]) #person's right side x_val
            im_s[i,3]= min(index[1]) #person's left side x_val
            im_s[i,4]= len(index[0]) #area of the person
    return im_s,im

def main_test():
    import scipy.io
    font = cv2.FONT_HERSHEY_COMPLEX
    data = scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')
    u_seq_abnormal = data['u_seq_abnormal']
    data = scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')
    v_seq_abnormal = data['v_seq_abnormal']
    weight = Weight_matrix().get_weight_matrix()
    thisSplitter = Spliter()

    fg_img = cv2.imread('../ref_data/fg_pics/108.bmp')
    im_s,im = poscal(fg_img)
    realPos=thisSplitter.split(im_s,im,weight)

    ab_img = cv2.imread('../ref_data/ab_fg_pics/108.bmp')

    for i,item in enumerate(realPos):
        cv2.rectangle(ab_img,(int(item[3]),int(item[1])),(int(item[2]),int(item[0])),(0, 0, 255))
        #cv2.putText(img, str(i), (int(item[3]),int(item[1])-5), font, 0.4, (255, 255, 0), 1)
    cv2.imshow('abnormal_with_posTag',ab_img)


    #np.savetxt('../ref_data/connectedFieldImg.txt',im_s,delimiter=',')
    #print(im_s)
    #plot
    img = cv2.imread('../ref_data/original_pics/001.tif')
    for i,item in enumerate(im_s):
        cv2.rectangle(img,(int(item[3]),int(item[1])),(int(item[2]),int(item[0])),(0, 0, 255))
        #cv2.putText(img, str(i), (int(item[3]),int(item[1])-5), font, 0.4, (255, 255, 0), 1)
    cv2.imshow('img',fg_img)
    cv2.imshow('im', im)
    cv2.imshow('img',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows();

if __name__=='__main__':
    main_test()