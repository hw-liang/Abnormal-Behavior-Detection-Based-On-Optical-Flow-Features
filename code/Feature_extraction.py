import cv2
from poscal import poscal
import numpy as np
from weight_matrix import *
from split import *
from getFeatureUV import *
from poscalNormal import *
from Classifiers import *
from labeling import *

class Feature_extractor(object):

    def __init__(self, originpics, forgpics, ab_forgpics, U, V, weigh):
        self.originpics = originpics
        self.forgpics = forgpics
        self.ab_forgpics = ab_forgpics
        self.U = U
        self.V = V
        self.weigh =weigh
        self.m = U.shape[0]
        self.n = U.shape[1]

    def getPosition(self, pics, index, style=True, mode=True):
        this_Spliter = Spliter()
        weight = self.weigh
        if mode:
            img = cv2.imread(pics[index])
            if img is None:
                img = np.zeros((self.m, self.n, 3), dtype=np.uint8)
            if style:
                ab_img = cv2.imread(self.ab_forgpics[index])
                im_s,mopho_img = poscalNormal(img,ab_img)
                splitPos = this_Spliter.split(im_s,mopho_img,weight)
                realPos, label = labeling(splitPos,ab_img)
            else:
                realPos,_ = poscal(img)
                label = np.ones(realPos.shape[0])
                mopho_img=None
        else:
            img = cv2.imread(self.forgpics[index])
            im_s,mopho_img = poscal(img)
            splitPos = this_Spliter.split(im_s,mopho_img,weight)
            realPos,label = labeling(splitPos,cv2.imread(self.ab_forgpics[index]))

        Img = cv2.imread(self.originpics[index])
        return realPos,Img,label,mopho_img

    def simgle_feature(self,pics,index,style=True,mode=True):
        realPos,_,label,_ = self.getPosition(pics,index,style,mode)

        feature = getFeaturesUV(realPos,self.U[:,:,index]* np.sqrt(self.weigh).reshape((self.m, 1)),self.V[:,:,index]* np.sqrt(self.weigh).reshape((self.m, 1)))

        return feature,label

    def get_features_and_labels(self, start, end, mode=True):
        datal = np.zeros((0,2))
        datalAb = np.zeros((0,2))
        label = np.zeros(0)
        labelAb = np.zeros(0)
        if mode:
            for i in range(start,end):
                data,labe = self.simgle_feature(self.forgpics,i)
                dataAb,labeAb = self.simgle_feature(self.ab_forgpics,i,False)
                datal = np.concatenate((datal, data), axis=0)
                datalAb = np.concatenate((datalAb, dataAb), axis=0)
                label = np.concatenate((label, labe), axis=0)
                labelAb = np.concatenate((labelAb, labeAb), axis=0)
            features = np.nan_to_num(np.concatenate((datal, datalAb), axis=0))
            labels = np.nan_to_num(np.concatenate((label, labelAb), axis=0))
        else:
            features = np.zeros((0,2))
            for i in range(start,end):
                data,_ = self.simgle_feature(self.forgpics,i,True,mode)
                features = np.nan_to_num(np.concatenate((features, data), axis=0))
            labels = None
        return features,labels