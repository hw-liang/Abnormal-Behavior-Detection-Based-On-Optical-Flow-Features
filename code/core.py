#Abnormal behavior detection project
#main features: optical flow
#classifiers: SVM, LinearRegression, KNN

from weight_matrix import *
from Feature_extraction import *
from Classifiers import *
import scipy.io
import cv2
import matplotlib.pyplot as plt

def uvPlot(u,v,labels,timerSet=False):
    fig=plt.figure()
    for ind,label in enumerate(labels):
        if label:
            plt.scatter(u[ind],v[ind],c='red')#red for abnormal person
        else:
            plt.scatter(u[ind], v[ind], c='blue')#blue for walking people
    if timerSet:
        timer = fig.canvas.new_timer(interval=10000)  # creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(plt.close)
        timer.start();
    plt.ylabel('v')
    plt.xlabel('u')
    plt.title('Optical Flow Features (U V) of training set \nPlease close it to continue')
    plt.legend()
    plt.show()

def load_data():
    #optical flow (u v) data generated from matlab script
    u_seq_abnormal = scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')['u_seq_abnormal']
    v_seq_abnormal = scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')['v_seq_abnormal']

    fg_imgs = ['../ref_data/fg_pics/' + str(i + 1) + '.bmp' for i in range(200)]#foreground extracted images
    original_imgs = ['../ref_data/original_pics/' + str(i + 1).zfill(3) + '.tif' for i in range(200)]#original images
    abnormal_fg_imgs = ['../ref_data/ab_fg_pics/' + str(i + 1) + '.bmp' for i in range(200)]#abnormal person position by manually drawing

    return u_seq_abnormal,v_seq_abnormal,fg_imgs,original_imgs,abnormal_fg_imgs

def plot(realPos,labels,img,classifier,timerSet=True):

    target=[None,0]#dummy start

    for i, item in enumerate(realPos):
        if labels[i]:#consider only abnormal people (label 1)
            if item[0] == item[1]:
                item[0]+=1;
            if item[2] == item[3]:
                item[2]+=1;
            score= item[-1]/((item[0]-item[1])*(item[2]-item[3]))

            if score >= target[-1]:#the largest fulfillment position
                target=[item,score]

    if target[-1]:#if no abnormal person detected, tag nothing
        item=target[0]
        cv2.rectangle(img, (int(item[3]), int(item[1])), (int(item[2]), int(item[0])), (0, 0, 255))
    cv2.imshow('classifier: {}'.format(classifier), img)
    if timerSet:
        if cv2.waitKey(100) & 0xff == 27:
            cv2.destroyAllWindows()
    else:
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

def main ():

    u_data,v_data,fg_imgs,original_imgs,abnormal_fg_imgs=load_data()#load data from ref_data

    weight = Weight_matrix().get_weight_matrix()#use normalization method for feature correction

    thisFeatureExtractor = Feature_extractor(original_imgs,fg_imgs,abnormal_fg_imgs,u_data,v_data,weight)

    train_data,train_labels = thisFeatureExtractor.get_features_and_labels(80,140)#training frames

    ########################## To see the train data features distribution, uncomment next line##################################
    #uvPlot(train_data[:,0],train_data[:,1],train_labels,False)
    #############################################################################################################################

    classifiers = Classifiers(train_data,train_labels)

    test_data, test_labels = thisFeatureExtractor.get_features_and_labels(140,199)#testing frames

    for name,model in classifiers.models.items():#get each classifier
        for ind,original_img in enumerate(original_imgs[:-1]):#get each frame

            pos,thisImg,_,_=thisFeatureExtractor.getPosition(fg_imgs,ind)#get the position of each person in this frame

            features,_=thisFeatureExtractor.get_features_and_labels(ind,ind+1,False)#get the features for each person in the frame

            labels=classifiers.models[name].predict(features)#predict label

            plot(pos,labels,thisImg,name)#show

        classifiers.prediction_metrics(test_data,test_labels,name)#metrics for each classifier based on the test data

if __name__ == '__main__':
    print (
        '''

        This version will continuously process 200 frames from a 20s video

        3 classifiers: SVM, LinearRegression, and KNN trained with hyper-parameters tuning

        Metrics: ROC curve, AUC of ROC, and prediction accuracy will be provided each classifier's process

        '''
    )
    main()