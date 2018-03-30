import numpy as np

gate=0.8

def labeling(pos,abnormal_fg_img):
    if abnormal_fg_img is None:
        abnormal_fg_img = np.zeros((158, 238, 3), dtype=np.uint8)
    labels=[]
    labeledPos=np.zeros((0,5))
    for thePos in pos:
        if abnormal_fg_img[int(thePos[1]):int(thePos[0]),int(thePos[3]):int(thePos[2])].any():
            #if the fulfillment from abnormal image in this tagged area is above a certain gate, tag 1
            if abnormal_fg_img[int(thePos[1]):int(thePos[0]),int(thePos[3]):int(thePos[2])].mean()>gate:
                labeledPos=np.concatenate((labeledPos,thePos.reshape(1,-1)))
                labels.append(1)
        else:
            labeledPos = np.concatenate((labeledPos, thePos.reshape(1, -1)))
            labels.append(0)

    return  labeledPos,np.array(labels)