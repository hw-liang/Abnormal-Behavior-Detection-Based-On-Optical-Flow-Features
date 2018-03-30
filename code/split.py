import numpy as np

class Spliter(object):

    normal = 120
    heightNorm = 20#'CHANGE TO INT'
    widthNorm = 6#'heightNorm / shapeParam'

    def __init__(self,discardFloor=0.08,splitCeil=1.8):
        self.floor=discardFloor*Spliter.normal;
        self.ceil=splitCeil*Spliter.normal;

    def split(self,pos,fg_img,weight):
        posArea,heights,widths = self.areaHeightWidthCompute(pos,weight)
        realPos = np.zeros((0,5))
        for ind,area in enumerate(posArea):

            height=heights[ind]
            width=widths[ind]

            if area < self.floor:
                #print('discard')
                continue
            if area > self.ceil:

                n_h=int(round(height[0]/Spliter.heightNorm));
                n_w=int(round(width[0]/Spliter.widthNorm));
                ##################old######################
                n = min(int(round(area[0] /Spliter.normal)),n_w*n_h);

                #print('n_h: ',n_h,'n_w: ',n_w,'n: ',n,'        -',n_h*n_w == n)

                #print('split',n)
                recArea = (pos[ind][0]-pos[ind][1])*(pos[ind][2]-pos[ind][3])
                shape = (pos[ind][0]-pos[ind][1])/(pos[ind][2]-pos[ind][3])
                step_y = (pos[ind][0] - pos[ind][1]) / n_h;
                step_x = (pos[ind][2] - pos[ind][3]) / n_w;

                res = []
                # diganol splitting
                for i in range(n_h):
                    pos1 = int(pos[ind][1] + i * step_y);
                    pos0 = int(pos[ind][1] + (i + 1) * step_y);
                    for j in range(n_w):
                        pos3 = int(pos[ind][3] + j * step_x);
                        pos2 = int(pos[ind][3] + (j + 1) * step_x);
                        res.append([pos0, pos1, pos2, pos3, fg_img[pos1:pos0, pos3:pos2].sum(),
                                    fg_img[pos1:pos0, pos3:pos2].mean()])
                res.sort(key=lambda x: x[-1])
                res = res[::-1]
                for i in range(n):
                    new = np.array(res[i][:-1])
                    realPos = np.concatenate((realPos, new.reshape(1, 5)), axis=0)
            else:
                realPos=np.concatenate((realPos,pos[ind].reshape(1,5)),axis=0)
        return realPos

    def areaHeightWidthCompute(self,pos,weight):
        area = np.zeros((pos.shape[0],1))
        height = np.zeros((pos.shape[0],1));
        width = np.zeros((pos.shape[0],1));
        for ind,eachPos in enumerate(pos):
            area[ind] = eachPos[-1]*weight[int((eachPos[0]+eachPos[1])//2)]
            height[ind] = (eachPos[0]-eachPos[1])*np.sqrt(weight[int((eachPos[0]+eachPos[1])//2)]);
            width[ind] = (eachPos[2]-eachPos[3])*np.sqrt(weight[int((eachPos[0]+eachPos[1])//2)]);
        return area,height,width;