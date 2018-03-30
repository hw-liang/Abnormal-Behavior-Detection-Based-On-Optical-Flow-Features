import re
import numpy as np
from xmlLoader_generator import *

class Weight_matrix:

    def __init__(self,n=3):
        tps=self.diff(Poi_handle().searchPic(n))
        self.y1 = tps[0][0]
        self.y2 = tps[-1][0]
        self.ab = tps[0][1];
        self.cd = tps[-1][1];
        connect=np.loadtxt('../ref_data/connectedFieldImg.txt',delimiter=',')
        self.h1 = connect[8][0]-connect[8][1];
        self.h2 = connect[15][0]-connect[15][1];
        #print (self.y1)
        self.compute_weight_matrix()

    def diff(self,pic):
        res = []
        if pic is not None:
            for y in pic:
                res.append((int(y.get('val')), abs(int(re.findall('\(.*,', y[0].text)[0][1:-1]) - \
                                                   int(re.findall('\(.*,', y[-1].text)[0][1:-1]))))
        return res;

    def y_weight(self,y):

        return ((y-self.y2)/(self.y1-self.y2)*(self.cd/self.ab)+(self.y1-y)/(self.y1-self.y2))* \
               ((y - self.y2) / (self.y1 - self.y2) * (self.h2 / self.h1)+ (self.y1 - y) / (self.y1 - self.y2) )

        #old
        #return (y - self.y2) / (self.y1 - self.y2) + (self.y1 - y) / (self.y1 - self.y2) * (self.ab / self.cd)

    def compute_weight_matrix(self):
        self.weight_matrix=np.vectorize(self.y_weight)(np.arange(158))
        np.savetxt('../ref_data/weight_matrix',self.weight_matrix,delimiter=',')

    def get_weight_matrix(self):
        return self.weight_matrix

    def y_query(self,n):
        return self.weight_matrix[n]

if __name__ == '__main__':
    Weight_matrix()