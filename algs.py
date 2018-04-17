# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:36:21 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import  cv2
from sklearn import linear_model


impath='thumb/frm_lsa20187x3.mp4416.png'

'''cv2:BGR   plt RGB'''
img=cv2.imread(impath)

def  sigmoid(inX):
    try:
        sigout=1.0/(1.0+np.exp(-inX)+1e-6)
    except:
        print inX.shape
        pass
            
    return sigout


HarrisParameter = [2,5,0.04]
HarrisRespThresh=0.02
ctl =linear_model.RANSACRegressor(linear_model.LinearRegression())
 

def hist():
    histB=cv2.calcHist([img],[0],None,[256],[0,256])
    histG=cv2.calcHist([img],[1],None,[256],[0,256])
    histR=cv2.calcHist([img],[2],None,[256],[0,256])
    plt.imshow(img)


def getCorner(imgIn):
    gray = cv2.cvtColor(imgIn,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(img[:,:,0],HarrisParameter[0],HarrisParameter[1],HarrisParameter[2])

    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255]
    
    
    return dst



def HarrisMap(imgIn):
    if imgIn.shape.__len__()>2:
        print 'need gray img'
    dst = cv2.cornerHarris(imgIn,HarrisParameter[0],HarrisParameter[1],HarrisParameter[2])
    dst = cv2.dilate(dst,None)
    cornerMap=np.copy(imgIn)*0
    
    '''try to calc the threshold by getThresh() later'''
    cornerMap[np.where(dst>HarrisRespThresh*dst.max())]=255
    plt.imshow(cornerMap)
    
    return conerMap

def getThresh(biImg,thresh=127,typ='adapt'):
    
    img = biImg/biImg.max()*255
    
    ret,biImg = cv2.threshold(img,thresh,255,0)
    blurCore=(1,1)
    burSigma=3
    if typ=='adapt':
        
        blur = cv2.GaussianBlur(img,blurCore,burSigma)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        #self.biImg=
        biImg = th3
        
    elif typ=='selfadapt':
        blur = cv2.GaussianBlur(img,blurCore,burSigma)
        th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,11,2)
        #self.biImg =
        biImg= th2
    elif typ=='simple':
        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        '''cv2.THRESH_BINARY
        cv2.THRESH_BINARY_INV
        cv2.THRESH_TRUNC
        cv2.THRESH_TOZERO
        cv2.THRESH_TOZERO_INV'''
        #self.biImg =
        biImg= thresh1
    elif typ=='otsu':
        #blur = cv2.GaussianBlur(img,blurCore,burSigma)
        ret,thresh1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        biImg= thresh1
#        plt.figure('th')
#        plt.imshow(biImg)
#        plt.figure('r')
#        plt.imshow(blur)

    return biImg

def contour(inBinary):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(inBinary.astype(np.uint8))
    
    return ret, labels, stats, centroids 

def raancDot():
          
    return


'''test script below'''

grayB=img[:,:,0]
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
      
conerMap=HarrisMap(gray)

biImg=threshMap=getThresh(conerMap)

ret, labels, stats, centroids = cv2.connectedComponentsWithStats(conerMap.astype(np.uint8))

dotCandi=[]
for i in xrange(centroids.__len__()):
    dotCandi=np.array([dotCandi,img[np.where(labels==i)]])


   