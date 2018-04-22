# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:36:21 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import  cv2
from sklearn import linear_model
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D 


def  sigmoid(inX):
    try:
        sigout=1.0/(1.0+np.exp(-inX)+1e-6)
    except:
        print inX.shape
        pass
            
    return sigout

def mophors(img,kernel= np.ones((5,5),np.uint8),tp='Opening'):
    #kernel = np.ones((5,5),np.uint8)
    #mp=img
    if tp=='Erosion':
        mp = cv2.erode(img,kernel,iterations = 1)
    elif tp=='Dilation':
        mp = cv2.dilate(img,kernel,iterations = 1)
    elif tp=='Opening':
       '''Opening is just another name of erosion followed by dilation.
       It is useful in removing noise. 
       '''
       mp = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif tp=='Closing':
        mp = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        '''Closing is reverse of Opening, Dilation followed by Erosion. 
        It is useful in closing small holes inside the foreground objects,
        or small black points on the object.
        '''        
    elif tp=='Gradient':
        mp = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return mp


def kerFunc(sigma,ker):
        #guassain func
        return np.exp(-0.5/(sigma**2) * (ker**2))
    
class imgPreProc:
    def __init__(self):
        self.impath='thumb/frm_lsa20187x3.mp4416.png'
        
        '''cv2:BGR   plt RGB'''
        self.img=cv2.imread(self.impath)[:,:,::-1]
        print self.img.shape,'imgshape'
            
        self.HarrisParameter = [2,9,0.02]
        self.HarrisRespThresh=0.03
        self.ctl =linear_model.RANSACRegressor(linear_model.LinearRegression())
        self.kmeans= KMeans(n_clusters=2, random_state=0)
        self.grayB=self.img[:,:,-1]
        self.gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)  
        
        self.bUseNormlizedImg=True
        
        '''project img to color vector or norm2'''
        self.strenthenByProj=True
        self.clustedColorVec=[0,0,0]
        
        self.strethenMagnitude=0.5



 

    def hist(self):
        histB=cv2.calcHist([self.img],[0],None,[256],[0,256])
        histG=cv2.calcHist([self.img],[1],None,[256],[0,256])
        histR=cv2.calcHist([self.img],[2],None,[256],[0,256])
        plt.figure('org')
        plt.imshow(self.img)


#    def getCorner(self,imgIn):
#        if imgIn.shape.__len__()>2:
#            gray = cv2.cvtColor(imgIn,cv2.COLOR_BGR2GRAY)
#            gray = np.float32(gray)
#        else:
#            gray = imgIn
#    
#        dst = cv2.cornerHarris(gray,HarrisParameter[0],HarrisParameter[1],HarrisParameter[2])
#    
#        dst = cv2.dilate(dst,None)
#    
#        imgIn[dst>0.01*dst.max()]=[0,0,255]
#        
#        
#        return dst



    def HarrisMap(self,imgIn):
        if imgIn.shape.__len__()>2:
            print 'need gray img'
            self.getGray(imgIn)
            gray = self.gray
            gray = np.float32(gray)
        else:
            gray = imgIn
        dst = cv2.cornerHarris(gray,self.HarrisParameter[0],self.HarrisParameter[1],self.HarrisParameter[2])
#        dst = cv2.dilate(dst,None)
        cornerMap=np.copy(imgIn)*0
        
        '''try to calc the threshold by getThresh() later'''
        cornerMap[np.where(dst>self.HarrisRespThresh*dst.max())]=255
        plt.figure('cornerMap')
        plt.imshow(dst)
        
        return cornerMap

    def getThresh(self,biImg,thresh=127,typ='otsu'):
        
        img = (biImg.astype(np.float32)/biImg.max().astype(np.float32)*255).astype(np.uint8)
        
        #ret,biImg = cv2.threshold(img,thresh,255,0)
        blurCore=(1,1)
        burSigma=0.1
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
            ret,thresh1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            biImg= thresh1

        return biImg

    def conductAreaByBinaImg(self,inBinary):
        ret, self.labels, self.stats, self.centroids = cv2.connectedComponentsWithStats(inBinary.astype(np.uint8))
        
        return ret, self.labels, self.stats, self.centroids

    def raancDot(self):
              
        return
    def calcImgMean(self):
        return  self.img.T.mean(2).mean(1)


    def findDotArea(self,gray):
        conerMap=self.HarrisMap(gray)
    
        self.threshMap=self.getThresh(conerMap)
        
        self.conductAreaByBinaImg(conerMap)
        #ret, self.labels, self.stats, self.centroids = cv2.connectedComponentsWithStats(conerMap.astype(np.uint8))
        
        plt.figure('labels')
        plt.imshow(self.labels)


    def clusterDotColor(self):

        nodots=np.where(self.stats[:,-1]>1000)
        dotsArea = self.img[np.where(self.labels != nodots)]
        dotCandi=[]
        dotAvg=[]
        for i in xrange(self.centroids.__len__()):
            if self.stats[i,-1]>500:
                continue
            dotCandi.append(self.img[np.where(self.labels==i)])
            dotAvg.append(self.img[np.where(self.labels==i)].mean(0))
        dotCandi=np.array(dotCandi)
        
        #''''calc everything with normlized img''''
        
        if self.bUseNormlizedImg:
            #print dotsArea,np.linalg.norm(dotsArea,2,1).reshape(-1,1)
            dotsArea=dotsArea/np.linalg.norm(dotsArea,2,1).astype(float).reshape(-1,1)
        else:
            pass
        self.kmeans.fit(dotsArea)
        
        print 'centers:',self.kmeans.cluster_centers_
        
        center0=self.labels[np.where(self.labels != nodots)][np.where(self.kmeans.labels_==0)]
        center1=self.labels[np.where(self.labels != nodots)][np.where(self.kmeans.labels_==1)]
        
        imgMean=self.calcImgMean()
        
        if self.bUseNormlizedImg:
            imgMean/=np.linalg.norm(imgMean,2)
        
        euDist0=np.linalg.norm((self.kmeans.cluster_centers_[0]-imgMean),2)
        euDist1=np.linalg.norm((self.kmeans.cluster_centers_[1]-imgMean),2)
        
        print 'cluster 0 norm 2 is :',euDist0,':',self.kmeans.cluster_centers_[0]
        print 'cluster 1 norm 2 is :',euDist1,':',self.kmeans.cluster_centers_[1]
        if euDist0>euDist1:
            print 'cluster 0 is the dot color'
            self.clustedColorVec=self.kmeans.cluster_centers_[0]/np.linalg.norm(self.kmeans.cluster_centers_[0])
            return 0,self.kmeans.cluster_centers_[0]
        else:
            print 'cluster 1 is the dot color'
            self.clustedColorVec=self.kmeans.cluster_centers_[1]/np.linalg.norm(self.kmeans.cluster_centers_[1])
            return 1,self.kmeans.cluster_centers_[1]
    
     
    def predictDot(self,inImg):
        
    
        pred1=[-1]
        pred2=[-1]
        for i in xrange(self.centroids.__len__()):
            if self.stats[i,-1]>500:
                continue
            pred1.append(self.kmeans.predict(inImg[np.where(self.labels==i)]).mean())
            pred2.append(self.kmeans.predict([inImg[np.where(self.labels==i)].mean(0)]))
        
        return np.array(pred1),np.array(pred2)
    
    def getGray(self,imgIn):
        self.gray=cv2.cvtColor(imgIn,cv2.COLOR_BGR2GRAY)
        self.grayB=self.img[:,:,-1]
        
    def calcStrenthenMap(self):   
        _,dotCluster=self.clusterDotColor()
        if self.bUseNormlizedImg:
            normMat = np.linalg.norm(self.img,2,2)+1e-2
            imgMap=self.img/(normMat.reshape(self.img.shape[0],self.img.shape[1],1))
            #imgMap[np.where(normMat>0)]=imgMap[np.where(normMat>0)].astype(float)/normMat[np.where(normMat>0)].reshape(-1,1).astype(float)
#            plt.figure('p')
#            plt.imshow(imgMap)
        else:
            imgMap=self.img
        if self.strenthenByProj:
            euDist = np.linalg.norm(np.cross(imgMap,dotCluster),2,2)
        else:
            euDist = np.linalg.norm(imgMap - dotCluster,2,2)
        imgStd=np.std(euDist)
        
        strenthenMap= kerFunc(self.strethenMagnitude*imgStd,euDist)
        
        return strenthenMap
    
    
    def strenthenImg(self):    
        strenthenMap =self.calcStrenthenMap()
        
        def showStrenImg():
            plt.figure('strenthened color')
            plt.imshow((self.img*strenthenMap.reshape(strenthenMap.shape[0],strenthenMap.shape[1],1)).astype(np.uint8))
            plt.figure('gray')
            plt.imshow(self.gray)
            plt.figure('strenthenMap')
            plt.imshow(strenthenMap)
            plt.figure('strenthened gray')
            plt.imshow((self.gray*strenthenMap))
            '''test script below'''
            
            gray2 = cv2.cvtColor((self.img*strenthenMap.reshape(strenthenMap.shape[0],strenthenMap.shape[1],1)).astype(np.uint8),cv2.COLOR_BGR2GRAY)
            gray2 = np.float32(gray2)  
            plt.figure('strenthened gray2')
            plt.imshow(gray2)
        return  (PreProc.img*strenthenMap.reshape(strenthenMap.shape[0],strenthenMap.shape[1],1)).astype(np.uint8)
    
    def projImg2ColorVect(self,imgIn):
        return
    
    def subPixelCorner(self,inImg):
        self.getGray(inImg)
        self.findDotArea(self.gray)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        self.corners = cv2.cornerSubPix(self.gray,np.float32(self.centroids),(3,3),(-1,-1),criteria)



PreProc=imgPreProc()

#
PreProc.getGray(PreProc.img)
PreProc.findDotArea(PreProc.gray)
PreProc.subPixelCorner(PreProc.img)
#
rst,dotCluster=PreProc.clusterDotColor() 
strentenedColorImg =PreProc.strenthenImg()

PreProc.subPixelCorner(strentenedColorImg)
pred1,pred2=PreProc.predictDot(strentenedColorImg)
PreProc.getGray(strentenedColorImg)


biImg=PreProc.getThresh(PreProc.gray)#,typ='adapt')
mpImg=mophors(biImg,kernel= np.ones((3,3),np.uint8),tp='Opening')
PreProc.conductAreaByBinaImg(mpImg)


plt.figure('0')
plt.imshow(PreProc.gray)
plt.plot(PreProc.centroids[:,0],PreProc.centroids[:,1],'*r')
plt.plot(PreProc.corners[:,0],PreProc.corners[:,1],'g.')
plt.figure('o')
plt.imshow(PreProc.img)
plt.figure('strenthened im')
plt.imshow(strentenedColorImg)


#kmeans = KMeans(n_clusters=2, random_state=0).fit(dotAvg)
#print 'centers:',kmeans.cluster_centers_




   