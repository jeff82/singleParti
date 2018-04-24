'''
2018-4-15
author--lz 
'''



import pylab
import imageio
import cv2
import numpy as np
import os

class vidInfo:
    def __init__(self):
        self.filename = 'lsa20187x3.mp4'
        self.curImg=np.eye(3)
        self.totalFrame=0
        self.curFrameNo=0
        self.fps=0
        self.outPutInfo='----------------------------'
    
    def procInfo(self,k):
        self.outPutInfo+=' \n fps:'+str(self.fps)+'\n'
        self.outPutInfo+=' frm:'+str(self.curFrameNo)+" / "+str(self.totalFrame)+'\n'
        print self.outPutInfo

  
    
class vdPlay(vidInfo):

    def __init__(self):
        vidInfo.__init__(self)

        self.player='cv2'

       
    def playVid(self):
        
        def ffmepgPlay(self):
            
            vid = imageio.get_reader('lsa20187x2.mp4',  'ffmpeg')
            fig = pylab.figure(0)
            nums = np.arange(200)
            for num in nums:
                self.curImg=image = vid.get_data(num)
                
                fig.suptitle('image #{}'.format(num), fontsize=20)
                cv2.imshow('',image)
    
        def cvPlay(self):
            cap = self.cap= cv2.VideoCapture(self.filename)  
            if cap.isOpened()==False:
                return
            self.totalFrame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.outPutInfo
            
#            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#            fgbg = cv2.createBackgroundSubtractorMOG2()
            
            cap.set(cv2.CAP_PROP_POS_FRAMES,self.curFrameNo)
            while(cap.isOpened()):  
                cmd=''
                self.curFrameNo+=1
                ret, frame = cap.read()  
                self.curImg=frame
                cv2.imshow('image', frame)  
                
#                fgmask = fgbg.apply(frame)
##                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
#                cv2.imshow('frame',fgmask)


                Key = cv2.waitKey(1)  
                if (Key & 0xff == ord('q')): 
                    self.release()
                elif Key & 0xff == ord('-'):
                    
#                    while(Key!=13):
#                       
#                        Key=cv2.waitKey(0)  
#                        print Key,'key',chr(Key & 0xff)
#                        cmd+=chr(Key & 0xff)                        
                    self.procInfo(cmd)
                    break
        
        def release(self):
            self.cap.release()  
            cv2.destroyAllWindows()
        
        if self.player =='cv2':
            return cvPlay(self)
  

class ctrl:
    
    def __init__(self):
        self.vids = vdPlay()
        self.cmd={
                'p':self.vids.playVid,
                's':self.svImg,
                'c':self.findCorner
                  }
        
    
        
    def svImg(self):
        cv2.imwrite('thumb/'+'frm_'+self.vids.filename+str(self.vids.curFrameNo)+'.png',self.vids.curImg)
    
    def findCorner():
        return
        

vidProc = ctrl()    
def proc():
   while True:
       cmdCode = raw_input("Please intput your cmd:")
       if vidProc.cmd.has_key(cmdCode):
           vidProc.cmd[cmdCode]()
       elif cmdCode=='q':
           break
       else:
           print 'invalid cmd'

                 
    
if __name__=="__main__" :

    proc()


    #sys.exit(app.exec_())