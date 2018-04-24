# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:06:27 2018

@author: Administrator
"""





import numpy as np

import cv2 

import algs


#from common import anorm2, draw_str

#from time import clock



lk_params = dict( winSize  = (15, 15),

                  maxLevel = 2,

                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))





feature_params = dict( maxCorners = 500,

                       qualityLevel = 0.3,

                       minDistance = 7,

                       blockSize = 7 )


algs.algsObj.getStrenthenImgMap()

class App:

    def __init__(self, video_src='lsa20187x3.mp4'):

        self.track_len = 10

        self.detect_interval = 5

        self.tracks = []

        self.cam = cv2.VideoCapture(video_src)  

        self.frame_idx = 0




    def run(self):

        while True:

            _ret, frame = self.cam.read()
            cv2.imshow('v',frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('v',frame)

            vis = frame.copy()
            
           # print 'frameid:',self.frame_idx



            if len(self.tracks) > 0:

                img0, img1 = self.prev_gray, frame_gray

                p0 = np.float32(self.tracks[-1]).reshape(-1, 1, 2)

                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
               
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                
                falseDot = d>1.4
                p1[np.where(falseDot)] = p0[np.where(falseDot)]
                self.tracks.append(p1)

                good = d < 1


                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
#
                    if not good_flag:

#                        continue
                        pass
#
#                    tr.append([x, y])
#
##                    if len(tr) > self.track_len:
##
##                        del tr[0]
#
#                    new_tracks.append(tr)

                    cv2.circle(vis, (x, y), 20, (0, 255, 0), 1,1)

#                self.tracks = new_tracks

                    #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

#                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))



            if self.frame_idx % self.detect_interval == 0:
                
                
                algs.algsObj.UpDateImg(frame)
                #algs.algsObj.strenthenImg()
                p=algs.algsObj.getDots()
                print self.frame_idx,'al pnt',p.shape
                
                if len(self.tracks) == 0:
                    self.tracks.append(p)
                else:
                    
                
                    for tr in self.tracks[-1]:
                        distNewPnt2old = np.linalg.norm((p-tr[-1]),2,1)
                        p=p[np.where(distNewPnt2old>1.4)]
#                        print 
#                        if np.linalg.norm((p-tr[-1]),2,1)<1.4:#the dots already in self.tracks
#                            continue
#                        else:
                    print '    new pnt',p.shape
                    if len(p)>0:
                        self.tracks.append(np.vstack([self.tracks[-1],p.reshape(-1,1,2)]))

#                mask = np.zeros_like(frame_gray)
#
#                mask[:] = 255
#
#                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
#
#                    cv2.circle(mask, (x, y), 25, 0, 1)
#
#                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
#                print '+++++++++'
#                print p
#                print '++======'
#
#                if p is not None:
#
#                    for x, y in np.float32(p).reshape(-1, 2):
#
#                        self.tracks.append([(x, y)])
#




            self.frame_idx += 1

            self.prev_gray = frame_gray

            cv2.imshow('lk_track', vis)
#            cv2.imshow('lk_track_mask', mask)



            ch = cv2.waitKey(1) & 0xff

            if ch ==ord('q'):

                break






def main():

    import sys

    try:

        video_src = sys.argv[1]

    except:

        video_src = 0



    print(__doc__)

    App().run()

    cv2.destroyAllWindows()




a=App()
a.run()

#if __name__ == '__main__':

#    main()