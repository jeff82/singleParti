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



lk_params = dict( winSize  = (21, 21),

                  maxLevel = 3,

                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.02))





feature_params = dict( maxCorners = 500,

                       qualityLevel = 0.3,

                       minDistance = 7,

                       blockSize = 7 )


algs.algsObj.getStrenthenImgMap()

class App:

    def __init__(self, video_src='data\lsa20187x3.mp4'):

        self.track_len = 10

        self.detect_interval = 5

        self.tracks = []

        self.cam = cv2.VideoCapture(video_src)

        self.frame_idx = 0




    def run(self):

        while True:

            _ret, frame = self.cam.read()
            frame=frame[:,200:-200,:]
            cv2.imshow('v',frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('v',frame)

            vis = frame.copy()

            # print 'frameid:',self.frame_idx



            if len(self.tracks) > 0:

                last_dots=[]

                for tr in self.tracks[0].values():
                    if tr['stat'][-1] > 0:
                        last_dots.append(tr['pos'][-1])

                img0, img1 = self.prev_gray, frame_gray



                p0 = np.array(last_dots).reshape(-1,1,2).astype(np.float32)
                print '========='



                # p0 = cv2.goodFeaturesToTrack(img0, mask=None, **feature_params)
                #
                # print p0,p0.shape,'f0adsf'

                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

                p0r, _st1, _err1 = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0-p0r).reshape(-1, 2).max(-1)

                good = d < 2

                #                print 'lk stat:', _st, _err
                #                print 'lk bk stat:', _st1, _err1
                print '  p1',p1.shape


                for tr,  [x, y], good_flag in zip(self.tracks[0].values(),p1.reshape(-1, 2), good):
                    #
                    if not good_flag:

                        tr['stat']=np.hstack([tr['stat'],-1])
                        tr['frm']=np.hstack([tr['frm'], self.frame_idx])
                        tr['pos']=np.vstack([tr['pos'], np.array([-1,-1])])


                    #                        continue
                    else:
                        tr['stat'] = np.hstack([tr['stat'], 1])
                        tr['frm'] = np.hstack([tr['frm'], self.frame_idx])
                        tr['pos'] = np.vstack([tr['pos'], np.array([x, y])])

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
                histOld=algs.algsObj.PreProc.histRGB
                histNew=algs.algsObj.PreProc.hist()

                hists=[]
                dotsTr={}
                for i in xrange(3):
                    hists.append(cv2.compareHist(histOld[i],histNew[i],0))
                th=np.linalg.norm(np.array(hists))
                if th<1.6:
                    algs.algsObj.getStrenthenImgMap()
                #algs.algsObj.strenthenImg()
                pts=algs.algsObj.getDots()

                print self.frame_idx,'al pnt',pts.shape

                if len(self.tracks) == 0:

                    for i,p in enumerate(pts):
                        dotsTr.update({i:{
                            'pos':np.array([p]),
                            'stat': np.array([1]),
                            'frm':np.array([0])}
                        })

                    self.tracks.append(dotsTr)
                else:

                    maxDotId = np.array(self.tracks[0].keys()).max()
                    print 'max dot Id', maxDotId

                    last_dots=[]

                    for tr in self.tracks[0].values():
                        if tr['stat'][-1]>0:
                            # last_dots.append(tr['pos'][-1])

                            distNewPnt2old = np.linalg.norm((pts-np.array(tr['pos'][-1])),2,1)
                            pts=pts[np.where(distNewPnt2old>1.4)]
                    #                        print
                    #                        if np.linalg.norm((pts-tr[-1]),2,1)<1.4:#the dots already in self.tracks
                    #                            continue
                    #                        else:
                    print '    new pnt', pts.shape
                    if len(pts)>0:
                        for i in xrange(len(pts)):
                            self.tracks[0].update(
                                {
                                    maxDotId+i+1:{
                                        'pos': np.array([pts[i]]),
                                        'stat': np.array([1]),
                                        'frm': np.array([self.frame_idx])
                                    }
                                }
                            )

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
            elif ch == ord('-'):
                cv2.waitKey(0)






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
