import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
from itertools import combinations
import pickle
import mediapipe as mp

class BWEManager:
    def __init__(self, CamOffset, CamViewDepth, NetHeight=224):
        self.CamOffset = CamOffset # cm
        self.CamViewDepth = CamViewDepth # pixels
        self.NetHeight = NetHeight # cm
        self.SceneManager = SceneManager(self)
    
    def BodyTriangulation(self, img1, img2, HomographyMatrix, debug=False):
        # [Bodies] [[Joints]] [[[x, y]]] [# bodies, 17, 2]
        img1arr = utils.BodyPose(img1)
        img2arr = utils.BodyPose(img2)
        img1BBoxes = np.array([np.min(img1arr[:, :, 0], axis=1), np.min(img1arr[:, :, 1], axis=1), np.max(img1arr[:, :, 0], axis=1), np.max(img1arr[:, :, 1], axis=1)]) # [4, # bodies, 33 joints]
        img2BBoxes = np.array([np.min(img2arr[:, :, 0], axis=1), np.min(img2arr[:, :, 1], axis=1), np.max(img2arr[:, :, 0], axis=1), np.max(img2arr[:, :, 1], axis=1)]) # [4, # bodies, 33 joints]
        if img1arr.size == 0 or img2arr.size == 0:
            return np.array([])

        norm = lambda x: (x - np.mean(x, axis=1)[:, np.newaxis, :])#/(np.max(x, axis=1)[:, np.newaxis, :]-np.min(x, axis=1)[:, np.newaxis, :])
        OffsetMatrix = np.mean((norm(img1arr)[:, np.newaxis, :, :] - norm(img2arr)[np.newaxis, :, :, :])**2, axis=(2,3))

        # Apply homography matrix
        img2arr = np.append(img2arr, (np.ones((img2arr.shape[0], img2arr.shape[1], 1))), axis = 2)
        for bInd, body in enumerate(img2arr):
            for jInd, joint in enumerate(body):
                img2arr[bInd][jInd] = (HomographyMatrix@joint[:, np.newaxis]).flatten()
        img2arr = img2arr[:, :, :-1]/img2arr[:, :, [-1]]
        
        # Normalize to center around (0,0)
        img1arr -= np.array(img1.shape)[[1, 0]][np.newaxis, np.newaxis, :]/2
        img2arr -= np.array(img2.shape)[[1, 0]][np.newaxis, np.newaxis, :]/2

        # [Bodies] [[Joints]] [[[x, y, z]]] [# bodies, 17, 3]
        img1arr = np.append(img1arr, (np.ones((img1arr.shape[0], img1arr.shape[1], 1))*self.CamViewDepth[0]), axis = 2)
        img2arr = np.append(img2arr, (np.ones((img2arr.shape[0], img2arr.shape[1], 1))*self.CamViewDepth[1]), axis = 2)

        joints = [] #[body] [[x, y, z]]
        
        img1BBoxesOut = []
        img2BBoxesOut = []
        while True:
            ind1, ind2 = np.unravel_index(np.argmin(OffsetMatrix, axis=None), OffsetMatrix.shape)

            x, y, z = [], [], []
            for joint in range(img1arr.shape[1]):
                (x_, y_, z_), cost = utils.intersection([0, 0, 0], img1arr[ind1][joint], self.CamOffset, np.array(self.CamOffset)+img2arr[ind2][joint], GetError=True)
                # print(set([tuple(i) for i in [[0, 0, 0], img1arr[ind1][joint], self.CamOffset, np.array(self.CamOffset)+img2arr[ind2][joint]]]))
                print(cost, end="\n\n")
                x.append(x_)
                y.append(y_)
                z.append(z_)
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            
            # Exit if offset too high
            if OffsetMatrix[ind1, ind2] == np.inf:
                break

            xyz = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1)

            if debug:
                connections = utils.BodyPoseConnections()
                plt.imshow(img1, alpha=0.5)
                for ind1_, ind2_ in connections:
                    x1, y1, _ = img1arr[ind1][ind1_] + np.array([*img1.shape[:-1], 0])[[1, 0, 2]]/2
                    x2, y2, _ = img1arr[ind1][ind2_] + np.array([*img1.shape[:-1], 0])[[1, 0, 2]]/2

                    plt.plot([x1, x2], [y1, y2])
                plt.show()
                plt.imshow(img2, alpha=0.5)
                for ind1_, ind2_ in connections:
                    x1, y1, _ = img2arr[ind2][ind1_] + np.array([*img2.shape[:-1], 0])[[1, 0, 2]]/2
                    x2, y2, _ = img2arr[ind2][ind2_] + np.array([*img2.shape[:-1], 0])[[1, 0, 2]]/2

                    plt.plot([x1, x2], [y1, y2])
                plt.show()
            
            mask = np.ones_like(OffsetMatrix).astype(np.bool_)
            mask[ind1] = False
            mask[:, ind2] = False
            OffsetMatrix[~mask] = np.inf
            joints.append(xyz)
            img1BBoxesOut.append(img1BBoxes[:, ind1])
            img2BBoxesOut.append(img2BBoxes[:, ind2])
        joints = np.array(joints) # [# bodies, 33 joints, 3 xyz]
        img1BBoxesOut = np.array(img1BBoxesOut) # [# bodies, 4 x1y1x2y2]
        img2BBoxesOut = np.array(img2BBoxesOut) # [# bodies, 4 x1y1x2y2]
        
        return joints, img1BBoxesOut, img2BBoxesOut
    
    def BallTriangulation(self, cap1, cap2, HomographyMatrix, length = None, YellowThresh1 = (0.8,130), YellowThresh2 = (0.8,120), BackgroundFrames = 150, UpdateMedian = False, MovementThresh = 20, debug=False, SaveAsVideo=True):
        def GetBackground(imgs1, imgs2):
            Background1 = np.median(np.array(imgs1), axis=0)
            Background2 = np.median(np.array(imgs2), axis=0)
            return Background1, Background2

        LastNImages = [[], []]
        if SaveAsVideo: imgs = [[], []]

        for t in range(BackgroundFrames):
            res1, frame1 = cap1.read()
            res2, frame2 = cap2.read()
            if not res1 or not res2:
                return "lenth of vid < background frames ):<"
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            LastNImages[0].append(frame1)
            LastNImages[1].append(frame2)
        Background1, Background2 = GetBackground(*LastNImages)

        xyz = []
        for t in range(BackgroundFrames):
            MovementMask1 = np.any((np.abs(LastNImages[0][t] - Background1) >= MovementThresh), axis=2)
            MovementMask2 = np.any((np.abs(LastNImages[1][t] - Background2) >= MovementThresh), axis=2)

            ColorMask1_ = (0.5*LastNImages[0][t][:, :, 0] + 0.5*LastNImages[0][t][:, :, 1] - LastNImages[0][t][:, :, 2])
            ColorMask1 = ((ColorMask1_-np.min(ColorMask1_))/(np.max(ColorMask1_)-np.min(ColorMask1_))>YellowThresh1[0]).astype(np.uint8)
            ColorMask1 = cv2.morphologyEx(ColorMask1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

            ColorMask2_ = (0.5*LastNImages[1][t][:, :, 0] + 0.5*LastNImages[1][t][:, :, 1] - LastNImages[1][t][:, :, 2])
            ColorMask2 = ((ColorMask2_-np.min(ColorMask2_))/(np.max(ColorMask2_)-np.min(ColorMask2_))>YellowThresh2[0]).astype(np.uint8)
            ColorMask2 = cv2.morphologyEx(ColorMask2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

            mask1 = np.all([MovementMask1, ColorMask1], axis = 0).astype(np.uint8)
            mask2 = np.all([MovementMask2, ColorMask2], axis = 0).astype(np.uint8)

            if np.count_nonzero(mask1) == 0 or np.count_nonzero(mask2) == 0:
                xyz.append(None)
                continue
            
            contour, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if np.max(ColorMask1_[mask1.astype(np.bool_)]) < YellowThresh1[1]:
                xyz.append(None)
                continue
            
            contour = max(contour, key=cv2.contourArea)[:, 0]
            if cv2.contourArea(contour) < 150:
                xyz.append(None)
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center1 = (x+w/2, y+h/2)

            contour, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if np.max(ColorMask2_[mask2.astype(np.bool_)]) < YellowThresh2[1]:
                xyz.append(None)
                continue
            
            contour = max(contour, key=cv2.contourArea)[:, 0]
            if cv2.contourArea(contour) < 150:
                xyz.append(None)
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center2 = (x+w/2, y+h/2)

            # Apply homography matrix
            b = np.array([*center2, 1])
            b = (HomographyMatrix@b[:, np.newaxis]).flatten()
            b = b[:-1]/b[-1]
            b = np.array([*b, self.CamViewDepth[1]])

            xyz.append(utils.intersection([0, 0, 0], [*center1, self.CamViewDepth[0]], self.CamOffset, np.array(self.CamOffset)+b))

            if SaveAsVideo:
                imgs[0].append(cv2.circle(LastNImages[0][t], center1, 5, (0, 0, 255), -1))
                imgs[1].append(cv2.circle(LastNImages[1][t], center2, 5, (0, 0, 255), -1))
            
            if debug:
                print("\n\n\n1:\n====================")
                plt.imshow(LastNImages[0][t])
                plt.scatter(*center1, color="red", s=1)
                plt.show()
                # plt.imshow(Background1/255)
                # plt.show()
                # plt.imshow(MovementMask1)
                # plt.show()
                # plt.imshow(ColorMask1)
                # plt.show()
                plt.imshow(mask1)
                plt.scatter(*center1, color="red", s=1)
                plt.show()
                
                print("\n2:\n====================")
                plt.imshow(LastNImages[1][t])
                plt.scatter(*center2, color="red", s=1)
                plt.show()
                # plt.imshow(Background2/255)
                # plt.show()
                # plt.imshow(MovementMask2)
                # plt.show()
                # plt.imshow(ColorMask2)
                # plt.show()
                plt.imshow(mask2)
                plt.scatter(*center2, color="red", s=1)
                plt.show()

        if not UpdateMedian: del(LastNImages)

        while True:
            if length: 
                if t >= length: break
            t += 1

            res1, frame1 = cap1.read()
            res2, frame2 = cap2.read()
            if not res1 or not res2:
                break
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            if UpdateMedian:
                LastNImages = [LastNImages[0][1:] + [frame1], LastNImages[1][1:] + [frame2]]
                Background1, Background2 = GetBackground(*LastNImages)

            MovementMask1 = np.sum((np.abs(frame1 - Background1) >= MovementThresh), axis=2).astype(np.bool_)
            MovementMask2 = np.sum((np.abs(frame2 - Background2) >= MovementThresh), axis=2).astype(np.bool_)

            ColorMask1 = (0.5*frame1[:, :, 0] + 0.5*frame1[:, :, 1] - frame1[:, :, 2])
            ColorMask1 = ((ColorMask1-np.min(ColorMask1))/(np.max(ColorMask1)-np.min(ColorMask1))>YellowThresh1[0]).astype(np.uint8)
            ColorMask1 = cv2.morphologyEx(ColorMask1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

            ColorMask2 = (0.5*frame2[:, :, 0] + 0.5*frame2[:, :, 1] - frame2[:, :, 2])
            ColorMask2 = ((ColorMask2-np.min(ColorMask2))/(np.max(ColorMask2)-np.min(ColorMask2))>YellowThresh2[0]).astype(np.uint8)
            ColorMask2 = cv2.morphologyEx(ColorMask2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

            mask1 = np.all([MovementMask1, ColorMask1], axis = 0).astype(np.uint8)
            mask2 = np.all([MovementMask2, ColorMask2], axis = 0).astype(np.uint8)

            if np.count_nonzero(mask1) == 0 or np.count_nonzero(mask2) == 0:
                xyz.append(None)
                continue
            if np.max(ColorMask2_[mask2.astype(np.bool_)]) < YellowThresh1[1]:
                xyz.append(None)
                continue
            if np.max(ColorMask2_[mask2.astype(np.bool_)]) < YellowThresh2[1]:
                xyz.append(None)
                continue

            contour, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            contour = max(contour, key=cv2.contourArea)[:, 0]
            if cv2.contourArea(contour) < 200:
                xyz.append(None)
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center1 = (x+w/2, y+h/2)

            contour, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            contour = max(contour, key=cv2.contourArea)[:, 0]
            if cv2.contourArea(contour) < 200:
                xyz.append(None)
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center2 = (x+w/2, y+h/2)

            # Apply homography matrix
            b = np.array([*center2, 1])
            b = (HomographyMatrix@b[:, np.newaxis]).flatten()
            b = b[:-1]/b[-1]
            b = np.array([*b, self.CamViewDepth[1]])
            
            xyz.append(utils.intersection([0, 0, 0], [*center1, self.CamViewDepth[0]], self.CamOffset, np.array(self.CamOffset)+b))

            if SaveAsVideo:
                imgs[0].append(cv2.circle(frame1, center1, 5, (0, 0, 255), -1))
                imgs[1].append(cv2.circle(frame2, center2, 5, (0, 0, 255), -1))

            if debug:
                print("\n\n\n1:\n====================")
                plt.imshow(frame1)
                plt.scatter(*center1, color="red", s=1)
                plt.show()
                # plt.imshow(Background1/255, aspect='auto')
                # plt.show()
                # plt.imshow(MovementMask1)
                # plt.show()
                # plt.imshow(ColorMask1)
                # plt.show()
                plt.imshow(mask1)
                plt.scatter(*center1, color="red", s=1)
                plt.show()
                
                print("\n2:\n====================")
                plt.imshow(frame2)
                plt.scatter(*center2, color="red", s=1)
                plt.show()
                # plt.imshow(Background2/255, aspect='auto')
                # plt.show()
                # plt.imshow(MovementMask2)
                # plt.show()
                # plt.imshow(ColorMask2)
                # plt.show()
                plt.imshow(mask2)
                plt.scatter(*center2, color="red", s=1)
                plt.show()
        
        if SaveAsVideo:
            video1=cv2.VideoWriter('BallOut1.mp4', 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   cap1.get(cv2.CAP_PROP_FPS),
                                   imgs[0][-1].shape[[1,0]])
            for img in imgs[0]:
                video1.write(img)
            video1.release()
            
            video2=cv2.VideoWriter('BallOut2.mp4', 
                                   cv2.VideoWriter_fourcc(*'mp4v'), 
                                   cap2.get(cv2.CAP_PROP_FPS),
                                   imgs[1][-1].shape[[1,0]])
            for img in imgs[1]:
                video1.write(img)
            video2.release()

            return xyz, (video1, video2)

        return xyz

    def AssignCourtValues(self, vid1, StartFrame1, vid2, StartFrame2, CourtStorageFile="CourtStorage.pkl"):
        cap1 = cv2.VideoCapture(vid1) # .mp4
        cap1.set(cv2.CAP_PROP_POS_FRAMES, StartFrame1-1)
        cap2 = cv2.VideoCapture(vid2)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, StartFrame2-1)

        res1, frame1 = cap1.read()
        res2, frame2 = cap2.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        if not res1 or not res2:
            return
        
        CourtPoints = self.SceneManager.CourtInput(frame1, frame2)
        file = open(CourtStorageFile,"wb")
        pickle.dump(CourtPoints, file)
        file.close()

        file = open(CourtStorageFile,"rb")
        CourtPoints = pickle.load(file)
        file.close()

        for x, y in CourtPoints[0]:
            print((x-frame1.shape[1]/2, y-frame1.shape[0]/2, self.CamViewDepth[0]))
        print("\n")

    def ProcessVideo(self, vid1, StartFrame1, vid2, StartFrame2, CourtRefPoints, CourtPointIndexes, StorageFile="storage.pkl", CourtStorageFile="CourtStorage.pkl", length=False, GetBodies=True, GetBall=True):
        cap1 = cv2.VideoCapture(vid1)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, StartFrame1-1)
        cap2 = cv2.VideoCapture(vid2)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, StartFrame2-1)

        res1, frame1 = cap1.read()
        res2, frame2 = cap2.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        if not res1 or not res2:
            return
        
        file = open(CourtStorageFile,"rb")
        CourtPoints = pickle.load(file)
        file.close()
        
        # Get CourtRefPoints pos on screen
        points = []
        for point in CourtRefPoints:
            points.append((point[:-1]-np.array(self.CamOffset)[:-1])*self.CamViewDepth[1]/(point[-1]-self.CamOffset[-1])+np.array(frame2.shape)[[1,0]]/2)
        points = np.array(points)
        
        # Get homography matrix
        HomographyMatrix = utils.HomographyMatrix(*CourtPoints[1].flatten(), *points.flatten())
        
        # Get court
        court = self.SceneManager.CourtInterpolation(CourtRefPoints, CourtPointIndexes, NetHeight=self.NetHeight)

        # Get bodies
        if GetBodies:
            bodies = []
            BodyBBoxes = [[], []]
            t = 0
            while True:
                if length:
                    if t >= length:
                        break
                t += 1
                
                out = self.BodyTriangulation(frame1, frame2, HomographyMatrix)
                bodies.append(out[0])
                BodyBBoxes[0].append(out[1])
                BodyBBoxes[1].append(out[2])
                
                res1, frame1 = cap1.read()
                res2, frame2 = cap2.read()
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                if not res1 or not res2:
                    break
        else:
            bodies = False
            BodyBBoxes = False
        
        if GetBall:
            # Get VBall
            cap1 = cv2.VideoCapture(vid1)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, StartFrame1-1)
            cap2 = cv2.VideoCapture(vid2)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, StartFrame2-1)
            balls = self.BallTriangulation(cap1, cap2, HomographyMatrix, length=length)
        else:
            balls = False

        eyes = False
        focuses = False


        out = []
        ind = 0
        while True:
            if length:
                if ind >= length:
                    break

            val = []

            val.extend([False]*5)
            
            if GetBodies:
                val.append(bodies[ind])
            else:
                val.append(False)

            if GetBall:
                val.append(balls[ind])
            else:
                val.append(False)
            
            out.append(val)

            ind += 1

        file = open(StorageFile,"wb")
        pickle.dump([out, court], file)
        file.close()
    
    def PlayVideo(self, StorageFile):
        file = open(StorageFile,"rb")
        self.SceneManager.MainLoop(*pickle.load(file))
        file.close()


class SceneManager:
    def __init__(self, BWEManager, framerate=60):
        self.BWEManager = BWEManager
        self.framerate = framerate
    
    def CourtInput(self, img1, img2):
        print("left click: place point")
        print("right click: undo point")
        print("enter: finish", end="\n"*3)

        CourtPoints = []
        for img in [img1, img2]:
            f1 = plt.figure()
            f2 = plt.figure()
            ax1 = f1.add_subplot(111)
            ax1.imshow(cv2.cvtColor(cv2.imread('VballCourtGuide.jpg'), cv2.COLOR_BGR2RGB))
            ax2 = f2.add_subplot(111)
            ax2.imshow(img)

            CourtPoints.append(np.array(f2.ginput(5, timeout=-1)))
            plt.close('all')
        
        return CourtPoints
    
    def CourtInterpolation(self, CourtPoints, CourtPointIndexes, NetHeight=224):
        # Estimate rest of points
        p1 = CourtPoints[0]
        p2 = CourtPoints[1]
        p3 = CourtPoints[2]
        # p4 = CourtPoints[3][1:]
        # print([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])
        
        norm = utils.GetPlaneNorm(p1, p2, p3)
        RotationMatrix = utils.GetVectorVectorRotationMatrix(norm, [0, 0, 1])
        p1, p2 = (RotationMatrix@np.array([p1, p2]).T).T # p1, p2, p4 = (RotationMatrix@np.array([p1, p2, p4]).T).T

        RefPoints = utils.CourtLocalCoordinates(NetHeight=NetHeight)
        rp1 = RefPoints[CourtPointIndexes[0]]
        RefPoints -= rp1[np.newaxis, :]
        rp1 = RefPoints[CourtPointIndexes[0]]
        rp2 = RefPoints[CourtPointIndexes[1]]
        theta = np.arctan2(p2[1]-p1[1], p2[0]-p1[0]) - np.arctan2(rp2[1]-rp1[1], rp2[0]-rp1[0])

        CourtPoints = (utils.RotationMatrix3D(0, theta, 0)@(RefPoints).T).T+p1[np.newaxis, :]
        CourtPoints = (RotationMatrix.T@np.array(CourtPoints).T).T

        return CourtPoints
    
    def MainLoop(self, vid, court):
        global t, ax, playing, x, y, z, MPLObjs

        # params
        MoveSpeed = 0.01
        RotateSpeed = 1

        # setup matplotlib
        def press(event):
            global t, ax, playing, x, y, z, MPLObjs

            if event.key.isupper() or event.key == ":":
                MoveSpeed_ = MoveSpeed * 20
                RotateSpeed_ = RotateSpeed * 20
            else:
                MoveSpeed_ = MoveSpeed
                RotateSpeed_ = RotateSpeed

            match event.key.lower():
                case "w":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[-MoveSpeed_], [0], [0]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "a":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[0], [-MoveSpeed_], [0]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "s":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[MoveSpeed_], [0], [0]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "d":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[0], [MoveSpeed_], [0]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "q":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[0], [0], [MoveSpeed_]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "e":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[0], [0], [-MoveSpeed_]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "o":
                    ax.elev -= RotateSpeed_
                    plt.draw()
                
                case "k":
                    ax.azim += RotateSpeed_
                    plt.draw()
                
                case "l":
                    ax.elev += RotateSpeed_
                    plt.draw()
                
                case ";" | ":":
                    ax.azim -= RotateSpeed_
                    plt.draw()
                
                case "i":
                    ax.roll += RotateSpeed_
                    plt.draw()
                
                case "p":
                    ax.roll -= RotateSpeed_
                    plt.draw()
                
                case "\\" | "|":
                    ax.elev, ax.azim, ax.roll = BaseElev, BaseAzim, BaseRoll
                    plt.draw()
                
                case "," | "<":
                    t -= 1

                    for obj in MPLObjs:
                        for obj_ in obj:
                            obj_.remove()
                    
                    LEyeCenter, REyeCenter, LIris, RIris, focus, body, ball = vid[t]
                    MPLObjs = []
                    if bodies:
                        for body in bodies: 
                            MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
                    
                    if ball:
                        MPLObjs.append(self.plot(ball, None, 'yellow'))
                    
                    if type(LEyeCenters) == np.ndarray:
                        for i in range(len(LEyeCenters)): 
                            LEyeCenter, REyeCenter, LIris, RIris, focus = LEyeCenters[i], REyeCenters[i], LIrises[i], RIrises[i], focuses[i]
                            
                            MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*10000], [[0, 1]], 'red')) 
                            MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*10000], [[0, 1]], 'red')) 
                            MPLObjs.append(self.plot(focus, None, 'red'))
                
                case "." | ">":
                    t += 1

                    for obj in MPLObjs:
                        for obj_ in obj:
                            obj_.remove()
                    
                    LEyeCenter, REyeCenter, LIris, RIris, focus, body, ball = vid[t]
                    MPLObjs = []
                    if bodies:
                        for body in bodies: 
                            MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
                    
                    if ball:
                        MPLObjs.append(self.plot(ball, None, 'yellow'))
                    
                    if type(LEyeCenters) == np.ndarray:
                        for i in range(len(LEyeCenters)): 
                            LEyeCenter, REyeCenter, LIris, RIris, focus = LEyeCenters[i], REyeCenters[i], LIrises[i], RIrises[i], focuses[i]
                            
                            MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*10000], [[0, 1]], 'red')) 
                            MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*10000], [[0, 1]], 'red')) 
                            MPLObjs.append(self.plot(focus, None, 'red'))
                
                case " ":
                    playing = True
                
                case "/" | "/":
                    t = 0
                    playing = False

                    for obj in MPLObjs:
                        for obj_ in obj:
                            obj_.remove()
                    
                    LEyeCenter, REyeCenter, LIris, RIris, focus, body, ball = vid[t]
                    MPLObjs = []
                    if bodies:
                        for body in bodies: 
                            MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
                    
                    if ball:
                        MPLObjs.append(self.plot(ball, None, 'yellow'))
                    
                    if type(LEyeCenters) == np.ndarray:
                        for i in range(len(LEyeCenters)): 
                            LEyeCenter, REyeCenter, LIris, RIris, focus = LEyeCenters[i], REyeCenters[i], LIrises[i], RIrises[i], focuses[i]
                            
                            MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*10000], [[0, 1]], 'red')) 
                            MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*10000], [[0, 1]], 'red')) 
                            MPLObjs.append(self.plot(focus, None, 'red'))
                
                case "`" | "~":
                    ax.set_xbound(x[0]-100, x[1]+100)
                    ax.set_ybound(y[0]-100, y[1]+100)
                    ax.set_zbound(z[0]-100, z[1]+100)
                
                case "up":
                    x1, x2 = ax.get_xbound()
                    y1, y2 = ax.get_ybound()
                    z1, z2 = ax.get_zbound()
                    ax.set_xbound(x1+(x2-x1)*0.1, x2-(x2-x1)*0.1)
                    ax.set_ybound(y1+(y2-y1)*0.1, y2-(y2-y1)*0.1)
                    ax.set_zbound(z1+(z2-z1)*0.1, z2-(z2-z1)*0.1)
                
                case "down":
                    x1, x2 = ax.get_xbound()
                    y1, y2 = ax.get_ybound()
                    z1, z2 = ax.get_zbound()
                    ax.set_xbound(x1-(x2-x1)*0.1, x2+(x2-x1)*0.1)
                    ax.set_ybound(y1-(y2-y1)*0.1, y2+(y2-y1)*0.1)
                    ax.set_zbound(z1-(z2-z1)*0.1, z2+(z2-z1)*0.1)

        fig = plt.figure()
        fig.tight_layout()
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot(projection='3d')
        ax.set_facecolor('black')
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

        # unbind keys that interfere with controls
        try:
            plt.rcParams['keymap.quit'].remove('q')
            plt.rcParams['keymap.save'].remove('s')
            plt.rcParams['keymap.xscale'].remove('k')
            plt.rcParams['keymap.yscale'] = []
            plt.rcParams['keymap.zoom'] = []
            plt.rcParams['keymap.xscale'] = []
        except:
            pass
        
        fig.canvas.mpl_connect('key_press_event', press)
        
        # initialize variables
        t=0
        playing = False
        
        # rotate resize window to include entire court + 1m padding
        plt.autoscale(False)
        ax.set_proj_type('persp')
        ax.elev, ax.azim, ax.roll = -30, -92, 104
        BaseElev, BaseAzim, BaseRoll = ax.elev, ax.azim, ax.roll
        BoundRange = max(np.max(court, axis=0)-np.min(court, axis=0))
        x, y, z = [None, None], [None, None], [None, None]
        x[0], y[0], z[0] = (np.max(court, axis=0)+np.min(court, axis=0))/2-BoundRange/2
        x[1], y[1], z[1] = (np.max(court, axis=0)+np.min(court, axis=0))/2+BoundRange/2
        ax.set_xbound(x[0]-100, x[1]+100)
        ax.set_ybound(y[0]-100, y[1]+100)
        ax.set_zbound(z[0]-100, z[1]+100)
        plt.draw()
        
        # plot
        LEyeCenters, REyeCenters, LIrises, RIrises, focuses, bodies, ball = vid[t]
        MPLObjs = []
        self.plot(court, utils.GetCourtConnections(), 'white')

        if bodies:
            for body in bodies: 
                MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
        
        if ball:
            MPLObjs.append(self.plot(ball, None, 'yellow'))
        
        if type(LEyeCenters) == np.ndarray:
            for i in range(len(LEyeCenters)): 
                LEyeCenter, REyeCenter, LIris, RIris, focus = LEyeCenters[i], REyeCenters[i], LIrises[i], RIrises[i], focuses[i]
                
                MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*10000], [[0, 1]], 'red')) 
                MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*10000], [[0, 1]], 'red')) 
                MPLObjs.append(self.plot(focus, None, 'red'))

        # mainloop
        while True:
            plt.pause(1/self.framerate)

            if playing:
                t += 1
                
                for obj in MPLObjs:
                    for obj_ in obj:
                        obj_.remove()
                
                LEyeCenters, REyeCenters, LIrises, RIrises, focuses, bodies, ball = vid[t]
                MPLObjs = []
                if bodies:
                    for body in bodies: 
                        MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
                
                if ball:
                    MPLObjs.append(self.plot(ball, None, 'yellow'))
                
                if type(LEyeCenters) == np.ndarray:
                    for i in range(len(LEyeCenters)): 
                        LEyeCenter, REyeCenter, LIris, RIris, focus = LEyeCenters[i], REyeCenters[i], LIrises[i], RIrises[i], focuses[i]
                        
                        MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*10000], [[0, 1]], 'red'))
                        MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*10000], [[0, 1]], 'red'))
                        MPLObjs.append(self.plot(focus, None, 'red'))
    
    def plot(self, points, connections=False, color='white'):
        if type(points) != np.ndarray:
            return [DummyMPLObj()]

        out = []
        if connections:
            for ind1, ind2 in connections:
                x1, y1, z1 = points[ind1]
                x2, y2, z2 = points[ind2]

                out.append(plt.plot([x1, x2], [y1, y2], [z1, z2], color=color)[0])
            return out
        else:
            if points.ndim == 1:
                points = points[np.newaxis, :]

            return plt.scatter(*np.array(points).T, color=color)


class DummyMPLObj:
    def remove(self):
        pass