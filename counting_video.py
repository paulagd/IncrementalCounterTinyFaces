import sys
#sys.path.append('./Tiny_Faces_in_Tensorflow/')
#import tiny_face_eval as tiny
sys.path.append('/home/paula/THINKSMARTER_/Model/ExtendedTinyFaces/')
# sys.path.append('/home/paula/THINKSMARTER_/Model/IncrementalCounterTinyFaces/')

# import evaluate
import evaluate as tiny_evaluate
from metrics import *
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error as mse
import glob
import os
import cv2
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import imp
import time
import random
import detect
#import dlib
from imgaug import augmenters as iaa
#imp.reload(tiny)
#imp.reload(detect)
# %matplotlib inline
from IPython import embed
from tqdm import tqdm

weights_path = '/home/paula/THINKSMARTER_/face-detectors/Tiny_Faces/hr_res101.pkl'

def createVideo(dir_path, videoName):
    # dir_path = './output_video_sample_all_faces'
    # videoName = 'test_ExtendedTinyFaces_allFaces.avi'
    listdir = os.listdir(dir_path)
    listdir.sort()
    images = []
    for f in tqdm(listdir):
        if f.endswith('.png'):
            images.append(f)

    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(videoName, fourcc, 20.0, (width, height))

    for image in tqdm(images):

        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

def cutVideo(video_path, t1, t2):
    cap = cv2.VideoCapture('/home/paula/THINKSMARTER_/videoplayback.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    initial = fps * t1
    final = fps * t2

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = cv2.VideoWriter('test_2.avi',fourcc, fps , size)
    count = 0
    while(count < final):

        ret, frame = cap.read()
        if ret :
            if count > initial:
                print(count)
                #frame = cv2.resize(frame,(size[0]//3,size[1]//3))
                out.write(frame)
            count += 1
        else:
            break

    cap.release()
    out.release()

#IDEA: GET ALL THE FRAMES AND IMAGES (neighbours)
def savingFrames(clip_path):
    # clip_path =
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    frames = []
    while(True):
        ret, frame = cap.read()
        if ret:
            frames.append(frame[:,:,::-1])
        else:
            break

    images = []
    for k in range(0, len(frames), 10):
        try:
            imgs = [frames[k], frames[k+1], frames[k+2], frames[k+10]]
        except IndexError:
            imgs = [frames[k], frames[k+1], frames[k+2], frames[len(frames)-1]]
        images.append(imgs)

    return [frames, images]
[frames, images] = savingFrames('test.avi')

#IDEA: GET ALL DETECTIONS IN IMAGES (LEN 26)
def getFrameAndNeighbourDetections(images):
    all_detections = []
    for row in tqdm(images):
        detections = []
        for frame in row:
            with tf.Graph().as_default():
                b = evaluate.evaluate(weight_file_path=weights_path,  img=frame)
            detections.append(b)
        all_detections.append(detections)

    np.save('numpy_alldetections',all_detections)
    return all_detections
all_detections = np.load('numpy_alldetections.npy')

#IDEA: GET MATCHEDS

def getMatcheds(images,all_detections, threshold = 0.55):
    matcheds = []
    t0 = time.time()
    for j in range(len(images)):
        frames = images[j]
        detections = all_detections[j]
        matched = 0
        t0bis = time.time()
        for p in range(len(detections[0])):
            neigh_detect, distances = detect.train_binclas(frames, detections, p)

            idx_max, val_max = np.argmax(distances[:,1]), np.max(distances[:,1])
            if val_max > threshold:
                matched += 1
        matcheds.append(matched)
        t1 = time.time()
        print('It took %.1f sec i.e %.2f/detection' % (t1-t0bis, (t1-t0bis)/len(detections[0])))
    print('Total : %.1f' % (time.time() - t0))

    np.save('matcheds',matcheds)
    return matcheds
matcheds = np.load('matcheds.npy')

#IDEA: COUNTING
s = 0
# for j in range(10):
for j in range(len(all_detections)):
    detections = all_detections[j]
    s += len(detections[0]) - matcheds[j]
s += len(detections[3])
print ('COUNTER S : '+ str(s))


#IDEA: GET ALL DETECTIONS in FRAMES (LEN 260)
def getFramesDetections(frames):
    detections = []                     # Detections for one face of each frame
    detections_faces = []               # Detections for all faces of each frame
    for i, frame in enumerate(tqdm(frames)):
        with tf.Graph().as_default():
            b = tiny_evaluate.evaluate(weight_file_path=weights_path, data_dir='.jpg', output_dir='', img=frame,
                              prob_thresh=0.5, nms_thresh=0.1, lw=3,
                              display=False, save=False, draw=False, print_=0)
        detections.append(b[0])
        detections_faces.append(b)
        time.sleep(0.5)

    np.save('numpy_detections_0',detections)
    np.save('numpy_detections_justFaces',detections_faces)
    return detections_faces
# detections = np.load('numpy_detections_0.npy')
detections_faces = np.load('numpy_detections_justFaces.npy')


#IDEA: COMPUTE THE INCREMENTAL COUNTER
def compute_nbs(all_detections,matcheds):
    nbs = []
    init = len(all_detections[0][0])

    #for j in range(1, 10):
    for j in range(1, len(all_detections)):
        nbs.append(init)
        detections_ = all_detections[j]

        init += len(detections_[0]) - matcheds[j-1]

    init += len(detections_[3]) - matcheds[j]
    nbs.append(init)
    return nbs
nbs = compute_nbs(all_detections,matcheds)


#IDEA: SAVE THE FRAMES
def saveFrames(frames,detections_faces, nbs, out_path):
    k = 0
    l = 0
    images = []
    ff = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    frames[:-2]

    for j, frame in enumerate(frames):
        img = frame.copy()
        for detect_ in detections_faces[j]:
            pt1, pt2 = tuple(detect_[:2]), tuple(detect_[2:])
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)

        bottomLeftCornerOfText = (img.shape[1]-650,img.shape[0]-50)
        cv2.putText(img, 'Incremental count : %d' % nbs[l], bottomLeftCornerOfText , font, 1.5, (0, 255, 0), 3)

        #if j in range(10, 89, 9):
        if j in range(len(all_detections), 89, 9):
            l += 1
        images.append(img)
        cv2.imwrite(out_path+'/frames_%05d.png' % j, img[:,:,::-1])
out_path = './output_video_sample_all_faces'
saveFrames(frames,detections_faces, nbs, out_path)


#IDEA: CREATE VIDEO
videoName = 'test_ExtendedTinyFaces_allFaces.avi'
createVideo(out_path, videoName)
