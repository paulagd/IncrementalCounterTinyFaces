import sys
# sys.path.append('/home/paula/THINKSMARTER_/Model/ExtendedTinyFaces/')
# sys.path.append('/home/paula/THINKSMARTER_/Model/IncrementalCounterTinyFaces/')
import evaluate as tiny_evaluate
from metrics import *
import tensorflow as tf
import os
import cv2
import numpy as np
import time
import detect
from IPython import embed
from tqdm import tqdm
from crowd_counting.crowd_counting import CrowdCounting


def createVideo(dir_path, videoName):
    """ Creates a video from the frames of a directory. """
    # dir_path = './output_video_sample_all_faces'
    # videoName = 'test_ExtendedTinyFaces_allFaces.avi'

    listdir = os.listdir(dir_path)
    listdir.sort()
    images = []
    for f in listdir:
        if f.endswith('.png'):
            images.append(f)

    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(videoName, fourcc, 20.0, (width, height))

    print("CREATING VIDEO --->")
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
    """Cut the video on video_path between the instants t1 and t2 in seconds."""

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

def gettingFrames(clip_path):
    """ Get all the frames and images"""
    # clip_path =
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    frames = []
    print("Getting all the frames ... ")

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

def getFrameAndNeighbourDetections(images, weights_path):
    """ Get all the detections in images(len 26)"""

    all_detections = []
    print("Getting all the detections of the main frame and its neighbours --->")
    for row in tqdm(images):
        detections = []
        for frame in row:
            with tf.Graph().as_default():
                b = tiny_evaluate.evaluate(weight_file_path=weights_path,  img=frame)
            detections.append(b)
        all_detections.append(detections)

    np.save('numpy_alldetections',all_detections)
    return all_detections

def getMatcheds(images,all_detections, threshold = 0.55):
    """Get the matcheds on the frames."""
    matcheds = []
    t0 = time.time()
    print("Getting matcheds ... --->")
    for j in tqdm(range(len(images))):
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

def counting(all_detections,matcheds):
    """ Get the total of detections computed."""
    s = 0
    # for j in range(10):
    for j in range(len(all_detections)):
        detections = all_detections[j]
        s += len(detections[0]) - matcheds[j]
    s += len(detections[3])
    print ('COUNTER S : '+ str(s))
    return s

def getFramesDetections(frames,weights_path):
    """ Get all the detections in frames (len 260)"""
    detections = []                     # Detections for one face of each frame
    detections_faces = []               # Detections for all faces of each frame
    print("Getting all the frames detections... --->")
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

def compute_nbs(all_detections,matcheds):
    """ Computes the incremental counter nbs"""
    nbs = []
    init = len(all_detections[0][0])

    #for j in range(1, 10):
    for j in range(1, len(all_detections)):
        nbs.append(init)
        detections_ = all_detections[j]

        # init += len(detections_[0]) - matcheds[j-1]
        init += len(detections_[0]) - matcheds[j]

    init += len(detections_[3]) - matcheds[j]
    nbs.append(init)
    return nbs

def predict_crowd(img):
    cc = CrowdCounting()
    number_person = cc.run(img)
    return number_person[0]

def saveFrames(frames,detections_faces, nbs, out_path):
    """Save all the frames with the information processed
    (bounding boxes and incremental counter)"""
    # k = 0
    l = 0
    images = []
    ff = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    frames[:-2]
    cc = 0
    # crowd_counter = []
    print("Saving frames ... --->")
    for j, frame in enumerate(tqdm(frames)):
        img = frame.copy()
        for detect_ in detections_faces[j]:
            pt1, pt2 = tuple(detect_[:2]), tuple(detect_[2:])
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)

        bottomLeftCornerOfText = (img.shape[1]-650,img.shape[0]-50)
        cv2.putText(img, 'Incremental count : %d' % nbs[l], bottomLeftCornerOfText , font, 1.5, (0, 255, 0), 3)
        # cv2.putText(img, 'Incremental count : %d' % cc, bottomLeftCornerOfText , font, 1.5, (0, 255, 0), 3)

        # if j in range(10, 89, 9):
        if j in range(10, len(frames), 10):
        # if j in range(len(all_detections), 89, 9):
            # print("Predicting crowd...")
            # cc = predict_crowd(frames[j])
            # crowd_counter.append(cc)
            l += 1
            # print("!!!!! DONE !!!!!")

        images.append(img)
        cv2.imwrite(out_path+'/frames_%05d.png' % j, img[:,:,::-1])

def main():
    # weights_path = '/home/paula/THINKSMARTER_/face-detectors/Tiny_Faces/hr_res101.pkl'
    weights_path = './weights/hr_res101.pkl'
    out_path = './output_video_sample_all_faces'
    videoName = 'test_incrementalCounter_new.avi'

    [frames, images] = gettingFrames('test.avi')

    if os.path.isfile("numpy_alldetections.npy"):
        all_detections = np.load('numpy_alldetections.npy')
    else:
        all_detections = getFrameAndNeighbourDetections(images, weights_path)

    if os.path.isfile("matcheds.npy"):
        matcheds = np.load('matcheds.npy')
    else:
        matcheds = getMatcheds(images,all_detections, threshold = 0.55)

    if os.path.isfile("numpy_detections_justFaces.npy"):
        # detections = np.load('numpy_detections_0.npy')
        detections_faces = np.load('numpy_detections_justFaces.npy')
    else:
        detections_faces = getFramesDetections(frames, weights_path)

    max_detections = counting(all_detections,matcheds)
    nbs = compute_nbs(all_detections,matcheds)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    saveFrames(frames,detections_faces, nbs, out_path)
    createVideo(out_path, videoName)

if __name__ == "__main__":
    main()
