
import cv2 as cv
import time
import numpy as np
import pickle
import amcrest
import torch
import net
import dlib
from align_dlib import AlignDlib
import imutils
import argparse
import os

ip = "192.168.1.109"
port = "80"
username = "admin"
password = "internsarethebest"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--embedding-model", default="net.pth",
                    help="path to the deep learning face embedding model")
    ap.add_argument("-r", "--recognizer", default="output/recognizer.pickle",
                    help="path to model trained to recognize faces")
    ap.add_argument("-l", "--le", default="output/le.pickle",
                    help="path to label encoder")
    ap.add_argument("-c", "--confidence", type=float, default=0.45,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-d", "--detector", default="face_detection_model",
                    help="path to OpenCV's deep learning face detector")

    args = vars(ap.parse_args())
    url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/cam/realmonitor?channel=1&subtype=0"
    video = cv.VideoCapture(url)


    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv.dnn.readNetFromCaffe(protoPath, modelPath)

    # You can download the required pre-trained face detection model here:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    face_aligner = AlignDlib(predictor_model)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    torch.set_grad_enabled(False)
    embedder = net.model
    embedder.load_state_dict(torch.load('net.pth'))
    embedder.to(device)
    embedder.eval()

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(args["recognizer"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())

    detect_timer = 3

    faces = []
    recognized = []

    while True:
        _, frame = video.read()

        if detect_timer == 0:
            # resize it to have a width of 600 pixels (while
            # maintaining the aspect ratio) 
            image = imutils.resize(frame, width=600)
            blob, box = create_face_blob(frame, detector, face_aligner)
            if blob is None:
                continue
            (x, y, endX, endY) = box.astype("int")
            recognized = []

            inputs = torch.from_numpy(blob).to(device)
            vec = embedder.forward(inputs).cpu().numpy()
            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)
            detect_timer = 3

            j = np.argmax(preds)
            proba = preds[0, j]
            name = le.classes_[j]
            if proba*100 < 40:
                text = "{}: {:.2f}%".format("Unknown", proba * 100)
            else:
                text = "{}: {:.2f}%".format(name, proba * 100)
            recognized.append((x, y, endX, endY, text))

        for face in recognized:
            cv.rectangle(frame, face[:2], face[2:4], (255,255,0), 2)  
            cv.putText(frame, face[4], face[:2], cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv.imshow(('Camera'), frame)

        key = cv.waitKey(1) & 0xff 
        if key == ord('q'):
             break
        detect_timer -= 1

    cv.destroyAllWindows()


def create_face_blob(image, detector, face_aligner):
    # grab the image dimensions
    (ih, iw) = image.shape[:2]
    # construct a blob from the image
    imageBlob = cv.dnn.blobFromImage(
        cv.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # extract the confidence (i.e., probability) associated with the
    # first prediction
    confidence = detections[0, 0, 0, 2]

    min_confidence = 0.6
    # filter out weak detections
    if confidence > min_confidence:
        # compute the (x, y)-coordinates of the bounding box for the
        # face
        box = detections[0, 0, 0, 3:7] * np.array([iw, ih, iw, ih])
        (startX, startY, endX, endY) = box.astype("int")
        if startX < 0 or startY < 0 or endX > iw or endY > ih:
            return None, None

        # align the face
        rect = dlib.rectangle(startX, startY, endX, endY)
        face = face_aligner.align(
                96,
                image,
                rect,
                landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE
        )

        # construct a blob for the face ROI, then pass the blob
        # through our face embedding model to obtain the 128-d
        # quantification of the face
        faceBlob = cv.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                         (0, 0, 0), swapRB=True, crop=False)
        return faceBlob, box
    else:
        return None, None



if __name__ == '__main__':
    main()
