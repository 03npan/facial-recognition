
import cv2 as cv
import time
import numpy as np
import pickle
import amcrest

from config import *

url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/cam/realmonitor?channel=1&subtype=0"
cam = amcrest.AmcrestCamera(ip, port, username, password).camera

video = cv.VideoCapture(url)
face_cascade = cv.CascadeClassifier()


if not face_cascade.load(cv.data.haarcascades + "haarcascade_frontalface_default.xml"):
    print('[!] Error loading face cascade')
    exit(0)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv.dnn.readNetFromTorch(EMBEDDING_PATH)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(RECOGNIZER_PATH, "rb").read())
le = pickle.loads(open(LABEL_ENCODER_PATH, "rb").read())

detect_timer = 10

faces = []
recognized = []

# Check optimization
print("[i] Checking optimization: " + str(cv.useOptimized()))

while True:
    _, frame = video.read()

    if detect_timer == 0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        recognized = []

        detect_timer = 10

        for i, (x,y,w,h) in enumerate(faces): 
            endX = x + w
            endY = y + h

            face = frame[y:endY, x:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
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
#      elif key == ord('w'):
#          cam.ptz_control_command(action="start", code="Up", arg1=0, arg2=0, arg3=0)
#      elif key == ord('a'):
#          cam.ptz_control_command(action="start", code="Left", arg1=0, arg2=0, arg3=0)
#      elif key == ord('s'):
#          cam.ptz_control_command(action="start", code="Down", arg1=0, arg2=0, arg3=0)
#      elif key == ord('d'):
#          cam.ptz_control_command(acti#on="start", code="Right", arg1=0, arg2=0, arg3=0)
#      elif key == ord('+'):
#          break
#      elif key == ord('-'):
#          break
#      else:
#          cam.ptz_control_command(action="stop", code="Up", arg1=0, arg2=0, arg3=0)
#          cam.ptz_control_command(action="stop", code="Down", arg1=0, arg2=0, arg3=0)
#          cam.ptz_control_command(action="stop", code="Left", arg1=0, arg2=0, arg3=0)
#          cam.ptz_control_command(action="stop", code="Right", arg1=0, arg2=0, arg3=0)
    detect_timer -= 1

cv.destroyAllWindows()
