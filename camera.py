import cv2 as cv
import time
import numpy as np
import amcrest

from config import *

speed = 8
fps = 30

url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/cam/realmonitor?channel=1&subtype=0"
cam = amcrest.AmcrestCamera(ip, port, username, password).camera
moved = False

video = cv.VideoCapture(url)


def move(dir): 
    moved = True
    cam.ptz_control_command(action="start", code=dir, arg1=0, arg2=speed, arg3=0)

while True:
    _, frame = video.read()

    cv.imshow(('Camera'), frame)

    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('w'):
        move("Up")
    elif key == ord('a'):
        move("Left")
    elif key == ord('s'):
        move("Down")
    elif key == ord('d'):
        move("Right")
    elif key == ord('+'):
        break
    elif key == ord('-'):
        break
    elif moved:
        moved = False
        cam.ptz_control_command(action="stop", code="Up", arg1=0, arg2=0, arg3=0)
        cam.ptz_control_command(action="stop", code="Down", arg1=0, arg2=0, arg3=0)
        cam.ptz_control_command(action="stop", code="Left", arg1=0, arg2=0, arg3=0)
        cam.ptz_control_command(action="stop", code="Right", arg1=0, arg2=0, arg3=0)
 
cv.destroyAllWindows()
