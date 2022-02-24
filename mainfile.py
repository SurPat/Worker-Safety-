import threading
import cv2
import cam1, cam2

from threading import *

def camra1():
    print("starting cam1..")
    cam1.cam1()

def camra2():
    print("starting cam2..")
    cam2.cam2()

thr1 = threading.Thread(target=camra1)
thr2 = threading.Thread(target=camra2)

thr1.start()
thr2.start()
