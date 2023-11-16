import cv2
import time
path='/path/to/your/videos.mp4'

def call(x):
    pass

try:
    vd=cv2.VideoCapture(path)
except:
    pass
if not vd.isOpened():    
    raise TypeError("!!!you have to change video path!!!")
    
cv2.namedWindow('config',cv2.WINDOW_FREERATIO)
cv2.createTrackbar('lower', 'config', 0, 255, call)
cv2.createTrackbar('height', 'config', 100, 1000, call)
cv2.createTrackbar('width', 'config', 100, 1000, call)
cv2.setTrackbarPos('lower', 'config', 147)
cv2.setTrackbarPos('height', 'config', 500)
cv2.setTrackbarPos('width', 'config', 800)

cv2.namedWindow('show',cv2.WINDOW_NORMAL)
count=0
while vd.isOpened():
    count+=1
    ret,frame=vd.read()
    if count%1==0:
        dst=cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
        _,u,_=cv2.split(dst)
        
        lower=cv2.getTrackbarPos('lower','config')
        height=cv2.getTrackbarPos('height','config')
        width=cv2.getTrackbarPos('width','config')
        
        dst=cv2.inRange(u,lower,255)
        dst=cv2.resize(dst,(width,height),interpolation=cv2.INTER_LINEAR)
        cv2.imshow('show',dst)
        
    if not ret or (cv2.waitKey(1) & 0xFF) == 27:
        break
    
