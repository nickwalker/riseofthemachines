import cv2
import numpy

# cascade = cv2.CascadeClassifier('cascades/hogcascade_pedestrians.xml')
# cascade = cv2.CascadeClassifier('cascades/lbpcascade_frontalface.xml')
cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
# cascade = cv2.CascadeClassifier('cascades/banana.xml')
# cascade = cv2.CascadeClassifier('cascades/lbpcascade_profileface.xml')


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Gray scale and normalise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect corners
    dst = cv2.cornerHarris(gray,2,3,0.04)

    # result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    frame[dst>0.01*dst.max()]=[0,0,255]


    #look for objects over the given image using the loaded cascade file
    objects = cascade.detectMultiScale(gray, 1.3, 5)

    print 'Detected {0} objects'.format(len(objects))
    for (x,y,w,h) in objects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #use opencv built in window to show the image
    cv2.imshow('image',frame)

    k = cv2.waitKey(10)
    if k==27:
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
