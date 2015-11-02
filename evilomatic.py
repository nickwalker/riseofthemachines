import io
import cv2
import numpy
import math
from UniversalAnalytics import Tracker

size = 1
cv2.namedWindow('Preview')
life=20


#face_cascade = cv2.CascadeClassifier('cascades/lbpcascade_frontalface.xml')
#face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('cascades/hogcascade_pedestrians.xml')
face_cascade = cv2.CascadeClassifier('cascades/banana.xml')

stream = io.BytesIO()
tracker = Tracker.create('UA-123944-30')

def distance(newpoint, oldpoint):
    return math.sqrt((newpoint[0] - oldpoint[0]) ** 2 + (newpoint[1] - oldpoint[1]) ** 2)

(oldpoints, opid) = ({}, 0)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #convert to grayscale, which is easier
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    #look for faces over the given image using the loaded cascade file
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(20, 20)
    )

    todo = [f for f in faces]
    sofar = {}
    for op in oldpoints:
        sofar[op] = ''
    extra = []
    while todo:
            np = todo.pop()
            done = False
            closest = [(op, distance(np, oldpoints[op][0])) for op in
                               oldpoints]
            closest = [x[0] for x in sorted(closest, key=lambda x: x[1])]
            for cop in closest:
                    if sofar[cop] == '' or distance(np, oldpoints[cop][0]) \
                            < distance(sofar[cop], oldpoints[cop][0]):
                            if sofar[cop] != '':
                                    todo.append(sofar[cop])
                            sofar[cop] = np
                            done = True
                            break
            if not done:
                    extra.append(np)

    # Update oldpoints

    for op in sofar:
            if sofar[op] != '':
                    oldpoints[op] = [sofar[op], life]
            else:
                    oldpoints[op] = [oldpoints[op][0], oldpoints[op][1] - 1]

    # Check for any new points

    for np in extra:
            oldpoints[opid] = [np, life]
            opid += 1

            #Ping Google Analytics
            tracker.send('event', 'Person', 'Detect')
            print 'New face detected: {0}'.format(opid);

    # Check to kill any

    todel = []
    for op in oldpoints:
            if oldpoints[op][1] == 0:
                    todel.append(op)

    for op in todel:
            del oldpoints[op]

    for f in oldpoints:
            (x, y, w, h) = [v * size for v in oldpoints[f][0]]
            dying = 255 // life * (life - oldpoints[f][1])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0 , 255 - dying, dying), 2)
            cv2.putText(frame, '%s' % f, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    cv2.imshow('Preview', frame)

    key = cv2.waitKey(10)
    if key == 27:  # exit on ESC
            break

cap.release()
cv2.destroyAllWindows()

