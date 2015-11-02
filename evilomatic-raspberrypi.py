import io
import picamera
import cv2
import numpy
import math
from UniversalAnalytics import Tracker

size = 1
cv2.namedWindow('Preview')
life=5


face_cascade = cv2.CascadeClassifier('/home/pi/people_counter/lbpcascade_frontalface.xml')
#face_cascade = cv2.CascadeClassifier('/home/pi/people_counter/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('hogcascade_pedestrians.xml')

stream = io.BytesIO()
tracker = Tracker.create('UA-123944-30')

def distance(newpoint, oldpoint):
    return math.sqrt((newpoint[0] - oldpoint[0]) ** 2 + (newpoint[1] - oldpoint[1]) ** 2)

(oldpoints, opid) = ({}, 0)

with picamera.PiCamera() as camera:
        #camera.start_preview()
        camera.resolution = (360,240)
        camera.rotation = 180


        #capture into stream
        count = 0
        for foo in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.truncate()
                stream.seek(0)

                #convert image into numpy array
                data = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

                #turn the array into a cv2 image
                frame = cv2.imdecode(data, 1)

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
        cv2.waitKey(0)
        cv2.destroyAllWindows()

