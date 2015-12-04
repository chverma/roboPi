###https://dzone.com/articles/face-and-eyes-detection-opencv
import picamera
import cv
import cv2
import io
import numpy as np
import facedetection
import eyedetection_bo

def getFrame(camera):
	#Create a memory stream so photos doesn't need to be saved in a file
	stream = io.BytesIO()
	camera.capture(stream, format='jpeg')
	#Convert the picture into a numpy array
	buff = np.fromstring(stream.getvalue(), dtype=np.uint8)

	#Now creates an OpenCV image
	image = cv2.imdecode(buff, 1)
	return image
camera = picamera.PiCamera() 
camera.resolution = (320, 240)	
while True:
    
    frame = getFrame(camera)
    #cv2.imshow("video", frame)
            
    cv.SaveImage("webcam.jpg", cv.fromarray(frame))
    
    imcolor = cv.LoadImage('webcam.jpg') # input image
    storage = cv.CreateMemStorage()
    faces = facedetection.getFaces(imcolor,storage)
    eyes  = eyedetection_bo.getEyes(imcolor,storage)
    
    cv.ShowImage('preview', imcolor) 

    key = cv2.waitKey(40)
    if key == 27 or key == 1048603: # exit on ESC
        break
    #if key == 32 or 1048608 == key: # press space to save images

