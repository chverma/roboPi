
import sys
#sys.path.append('/home/chverma/Descargas/opencv-3.0.0')

import cv2.cv as cv
import io
import cv2
import time
from PIL import Image
import numpy as np
import defaults
import logistic
import faceDetect
if defaults.raspberry_plt:
    import picamera
from time import sleep
from loadTrainingData import getTrainingRes

    
"""
pop up an image showing the mouth with a blue rectangle
"""
def show(area): 
    cv.Rectangle(img,(area[0][0],area[0][1]),
                     (area[0][0]+area[0][2],area[0][1]+area[0][3]),
                    (255,0,0),2)
    cv.NamedWindow('Face Detection', cv.CV_WINDOW_NORMAL)
    cv.ShowImage('Face Detection', img) 
    cv.WaitKey()

"""
given an area to be cropped, crop() returns a cropped image
"""
def crop(area): 
    crop = img[area[0][1]:area[0][1] + area[0][3], area[0][0]:area[0][0]+area[0][2]] #img[y: y + h, x: x + w]
    return crop

"""
given a jpg image, vectorize the grayscale pixels to 
a (width * height, 1) np array
it is used to preprocess the data and transform it to feature space
"""   
def vectorize(pil_im):
    size = defaults.WIDTH, defaults.HEIGHT # (width, height)

    resized_im = pil_im.resize(size, Image.ANTIALIAS) # resize image
    
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    
    return oned_array
def getFrameFromWebCam(cap):
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return gray
def getFrame(camera):
	#Create a memory stream so photos doesn't need to be saved in a file
	stream = io.BytesIO()
	camera.capture(stream, format='jpeg')
	#Convert the picture into a numpy array
	buff = np.fromstring(stream.getvalue(), dtype=np.uint8)

	#Now creates an OpenCV image
	image = cv2.imdecode(buff, 1)
	return image

if __name__ == '__main__':

    ## Exists weigths
    if sys.argv[1]=="train":
        #Load training data
        lr=getTrainingRes()
    else:
        weigths = np.load(defaults.weigths_file)
        lr = logistic.Logistic(defaults.dim,weigths)
    """
    open webcam and capture images
    """
    if defaults.SHOW_PREVIEW:
        cv2.namedWindow("preview")
  
    print "\n--Starting to get images"
    if defaults.raspberry_plt:
        cam = picamera.PiCamera() 
        cam.resolution = (320, 240)
    else:
        cam = cv.CaptureFromCAM(0)
    
    while True:
        if defaults.raspberry_plt:
            frame = getFrame(cam)
            img = cv.fromarray(frame)
        else:
            frame = cv.QueryFrame(cam)
            img=frame
        
        if defaults.SHOW_PREVIEW:
            cv2.imshow("preview", frame)
        key = cv2.waitKey(1)

        if key == 27 or key == 1048603: # exit on ESC
            break
        #if key == 32 or 1048608 == key: # press space to save images
        #cv.SaveImage("webcam.jpg", cv.fromarray(frame))
        #img = cv.LoadImage("webcam.jpg") # input image
        
         
        storage = cv.CreateMemStorage()
        detectedFace = faceDetect.getFaces(img,storage,defaults.SHOW_RECTANGLES)
        if detectedFace:
            mouth = faceDetect.getMouth(img,detectedFace,storage,defaults.SHOW_RECTANGLES)
            detectedEye = faceDetect.getEyes(img,storage,defaults.SHOW_RECTANGLES)
            #if defaults.SHOW_PREVIEW:
            #    cv.ShowImage("preview", img)
            if mouth != 2: # did not return error
                    mouthimg = crop(mouth)
                    ##Comprobar rapidesa i min error
                    a = np.asarray(mouthimg) 
                    cv2_im = cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
                    
                    #pil_im2 = Image.fromarray(a) # img with low intensity
                    pil_im = Image.fromarray(cv2_im) # img with high intensity
                    
                    #cv.SaveImage("webcam-m.jpg", mouthimg)
                    # predict the captured emotion
                    result = lr.predict(vectorize(pil_im))
                    if result == 0:
                        print "you are smiling! :-) "
                    elif result==1:
                        print "you are neutral :-| "
                    elif result==2:
                        print "you are disgust :-$ "
                    else:
                        print "upss, I don't know... "
            else:
                    print "failed to detect mouth. 1 face only and good posture"
        else:
            print "failed to detect face."
    
    cv2.destroyWindow("preview")
