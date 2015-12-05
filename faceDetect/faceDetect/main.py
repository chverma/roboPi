
import sys
#sys.path.append('/home/chverma/Descargas/opencv-3.0.0')

import cv2.cv as cv
import io
import cv2
import time
from PIL import Image
import numpy as np

import logistic
import faceDetect
import picamera
from time import sleep
from loadTrainingData import getTrainingRes
import defaults
    
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
    camera = picamera.PiCamera() 
    camera.resolution = (320, 240)
	
    print "\n\n\n\n\nStarting to get images"

    while True:
        
        frame = getFrame(camera)
        if defaults.SHOW_PREVIEW:
            cv2.imshow("preview", frame)
        key = cv2.waitKey(1)

        if key == 27 or key == 1048603: # exit on ESC
            break
        #if key == 32 or 1048608 == key: # press space to save images
        #cv.SaveImage("webcam.jpg", cv.fromarray(frame))
        #img = cv.LoadImage("webcam.jpg") # input image
        img = cv.fromarray(frame)
        
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
                    if result == 1:
                        print "you are smiling! :-) "
                    else:
                        print "you are not smiling :-| "
            else:
                    print "failed to detect mouth. 1 face only and good posture"
        else:
            print "failed to detect face."
    
    cv2.destroyWindow("preview")
