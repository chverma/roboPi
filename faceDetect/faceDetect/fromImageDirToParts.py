
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
import utils    
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
    

if __name__ == '__main__':

    print "\n\n\n\n\nStarting to get images"

    list_files = utils.listImagePathFromDir(sys.argv[1])
    cv.NamedWindow('Face Detection', cv.CV_WINDOW_NORMAL)
    mouth_ok=0
    eye_ok=0
    total_processed=0
    list_length=len(list_files)
    for file_name in list_files:
        
        img = Image.open(file_name)
        im_array = np.array(img)
        img = cv.fromarray(im_array)
        total_processed+=1
        storage = cv.CreateMemStorage()
        detectedFace = faceDetect.getFaces(img,storage,defaults.SHOW_RECTANGLES)

        if detectedFace:
            detectedMouth = faceDetect.getMouth(img,detectedFace,storage,defaults.SHOW_RECTANGLES)
            #detectedMouth = faceDetect.getMouth2(img,detectedFace,storage,defaults.SHOW_RECTANGLES)
            detectedEye = faceDetect.getEyes(img,storage,defaults.SHOW_RECTANGLES)
            
            #if defaults.SHOW_PREVIEW:
            #    cv.ShowImage('Face Detection', mouth) 
            #    cv2.waitKey()
            if detectedMouth != 2: # did not return error
                mouthimg = crop(detectedMouth)
                mouth_ok+=1
                cv.SaveImage(sys.argv[2]+"/mouths/"+file_name[len(sys.argv[1]):-4]+"-m.pgm", mouthimg)
            else:
                print "failed to detect mouth. 1 face only and good posture"
                    
            if detectedEye:
                print "tamanty de ulls:",len(detectedEye)
                print detectedEye
                i=0
                for eye in detectedEye:
                    eye_img=crop(eye)
                    eye_ok+=1
                    cv.SaveImage(sys.argv[2]+"/eyes/"+file_name[len(sys.argv[1]):-4]+"-"+str(i)+"-e.pgm", eye_img)
                    i+=1
            else:
                print "failed to detect eyes. 1 face only and good posture"
            print "Progression:",(float(total_processed)/list_length)*100
        else:
            print "failed to detect face."
            
    print "Final mouth_ok: %d; bad:%d: error_percent:%f per cent"%( mouth_ok, list_length-mouth_ok, float(((list_length-mouth_ok))/list_length)*100)
    print "Final eye_ok: %d; bad:%d: error_percent:%f per cent"%( eye_ok, list_length-eye_ok, float(((list_length-eye_ok))/list_length)*100)
    cv2.destroyWindow("preview")
