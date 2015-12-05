"""
input: a loaded image; 
output: [[x,y],[width,height]] of the detected mouth area
"""
from time import sleep
import cv2.cv as cv
import cv2
import defaults
def showRectangle(imcolor, obj, color):
    cv.Rectangle( imcolor, (obj[0][0],obj[0][1]), (obj[0][0]+obj[0][2],obj[0][1]+obj[0][3]), color)
    cv.NamedWindow('Face Detection', cv.CV_WINDOW_NORMAL)
    cv.ShowImage('Face Detection', imcolor)
   
    
    
def getFaces(imcolor,storage=None,DEBUG=None):
    if not storage:
        storage = cv.CreateMemStorage()
    # loading the classifiers
    haarFace = cv.Load(defaults.haar_face)
    
    # running the classifiers
    detectedFace = cv.HaarDetectObjects(imcolor, haarFace, storage)
    
    # draw a green rectangle where the face is detected
    if DEBUG and detectedFace:
        for face in detectedFace:
            showRectangle(imcolor,face,cv.RGB(155, 55, 200))

    return detectedFace

def getEyes(imcolor,storage=None,DEBUG=None):
    if not storage:
        storage = cv.CreateMemStorage()
            
    # loading the classifiers
    haarEyes = cv.Load(defaults.haar_eye)

    # running the classifiers
    detectedEye = cv.HaarDetectObjects(imcolor, haarEyes, storage)
    
    # draw a green rectangle where the face is detected
    if DEBUG and detectedEye:
        for eye in detectedEye:
            showRectangle(imcolor,eye,cv.RGB(155, 10, 110))

    return detectedEye
    
def getMouth(imcolor,detectedFace,storage=None,DEBUG=None):
    if not storage:
        storage = cv.CreateMemStorage()
    # loading the classifiers
    haarMouth = cv.Load(defaults.haar_mouth)    
    # running the classifiers
    detectedMouth = cv.HaarDetectObjects(imcolor, haarMouth, storage)

    # FACE: find the largest detected face as detected face
    maxFaceSize = 0
    maxFace = 0
    if detectedFace:
        for face in detectedFace: # face: [0][0]: x; [0][1]: y; [0][2]: width; [0][3]: height 
            if face[0][3]* face[0][2] > maxFaceSize:
                maxFaceSize = face[0][3]* face[0][2]
                maxFace = face
  
    ##Mouth
    def mouth_in_lower_face(mouth,face):
        # if the mouth is in the lower 2/5 of the face 
        # and the lower edge of mouth is above that of the face
        # and the horizontal center of the mouth is the center of the face
        if (mouth[0][1] > face[0][1] + face[0][3] * 3 / float(5) and mouth[0][1] + mouth[0][3] < face[0][1] + face[0][3] and abs((mouth[0][0] + mouth[0][2] / float(2)) - (face[0][0] + face[0][2] / float(2))) < face[0][2] / float(10)):
            return True
        else:
            return False

    # FILTER MOUTH
    filteredMouth = []
    if detectedMouth:
        for mouth in detectedMouth:
            if mouth_in_lower_face(mouth,maxFace):
                filteredMouth.append(mouth) 
  
    maxMouthSize = 0
    
    for mouth in filteredMouth:
        if mouth[0][3]* mouth[0][2] > maxMouthSize:
            maxMouthSize = mouth[0][3]* mouth[0][2]
            maxMouth = mouth
      
    if DEBUG and maxMouthSize>0:
        showRectangle(imcolor,maxMouth,cv.RGB(20, 50, 100))
    try:
        return maxMouth
    except UnboundLocalError:
        return 2

