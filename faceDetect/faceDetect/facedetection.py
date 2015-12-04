import cv2.cv as cv

def getFaces(imcolor,,storage=None,DEBUG=None):
    # loading the classifiers
    haarFace = cv.Load('haarcascade_frontalface_default.xml')
    if not storage:
        storage = cv.CreateMemStorage()
    # running the classifiers
    detectedFace = cv.HaarDetectObjects(imcolor, haarFace, storage)
    
    # draw a green rectangle where the face is detected
    if DEBUG and detectedFace:
         for face in detectedFace:
              cv.Rectangle(imcolor,(face[0][0],face[0][1]),
                           (face[0][0]+face[0][2],face[0][1]+face[0][3]),
                           cv.RGB(155, 55, 200,2)
    return detectedFace
                   
