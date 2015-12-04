import cv2.cv as cv
def getEyes(imcolor,storage=None,DEBUG=None):
    # loading the classifiers
    haarEyes = cv.Load('haarcascade_eye.xml')
    if not storage:
        storage = cv.CreateMemStorage()
    # running the classifiers
    detectedEye = cv.HaarDetectObjects(imcolor, haarEyes, storage)
    
    # draw a green rectangle where the face is detected
    if DEBUG and detectedEye:
        for eye in detectedEye:
            cv.Rectangle(imcolor,(eye[0][0],eye[0][1]),
                   (eye[0][0]+face[0][2],eye[0][1]+eye[0][3]),
                   cv.RGB(155, 255, 25),2)

    return detectedEye
                   
