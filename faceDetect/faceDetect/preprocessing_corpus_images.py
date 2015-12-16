##INFO: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import cv2.cv as cv
from PIL import Image
import defaults
def normalize2(matrix):
    sum = np.sum(matrix)
    if sum > 0.:
        return matrix / float(sum)
    else:
        return matrix
def PCA(PCAInput):
    # The following mimics PCA::operator() implementation from OpenCV's
    # matmul.cpp() which is wrapped by Python cv2.PCACompute(). We can't
    # use PCACompute() though as it discards the eigenvalues.

    # Scrambled is faster for nVariables >> nObservations. Bitmask is 0 and
    # therefore default / redundant, but included to abide by online docs.
    covar, mean = cv2.calcCovarMatrix(PCAInput, cv2.cv.CV_COVAR_SCALE |
                                                cv2.cv.CV_COVAR_ROWS  |
                                                cv2.cv.CV_COVAR_SCRAMBLED)

    eVal, eVec = cv2.eigen(covar, computeEigenvectors=True)[1:]

    # Conversion + normalisation required due to 'scrambled' mode
    eVec = cv2.gemm(eVec, PCAInput - mean, 1, None, 0)
    # apply_along_axis() slices 1D rows, but normalize() returns 4x1 vectors
    eVec = np.apply_along_axis(lambda n: cv2.normalize(n).flat, 1, eVec)

    return mean,  eVec    
def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr
def histo(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
def equalize_hist(img):
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    #cv2.imshow('pepe',res)
    #cv2.waitKey(0)
    return equ 
def denoise(img):
    #dst = cv2.fastNlMeansDenoisingColored(src=img,dst=None,h=10,h_color=10,templ_win_size==7,win_size=21)
    #dst=cv2.fastNlMeansDenoising(src=img, dst=None,h=p,templateWindowSize = 7, searchWindowSize = 21 ) 
    plt.subplot(121),plt.imshow(img)
    i=122
    destA=None
    dst=None
    for p in xrange(1,10):
        for t in xrange(1,7):
            for s in xrange(1,21):
                destA=dst
                dst=cv2.fastNlMeansDenoising(src=img, dst=None,h=p,templateWindowSize = t, searchWindowSize = s ) 
                print "p:%d, t:%d, s:%d"%(p,t,s) 	
                if destA!=None: 
                    print str(dst==destA)
                cv2.imshow("pepe",dst)
                cv2.waitKey(2000)
                #plt.subplot(122),plt.imshow(dst)
                #plt.show()
                
                
    #plt.show()
'''
img = cv2.imread(sys.argv[1],0)
res=equalize_hist(img)
blur = cv2.bilateralFilter(res,9,75,75)
blur2 = cv2.bilateralFilter(img,9,75,75)
res2=equalize_hist(blur2)
cv2.imshow("from hist",blur)
cv2.imshow("from img",blur2)
cv2.imshow("from img to blur to hist",res2)
cv2.waitKey(0)
#denoise(res)'''
def preprocess_image(file_name):
    ## Grayscale
    im_gray = cv2.imread(file_name, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    ## Scale
    a = np.asarray(im_gray) 
    pil_im = Image.fromarray(a) # img with high intensity
    
    size = defaults.WIDTH, defaults.HEIGHT # (width, height)
    resized_im = pil_im.resize(size, Image.ANTIALIAS) # resize image
    #resized_im = pil_im.thumbnail(maxsize, PIL.Image.ANTIALIAS)
    im_array = np.array(resized_im) # convert to np array
    #cv2.imshow("resized_img",im_array)

    
    
    ## Normalize
    norm_img = normalize(im_array).astype('uint8')
    
    ## Histogram equalization
    res=equalize_hist(im_array)
    #cv2.imshow("equalize_hist",res)
    
    ## Noise filter and image segmentation
    kernel = np.ones((5,5),np.float32)/25
    denoised_img = cv2.filter2D(res,-1,kernel)
    #cv2.imshow("denoised",denoised_img)

    ## Thresold
    (thresh, im_bw) = cv2.threshold(denoised_img, 6, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("threshold",im_bw)
    (thresh, im_bw) = cv2.threshold(norm_img, 6, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("threshold2",im_bw)
    img_bw_inv = np.invert(im_bw)
    cv2.imshow("threshold3",img_bw_inv)
    #cv2.waitKey()
    
    #img_vector = denoised_img.reshape(1, size[0] * size[1])
    img_vector = img_bw_inv.reshape(1, size[0] * size[1])
    return img_vector


import os

matrix_test = None
label=[]
count_label=0
matrix_label = []
for directory in os.listdir(sys.argv[1]):
    label.append(directory)
    #print "generating class..: ",label[-1]
    for image in os.listdir(os.path.join(sys.argv[1],directory)):
        imgvector = preprocess_image(os.path.join(os.path.join(sys.argv[1],directory), image))
        try:
            matrix_test = np.vstack((matrix_test, imgvector))
            matrix_label.append(count_label)
        except:
            matrix_test = imgvector
    count_label+=1
print
print "Matrix builded"
for c in xrange(count_label):
    print "class %d:%s with %d samples"%(c,label[c],len([x for x in matrix_label if x==c]))

# PCA
#mean, eigenvectors = cv.PCACompute(matrix_test, np.mean(matrix_test, axis=0))
mean,eigenvectors = PCA(matrix_test)
#print "mean", mean
#print "eig", eigenvectors[0][0]
#print eigenvectors  
print "len-eigenvectors",len(eigenvectors),len(eigenvectors[0])
eign = eigenvectors[2]
size = defaults.HEIGHT, defaults.WIDTH
img_vector = eign.reshape(size[0], size[1])

cv2.imshow("eigen",img_vector)
cv2.waitKey()
quit()
#NORM_INF, NORM_L1, or NORM_L2
# Where \alpha defines the width of the input intensity range, and \beta defines the intensity around which the range is centered.[2]
#src, dst,alpha, beta, normType, dtype=-1:still same format
#res=cv2.normalize(img,None, 0, 255, cv2.NORM_INF,-1)





