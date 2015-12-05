##########DEFAULTS

SHOW_PREVIEW=False # To show real time video
SHOW_RECTANGLES=False # To show preview image with objects detected

WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector

smile_csv   = '../data/csv/smiles.csv'
neutral_csv = '../data/csv/neutral.csv'

neutral_imgs = '../data/mouths/smile/'
smile_imgs = '../data/mouths/neutral/'


haar_face = '../data/haarcascade/haarcascade_frontalface_default.xml'
haar_mouth = '../data/haarcascade/haarcascade_mouth.xml'
haar_eye = '../data/haarcascade/haarcascade_eye.xml'

weigths_file = 'weights.npy'
