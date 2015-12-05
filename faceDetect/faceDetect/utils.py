import os
def listImagePathFromDir(path):
    file_names=[]
    for dirname, dirnames, filenames in os.walk(path):
        # print path to all filenames.
        for filename in filenames:
            if filename[-3:len(filename)]=='pgm':
                file_names.append(os.path.join(dirname, filename))

    return file_names
