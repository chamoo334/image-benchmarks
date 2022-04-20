import sys, cv2 as cv, numpy as np
import ImageProcessing as imgProc
from time import perf_counter
from matplotlib import pyplot as plt
from numba import cuda

def seq_run(imgFile):
    img = imgProc.ImageHandlerSeq(imgFile)
    lineData = img.detectLinesSeq('hori')
    newImg = imgProc.ImageHandlerSeq(None, img.data.mode, img.data.size)
    newImg.updatePixels(lineData)
    newImg.saveImage(img.seq)
    return img.data.size

def par_run(imgFile):
    return 0, 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("missing argument")

    elif sys.argv[2] == "seq":
        seq_size = seq_run(sys.argv[1])
        print(seq_size[0] * seq_size[1])
    else:
        print("finish python parallel")
        # par_size = par_run(sys.argv[1])
        # print(par_size[0] * par_size[1])