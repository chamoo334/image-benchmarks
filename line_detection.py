import sys, cv2 as cv, numpy as np
import ImageProcessing as imgProc
from time import perf_counter
from matplotlib import pyplot as plt


def seq_run(imgFile):
    img = imgProc.ImageHandlerSeq(imgFile)
    lineData = img.detectLines('hori')
    newImg = imgProc.ImageHandlerSeq(None, img.data.mode, img.data.size)
    newImg.updatePixels(lineData)
    newImg.saveImage(img.seq)
    return img.start, img.data.size


if __name__ == '__main__':
    t1_start, t1_size = seq_run(sys.argv[1])
    t1_stop = perf_counter()
    t1_elapsed = t1_stop-t1_start
    t1_pixelsPerSec = (t1_size[0] * t1_size[1])/t1_elapsed
    print(t1_elapsed, t1_pixelsPerSec)
