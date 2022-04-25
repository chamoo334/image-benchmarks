import sys
import ImageProcessing as imgProc
# from time import perf_counter
# from matplotlib import pyplot as plt
# from numba import cuda

def seq_run(imgFile):
    img = imgProc.ImageHandler(imgFile)
    img.closeMainImage()
    lineData = img.detectLinesSeq('hori')
    newImg = imgProc.ImageHandler(None, img.data.mode, img.data.size)
    newImg.updatePixels(lineData)
    newImg.saveImage(img.seq)
    return img.data.size

def par_run(imgFile):
    img = imgProc.ImageHandler(imgFile)
    lineData = img.detectLinesPar('hori')
    newImg = imgProc.ImageHandler(None, img.data.mode, img.data.size)
    newImg.updatePixels(lineData)
    newImg.saveImage(img.par)
    return img.data.size


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("missing argument")

    elif sys.argv[2] == "seq":
        seq_size = seq_run(sys.argv[1])
        print(seq_size[0] * seq_size[1])
    else:
        par_size = par_run(sys.argv[1])
        print(par_size[0] * par_size[1])