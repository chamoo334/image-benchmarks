from PIL import Image
from itertools import chain
from time import monotonic_ns
from numba import cuda
import numpy as np
# import sys, cv2 as cv, numpy as np

masks = {"hori": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
        "vert": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
        "ldia": [[-2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
        "rdia": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]}

@cuda.jit
def gpuLineDetect(rows, cols, mask, inImg, outImg):
    """
    Code for kernel.
    """
    elRow = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1
    elCol = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1

    if elRow <= rows and elCol <= cols:
        sum = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                sum = sum + inImg[elCol + i + (elRow + j) * cols] * mask[(i + 1) * 3 + (j + 1)]
        if sum > 255:
            sum = 255
        if sum < 0:
            sum = 0
        outImg[elCol + (elRow * cols)] = sum


def elapsedTimeMS(first, last):
    return (last - first)/1000000

def calcPixelsPerSecond(first, last, numPixels):
    elTime = elapsedTimeMS(first, last)
    return numPixels/elTime

class FileHandler:
    def __init__(self, src):
        self.main = src
        self.seq = src.split('.',1)[0] + '_seq_py.' + src.split('.',1)[1]
        self.par = src.split('.',1)[0] + '_par_py.' + src.split('.',1)[1]


class ImageHandler (FileHandler):
    def __init__(self, src=None, newMode=None, newSize=None):
        if src is not None:
            super().__init__(src)
            self.data = self.openMainImage()
            self.pixels = self.setPixelData()
            self.width = self.setWidth()
            self.height = self.setHeight()
            self.size = self.width * self.height
        elif newMode is not None and newSize is not None:
            self.data = self.createNewImage(newMode, newSize)
        else:
            print('Must provide an image or mode and size')
            quit()


    def openMainImage(self):

        try:
            imgOpen = Image.open(self.main)
            
            if imgOpen.mode != 'L' and imgOpen.mode != 1:
                print('Unable to proceed with RGB image')
                quit()

            return imgOpen
        except:
            print("Unable to open/process image")
            quit()

    def closeMainImage(self):
        self.data.close()

    def createNewImage(self, newm, news):
        return Image.new(newm, news)
    
    def setPixelData(self):
        return self.data.load()

    def setWidth(self):
        return self.data.size[0]

    def setHeight(self):
        return self.data.size[1]

    def detectLinesSeq(self, type):
        self.closeMainImage()
        mask = masks[type]
        outBuff = [[None]*self.width for _ in range(self.height)]
        rows = self.height - 1
        cols = self.width - 1
        sum = None

        start = monotonic_ns()

        for y in range(1, rows):
            for x in range(1, cols):
                sum = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        sum += self.pixels[(y+j), (x+i)] * mask[(j+1)][(i+1)]
                if sum > 255:
                    sum = 255
                elif sum < 0:
                    sum = 0
                outBuff[x-1][y-1] = sum

        end = monotonic_ns()
        pixelsPerMs = calcPixelsPerSecond(start, end, self.size)

        print(pixelsPerMs)
        
        
        return outBuff

    # TODO: parallel with CUDA
    def detectLinesPar(self, type, threads1d):
        
        blocks1d = (self.size + threads1d - 1) // threads1d
        rows = self.height
        cols = self.width
        blocks1d = (self.size + threads1d - 1) // threads1d
        threads2d = threads1d, threads1d
        blocks2d = blocks1d, blocks1d
        
        start1 = monotonic_ns()
        
        # reshape data and allocate memory
        h_in = np.array(self.data).reshape([self.width*self.height])
        h_out = np.empty_like(h_in)  # TODO: consider zeros or initialize to NULL
        mask = np.array(masks[type]).reshape([9])
        self.closeMainImage()


        start2 = monotonic_ns()
        gpuLineDetect[blocks2d, threads2d](rows, cols, mask, h_in, h_out)
        end2 = monotonic_ns()
        
        outBuff = h_out.reshape([self.width, self.height])
        end1 = monotonic_ns()

        pixelsPerMsTotal = calcPixelsPerSecond(start1, end1, self.size)
        pixelsPerMsGPU = calcPixelsPerSecond(start2, end2, self.size)
        pixelsDif = pixelsPerMsTotal - pixelsPerMsGPU
        # blocks, threads, totalTime, gpuTime, dif
        print("{%d,%d};{%d,%d};%.6f;%.6f;%.6f" % (blocks1d,blocks1d,threads1d,threads1d,pixelsPerMsTotal,pixelsPerMsGPU,pixelsDif))

        return outBuff

    # TODO: parallel with openCL / pyOpenCL

    def updatePixels(self, newData):
        if len(newData) != (self.data.size[0] * self.data.size[1]):
            flattened = list(chain.from_iterable(newData))
            newData = list(chain.from_iterable(newData))
        self.data.putdata(newData)

    def saveImage(self, newPath):
        self.data.save(newPath)
