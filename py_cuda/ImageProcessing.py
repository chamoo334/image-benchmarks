from PIL import Image
from itertools import chain
from time import perf_counter
from numba import cuda
import numpy as np
# import sys, cv2 as cv, numpy as np

masks = {"hori": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
        "vert": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
        "ldia": [[-2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
        "rdia": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]}

@cuda.jit
def my_kernel(inArr, outArr, rows, cols, mask):
    """
    Code for kernel.
    """
    for y in range(1, rows):
        for x in range(1, cols):
            sum = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    sum += inArr[((y+j) * rows) + (x+i)] * mask[(i + 1)*3 + (j + 1)]
            if sum > 255:
                sum = 255
            elif sum < 0:
                sum = 0
            outArr[((y-1)*rows) + (x-1)] = sum

class FileHandler:
    def __init__(self, src):
        self.main = src
        self.seq = src.split('.',1)[0] + '_seq_py.' + src.split('.',1)[1]
        self.par = src.split('.',1)[0] + '_par_py.' + src.split('.',1)[1]


class ImageHandler (FileHandler):
    def __init__(self, src=None, newMode=None, newSize=None):
        if src is not None:
            super().__init__(src)
            self.start1 = None
            self.start2 = None
            self.stop = None
            self.data = self.openMainImage()
            self.pixels = self.setPixelData()
            self.width = self.setWidth()
            self.height = self.setHeight()
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
        mask = masks[type]
        outBuff = [[None]*self.width for _ in range(self.height)]
        rows = self.height - 1
        cols = self.width - 1
        sum = None

        self.start1 = perf_counter()

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

        self.stop = perf_counter()
        
        return outBuff

    # TODO: parallel with CUDA
    def detectLinesPar(self, type):
        h_in = np.array(self.data).reshape([self.width*self.height])
        h_out = np.empty_like(h_in)  # TODO: consider zeros or initialize to NULL
        mask = np.array(masks[type]).reshape([9])

        self.closeMainImage()

        # rows = self.height - 2
        # cols = self.width - 2

        # Set the number of threads in a block
        threadsperblock = 1

        # Calculate the number of thread blocks in the grid
        blockspergrid = 1
        rows = self.height - 1
        cols = self.width - 1
        sum = None

        self.start1 = perf_counter()

        my_kernel[blockspergrid, threadsperblock](h_in, h_out, rows, cols, mask)

        # for y in range(1, rows):
        #     for x in range(1, cols):
        #         sum = 0
        #         for i in range(-1, 2):
        #             for j in range(-1, 2):
        #                 sum += h_in[((y+j) * rows) + (x+i)] * mask[(i + 1)*3 + (j + 1)]
        #         if sum > 255:
        #             sum = 255
        #         elif sum < 0:
        #             sum = 0
        #         h_out[((y-1)*rows) + (x-1)] = sum
        
        self.stop = perf_counter()
        
        outBuff = h_out.reshape([self.width, self.height])
        return outBuff

    # TODO: parallel with openCL / pyOpenCL
    # TODO: verify flattened vs newData
    def updatePixels(self, newData):
        if len(newData) != (self.data.size[0] * self.data.size[1]):
            flattened = list(chain.from_iterable(newData))
            newData = list(chain.from_iterable(newData))
        self.data.putdata(newData)

    def saveImage(self, newPath):
        self.data.save(newPath)
