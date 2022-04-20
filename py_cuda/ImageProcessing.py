from PIL import Image
from itertools import chain
from time import perf_counter

#TODO: convert picture imae to numpy array
#TODO: convert masks to

masks = {"hori": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
        "vert": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
        "ldia": [[-2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
        "rdia": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]}

def seq_data(pxl_in, pxl_out, mask, r, c):
    sum = None

    for y in range(1, r):
        for x in range(1, c):
            sum = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    sum += pxl_in[(y+j), (x+i)] * mask[(j+1)][(i+1)]
            if sum > 255:
                sum = 255
            elif sum < 0:
                sum = 0
            pxl_out[x-1][y-1] = sum

def par_data(pxl_in, pxl_out, mask, r, c):
    sum = None
    rows = r
    cols = c
    for y in range(rows):
        for x in range(cols):
            sum = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    sum += pxl_in[(y+j), (x+i)] * mask[(j+1)][(i+1)]

            if sum > 255:
                sum = 255
            elif sum < 0:
                sum = 0
            pxl_out[x][y] = sum

class FileHandler:
    def __init__(self, src):
        self.main = src
        self.seq = src.split('.',1)[0] + '_seq_py.' + src.split('.',1)[1]
        self.par = src.split('.',1)[0] + '_par_py.' + src.split('.',1)[1]


class ImageHandlerSeq (FileHandler):
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
        seq_data(self.pixels, outBuff, mask, rows, cols)
        self.stop = perf_counter()
        
        return outBuff

    # TODO: parallel with CUDA
    def detectLinesPar(self, type):
        mask = masks[type]
        outBuff = [[None]*self.width for _ in range(self.height)]
        rows = self.height - 2
        cols = self.width - 2
        sum = None

        self.start1 = perf_counter()
        par_mask(self.pixels, outBuff, mask, rows, cols)
        self.stop = perf_counter()
        
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
