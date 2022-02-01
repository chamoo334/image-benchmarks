from PIL import Image
from itertools import chain
from time import perf_counter

masks = {"hori": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
        "vert": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
        "ldia": [[-2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
        "rdia": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]}

class FileHandler:
    def __init__(self, src):
        self.main = src
        self.seq = src.split('.',1)[0] + '_seq_py.' + src.split('.',1)[1]
        self.par = src.split('.',1)[0] + '_par_py.' + src.split('.',1)[1]


class ImageHandlerSeq (FileHandler):
    def __init__(self, src=None, newMode=None, newSize=None):
        if src is not None:
            super().__init__(src)
            self.start = perf_counter()
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

    def detectLines(self, type):
        mask = masks[type]
        outBuff = [[None]*self.width for _ in range(self.height)]
        rows = self.height - 2
        cols = self.width - 2
        sum = None

        for y in range(rows):
            for x in range(cols):
                sum = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        sum += self.pixels[(y+j), (x+i)] * mask[(j+1)][(i+1)]

                if sum > 255:
                    sum = 255
                elif sum < 0:
                    sum = 0
                outBuff[x][y] = sum
        
        return outBuff

    def updatePixels(self, newData):
        if len(newData) != (self.data.size[0] * self.data.size[1]):
            flattened = list(chain.from_iterable(newData))
            newData = list(chain.from_iterable(newData))
        self.data.putdata(newData)

    def saveImage(self, newPath):
        self.data.save(newPath)
