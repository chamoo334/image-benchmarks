import sys
import ImageProcessing as imgProc

NUM_THREADS = 1
NUM_BLOCKS = 1

def createSaveNewImg(img, imgMode, imgSize, imgData, newImgFile):
    newImg = imgProc.ImageHandler(None, imgMode, imgSize)
    newImg.updatePixels(imgData)
    newImg.saveImage(newImgFile)


def seq_run(imgFile):
    img = imgProc.ImageHandler(imgFile)
    lineData = img.detectLinesSeq('hori')
    createSaveNewImg(img, img.data.mode, img.data.size, lineData, img.seq)


def par_run(imgFile):
    img = imgProc.ImageHandler(imgFile)
    lineData = img.detectLinesPar('hori', NUM_THREADS)
    createSaveNewImg(img, img.data.mode, img.data.size, lineData, img.par)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("missing argument")

    elif sys.argv[2] == "seq":
        seq_run(sys.argv[1])
    else:
        NUM_THREADS = int(sys.argv[3])
        par_run(sys.argv[1])