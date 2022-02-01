#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

static const int  _512by512_IMG_SIZE = 262144;
static const int BMP_COLOR_TABLE_SIZE = 1024;
static const int BMP_HEADER_SIZE = 54;
static const int WIDTH_OFFSET = 18;
static const int HEIGHT_OFFSET = 22;
static const int BITDEPTH_OFFSET = 28;
static const int GRAYSCALE_BITDEPTH  = 8;
static const int MAX_COLOR = 255;
static const int MIN_COLOR = 0;
static const int WHITE = MAX_COLOR;
static const int BLACK = MIN_COLOR;
static const int RGB_COLS = 3;

class ImageProcessing
{
    public:
        ImageProcessing(
                        char *_inImgName,
                        char *_outImgName,
                        int * _height,
                        int * _width,
                        int * _bitDepth,
                        int * _size,
                        bool * _isColor,
                        unsigned char * _iHeader,
                        unsigned char * _iColorTable,
                        unsigned char ** _inBuf,
                        unsigned char ** _outBuf);

    void readImage();
    void readColorImage();
    void writeImage();
    void copyImageData(unsigned char *_srcBuf, unsigned char *_destBuf, int bufSize);
    void detectLines(int option);

    virtual ~ImageProcessing();

protected:

private:
    const char *inImgName;
    const char *outImgName;
    int * height;
    int * width;
    int * bitDepth;
    int * size;
    bool * isColor;
    unsigned char * iHeader;
    unsigned char * iColorTable;
    unsigned char ** inBuf;
    unsigned char ** outBuf;
};

#endif