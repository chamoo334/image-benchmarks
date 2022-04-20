#ifndef IMAGEPROCESSING_CUH
#define IMAGEPROCESSING_CUH

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

using namespace std;

static const int _512by512_IMG_SIZE = 262144;
static const int BMP_COLOR_TABLE_SIZE = 1024;
static const int BMP_HEADER_SIZE = 54;
static const int WIDTH_OFFSET = 18;
static const int HEIGHT_OFFSET = 22;
static const int BITDEPTH_OFFSET = 28;
static const int GRAYSCALE_BITDEPTH = 8;
static const int MAX_COLOR = 255;
static const int MIN_COLOR = 0;
static const int WHITE = MAX_COLOR;
static const int BLACK = MIN_COLOR;
static const int RGB_COLS = 3;

class ImageProcessing
{
public:
    // typedef int mask_array[3][3];
    typedef int mask_array[3][3];

    ImageProcessing(
        char *_inImgName,
        char *_outImgName,
        int *_height,
        int *_width,
        int *_bitDepth,
        int *_size,
        bool *_isColor,
        unsigned char *_iHeader,
        unsigned char *_iColorTable,
        unsigned char **_inBuf,
        unsigned char **_outBuf);

    void readImage();
    void readColorImage();
    void writeImage();
    void copyImageData(unsigned char *_srcBuf, unsigned char *_destBuf, int bufSize);
    mask_array *setMask(int option);
    void detectLinesSeq(int option);
    void detectLinesPar(int option);

    virtual ~ImageProcessing();

protected:
private:
    const char *inImgName;
    const char *outImgName;
    int *height;
    int *width;
    int *bitDepth;
    int *size;
    bool *isColor;
    unsigned char *iHeader;
    unsigned char *iColorTable;
    unsigned char **inBuf;
    unsigned char **outBuf;
    mask_array hori_mask = {{-1, 2, -1},
                            {-1, 2, -1},
                            {-1, 2, -1}};
    mask_array vert_mask = {{-1, -1, -1},
                            {2, 2, 2},
                            {-1, -1, -1}};
    mask_array ldia_mask = {{2, -1, -1},
                            {-1, 2, -1},
                            {-1, -1, 2}};
    mask_array rdia_mask = {{-1, -1, 2},
                            {-1, 2, -1},
                            {2, -1, -1}};
};

ImageProcessing::ImageProcessing(char *_inImgName, char *_outImgName, int *_height,
                                 int *_width, int *_bitDepth, int *_size, bool *_isColor, unsigned char *_iHeader,
                                 unsigned char *_iColorTable, unsigned char **_inBuf, unsigned char **_outBuf)
{
    inImgName = _inImgName;
    outImgName = _outImgName;
    height = _height;
    width = _width;
    bitDepth = _bitDepth;
    size = _size;
    isColor = _isColor;
    iHeader = _iHeader;
    iColorTable = _iColorTable;
    inBuf = _inBuf;
    outBuf = _outBuf;
}

/***************************************************************
 * readImage
 * Parameters: n/a
 * Opens inImageName and extracts width, height, bitdepth
 * from image header as well as image's color table and data.
 ****************************************************************/
void ImageProcessing::readImage()
{
    FILE *streamIn = fopen(inImgName, "rb");

    if (streamIn == (FILE *)0)
    {
        cout << "Unable to open file" << endl;
        cout << streamIn << endl;
        exit(0);
    }

    for (int i = 0; i < BMP_HEADER_SIZE; i++)
    {
        iHeader[i] = getc(streamIn);
    }

    *width = *(int *)&iHeader[WIDTH_OFFSET];
    *height = *(int *)&iHeader[HEIGHT_OFFSET];
    *bitDepth = *(int *)&iHeader[BITDEPTH_OFFSET];
    *size = (*width) * (*height);

    if (*bitDepth <= GRAYSCALE_BITDEPTH)
    {
        fread(iColorTable, sizeof(unsigned char), BMP_COLOR_TABLE_SIZE, streamIn);
        *isColor = false;

        *inBuf = (unsigned char *)malloc((*size));
        *outBuf = (unsigned char *)malloc((*size));

        *inBuf = new unsigned char[*size];
        *outBuf = new unsigned char[*size];

        fread(*inBuf, sizeof(unsigned char), (*size), streamIn);
    }
    else
    {
        *isColor = true;
    }

    fclose(streamIn);
}

/***************************************************************
 * writeImage
 * Parameters: n/a
 * Opens outImageName and writes header info to file as well as
 * image's color table and data.
 ****************************************************************/
void ImageProcessing::writeImage()
{
    FILE *streamOut = fopen(outImgName, "wb");

    fwrite(iHeader, sizeof(unsigned char), BMP_HEADER_SIZE, streamOut);

    if (*bitDepth <= GRAYSCALE_BITDEPTH)
    {
        fwrite(iColorTable, sizeof(unsigned char), BMP_COLOR_TABLE_SIZE, streamOut);
    }

    fwrite(*outBuf, sizeof(unsigned char), (*size), streamOut);

    fclose(streamOut);
}

/**************************************************************************
 * copyImageData
 * Parameters: unsigned char *_srcBuf, unsigned char *_destBuf, int bufSize
 * Copies _srcBuf address to _destBuf for bufSize elements
 ***************************************************************************/
void ImageProcessing::copyImageData(unsigned char *_srcBuf, unsigned char *_destBuf, int bufSize)
{
    for (int i = 0; i < bufSize; i++)
    {
        _destBuf[i] = _srcBuf[i];
    }
}

/**************************************************************************
 * setMask
 * Parameters: int option
 * returns the appropriate mask for line detection based on option value
 ***************************************************************************/
ImageProcessing::mask_array *ImageProcessing::setMask(int option)
{

    switch (option)
    {
    case 1:
        return &hori_mask;
    case 2:
        return &vert_mask;
    case 3:
        return &ldia_mask;
    case 4:
        return &rdia_mask;
    default:
        cout << "Error processing option" << endl;
        exit(0);
    }
}

/***************************************************************
 * detectLinesSeq
 * Parameters: int option
 *
 ****************************************************************/
void ImageProcessing::detectLinesSeq(int option)
{
    int(*mask)[3][3] = setMask(option);

    int sum;
    int rows = (*height);
    int cols = (*width);
    for (int y = 1; y <= rows - 1; y++)
    {
        for (int x = 1; x <= cols; x++)
        {
            sum = 0;
            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    sum = sum + (*inBuf)[x + i + (long)(y + j) * cols] * (*mask)[i + 1][j + 1];
                }
            }
            if (sum > 255)
                sum = 255;
            if (sum < 0)
                sum = 0;

            (*outBuf)[x + (long)y * cols] = sum;
        }
    }
}

/***************************************************************
 * detectLinesPar
 * Parameters: int option
 *
 ****************************************************************/
void ImageProcessing::detectLinesPar(int option)
{
    int(*mask)[3][3] = setMask(option);

    int sum;
    int rows = (*height);
    int cols = (*width);
    for (int y = 1; y <= rows - 1; y++)
    {
        for (int x = 1; x <= cols; x++)
        {
            sum = 0;
            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    sum = sum + (*inBuf)[x + i + (long)(y + j) * cols] * (*mask)[i + 1][j + 1];
                }
            }
            if (sum > 255)
                sum = 255;
            if (sum < 0)
                sum = 0;

            (*outBuf)[x + (long)y * cols] = sum;
        }
    }
}

ImageProcessing::~ImageProcessing()
{
    // dtor
}

#endif