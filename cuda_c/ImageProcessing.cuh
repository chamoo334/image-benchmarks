#ifndef IMAGEPROCESSING_CUH
#define IMAGEPROCESSING_CUH

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <cmath>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "cuda.h"

using namespace std;
using namespace std::chrono;

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

__global__ void gpuLineDetect(int perThread, int perBlock, int rows, int cols, int size, int *mask, unsigned char *inImg, unsigned char *outImg);
long elapsedTime(steady_clock::time_point first, steady_clock::time_point last);

class ImageProcessing
{
public:
    typedef int mask_array[3][3];

    ImageProcessing(
        char *_inImgName,
        char *_outImgName,
        int *_height,
        int *_width,
        int *_bitDepth,
        int *_size,
        bool *_isColor,
        bool *_usingGPU,
        unsigned char *_iHeader,
        unsigned char *_iColorTable,
        unsigned char **_inBuf,
        unsigned char **_outBuf);

    int readImage();
    void readColorImage();
    void writeImage();
    void copyImageData(unsigned char *_srcBuf, unsigned char *_destBuf, int bufSize);
    mask_array *setMask(int option);
    void detectLines(int option, int threads_1d, int blocks_1d);
    void detectLinesPar(mask_array mask, int threads_1d, int blocks_1d);
    void verifyLimitsAndRun(dim3 threadsDim, dim3 blocksDim, int maxThreads, int maxBlocks, mask_array mask);

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
    bool *usingGPU;
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
                                 int *_width, int *_bitDepth, int *_size, bool *_isColor, bool *_usingGPU, unsigned char *_iHeader,
                                 unsigned char *_iColorTable, unsigned char **_inBuf, unsigned char **_outBuf)
{
    inImgName = _inImgName;
    outImgName = _outImgName;
    height = _height;
    width = _width;
    bitDepth = _bitDepth;
    size = _size;
    isColor = _isColor;
    usingGPU = _usingGPU;
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
int ImageProcessing::readImage()
{
    FILE *streamIn = fopen(inImgName, "rb");

    if (streamIn == (FILE *)0)
    {
        return 3;
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
        *outBuf = (unsigned char *)malloc(sizeof(unsigned char*) * (*size));

        fread(*inBuf, sizeof(unsigned char), (*size), streamIn);
    }
    else
    {
        *isColor = true;
        return 4;
    }

    fclose(streamIn);
    return 0;
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
        fprintf( stderr, "Error processing option\n" );
        exit(0);
    }
}

/***************************************************************
 * detectLines
 * Parameters: int option
 * Hough transform to detect lines based on option.
 ****************************************************************/
void ImageProcessing::detectLines(int option, int threads_1d = 0, int blocks_1d = 0)
{
    int(*mask)[3][3] = setMask(option);

    if (*usingGPU){
        detectLinesPar(*mask, threads_1d, blocks_1d);
    } else {
        int sum;
        int rows = (*height);
        int cols = (*width);
        steady_clock::time_point start, end;
        long totalTime;
        float pixelsPerMS;
        *outBuf = new unsigned char[*size];

        start = steady_clock::now();

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

        end = steady_clock::now();
        totalTime = elapsedTime(start, end);
        pixelsPerMS = (float)(*size / totalTime);
        fprintf(stdout, "%.6f", pixelsPerMS);
    }
}

/***************************************************************
 * detectLinesPar
 * Parameters: mask_array mask, int threads_1d, int blocks_1D
 * 
 ****************************************************************/
void ImageProcessing::detectLinesPar(mask_array mask, int threads_1d, int blocks_1d)
{

    // TODO: verify compute capability and set limits approrpiately
    int maxThreadsPerBlock = 1024;
    int maxBlocksPerGrid = 65535;

    // TODO: consider/switch to grid offset to avoid predetermined perThread and perBlock
    // int rows = (*height);
    // int cols = (*width);
    dim3 threadsDim(threads_1d, 1, 1);
    dim3 threadsDim2D(threads_1d, threads_1d, 1);
    // dim3 threadsDim3D(threads_1d, threads_1d, threads_1d);
    dim3 blocksDim(blocks_1d, 1, 1);
    dim3 blocksDim2D(blocks_1d, blocks_1d, 1);
    // dim3 blocksDim3D(blocks_1d, blocks_1d, blocks_1d);

    verifyLimitsAndRun(threadsDim, blocksDim, maxThreadsPerBlock, maxBlocksPerGrid, mask);
    verifyLimitsAndRun(threadsDim2D, blocksDim2D, maxThreadsPerBlock, maxBlocksPerGrid, mask);
}

/***************************************************************
 * verifyLimitsAndRun
 * Parameters: dim3 threadsDim, dim3 blocksDim, int maxThreads, 
 *             int maxBlocks, mask_array mask
 * Verifies thread & block dimensions are within GPU's limits and 
 * prepares data required for line detection using GPULineDetect
 ****************************************************************/
void ImageProcessing::verifyLimitsAndRun(dim3 threadsDim, dim3 blocksDim, int maxThreads, int maxBlocks, mask_array mask)
{
    int tDimTotal = threadsDim.x * threadsDim.y * threadsDim.z;
    int bDimTotal = blocksDim.x * blocksDim.y * blocksDim.z;

    if (tDimTotal > maxThreads){
        fprintf(stdout, "{%d,%d,%d};{%d,%d,%d};%d;%d;%.d;%.d\n",blocksDim.x, blocksDim.y, blocksDim.z, threadsDim.x, threadsDim.y, threadsDim.z, -1, -1, -1, -1);
        return;
    } else if (bDimTotal > maxBlocks){
        fprintf(stdout, "{%d,%d,%d};{%d,%d,%d};%d;%d;%.d;%.d\n",blocksDim.x, blocksDim.y, blocksDim.z, threadsDim.x, threadsDim.y, threadsDim.z, -2, -2, -2, -2);
        return;
    }

    // fprintf(stdout, "{%d,%d,%d};{%d,%d,%d};%d;%d;%d;%d\n",blocksDim.x, blocksDim.y, blocksDim.z, threadsDim.x, threadsDim.y, threadsDim.z, 0, 0, 0, 0);

    int pixelsPerThread = ceil((double)(*size) / (double)(tDimTotal * bDimTotal));
    int pixelsPerBlock = pixelsPerThread * tDimTotal;

    int byte_size_img = sizeof(unsigned char*) * (*size);
    int byte_size_mask = sizeof(int*) * 9;
    int * d_mask;
    unsigned char *d_in_img, *d_out_img;
    steady_clock::time_point start1, start2, end1, end2;
    long  gpuTime, totalTime;
    float pixelsPerMSTotal, pixelsPerMSGPU;

    // allocate & set memory & time operations
    start1 = steady_clock::now();
    cudaMalloc((void**)&d_mask, byte_size_mask);
	cudaMalloc((void**)&d_in_img, byte_size_img);
	cudaMalloc((void**)&d_out_img, byte_size_img);
    cudaMemcpy(d_mask, *mask, byte_size_mask, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_img, (*inBuf), byte_size_img, cudaMemcpyHostToDevice);

    // gpuLinedetect and synchronize
    start2 = steady_clock::now();

    gpuLineDetect<<<blocksDim, threadsDim>>>(pixelsPerThread, pixelsPerBlock, (*height), (*width), *size, d_mask, d_in_img, d_out_img);


    cudaDeviceSynchronize();
    end2 = steady_clock::now();
    cudaMemcpy((*outBuf), d_out_img, byte_size_img, cudaMemcpyDeviceToHost);

    // free and reset
    cudaFree(d_mask);
    cudaFree(d_in_img);
    cudaFree(d_out_img);
    cudaDeviceReset();

    end1 = steady_clock::now();

    elapsedTime(start1, end1) == 0 ? totalTime = 1 : totalTime = elapsedTime(start1, end1);
    elapsedTime(start2, end2) == 0 ? gpuTime = 1 : gpuTime = elapsedTime(start2, end2);
    pixelsPerMSTotal = (float)(*size / totalTime);
    pixelsPerMSGPU = (*size / gpuTime);

    // blocks, threads, pixelsPerThread, pixelsPerBlock, totalTime, gpuTime
    fprintf(stdout, "{%d,%d,%d};{%d,%d,%d};%d;%d;%.6f;%.6f::",blocksDim.x, blocksDim.y, blocksDim.z, threadsDim.x, threadsDim.y, threadsDim.z, 
    pixelsPerThread,pixelsPerBlock, pixelsPerMSTotal, pixelsPerMSGPU);
}

ImageProcessing::~ImageProcessing()
{
    // DTOR
}

/***************************************************************
 * elapsedTime
 * Parameters: steady_clock::time_point first, steady_clock::time_point last
 * Determine time elapsed between two parameters and returns the value in ms
 ****************************************************************/
long elapsedTime(steady_clock::time_point first, steady_clock::time_point last)
{
  
//   steady_clock::duration time_span = last - first;
//   milliseconds ms = duration_cast< milliseconds >(time_span);
  duration<float> fs = last - first;
  milliseconds ms2 = duration_cast< milliseconds >(fs);
  return ms2.count();
}


/***************************************************************
 * gpuLineDetect
 * Parameters: int perThread, int rows, int cols, int size, 
 *      int *mask, unsigned char *inImg, unsigned char *outImg
 * Each thread accumalates edge points for pixels perThread
 ****************************************************************/
__global__ void gpuLineDetect(int perThread, int perBlock, int rows, int cols, int size, int *mask, unsigned char *inImg, unsigned char *outImg)
{
    int threadAdj = (threadIdx.x + threadIdx.y * blockDim.x) * perThread;
    int blockAdj = (blockIdx.x + blockIdx.y) * perBlock;
    int totalAdj = threadAdj + blockAdj;
    int tCol = (totalAdj % cols) + 1;
    int tRow = (floor((double) totalAdj / (double) cols)) + 1;


    for (int threadPos = 0; threadPos < perThread; threadPos++){
        int tPCol = tCol + (threadPos % cols);
        int tPRow = tRow + floor((double) threadPos / (double) cols);

        if (tPCol < cols && tPRow < rows){
            int sum = 0;

            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    sum = sum + inImg[tPCol + i + (long)(tPRow + j) * cols] * mask[(i + 1) * 3 + (j + 1)];
                }
            }

            if (sum > 255)
                sum = 255;
            if (sum < 0)
                sum = 0;
            
            outImg[tPRow * cols + tPCol] = sum;

        }
    }
}


#endif