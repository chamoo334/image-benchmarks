
#include <fstream>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
// #include <string.h>
#include <unistd.h>
#include "ImageProcessing.cuh"
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// #include "cuda.h"

using namespace std;

const char *SEQ = "_seq_c.";
const char *PAR = "_par_c.";

void createFileNames(char *src, char *i, char *i_seq, char *i_par);
void deleteEverything(int count, ...);
void freeEverything(int count, ...);
void errorExit(int err);
void TimeOfDaySeed();

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        errorExit(1);
    }

    char *image_main = new char[strlen(argv[1]) + 1]();
    char *image_seq = new char[strlen(argv[1]) + strlen(SEQ)]();
    char *image_par = new char[strlen(argv[1]) + strlen(PAR)]();

    createFileNames(argv[1], image_main, image_seq, image_par);

    if (access(image_main, R_OK) == 0)
    {

        int imgWidth, imgHeight, imgBitDepth, imgSize;
        bool imgIsColor;
        unsigned char imgHeader[BMP_HEADER_SIZE];
        unsigned char imgColorTable[BMP_COLOR_TABLE_SIZE];
        unsigned char *imgInBuffer = NULL;
        unsigned char *imgOutBuffer = NULL;

        ImageProcessing *new_image = new ImageProcessing(image_main,
                                                         image_seq,
                                                         &imgHeight,
                                                         &imgWidth,
                                                         &imgBitDepth,
                                                         &imgSize,
                                                         &imgIsColor,
                                                         &imgHeader[0],
                                                         &imgColorTable[0],
                                                         &imgInBuffer,
                                                         &imgOutBuffer);

        new_image->readImage();
        if (imgIsColor)
            errorExit(3);

        if (strcmp(argv[2], "seq") == 0)
        {
            new_image->detectLinesSeq(1);
        }
        else
        {
            // new_image->detectLinesPar(1);
            cout << "finish c parallel" << endl;
            cout << imgSize << endl;
            exit(0);
        }

        new_image->writeImage();
        cout << imgSize << endl;

        deleteEverything(2, imgInBuffer, imgOutBuffer);
    }
    else
    {
        errorExit(2);
    }

    deleteEverything(3, image_main, image_seq, image_par);
}

void createFileNames(char *src, char *i, char *i_seq, char *i_par)
{
    char delim[] = ".";
    char *saveptr = NULL;

    strncpy(i, src, strlen(src));

    char *token = strtok_r(src, delim, &saveptr);

    strncpy(i_seq, token, strlen(token));
    strncat(i_seq, SEQ, strlen(SEQ));
    strncat(i_seq, saveptr, strlen(saveptr));

    strncpy(i_par, token, strlen(token));
    strncat(i_par, PAR, strlen(PAR));
    strncat(i_par, saveptr, strlen(saveptr));
}

void deleteEverything(int count, ...)
{
    va_list list;
    int j = 0;
    va_start(list, count);

    for (j = 0; j < count; j++)
    {
        delete[] va_arg(list, char *);
        // char *temp = va_arg(list, char *);
        // *temp = 0;
    }

    va_end(list);
}

void freeEverything(int count, ...)
{
    va_list list;
    int j = 0;
    va_start(list, count);

    for (j = 0; j < count; j++)
    {
        free(va_arg(list, char *));
    }

    va_end(list);
}

void errorExit(int err)
{
    switch (err)
    {
    case 1:
        cerr << "Unable to continue due to missing image or results filename\n";
        break;
    case 2:
        cerr << "Unable to open the requested image\n";
        break;
    case 3:
        cerr << "Unable to process RGB images\n";
        break;
    default:
        cerr << "An error occured\n";
        break;
    }

    exit(0);
}

void TimeOfDaySeed()
{
    struct tm y2k = {0};
    y2k.tm_hour = 0;
    y2k.tm_min = 0;
    y2k.tm_sec = 0;
    y2k.tm_year = 100;
    y2k.tm_mon = 0;
    y2k.tm_mday = 1;

    time_t timer;
    time(&timer);
    double seconds = difftime(timer, mktime(&y2k));
    unsigned int seed = (unsigned int)(1000. * seconds); // milliseconds
    srand(seed);
}