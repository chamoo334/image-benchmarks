
#include <fstream>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
// #include <string.h>
#include <unistd.h>
#include "ImageProcessing.cuh"
#ifndef NUM_THREADS
#define NUM_THREADS		1
#endif
#ifndef NUM_BLOCKS
#define NUM_BLOCKS		1
#endif


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

    cout << NUM_THREADS << "  " << NUM_BLOCKS << endl;

    char *image_main = new char[strlen(argv[1]) + 1]();
    char *image_seq = new char[strlen(argv[1]) + strlen(SEQ)]();
    char *image_par = new char[strlen(argv[1]) + strlen(PAR)]();

    createFileNames(argv[1], image_main, image_seq, image_par);

    if (access(image_main, R_OK) == 0)
    {

        int imgWidth, imgHeight, imgBitDepth, imgSize;
        bool imgIsColor, useGPU=false;
        unsigned char imgHeader[BMP_HEADER_SIZE];
        unsigned char imgColorTable[BMP_COLOR_TABLE_SIZE];
        unsigned char *imgInBuffer = NULL;
        unsigned char *imgOutBuffer = NULL;
        char *image_out_file = image_seq;

        if (strcmp(argv[2], "par") == 0) { 
            useGPU = true; 
            image_out_file = image_par;
        }

        ImageProcessing *new_image = new ImageProcessing(image_main,
                                                         image_out_file,
                                                         &imgHeight,
                                                         &imgWidth,
                                                         &imgBitDepth,
                                                         &imgSize,
                                                         &imgIsColor,
                                                         &useGPU,
                                                         &imgHeader[0],
                                                         &imgColorTable[0],
                                                         &imgInBuffer,
                                                         &imgOutBuffer);

        new_image->readImage();

        if (imgIsColor)
            errorExit(3);
        if (useGPU)
            new_image->detectLines(1, NUM_THREADS, NUM_BLOCKS);
        else
            new_image->detectLines(1);


        // new_image->writeImage();

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
        fprintf( stderr, "Unable to continue due to missing image or results filename\n" );
        break;
    case 2:
        fprintf( stderr, "Unable to open the requested image\n" );
        break;
    case 3:
        fprintf( stderr, "Unable to process RGB images\n" );
        break;
    default:
        fprintf( stderr, "An error occured\n" );
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