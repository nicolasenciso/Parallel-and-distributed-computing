#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <pthread.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//g++ blur-effect.cpp -o w `pkg-config --cflags --libs opencv` -lpthread 
//./w sl.jpg results2/hd-blurred.jpg $kernelSize $num_threads | tee -a times.in

using namespace std;
using namespace cv; //for tratement of images in OpenCV

int NUM_THREADS;
int KERNEL_SIZE;
int KERNEL_OFFSET;
double **KERNEL_MATRIX;

Mat inputIMG;
Mat outputIMG;

#define SIZE_IMG_ROWS inputIMG.rows
#define SIZE_IMG_COLS inputIMG.cols


//Function to set the limits of the kernel on the image matrix. If is out of bound, it sets
// the limit to the start to the end of the same size of the kernel
int *limitSetter(int i, int j, int KERNEL_SIZE){
    int limitImgX = SIZE_IMG_ROWS - 1;
    int limitImgY = SIZE_IMG_COLS - 1;
    int KERNEL_OFFSET = KERNEL_SIZE / 2;

    int xStart, yStart, xEnd, yEnd;
    int *finalLimits = new int[4]; //array which will contain the x and y limits, start to end

    //Conditionals for the X axis of the kernel on the image matrix
    
    if( i + KERNEL_OFFSET > limitImgX){
        xStart = 0;
        xEnd = (limitImgX - i) + KERNEL_OFFSET;

    }else if(i - KERNEL_OFFSET < 0){
        xStart = KERNEL_OFFSET - i;
        xEnd = KERNEL_SIZE;
    }else{
        xStart = 0;
        xEnd = KERNEL_SIZE;
    }

    finalLimits[0] = xStart;
    finalLimits[1] = xEnd;

    //Conditionals fot the Y axis of the kernel on the image matrix
    if(j + KERNEL_OFFSET > limitImgY){
        yStart = 0;
        yEnd = (limitImgY - j) + KERNEL_OFFSET;

    }else if(j - KERNEL_OFFSET < 0 ){
        yStart = KERNEL_OFFSET - j;
        yEnd = KERNEL_SIZE;
        
    }else{
        yStart = 0;
        yEnd = KERNEL_SIZE;
    }

    finalLimits[2] = yStart;
    finalLimits[3] = yEnd;
    
    return finalLimits;
}


//Gaussian function to get the value for the i,j position on the kernel matrix
// stdDev is the standard deviation of the gaussian function ( sigma )
//https://es.wikipedia.org/wiki/Funci%C3%B3n_gaussiana
double gaussianFunction(double x, double y, double stdDev){ 
    return exp( -(((x-y)*((x-y)))/(2.0*stdDev*stdDev)));
}


//Function to create the kernel matrix which will be use to make the blur effect.
//It uses the gaussian value weighted, it means, multiplied the blur in horizontal
//  with the blur on vertical, and later divide the value in the average of
//  all the matrix values;
double **createKernelMatrix(int KERNEL_SIZE){
    if(KERNEL_SIZE%2 == 0){
        KERNEL_SIZE += 1; // to make sure of kernel with odd size
    }

    double radioEffect = (KERNEL_SIZE - 1) / 2; //radioEffect is a metric to know the scope of the effect
    double stdDev = radioEffect / 2.;
    double average = 0;
    double **kernelMatrix = new double *[KERNEL_SIZE]; //Due to c++ is unable to return a matrix
                                                       //it has to be a pointer to an array
    
    for(int i = 0; i < KERNEL_SIZE; i++){
        kernelMatrix[i] = new double [KERNEL_SIZE];

        for(int j = 0; j < KERNEL_SIZE; j++){ // Multiplied horizontal and vertical blur values

            double blurHorizontal = gaussianFunction(i, radioEffect, stdDev);
            double blurVertical = gaussianFunction(j, radioEffect, stdDev);

            kernelMatrix[i][j] = blurHorizontal * blurVertical;
            average += kernelMatrix[i][j];
        }
    }

    for(int i = 0; i < KERNEL_SIZE; i++){
        for(int j = 0; j < KERNEL_SIZE; j++){
            kernelMatrix[i][j] /= average; // divide each value over the average
        }
    }
    
    /*for(int i = 0; i < KERNEL_SIZE; i++){ //print the kernel matrix
        for(int j = 0; j < KERNEL_SIZE; j++){
            cout<<kernelMatrix[i][j]<<" | ";
        }
        cout << "\n";
    }*/
    return kernelMatrix;
}

//Function to apply the blur effect of the kernel matrix to the image matrix.
// - There are 3 color channels for each pixels according to the RGB color mode
//   each color channel will be update with the convolution value betwenn both matrixes.

// - The functions has to be a pointer to void due to the argument parameter for pthread.
// - The main idea is allocate to each thread, a number multiple of the total number 
//   of threads n, so the ith thread will have to process every jth + n row
void *makeBlurEffect(void *id){
    int *kernelLimits;

    double redPixel; 
    double bluePixel;
    double greenPixel;

    long threadID = (long) id;

    for(int i = threadID; i < SIZE_IMG_ROWS; i += NUM_THREADS){

        for(int j = 0; j < SIZE_IMG_COLS; j++){

            redPixel = 0;
            bluePixel = 0;
            greenPixel = 0;

            kernelLimits = limitSetter(i, j, KERNEL_SIZE);

            for(int kernelX = kernelLimits[0]; kernelX < kernelLimits[1]; kernelX++){

                for(int kernelY = kernelLimits[2]; kernelY < kernelLimits[3]; kernelY++){

                    bluePixel += inputIMG.at<Vec3b>(kernelX + i, kernelY + j)[0] * 
                                KERNEL_MATRIX[kernelX + KERNEL_OFFSET] [kernelY + KERNEL_OFFSET];
                    
                    greenPixel += inputIMG.at<Vec3b>(kernelX + i, kernelY + j)[1] * 
                                KERNEL_MATRIX[kernelX + KERNEL_OFFSET] [kernelY + KERNEL_OFFSET];

                    redPixel += inputIMG.at<Vec3b>(kernelX + i, kernelY + j)[2] * 
                                KERNEL_MATRIX[kernelX + KERNEL_OFFSET] [kernelY + KERNEL_OFFSET];
                }
            }

            outputIMG.at<Vec3b>(i, j)[0] = bluePixel;
            outputIMG.at<Vec3b>(i, j)[1] = greenPixel;
            outputIMG.at<Vec3b>(i, j)[2] = redPixel;
        }
    }
}

int main( int argc, char** argv ) {
    
    //Data capture
    string imageToOpen = argv[1];
    string imageToSave = argv[2];
    KERNEL_SIZE = atoi(argv[3]);
    NUM_THREADS = atoi(argv[4]);

    //Reading the image and setting sizes of the output image
    inputIMG = imread(imageToOpen, CV_LOAD_IMAGE_COLOR);
    outputIMG = Mat(SIZE_IMG_ROWS, SIZE_IMG_COLS, CV_8UC3, Scalar(0, 0, 0));

    KERNEL_MATRIX = createKernelMatrix(KERNEL_SIZE);

    pthread_t threads[NUM_THREADS];
    int thread_error;

    //Starting the threads and setting work
    auto startClock = chrono::steady_clock::now();

    for(int i = 0; i < NUM_THREADS; i++){
        thread_error = pthread_create(&threads[i], NULL, makeBlurEffect, (void*)i );

        if(thread_error){
            perror("Error: ");
            exit(-1);
        }
    }

    //Stop threads
    for(int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }
    //clock_t endClock = clock();
    auto endClock = chrono::steady_clock::now();

    auto finalClock = endClock - startClock;
    cout << KERNEL_SIZE << "\t" << NUM_THREADS << "\t" << chrono::duration <double, milli> (finalClock).count() << " ms" <<"\n" <<endl;
    imwrite(imageToSave, outputIMG);

    return 0;
}