#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "math.h"
#include "png.h"
#include <sys/time.h>
#include <malloc.h>
#include <chrono>
#include <cmath>

//nvcc gauss_cuda.cu -o gauss_cuda -I /usr/local/cuda/samples/common/inc/ -lpng
//./gauss_cuda hdLancia.png blur/hdLanciaBlur.png 15 | tee -a times/hd-times.txt

using namespace std;

int width, height;

png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers;

size_t size;

unsigned char *d_Red, *d_Green, *d_Blue;
unsigned char *h_Red, *h_Green, *h_Blue;

 __global__ void blurEffect(double *d_kernel, int height, int width,  unsigned char *d_Red,  unsigned char *d_Green,unsigned char *d_Blue, int radius, int kernelSize, int operationPerThread){
    int index = ((blockDim.x * blockIdx.x + threadIdx.x));
    if( index < (height*width) )
    {
        for(int count = 0; count < operationPerThread; count ++){
            int i = (index*operationPerThread + count) / width;// fila del pixel al que se le hara gauss
            int j = (index*operationPerThread + count) % width;//columna del pixel al que se le hara gauss

            double redTemp = 0;
            double blueTemp = 0;
            double greenTemp = 0;
            double acum = 0;
            for (int k = 0; k < kernelSize; k++ )
            {
                int y = (i - radius + k + height)%height;
                for (int l = 0; l < kernelSize; l++)
                {
                    int x = (j - radius + l + width )% width;
                    redTemp += d_Red[y*width + x] * d_kernel[k*kernelSize + l];
                    greenTemp += d_Green[y*width + x] * d_kernel[k*kernelSize + l];
                    blueTemp += d_Blue[y*width + x] * d_kernel[k*kernelSize + l];
                    acum += d_kernel[k*kernelSize + l];
                    
                }
            }

            d_Red[i*width + j] = redTemp/acum;
            d_Green[i*width + j] = greenTemp/acum;
            d_Blue[i*width + j] = blueTemp/acum;
        }
    }
}

void read_png_file(char *filename){
    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
        abort();

    png_infop info = png_create_info_struct(png);
    if (!info)
        abort();

    if (setjmp(png_jmpbuf(png)))
        abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth = png_get_bit_depth(png, info);

    // Read any color_type into 8bit depth, RGBA format.
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++)
    {
        row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers);

    fclose(fp);
}

void write_png_file(char *filename){
    FILE *fp = fopen(filename, "wb");
    if (!fp)
        abort();

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
        abort();

    png_infop info = png_create_info_struct(png);
    if (!info)
        abort();

    if (setjmp(png_jmpbuf(png)))
        abort();

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    for (int y = 0; y < height; y++)
    {
        free(row_pointers[y]);
    }
    free(row_pointers);

    fclose(fp);
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
double **createKernel(int KERNEL_SIZE){
    if(KERNEL_SIZE%2 == 0){
        KERNEL_SIZE += 1; // to make sure of kernel with odd size
    }

    double matrixOffsetEffect = (KERNEL_SIZE - 1) / 2; //matrixOffsetEffect is a metric to know the scope of the effect
    double stdDev = matrixOffsetEffect / 2.;
    double average = 0;
    double **kernelMatrix = new double *[KERNEL_SIZE]; //Due to c++ is unable to return a matrix
                                                       //it has to be a pointer to an array
    
    for(int i = 0; i < KERNEL_SIZE; i++){
        kernelMatrix[i] = new double [KERNEL_SIZE];

        for(int j = 0; j < KERNEL_SIZE; j++){ // Multiplied horizontal and vertical blur values

            double blurHorizontal = gaussianFunction(i, matrixOffsetEffect, stdDev);
            double blurVertical = gaussianFunction(j, matrixOffsetEffect, stdDev);

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

double *matrixToArray(double **matrix, int rows, int cols){
    double *array = (double *)malloc(rows * cols * sizeof(double));
    int k = 0;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            array[k++] = matrix[i][j];
        }
    }
    return array;
}
void getChannels(){
    for (int i = 0; i < height; i++)
    {
        png_bytep row = row_pointers[i];
        for (int j = 0; j < width; j++)
        {
            png_bytep px = &(row[j * 4]);
            h_Red[i * width + j] = (char)px[0];
            h_Green[i * width + j] = (char)px[1];
            h_Blue[i * width + j] = (char)px[2];
        }
    }
}

void makeRowPointer(){
    for (int i = 0; i < height; i++)
    {
        png_bytep row = row_pointers[i];
        for (int j = 0; j < width; j++)
        {
            png_bytep px = &(row[j * 4]);
            px[0] = h_Red[i * width + j];
            px[1] = h_Green[i * width + j];
            px[2] = h_Blue[i * width + j];
        }
    }
}
int main(int argc, char *argv[]){
    
    cudaError_t err = cudaSuccess;

    //To get the number of core(threads) on each block, and the number of blocks per grid
    //this numbers depends on the GPU, to run this part, it is necessary to call the flag:
    //-I /usr/local/cuda/samples/common/inc/   which comes from CUDA libraries

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int threadsPerBlock = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	threadsPerBlock = threadsPerBlock*2;
    int blocksPerGrid =   deviceProp.multiProcessorCount;
    


    int KERNEL_SIZE = atoi(argv[3]);
    char matrixOffset = (char)floor(KERNEL_SIZE / 2);
    read_png_file(argv[1]);
    int opt = (int)(ceil(height * width/ (threadsPerBlock*blocksPerGrid)));
    
    size_t size = height * width*sizeof(unsigned char);
    // Asignar memoria para cpu
    h_Red = (unsigned char *)malloc( size );
    h_Blue = (unsigned char *)malloc(  size );
    h_Green = (unsigned char *)malloc( size );
  
    
    
    if (h_Red == NULL || h_Blue == NULL || h_Green == NULL){
        fprintf(stderr, "Failed to allocate host variables\n");
        exit(EXIT_FAILURE);
    }
    getChannels();
    
    double *h_kernel;
    double *d_kernel;
    h_kernel = matrixToArray(createKernel(KERNEL_SIZE), KERNEL_SIZE, KERNEL_SIZE);
    
    //Memory allocation on device
    err = cudaMalloc((void **)&d_Red, size);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device variable d_Red (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_Green, size);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device variable d_Green (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_Blue, size);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device variable d_Blue (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_kernel, KERNEL_SIZE*KERNEL_SIZE*sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device matrix kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copy memory from host to device
    err = cudaMemcpy(d_Red, h_Red, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy variable d_Red from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_Green, h_Green, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy variable d_Green from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_Blue, h_Blue, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy variable d_Blue from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE*KERNEL_SIZE*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector kernel from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //starting the threads and setting work
    auto startClock = chrono::steady_clock::now();

    //starting kernel on GPU
    blurEffect<<<blocksPerGrid,threadsPerBlock>>>(d_kernel, height, width, d_Red, d_Green, d_Blue, matrixOffset, KERNEL_SIZE, opt);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch Blur effect Kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy results vector from device to host
    err = cudaMemcpy(h_Red, d_Red, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){    
        fprintf(stderr, "Failed to copy variable Red from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_Green, d_Green, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){    
        fprintf(stderr, "Failed to copy variable Green from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_Blue, d_Blue, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){    
        fprintf(stderr, "Failed to copy variable Blue from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(d_Red);
    cudaFree(d_Green);
    cudaFree(d_Blue);
    cudaFree(d_kernel);

    free(h_kernel);
    makeRowPointer();
    cudaFree(h_Red);
    cudaFree(h_Green);
    cudaFree(h_Blue);
    
    //end clock timing
    auto endClock = chrono::steady_clock::now();
    auto finalClock = endClock - startClock;
    cout<< KERNEL_SIZE << "," << threadsPerBlock << "," << chrono::duration <double, milli> (finalClock).count()<<endl;
    write_png_file(argv[2]);
    
    return 0;
}