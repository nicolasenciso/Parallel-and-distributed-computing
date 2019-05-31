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

using namespace std;

int width, height;
unsigned char *d_R, *d_G, *d_B;
unsigned char *h_R, *h_G, *h_B;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers;
size_t size;
 __global__ void
blurEffect(double *d_kernel, int height, int width,  unsigned char *d_R,  unsigned char *d_G,unsigned char *d_B, int radius, int kernelSize, int operationPerThread)
{
    
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
                    // x = x < 0 ? 0 : x < width ? x : width - 1;
                    redTemp += d_R[y*width + x] * d_kernel[k*kernelSize + l];
                    greenTemp += d_G[y*width + x] * d_kernel[k*kernelSize + l];
                    blueTemp += d_B[y*width + x] * d_kernel[k*kernelSize + l];
                    acum += d_kernel[k*kernelSize + l];
                    
                }
            }

            d_R[i*width + j] = redTemp/acum;
            d_G[i*width + j] = greenTemp/acum;
            d_B[i*width + j] = blueTemp/acum;
        }
    }
}

void read_png_file(char *filename)
{

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

void write_png_file(char *filename)
{
    

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

double **createKernel(int tamanio)
{
    double **matriz = (double **)malloc(tamanio * sizeof(double *));
    for (int i = 0; i < tamanio; i++)
        matriz[i] = (double *)malloc(tamanio * sizeof(double));
    int radio = floor(tamanio / 2);
    double sigma = radio * radio;
    for (int fila = 0; fila < tamanio; fila++)
    {
        for (int columna = 0; columna < tamanio; columna++)
        {
            double square = (columna - radio) * (columna - radio) + (fila - radio) * (fila - radio);
            double weight = (exp(-square / (2 * sigma))) / (3.14159264 * 2 * sigma);
            matriz[fila][columna] = weight;
        }
    }
    return matriz;
}

double *matrix_to_arr(double **M, int rows, int cols){
    double *arr = (double *)malloc(rows * cols * sizeof(double));
    int k = 0;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            arr[k++] = M[i][j];
        }
    }

    return arr;
}
void getChannels()
{
    for (int i = 0; i < height; i++)
    {
        png_bytep row = row_pointers[i];
        for (int j = 0; j < width; j++)
        {
            png_bytep px = &(row[j * 4]);
            h_R[i * width + j] = (char)px[0];
            h_G[i * width + j] = (char)px[1];
            h_B[i * width + j] = (char)px[2];

        }
    }
}

void makeRowPointer()
{
    for (int i = 0; i < height; i++)
    {
        png_bytep row = row_pointers[i];
        for (int j = 0; j < width; j++)
        {
            png_bytep px = &(row[j * 4]);
            px[0] = h_R[i * width + j];
            px[1] = h_G[i * width + j];
            px[2] = h_B[i * width + j];
        }
    }
}
int main(int argc, char *argv[])
{

    if (argc != 4)
        abort();  
    cudaError_t err = cudaSuccess;
// declarar  la cantidad de hilos segun la gpu
//-------------------------------------------------
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int threadsPerBlock = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	threadsPerBlock = threadsPerBlock*2;
    int blocksPerGrid =   deviceProp.multiProcessorCount;
    
//-------------------------------------------------
    int tamanio = atoi(argv[3]);
    char radio = (char)floor(tamanio / 2);
    read_png_file(argv[1]);
    int opt = (int)(ceil(height * width/ (threadsPerBlock*blocksPerGrid)));
    
    size_t size = height * width*sizeof(unsigned char);
    // Asignar memoria para cpu
    h_R = (unsigned char *)malloc( size );
    h_B = (unsigned char *)malloc(  size );
    h_G = (unsigned char *)malloc( size );
  
    
    
    if (h_R == NULL || h_B == NULL || h_G == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    getChannels();
    
 
  
    
    double *h_kernel;
    double *d_kernel;
    h_kernel = matrix_to_arr(createKernel(tamanio), tamanio, tamanio);
    
    //Asignacion de memoria para cuda
    
    err = cudaMalloc((void **)&d_R, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector R (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_G, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector G (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_kernel, tamanio*tamanio*sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copiar memoria de host a device
    err = cudaMemcpy(d_R, h_R, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector R from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_G, h_G, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector G from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_kernel, h_kernel, tamanio*tamanio*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //starting the threads and setting work
    auto startClock = chrono::steady_clock::now();
    //Se lanza el kernel
    //blurEffect(double **kernel, int height, int width,  char *d_R,  char *d_G,char *d_B, int radius, int kernelSize, int operationPerThread)
    blurEffect<<<blocksPerGrid,threadsPerBlock>>>(d_kernel, height, width, d_R, d_G, d_B, radio, tamanio, opt);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(h_R, d_R, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {    
        fprintf(stderr, "Failed to copy vector R from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    err = cudaMemcpy(h_G, d_G, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {    
        fprintf(stderr, "Failed to copy vector G from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {    
        fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(d_R);
    cudaFree(d_G);
    cudaFree(d_B);
    cudaFree(d_kernel);


  
    free(h_kernel);
    makeRowPointer();
    cudaFree(h_R);
    cudaFree(h_G);
    cudaFree(h_B);
    //end clock timing
    auto endClock = chrono::steady_clock::now();
    auto finalClock = endClock - startClock;
    cout<< tamanio << "," << threadsPerBlock << "," << chrono::duration <double, milli> (finalClock).count()<<endl;
    write_png_file(argv[2]);
    
    return 0;
}