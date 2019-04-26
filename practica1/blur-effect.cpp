#include <stdio.h>
#include <iostream>

int marco(int sizeKernel){
    return int(sizeKernel/2);
}


int main(){

int sizeKernel = 3;
int sizeImgX = 7;
int sizeImgY = 6;
int imagen[sizeImgX][sizeImgY] = { {3,3,3,3,3,3},
                                   {3,4,4,4,4,3},
                                   {3,4,1,1,4,3},
                                   {3,4,1,1,4,3},
                                   {3,4,1,1,4,3},
                                   {3,4,4,4,4,3},
                                   {3,3,3,3,3,3}} ;

int kernel [sizeKernel][sizeKernel] = {{0,-1,0},
                                       {-1,5,-1},
                                       {0,-1,0}};                          

int count = 0;
int blur [sizeKernel][sizeKernel];
    for(int i = marco(sizeKernel); i < sizeImgX-marco(sizeKernel); i++){
        for(int j = marco(sizeKernel); j < sizeImgY-marco(sizeKernel); j++){
            //printf("%d\n", imagen[i][j]);
            int convolution = 0;
            for(int k = 0; k < sizeKernel; k++){
                for(int l = 0; l < sizeKernel; l++){
                    
                }
            }
            imagen[i][j] = convolution;
        }
    }
    for(int i=0; i < sizeImgX; i++){
        for(int j=0; j < sizeImgY; j++){
            printf("%d \n",imagen[i][j]);
        }
    }
        
    return 0;
}
