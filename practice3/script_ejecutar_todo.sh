#!/bin/bash

nvcc gauss_cuda.cu -o gauss_cuda -I /usr/local/cuda/samples/common/inc/ -lpng

printf "\n --- HD blur processing --- \n"

kernelsize=3

while [ $kernelsize -le 15 ]
do  
    ./gauss_cuda hdLancia.png blur/hdLanciaBlur.png $kernelsize | tee -a times/hd-times.txt    
    kernelsize=$((kernelsize+1))
done

printf "\n --- FHD blur processing --- \n"

kernelsize=3

while [ $kernelsize -le 15 ]
do  
    ./gauss_cuda fhdLancia.png blur/fhdLanciaBlur.png $kernelsize | tee -a times/fhd-times.txt    
    kernelsize=$((kernelsize+1))
done


printf "\n --- 4k blur processing --- \n"

kernelsize=3

while [ $kernelsize -le 15 ]
do  
    ./gauss_cuda 4kLancia.png blur/4kLanciaBlur.png $kernelsize | tee -a times/4k-times.txt    
    kernelsize=$((kernelsize+1))
done



printf "\n --- HD blur processing --- \n"

kernelsize=3

while [ $kernelsize -le 100 ]
do  
    ./gauss_cuda hdLancia.png blur/hdLanciaBlur.png $kernelsize | tee -a times/hd-times100.txt    
    kernelsize=$((kernelsize+1))
done

printf "\n --- FHD blur processing --- \n"

kernelsize=3

while [ $kernelsize -le 100 ]
do  
    ./gauss_cuda fhdLancia.png blur/fhdLanciaBlur.png $kernelsize | tee -a times/fhd-times100.txt    
    kernelsize=$((kernelsize+1))
done


printf "\n --- 4k blur processing --- \n"

kernelsize=3

while [ $kernelsize -le 100 ]
do  
    ./gauss_cuda 4kLancia.png blur/4kLanciaBlur.png $kernelsize | tee -a times/4k-times100.txt    
    kernelsize=$((kernelsize+1))
done





