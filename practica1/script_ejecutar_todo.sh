#!/bin/bash
g++ blur-effect.cpp -o y `pkg-config --cflags --libs opencv` -lpthread

printf "\n --- HD blur processing --- \n"
threads=1
kernelsize=3

while [ $kernelsize -le 15 ]
do  
    while [ $threads -le 16 ]
    do
        ./y hdLancia.jpg blur/hd-blur.jpg $kernelsize $threads | tee -a times/hd-times.in
        threads=$((threads+1))
    done
    kernelsize=$((kernelsize+1))
done

printf "\n --- FHD blur processing --- \n"
threads=1
kernelsize=3

while [ $kernelsize -le 15 ]
do  
    while [ $threads -le 16 ]
    do
        ./y fhdLancia.jpg blur/full_hd-blur.jpg $kernelsize $threads | tee -a times/fhd-times.in
        threads=$((threads+1))
    done
    kernelsize=$((kernelsize+1))
done

printf "\n --- 4K blur processing --- \n"
threads=1
kernelsize=3

while [ $kernelsize -le 15 ]
do  
    while [ $threads -le 16 ]
    do
        ./y 4kLancia.jpg blur/4k-blur.jpg $kernelsize $threads | tee -a times/4k-times.in
        threads=$((threads+1))
    done
    kernelsize=$((kernelsize+1))
done
