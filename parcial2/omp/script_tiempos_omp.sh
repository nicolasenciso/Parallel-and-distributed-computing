#!/bin/bash
g++ p2_matrixMult_omp.cpp -o p2_matrixMult_omp -fopenmp

printf "\n --- Times OMP --- \n"
echo "">times.in
threads=1
theN=8
while [ $theN -le 1024 ]
do  
	./p2_matrixMult_omp $theN  $threads| tee -a times.in   
    # ./y hdLancia.jpg blur/hd-blur.jpg $theN $threads | tee -a times/hd-times.in    
    theN=$((theN*2))
done

threads=2
theN=8
while [ $theN -le 1024 ]
do  
   ./p2_matrixMult_omp $theN  $threads| tee -a times.in   
    theN=$((theN*2))
done

threads=3
theN=8
while [ $theN -le 1024 ]
do  
    ./p2_matrixMult_omp $theN  $threads| tee -a times.in   
    theN=$((theN*2))
done

threads=4
theN=8
while [ $theN -le 1024 ]
do  
   ./p2_matrixMult_omp $theN  $threads| tee -a times.in   
    theN=$((theN*2))
done
