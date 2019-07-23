#!/bin/bash
# g++ blur-effect.cpp -o y `pkg-config --cflags --libs opencv` -lpthread

printf "\n --- Times MPI --- \n"
echo "">times.in
threads=1
theN=8
while [ $theN -le 1024 ]
do  
    mpirun -np $threads --hostfile /home/mpiuser/mpi_hosts /home/mpiuser/Parcial2/mpiexe $theN | tee -a times.in   
    # ./y hdLancia.jpg blur/hd-blur.jpg $theN $threads | tee -a times/hd-times.in    
    theN=$((theN*2))
done

threads=2
theN=8
while [ $theN -le 1024 ]
do  
    mpirun -np $threads --hostfile /home/mpiuser/mpi_hosts /home/mpiuser/Parcial2/mpiexe $theN | tee -a times.in   
    theN=$((theN*2))
done

threads=3
theN=8
while [ $theN -le 1024 ]
do  
    mpirun -np $threads --hostfile /home/mpiuser/mpi_hosts /home/mpiuser/Parcial2/mpiexe $theN | tee -a times.in   
    theN=$((theN*2))
done

threads=4
theN=8
while [ $theN -le 1024 ]
do  
    mpirun -np $threads --hostfile /home/mpiuser/mpi_hosts /home/mpiuser/Parcial2/mpiexe $theN | tee -a times.in   
    theN=$((theN*2))
done
