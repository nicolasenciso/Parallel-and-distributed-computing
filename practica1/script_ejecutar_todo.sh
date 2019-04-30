g++ blur-effect.cpp -o y `pkg-config --cflags --libs opencv` -lpthread

printf "\n --- HD blur processing --- \n"
for i in {1..16}
do
    for j in {3..15}
    do
    ./y hdLancia.jpg blur/hd-blur.jpg $j $i | tee -a times/hd-times.in
    done
done

 printf "\n --- Full HD blur processing --- \n"
 for i in {1..16}
 do
    for j in {3..15}
    do
     ./y fhdLancia.jpg blur/full_hd-blur.jpg $j $i | tee -a times/fhd-times.in
     done
 done

 printf "\n --- 4K blur processing--- \n"
 for i in {1..16}
 do
    for j in {3..15}
    do
     ./y 4kLancia.jpg blur/4k-blur.jpg $j $i | tee -a times/4k-times.in
    done
 done
