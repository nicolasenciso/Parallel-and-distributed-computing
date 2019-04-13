
// Compilacion
// clear && g++ pi.cpp -lpthread -w -o x && ./x

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <pthread.h>

using namespace std;

#define NUM_THREADS 1
#define NFRACTIONS 1000000000
double pi = 0;

void *calc_pi(void * tid){
    long id;
    id = (long) tid;

    long start = NFRACTIONS / NUM_THREADS * (id);
    long end = start + NFRACTIONS / NUM_THREADS;

    double sum = 0;
    for(long i = start; i < end; i++){
        sum += (i % 2 == 0 ? 1 : -1) / (2.0 * i + 1);
    }

    pi += sum;

    pthread_exit(NULL);
}

int main(int argc, char** argv){

    // int* NUM_THREADS = (int*) argv[1];
    pthread_t threads[NUM_THREADS];
    int error_thread;

    for(int i = 0; i < NUM_THREADS; i++){
        error_thread = pthread_create(&threads[i], NULL, calc_pi, (void*)i);

        if(error_thread){
            cout << "Error: al crear el hilo" << error_thread << "\n";
            exit(-1);
        }
    }

    char *b;
    for(int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], (void **)&b);
    }

    double f = pi * 4;
    cout << setprecision(30) << "<*> PI: " << f << "\n";

    pthread_exit(NULL);

    return 0;
}
