// omp_schedule.c
//gcc omp_schedule.c -o omp_schedule -fopenmp
#include <stdio.h>
#include <omp.h>

#define NUM_THREADS 4
#define STATIC_CHUNK 5
#define DYNAMIC_CHUNK 5
#define NUM_LOOPS 20
#define SLEEP_EVERY_N 3

int main( ) 
{
    int nStatic1[NUM_LOOPS], 
        nStaticN[NUM_LOOPS];
    int nDynamic1[NUM_LOOPS], 
        nDynamicN[NUM_LOOPS];
    int nGuided[NUM_LOOPS];
	int i;
    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel
    {
        #pragma omp for schedule(static, 1)
        for ( i = 0 ; i < NUM_LOOPS ; ++i) 
        {
            if ((i % SLEEP_EVERY_N) == 0) 
                sleep(0);
            nStatic1[i] = omp_get_thread_num( );
        }

        #pragma omp for schedule(static, STATIC_CHUNK)
        for ( i = 0 ; i < NUM_LOOPS ; ++i) 
        {
            if ((i % SLEEP_EVERY_N) == 0) 
                sleep(0);
            nStaticN[i] = omp_get_thread_num( );
        }

        #pragma omp for schedule(dynamic, 1)
        for ( i = 0 ; i < NUM_LOOPS ; ++i) 
        {
            if ((i % SLEEP_EVERY_N) == 0) 
                sleep(0);
            nDynamic1[i] = omp_get_thread_num( );
        }

        #pragma omp for schedule(dynamic, DYNAMIC_CHUNK)
        for ( i = 0 ; i < NUM_LOOPS ; ++i) 
        {
            if ((i % SLEEP_EVERY_N) == 0) 
                sleep(0);
            nDynamicN[i] = omp_get_thread_num( );
        }

        #pragma omp for schedule(guided)
        for ( i = 0 ; i < NUM_LOOPS ; ++i) 
        {
            if ((i % SLEEP_EVERY_N) == 0) 
                sleep(0);
            nGuided[i] = omp_get_thread_num( );
        }
    }

    printf("------------------------------------------------\n");
    printf("| static | static | dynamic | dynamic | guided |\n");
    printf("|    1   |    %d   |    1    |    %d    |        |\n",
             STATIC_CHUNK, DYNAMIC_CHUNK);
    printf("------------------------------------------------\n");

    for ( i=0; i<NUM_LOOPS; ++i) 
    {
        printf("|    %d   |    %d   |    %d    |    %d    |"
                 "    %d   |\n",
                 nStatic1[i], nStaticN[i],
                 nDynamic1[i], nDynamicN[i], nGuided[i]);
    }

    printf("------------------------------------------------\n");
}