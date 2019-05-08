//programa para mostrar sentencia nowait y atomic
//mas de 1 hilo tarda mas debido al atomic, dado que no deja que se ejecuten al tiempo
//solucion, partir el trabajo de forma manual
//gcc nowait.c -o nowait -fopenmp -lm

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define	SIZE	1e03

void nowait_example(long n, long m, double *a, double *b)
{ 	 int i, id;


	 #pragma omp parallel private(id)
	 {
	 	id = omp_get_thread_num();
		#pragma omp for nowait
		 for (i=1; i<n; i++){
			 //#pragma omp atomic	//es necesario proteger la acumulación
			*a = *a + sin((double)i);
		}

		printf("\nHilo %i ha terminado 1er loop", id);
		fflush(stdout);

		#pragma omp for nowait
		 for (i=0; i<m; i++){
			 #pragma omp atomic
			*b = *b + sin((double)i);
		}
		printf("\nHilo %i ha terminado 2do loop", id);
		fflush(stdout);
	}
}

int main()
{
	double *a, *b;
	a = malloc(sizeof(double));
	b = malloc(sizeof(double));

	omp_set_num_threads(8);
	*a = 0.0;
 	*b = 0.0;
	nowait_example(SIZE, SIZE, a, b);
	printf("\n Total a: %lf  b: %lf...", *a, *b); fflush(stdout);

	free(a);
	free(b);

}
