//gcc pi_omp.c -o pi_omp -fopenmp
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"


void MatrixMultiply(double ** A, double ** B, double ** C, int ID, int threads , int n ) {
	for (int i = ID ; i < n; i += threads ) {
		//printf("fila: %d \n",i);
		for (int j = 0 ; j < n ; j++	) {
			C[i][j] = 0;
			
			for (int k = 0; k < n; k++) {
				C[i][j] += A[i][k] * B[k][j];
				//printf("A :  %f --- ", A[i][k]);
				//printf("B :  %f \n", B[k][j]);

			}
		}
	}
}

double ** CreateZerosMatrix(int _n) {
	double **result = new double *[_n] ;

	for (int h = 0; h < _n; h++) {
		result[h] = new double [_n] ;

		for (int w = 0; w < _n; w++)
			result[h][w] = 0.0;
	}

	return result;
}
double ** CreateRandomMatrix (int _n) {
	double **result = new double *[_n] ;

	for (int h = 0; h < _n; h++) {
		result[h] = new double [_n] ;

		for (int w = 0; w < _n; w++)
			result[h][w] = rand();
	}

	return result;
}

void MatrixToString(double ** m, int N) {
	for (int h = 0; h < N; h++) {
		for (int w = 0; w < N; w++)
			printf("%0.1f , ", m[h][w] );

		printf("\n" );
	}
}
int main(int argc, char** argv) {
	int threads = atoi(argv[2]);
	int N = atoi(argv[1]);
	double **iniMatrixA = new double *[N] ;
	double **iniMatrixB = new double *[N] ;
	double **resultMatrix = new double *[N] ;
	srand(time_t(NULL));
	iniMatrixA =  CreateRandomMatrix(N);
	iniMatrixB =  CreateRandomMatrix(N);
	resultMatrix =  CreateZerosMatrix(N);
	//MatrixToString(iniMatrixA,N);
	//printf("\n");
	//MatrixToString(iniMatrixB,N);
	double start = omp_get_wtime();
	#pragma omp parallel num_threads(threads)
	{
		int ID = omp_get_thread_num();
		MatrixMultiply(iniMatrixA, iniMatrixB, resultMatrix, ID, threads, N);
	}
	//printf(" %f \n",(omp_get_wtime() - start) );
	printf("%d  %d  %f \n",N,threads,(omp_get_wtime() - start));
}



