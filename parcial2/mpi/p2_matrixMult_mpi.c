#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>



double * CreateZerosMatrix(int _n) {
	double *result = (double *)malloc(_n * _n * sizeof(double)) ;

	for (int h = 0; h < _n; h++)
		for (int w = 0; w < _n; w++)
			result[h * _n + w] = 0.0;

	return result;
}
double * CreateRandomMatrix (int _n) {
	double *result = (double *)malloc(_n * _n * sizeof(double)) ;

	for (int h = 0; h < _n; h++)
		for (int w = 0; w < _n; w++)
			result[h * _n + w] = rand();

	return result;
}

void MatrixToString(double * m, int N) {
	for (int h = 0; h < N; h++) {
		for (int w = 0; w < N; w++)
			printf("%0.1f ", m[h * N + w] );

		printf("\n" );
	}
}
int main(int argc, char** argv) {
	int  tasks, iam;
	struct timeval start_time, stop_time, elapsed_time;
	int N = atoi(argv[1]);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &tasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &iam);
	double *iniMatrixA = (double *)malloc(N * N * sizeof(double)) ;
	double *iniMatrixB = (double *)malloc(N * N * sizeof(double)) ;
	double *resultLocalMatrix = (double *)malloc(N * N * sizeof(double)) ;
	double *resultGlobalMatrix = (double *)malloc(N * N * sizeof(double)) ;

	srand(time(NULL));
	gettimeofday(&start_time, NULL);
	resultLocalMatrix =  CreateZerosMatrix(N);
	resultGlobalMatrix =  CreateZerosMatrix(N);

	if (iam == 0) {
		iniMatrixA =  CreateRandomMatrix(N);
		iniMatrixB =  CreateRandomMatrix(N);
		// printf("A: \n" );
		// MatrixToString(iniMatrixA, N);
		// printf("B: \n" );
		// MatrixToString(iniMatrixB, N);
	}

	MPI_Bcast(iniMatrixA, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(iniMatrixB, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	for (int i = iam ; i < N; i += tasks ) {
		for (int j = 0 ; j < N ; j++	) {
			resultLocalMatrix[i * N + j] = 0;

			for (int k = 0; k < N; k++) {
				resultLocalMatrix[i * N + j] += iniMatrixA[i * N + k] * iniMatrixB[k * N + j];
			}
		}
	}

	MPI_Reduce(resultLocalMatrix, resultGlobalMatrix, N  * N, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (iam == 0) {
		char tiempo[10];
		gettimeofday(&stop_time, NULL);
		timersub(&stop_time, &start_time, &elapsed_time);
		sprintf(tiempo, "%f", elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);
		printf("%d\t%d\t%s ms\n",tasks, N, tiempo  );
		// MatrixToString(resultGlobalMatrix, N);
	}

	MPI_Finalize();
	return 0;
}



