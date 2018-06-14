#include "CUDAmatrixVector.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

//Eigen + CUDA
//https://stackoverflow.com/questions/23802209/how-to-work-with-eigen-in-cuda-kernels/41120980#41120980


__global__ void matrixVectorMultiplication(int vectorSize, int matrixRows, int matrixColums, float* vector, float* matrix, float* result) {
	int i_thread = blockIdx.x*blockDim.x + threadIdx.x;

	//Matrix-Vector-Multiplication not possible for the given dimensios
	if (vectorSize != matrixColums) {
		return;
	}

	//Result vector will have size of matrixRows
	if (i_thread >= matrixRows) {
		return;
	}

	//Make sure result doesnt contain any value
	result[i_thread] = 0.0f;

	for (int v = 0; v < vectorSize; v++) {
		int matrixIndex = i_thread * matrixColums + v;
		result[i_thread] += vector[v] * matrix[matrixIndex];
	}
}

void runMatrixVectorMult_test() {


	/*
	(a x b) * (b x c) = (a x c)

	(--- 199 --- )   (     )    (	   )
	(    159k    ) * ( 199 )  = ( 159k )
	(	         )   (     )    (	   )

	*/


	float* vector = (float*)malloc(5 * sizeof(float));
	float* matrix = (float*)malloc(10 * sizeof(float));
	float* result = (float*)malloc(2 * sizeof(float));

	for (int i = 0; i < 5; i++) {
		vector[i] = float(i);
	}

	for (int i = 0; i < 10; i++) {
		matrix[i] = float(i);
	}

	float *d_result, *d_vector, *d_matrix;

	cudaMalloc(&d_result, 2 * sizeof(float));
	cudaMalloc(&d_vector, 5 * sizeof(float));
	cudaMalloc(&d_matrix, 10 * sizeof(float));

	cudaMemcpy(d_vector, vector, 5 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix, matrix, 10 * sizeof(float), cudaMemcpyHostToDevice);

	const int problemSize = 5; //problem vector
	const int blockSize = 256;
	const int numBlocks = int((problemSize + blockSize - 1) / blockSize); //if it divides EXACTLY this is better. (blocksize-1)/blocksize is just under 1
	//int numBlocks = int(problemSize/blockSize)+1; //this is simple, but might be 1 too big if it divides exactly

	matrixVectorMultiplication <<<numBlocks, blockSize>>> (5, 2, 5, d_vector, d_matrix, d_result);

	cudaMemcpy(result, d_result, 2 * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Result: " << result[0] << " , " << result[1] << std::endl;

	cudaFree(d_vector);
	cudaFree(d_matrix);
	cudaFree(d_result);
}