#pragma once
#include "CUDAexample.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
	std::printf("F Kernel \n");
    c[i] = a[i] + b[i];
}

void Eigen_test01()
{
	std::cout << "Eigen test:" << std::endl;

	MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2.5;
	m(0, 1) = -1;
	m(1, 1) = m(1, 0) + m(0, 1);
	std::cout << m << std::endl;
}

void CUDA_test01()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	std::cout << "CUDA Add vectors in parallel:" << std::endl;

	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}



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
	//const int numBlocks = int(problemSize/blockSize)+1; //this is simple, but might be 1 too big if it divides exactly

	matrixVectorMultiplication <<<numBlocks, blockSize >>> (5, 2, 5, d_vector, d_matrix, d_result);

	cudaMemcpy(result, d_result, 2 * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Result: " << result[0] << " , " << result[1] << std::endl;

	cudaFree(d_vector);
	cudaFree(d_matrix);
	cudaFree(d_result);
}