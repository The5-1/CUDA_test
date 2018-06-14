#include "CUDAmatrixVector.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <Eigen\Eigen>
#include <Eigen\Dense>

//Eigen + CUDA
//https://stackoverflow.com/questions/23802209/how-to-work-with-eigen-in-cuda-kernels/41120980#41120980

/*
__global__ void cu_dot(Eigen::Vector3f *v1, Eigen::Vector3f *v2, double *out, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		out[idx] = v1[idx].dot(v2[idx]);
	}
	return;
}
*/

/*
__global__ void cu_getOffsetVector(Eigen::MatrixXf *basis, Eigen::VectorXf *variance, Eigen::VectorXf *sample, Eigen::VectorXf *out, size_t N)
{
	return;
}
*/

__global__ void outputHelper(){
	printf("TEST!!!");
}

__global__ void cu_getOffsetVector(int numVerts, int numVariance, float* basisMatrix, float* varianceVector, float* sampleVector,  float* outOffsetVector)
{
	int i_thread = blockIdx.x*blockDim.x + threadIdx.x;

	//Result vector will have size of outOffsetVector
	if (i_thread >= numVerts) {
		return;
	}

	if (i_thread == 0) {
		printf("basisMatrix: %f , %f, %f \n", basisMatrix[0], basisMatrix[numVerts], basisMatrix[2 * numVerts]);

		printf("varianceVector: %f , %f, %f \n", varianceVector[0], varianceVector[1], varianceVector[2]);
		printf("sampleVector: %f , %f, %f \n", sampleVector[0], sampleVector[1], sampleVector[2]);
	
	}

	//Make sure result doesnt contain any value
	outOffsetVector[i_thread] = 0.0f;

	for (int v = 0; v < numVariance; v++) {

		int matrixIndex = v * numVerts + i_thread;

		outOffsetVector[i_thread] += basisMatrix[matrixIndex] * (varianceVector[v] * sampleVector[v]);
	}

	return;
}

void getOffsetVector(const Eigen::MatrixXf &basis, const Eigen::MatrixXf &variance, const Eigen::MatrixXf &sample, Eigen::MatrixXf &out)
{
	if (basis.cols() != variance.size() || variance.size() != sample.size()) {
		printf("Dimension missmatch!");
		return;
	}



}




void runEigenCudaTest01() {

	Eigen::Matrix2f m;
	m << 1.0f, 2.0f, 3.0f, 4.0f; //initializes row-wise (left to right)

	float* m_data = m.data();

	for (int i = 0; i < 4; i++) {
		std::cout << m_data[i] << std::endl; //acesses column wise 1,3,2,4 (top-down)
	}

	const int varianceSize = 3;
	const int verticesSize = 5;

	Eigen::Matrix<float, verticesSize, varianceSize> basisMat;
	basisMat << 1.0f, 2.0f, 3.0f,
				1.0f, 2.0f, 3.0f,
				1.0f, 2.0f, 3.0f,
				1.0f, 2.0f, 3.0f,
				1.0f, 2.0f, 3.0f;

	Eigen::Matrix<float, varianceSize, 1> varianceVec;
	varianceVec <<	1.0f,
					2.0f,
					3.0f;

	Eigen::Matrix<float, varianceSize, 1> sampleVec;
	sampleVec <<	2.0f,
					2.0f,
					2.0f;


	Eigen::Matrix<float, 1, verticesSize> resultVec;
	resultVec.setZero();


	float* d_basisMatrix, float* d_varianceVector, float* d_sampleVector, float* d_outOffsetVector;

	cudaMalloc(&d_basisMatrix, varianceSize * verticesSize * sizeof(float));
	cudaMalloc(&d_varianceVector, varianceSize * sizeof(float));
	cudaMalloc(&d_sampleVector, varianceSize * sizeof(float));
	cudaMalloc(&d_outOffsetVector, verticesSize * sizeof(float));

	cudaMemcpy(d_basisMatrix, basisMat.data(), varianceSize * verticesSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_varianceVector, varianceVec.data(), varianceSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sampleVector, sampleVec.data(), varianceSize * sizeof(float), cudaMemcpyHostToDevice);


	const int problemSize = verticesSize;
	const int blockSize = 256;
	const int numBlocks = int((problemSize + blockSize - 1) / blockSize); //if it divides EXACTLY this is better. (blocksize-1)/blocksize is just under 1



	std::cout << "Starting Kernel" << std::endl; 
	cu_getOffsetVector <<<numBlocks, blockSize>>> (verticesSize, varianceSize, d_basisMatrix, d_varianceVector, d_sampleVector, d_outOffsetVector);
	std::cout << "Finished Kernel" << std::endl;
	
	float *resultPointer = new float[verticesSize];
	cudaMemcpy(resultPointer, d_outOffsetVector, verticesSize * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Resultat: " << std::endl;
	for (int i = 0; i < verticesSize; i++) {
		std::cout << resultPointer[i] << std::endl;
	}

	//https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
	//See: How to write generic, but non-templated function?
	//getOffsetVector(basisMat, varianceVec, sampleVec, resultVec);



	return;
}