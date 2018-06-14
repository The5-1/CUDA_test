#include <iostream>

#include "CUDAexample.h"
#include "CUDAmatrixVector.h"
#include "EigenMatrixVector.h"
//#include "BFM_Mult.cuh"


#include <Eigen\Eigen>
#include <Eigen\Dense>


int main()
{
	//CUDA_test01();
	//Eigen_test01();

	
	std::cout << "Matrix mult Test3" << std::endl;
	runMatrixVectorMult_test();

	Eigen::MatrixXf basis_matrix;
	//Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> basis_matrix;

	std::cout << "Cuda + Eigen Test: 0" << std::endl;
	runEigenCudaTest01();

	//std::cin.get();
	return 0;
}