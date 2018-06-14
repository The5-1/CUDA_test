#include <iostream>

#include "CUDAexample.h"


int main()
{
	//CUDA_test01();
	//Eigen_test01();

	
	std::cout << "Matrix mult Test3" << std::endl;
	runMatrixVectorMult_test();

	std::cin.get();
	return 0;
}