#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string>
#define epsilon 0.000001

using namespace std;

void fillArrays(int* data, int* data2, int* backup, int size);
void copyArray(int* data, int* backup, int size);

void unoptimizedSort(int* randomNumbers, int size, FILE* file);
void testIfSorted(int* randomNumbers);

void cudaSort(int* data2, int size, int blocks, int tasksPerThread, FILE* file);
__global__ void oddEvenKernel(const int *randomNumberOptimized, int *sortedNumbers);

int main()
{
	srand(time(NULL));

	cudaError_t cudaStatus = cudaSuccess;
	FILE* file = fopen("data.txt", "w+");

	int* data, *data2, *backup;

	fprintf(file, "ODD-EVEN SORTING DATA\n---------------------------------------------\n");
	// Sorting, size 100, 1000, 10000, 100000
	for (int size = 100; size < 100001; size *= 10)
	{
		std::cout << "Working on size: " << size << std::endl;

		// Allocate memory for arrays
		data = (int*)malloc((size + 1) * size * sizeof(int));
		backup = (int*)malloc((size + 1) * size * sizeof(int));
		data2 = (int*)malloc((size + 1) * size * sizeof(int));

		// Fill arrays
		fillArrays(data, data2, backup, size);
		
		// CPU SORTING
		unoptimizedSort(data, size, file);

		// GPU SORTING
		//for (int tasksPerThread = 1; tasksPerThread < 2; ++tasksPerThread)
		//{
		//	std::cout << "Tasks per thread: " << tasksPerThread << std::endl;

		//	int threads = (size + 1) / tasksPerThread;
		//	int blocks = (threads - 1) / 1024 + 1; // 1024 to match current GPU limitations

		//	// Call GPU helper function
		//	cudaSort(data2, size, blocks, tasksPerThread, file);
		//}
		std::cout << std::endl << "------------------------------------------------------------------" << std::endl;
	}
	

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	/*cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset function in main failed.");
		return 1;
	}*/

	fclose(file);
	std::cout << "FINISHED! Press any key to exit." << std::endl;
	std::cin.get();

	return 0;
}


void fillArrays(int* data, int* data2, int* backup, int size)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = data2[i] = backup[i] = rand() % size + 1;
	}
}

void copyArray(int* data, int* backup, int size)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = backup[i];
	}
}

void unoptimizedSort(int* randomNumbers, int size, FILE* file)
{
	clock_t t;
	t = clock();

	bool sorted = false;

	// Loop until sorted
	while (!sorted)
	{
		int index = 0;
		sorted = true;

		// Sort even indices
		for (index; index < size - 2; index += 2)
		{
			if (randomNumbers[index] > randomNumbers[index + 1])
			{
				int temp = randomNumbers[index];
				randomNumbers[index] = randomNumbers[index + 1];
				randomNumbers[index + 1] = temp;
				sorted = false;
			}
		}
		//std::cout << "CPU - Finished sorting even indices" << std::endl;
		// Sort odd indices
		index = 1;
		for (index; index < size - 2; index += 2)
		{
			if (randomNumbers[index] > randomNumbers[index + 1])
			{
				int temp = randomNumbers[index];
				randomNumbers[index] = randomNumbers[index + 1];
				randomNumbers[index + 1] = temp;
				sorted = false;
			}
		}
		//std::cout << "CPU - Finished sorting odd indices" << std::endl;
	}
	std::cout << "CPU - Finished Sorting" << std::endl;
	t = clock() - t;

	std::cout << "CPU Odd-Even Sorting took: " << t << " clicks and " << ((float)t)/CLOCKS_PER_SEC << " seconds." << std::endl;
	fprintf(file, "CPU: %i %.4f \n", size,((float)t) / CLOCKS_PER_SEC);
	
	testIfSorted(randomNumbers);
}

void testIfSorted(int* randomNumbers)
{
	// Loop through array and check if sorted
	bool sorted = true;
	for (int i = 1; i < sizeof(randomNumbers); ++i)
	{
		if (randomNumbers[i] < randomNumbers[i - 1])
			sorted = false;
	}
	if (sorted)
		cout << "The array is sorted!" << endl;
	else
		cout << "The array is not sorted..." << endl;
}


// CUDA allocating function
void cudaSort(int* data2, int size, int blocks, int tasksPerThread, FILE* file)
{
	//	int *dev_randomNumbers = 0;
	//	int *dev_sortedNumbers = 0;
	//	int *sortedNumbers = 0;
	//	cudaError_t cudaStatus;
	//
	//	//Link CUDA with GPU and error-check.
	//	cudaStatus = cudaSetDevice(0);
	//	if (cudaStatus != cudaSuccess)
	//	{
	//		fprintf(stderr, "cudaSetDevice function failed. Could not find capable GPU.");
	//		goto Error;
	//	}
	//
	//	// Allocate GPU buffer for random int array and error-check.
	//	cudaStatus = cudaMalloc((void**)&dev_randomNumbers, sizeRandoms * sizeof(int));
	//	if (cudaStatus != cudaSuccess)
	//	{
	//		fprintf(stderr, "cudaMalloc of random int array failed.");
	//		goto Error;
	//	}
	//
	//	// Allocate GPU buffer for sorted int array and error-check.
	//	cudaStatus = cudaMalloc((void**)&dev_sortedNumbers, sizeRandoms * sizeof(int));
	//	if (cudaStatus != cudaSuccess)
	//	{
	//		fprintf(stderr, "cudaMalloc of sorted int array failed.");
	//		goto Error;
	//	}
	//
	//	//Copy input int array from host memory to GPU buffers.
	//	cudaStatus = cudaMemcpy(dev_randomNumbers, randomNumbersOptimized, sizeRandoms * sizeof(int), cudaMemcpyHostToDevice);
	//	if (cudaStatus != cudaSuccess)
	//	{
	//		fprintf(stderr, "cudaMemcpy of random ints failed.");
	//		goto Error;
	//	}
	//
	//
	//	// Launch a kernel on the GPU with one thread for each element.
	//	double start = clock();
	//	oddEvenKernel <<<2, 1024>>> (dev_randomNumbers, dev_sortedNumbers); //2 blocks, 1024 threads per block?
	//	double finish = clock() - start;
	//	finish /= CLOCKS_PER_SEC;
	//
	//	// Check for any errors launching the kernel
	//	cudaStatus = cudaGetLastError();
	//	if (cudaStatus != cudaSuccess)
	//	{
	//		fprintf(stderr, "oddEvenKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	//		goto Error;
	//	}
	//
	//	cout << std::endl << "Optimized time taken: " << finish << std::endl;
	//
	//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	//	// any errors encountered during the launch.
	//	cudaStatus = cudaDeviceSynchronize();
	//	if (cudaStatus != cudaSuccess)
	//	{
	//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching oddEvenKernel!\n", cudaStatus);
	//		goto Error;
	//	}
	//
	//	//Copy sorted int array back to CPU.
	//	cudaStatus = cudaMemcpy(sortedNumbers, dev_sortedNumbers, sizeRandoms * sizeof(int), cudaMemcpyDeviceToHost);
	//	if (cudaStatus != cudaSuccess)
	//	{
	//		fprintf(stderr, "cudaMemcpy from GPU to CPU failed.");
	//		goto Error;
	//	}
	//	//testIfSorted(sortedNumbers);
	//Error:
	//	cudaFree(dev_randomNumbers);
	//	cudaFree(dev_sortedNumbers);
	//
	//	return cudaStatus;
	return;
}

// GPU Kernel function
__global__ void oddEvenKernel(const int *randomNumberOptimized, int *sortedNumbers)
{
	int i = threadIdx.x;
	//c[i] = a[i] + b[i];
	sortedNumbers[i] = randomNumberOptimized[i];

	//Odd Even Sort with optimization
	/*bool sorted = false;
	while (!sorted)
	{
		int index = 0;
		sorted = true;

		for (index; index < 99998; index += 2)
		{
			if (randomNumbers[index] > randomNumbers[index + 1])
			{
				int temp = randomNumbers[index];
				randomNumbers[index] = randomNumbers[index + 1];
				randomNumbers[index + 1] = temp;
				sorted = false;
			}
		}

		index = 1;

		for (index; index < 99998; index += 2)
		{
			if (randomNumbers[index] > randomNumbers[index + 1])
			{
				int temp = randomNumbers[index];
				randomNumbers[index] = randomNumbers[index + 1];
				randomNumbers[index + 1] = temp;
				sorted = false;
			}
		}
	}*/
}