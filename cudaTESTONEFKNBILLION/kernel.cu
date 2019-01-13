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
bool gpuSortingTest(int* data);

void cudaSort(int* &data, int size, int blocks, int tasksPerThread, FILE* file);
__global__ void oddEvenKernel(int* data, int size, int tasksPerThread, int index);

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
		data = (int*)malloc((size + 1) * sizeof(int));
		backup = (int*)malloc((size + 1) * sizeof(int));
		data2 = (int*)malloc((size + 1) * sizeof(int));

		// Fill arrays
		fillArrays(data, data2, backup, size);
		
		// CPU SORTING
		unoptimizedSort(data, size, file);

		// GPU SORTING
		for (int tasksPerThread = 1; tasksPerThread < 9; tasksPerThread *= 2)
		{
			std::cout << "Tasks per thread: " << tasksPerThread << std::endl;

			int threads = (size + 1) / tasksPerThread;
			int blocks = (threads - 1) / 1024 + 1; // 1024 to match current GPU limitations

			// Call GPU helper function
			cudaSort(data2, size, blocks, tasksPerThread, file);
		}
		std::cout << std::endl << "------------------------------------------------------------------" << std::endl;

		// Release array memory
		free(data);
		free(data2);
		free(backup);
	}
	

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset function in main failed.");
		return 1;
	}

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
		cout << endl << "The array is sorted!" << endl;
	else
		cout << endl << "The array is not sorted..." << endl;
}

bool gpuSortingTest(int* data)
{
	// Loop through array and check if sorted
	bool sorted = true;
	for (int i = 1; i < sizeof(data); ++i)
	{
		if (data[i] < data[i - 1])
			sorted = false;
	}
	return sorted;
}

// CUDA allocating function
void cudaSort(int* &data, int size, int blocks, int tasksPerThread, FILE* file)
{
	int* devArray = 0;
	clock_t t;
	t = clock();

	// Allocate array to GPU
	cudaError_t cudaStatus = cudaMalloc((void**)&devArray, (size + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for array\n");
		return;
	}
	// Copy array data to GPU
	cudaStatus = cudaMemcpy(devArray, data, (size + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for CPU -> GPU\n");
		return;
	}

	// Create temp array to retrieve array back from GPU
	int* tempArray = (int*)malloc((size + 1) * sizeof(int));

	// Call kernel function
	bool sorted = false;
	while (!sorted)
	{
		for (int i = 0; i < (size - 2); i += 2) // change how often its called
		{
			//cout << "Call GPU for even, current index: " << i << endl;
			oddEvenKernel << <blocks, 1024 >> > (devArray, size, tasksPerThread, i);
			//oddEvenKernel << <blocks, 1024 >> > (devArray, size, tasksPerThread, i * tasksPerThread);
		}
		for (int i = 1; i < (size - 2); i += 2) // change how often its called
		{
			oddEvenKernel << <blocks, 1024 >> > (devArray, size, tasksPerThread, i);
			//oddEvenKernel << <blocks, 1024 >> > (devArray, size, tasksPerThread, i * tasksPerThread);
		}

		// Retreive sorted array back from GPU
		cudaStatus = cudaMemcpy((void*)tempArray, (void*)devArray, (size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed for GPU -> CPU\n");
			return;
		}
		sorted = gpuSortingTest(tempArray);
		//cout << "Sorted: " << sorted << endl;
	}


	data = tempArray;
	testIfSorted(data);

	t = clock() - t;
	std::cout << "GPU sorting took: " << t << "clicks (" << ((int)t) / CLOCKS_PER_SEC << " seconds.)" << endl;
	fprintf(file, "GPU %i: %.4i,", size, ((int)t) / CLOCKS_PER_SEC);

	cudaFree(devArray);
	cudaFree(tempArray);
}

// GPU Kernel function
__global__ void oddEvenKernel(int* data, int size, int tasksPerThread, int rowIndex)
{
	// Sort even indices
	if (data[rowIndex] > data[rowIndex + 1])
	{
		int temp = data[rowIndex];
		data[rowIndex] = data[rowIndex + 1];
		data[rowIndex + 1] = temp;
	}
}