#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

/*
Not use SMEM.
*/
__global__ void computeHistKernel1(int * in, int n, int * hist, int nBins)
{
	// TODO
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		atomicAdd(&hist[in[i]], 1);
	}
}

/*
Use SMEM.
*/
__global__ void computeHistKernel2(int * in, int n, int * hist, int nBins)
{
	// TODO
	// Each block computes its local hist using atomic on SMEM
	extern __shared__ int s_hist[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (int stride = threadIdx.x; stride < nBins; stride += blockDim.x) {
		s_hist[stride] = 0;
	}
	__syncthreads();
	if (i < n)
		atomicAdd(&s_hist[in[i]], 1);
	__syncthreads();

	// Each block adds its local hist to global hist using atomic on GMEM
	for (int stride = threadIdx.x; stride < nBins; stride += blockDim.x) {
		atomicAdd(&hist[stride], s_hist[stride]);
	}
}

void computeHist(int * in, int n, int * hist, int nBins,
	bool useDevice = false, dim3 blockSize = dim3(1), int kernelType = 1)
{
	GpuTimer timer;
	timer.Start();

	if (useDevice == false)
	{
		printf("\nHistogram by host\n");
		memset(hist, 0, nBins * sizeof(int));
		for (int i = 0; i < n; i++)
			hist[in[i]]++;
	}
	else // Use device
	{
		printf("\nHistogram by device, kernel %d, ", kernelType);

		// Allocate device memories
		int * d_in, *d_hist;
		CHECK(cudaMalloc(&d_in, n * sizeof(int)));
		CHECK(cudaMalloc(&d_hist, nBins * sizeof(int)));

		// Copy data to device memories
		CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

		// TODO: Initialize d_hist using cudaMemset
		CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));

		// Call kernel
		dim3 gridSize((n - 1) / blockSize.x + 1);
		printf("block size: %d, grid size: %d\n", blockSize.x, gridSize.x);
		if (kernelType == 1)
		{
			computeHistKernel1 << <gridSize, blockSize >> > (d_in, n, d_hist, nBins);
		}
		else // kernelType == 2
		{
			size_t smemSize = nBins * sizeof(int);
			computeHistKernel2 << <gridSize, blockSize, smemSize >> > (d_in, n, d_hist, nBins);
		}
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		// Copy result from device memories
		CHECK(cudaMemcpy(hist, d_hist, nBins * sizeof(int), cudaMemcpyDeviceToHost));

		// Free device memories
		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_hist));
	}

	timer.Stop();
	printf("Processing time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
	CHECK(cudaGetDeviceProperties(&devProv, 0));
	printf("**********GPU info**********\n");
	printf("Name: %s\n", devProv.name);
	printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
	printf("Num SMs: %d\n", devProv.multiProcessorCount);
	printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
	printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
	printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
	printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
	printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
	printf("****************************\n");
}

void checkCorrectness(int * out, int * correctOut, int n)
{
	for (int i = 0; i < n; i++)
	{
		if (out[i] != correctOut[i])
		{
			printf("INCORRECT :(\n");
			return;
		}
	}
	printf("CORRECT :)\n");
}

int main(int argc, char ** argv)
{
	// PRINT OUT DEVICE INFO
	printDeviceInfo();

	// SET UP INPUT SIZE AND THE NUMBER OF BINS
	int n = (1 << 24) + 1;
	int nBins = 256;
	printf("\nInput size: %d\n", n);

	// ALLOCATE MEMORIES
	int * in = (int *)malloc(n * sizeof(int));
	int * hist = (int *)malloc(nBins * sizeof(int)); // Device result
	int * correctHist = (int *)malloc(nBins * sizeof(int)); // Host result

	// SET UP INPUT DATA
	for (int i = 0; i < n; i++)
		in[i] = (int)(rand() & (nBins - 1)); // random int in [0, nBins-1]

	// DETERMINE BLOCK SIZE
	dim3 blockSize(512); // Default
	if (argc == 2)
		blockSize.x = atoi(argv[1]);

	// COMPUTE HISTOGRAM BY HOST
	computeHist(in, n, correctHist, nBins);

	// COMPUTE HISTOGRAM BY DEVICE, KERNEL 1
	computeHist(in, n, hist, nBins, true, blockSize, 1);
	checkCorrectness(hist, correctHist, nBins);

	// COMPUTE HISTOGRAM BY DEVICE, KERNEL 2
	memset(hist, 0, nBins * sizeof(int)); // Reset output
	computeHist(in, n, hist, nBins, true, blockSize, 2);
	checkCorrectness(hist, correctHist, nBins);

	// FREE MEMORIES
	free(in);
	free(hist);
	free(correctHist);

	return EXIT_SUCCESS;
}
