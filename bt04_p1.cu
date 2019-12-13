#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
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
		cudaEventSynchronize(start);
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
Scan within each block's data (work-inefficient), write results to "out", and
write each block's sum to "blkSums" if "blkSums" is not NULL.
*/
__global__ void scanBlkKernel(int * in, int n, int * out, int * blkSums)
{
	// TODO
	extern __shared__ int s_data[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_data[threadIdx.x] = (i < n) ? in[i] : 0;
	__syncthreads();

	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		int temp = 0;
		if (threadIdx.x >= stride) {
			temp = s_data[threadIdx.x - stride];
		}
		__syncthreads();
		if (threadIdx.x >= stride) {
			s_data[threadIdx.x] += temp;
		}
		__syncthreads();
	}

	out[i] = s_data[threadIdx.x];


	if (blkSums != NULL && threadIdx.x == 0) {
		blkSums[blockIdx.x] = s_data[blockDim.x - 1];
	}
}

// TODO: You can define necessary functions here
__global__ void scanSumKernel(int *in, int *blkSums, int n) {
	if (blockIdx.x >= 1) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		in[i] += blkSums[blockIdx.x - 1];
	}
}

void scan(int * in, int n, int * out,
	bool useDevice = false, dim3 blkSize = dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		printf("\nScan by host\n");
		out[0] = in[0];
		for (int i = 1; i < n; i++)
		{
			out[i] = out[i - 1] + in[i];
		}
	}
	else // Use device
	{
		printf("\nScan by device\n");
		// TODO

		// Khoi tao kich thuoc grid va block
		dim3 gridSize((n - 1) / blkSize.x + 1);
		int smem_size = blkSize.x * sizeof(int);

		// Khoi tao bien blkSums
		int *blkSums = (int*)malloc(gridSize.x * sizeof(int));

		// Cap phat bo nho
		int *d_in; CHECK(cudaMalloc(&d_in, n * sizeof(int)));
		int *d_out; CHECK(cudaMalloc(&d_out, n * sizeof(int)));
		int *d_blkSums; CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));

		// Sao chep du lieu tu host sang device
		CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

		// Goi ham kernel scan
		scanBlkKernel << <gridSize, blkSize, smem_size >> > (d_in, n, d_out, d_blkSums);
		CHECK(cudaGetLastError());

		// Chep du lieu tu device sang host
		CHECK(cudaMemcpy(blkSums, d_blkSums, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));

		// Goi ham scan tai host cho mang blkSums
		int *scan_blkSums = (int*)malloc(gridSize.x * sizeof(int));
		scan_blkSums[0] = blkSums[0];
		for (int i = 1; i < gridSize.x; i++)
		{
			scan_blkSums[i] = scan_blkSums[i - 1] + blkSums[i];
		}

		// Chep du lieu tu host sang device
		CHECK(cudaMemcpy(d_blkSums, scan_blkSums, gridSize.x * sizeof(int), cudaMemcpyHostToDevice));

		// Goi ham kernel de tinh tong
		scanSumKernel << <gridSize, blkSize >> > (d_out, d_blkSums, n);
		CHECK(cudaGetLastError());

		// Chep du lieu tu device sang host
		CHECK(cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

		// Giai phong du lieu
		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));
		CHECK(cudaFree(d_blkSums));
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

	// SET UP INPUT SIZE
	int n = (1 << 24) + 1;
	printf("\nInput size: %d\n", n);

	// ALLOCATE MEMORIES
	size_t bytes = n * sizeof(int);
	int * in = (int *)malloc(bytes);
	int * out = (int *)malloc(bytes); // Device result
	int * correctOut = (int *)malloc(bytes); // Host result

	// SET UP INPUT DATA
	for (int i = 0; i < n; i++)
		in[i] = (int)(rand() & 0xFF) - 127; // random int in [-127, 128]

	// DETERMINE BLOCK SIZE
	dim3 blockSize(512);
	if (argc == 2)
	{
		blockSize.x = atoi(argv[1]);
	}

	// SCAN BY HOST
	scan(in, n, correctOut);

	// SCAN BY DEVICE
	scan(in, n, out, true, blockSize);
	checkCorrectness(out, correctOut, n);

	// FREE MEMORIES
	free(in);
	free(out);
	free(correctOut);

	return EXIT_SUCCESS;
}

