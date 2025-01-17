#include <stdio.h>
#include <stdint.h>

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

// Sequential radix sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
void sortByHost(const uint32_t * in, int n,
	uint32_t * out,
	int nBits)
{
	int nBins = 1 << nBits; // 2^nBits
	int * hist = (int *)malloc(nBins * sizeof(int));
	int * histScan = (int *)malloc(nBins * sizeof(int));

	// In each counting sort, we sort data in "src" and write result to "dst"
	// Then, we swap these 2 pointers and go to the next counting sort
	// At first, we assign "src = in" and "dest = out"
	// However, the data pointed by "in" is read-only 
	// --> we create a copy of this data and assign "src" to the address of this copy
	uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
	memcpy(src, in, n * sizeof(uint32_t));
	uint32_t * originalSrc = src; // Use originalSrc to free memory later
	uint32_t * dst = out;

	// Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
	// (Each digit consists of nBits bits)
	// In each loop, sort elements according to the current digit 
	// (using STABLE counting sort)
	for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
	{
		// TODO: Compute "hist" of the current digit
		memset(hist, 0, nBins * sizeof(int));
		for (int i = 0; i < n; i++) {
			int bin = (src[i] >> bit) & (nBins - 1);
			hist[bin]++;
		}
		// TODO: Scan "hist" (exclusively) and save the result to "histScan"
		histScan[0] = 0;
		for (int i = 1; i < nBins; i++) {
			histScan[i] = histScan[i - 1] + hist[i - 1];
		}
		// TODO: From "histScan", scatter elements in "src" to correct locations in "dst"
		for (int i = 0; i < n; i++) {
			int bin = (src[i] >> bit) & (nBins - 1);
			dst[histScan[bin]] = src[i];
			histScan[bin]++;
		}
		// TODO: Swap "src" and "dst"
		uint32_t *temp = src;
		src = dst;
		dst = temp;
	}

	// TODO: Copy result to "out"
	memcpy(out, src, n * sizeof(uint32_t));

	// Free memories
	free(hist);
	free(histScan);
	free(originalSrc);
}


__global__ void createBits(uint32_t *d_in, int n, int *d_bits, int bit) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		d_bits[i] = (d_in[i] >> bit) & 1;
	}
}

__global__ void scanBlkKernel(int * d_in, int n, int *d_out, int *d_blkSums)
{
	extern __shared__ uint32_t s_data[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	s_data[threadIdx.x] = (i < n) ? d_in[i] : 0;
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

	d_out[i] = s_data[threadIdx.x];


	if (d_blkSums != NULL && threadIdx.x == 0) {
		d_blkSums[blockIdx.x] = s_data[blockDim.x - 1];
	}
}

__global__ void scanSumKernel(int *in, int *blkSums, int n) {
	if (blockIdx.x >= 1) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		in[i] += blkSums[blockIdx.x - 1];
	}
}

__global__ void scatter(uint32_t *d_in, int n, uint32_t *d_out, int *d_bitScan, int bit, int nZeros) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		int digit = (d_in[i] >> bit) & 1;
		if (digit == 0) {
			int rank = i - d_bitScan[i];
			d_out[rank] = d_in[i];
		}
		else {
			int rank = nZeros + d_bitScan[i];
			d_out[rank] = d_in[i];
		}
	}
}

void swapMemories(uint32_t *&a, uint32_t *&b) {
	uint32_t* temp = a;
	a = b;
	b = temp;
}

void scanBlockSum(int *in, int *out, int size) {
	out[0] = in[0];
	for (int i = 1; i < size; i++)
	{
		out[i] = out[i - 1] + in[i];
	}
}

void createHistExclusiveScan(int *histScan, int *temp, int nBins) {
	memcpy(temp, histScan, nBins * sizeof(int));
	histScan[0] = 0;
	for (int i = 1; i < nBins; i++) {
		histScan[i] = temp[i - 1];
	}
}
void sortByDevice(const uint32_t * in, int n,
	uint32_t * out,
	int * blockSizes)
{
	// TODO
	// Tao mang bitScan, d_bit, d_bitScan va cap phat bo nho
	int *d_bit; CHECK(cudaMalloc(&d_bit, n * sizeof(int)));
	int *bitScan = (int *)malloc(n * sizeof(int));
	int *d_bitScan; CHECK(cudaMalloc(&d_bitScan, n * sizeof(int)));
	
	// Tao mang d_in va sao chep du lieu tu in sang d_in
	uint32_t *d_in; CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
	CHECK(cudaMemcpy(d_in, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

	// Tao mang d_out 
	uint32_t *d_out; CHECK(cudaMalloc(&d_out, n * sizeof(uint32_t)));

	// Initialize gridSizeBit, gridSizeScan, gridSizeScatter
	dim3 gridSizeBit((n - 1) / blockSizes[0] + 1);
	dim3 gridSizeScan((n - 1) / blockSizes[1] + 1);
	dim3 gridSizeScatter((n - 1) / blockSizes[2] + 1);

	// Initialize smem_size
	int smem_size = blockSizes[1] * sizeof(int);

	// Initialize d_blkSums
	int *blkSums = (int*)malloc(gridSizeScan.x * sizeof(int));
	int *d_blkSums; CHECK(cudaMalloc(&d_blkSums, gridSizeScan.x * sizeof(int)));
	int *scan_blkSums = (int*)malloc(gridSizeScan.x * sizeof(int));
	int *bitScanTemp = (int *)malloc(n * sizeof(int));

	for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += 1)
	{
		// TODO: Create bits
		createBits << <gridSizeBit, blockSizes[0] >> > (d_in, n, d_bit, bit);
		CHECK(cudaGetLastError());
		
		// TODO: Scan "d_bit" (exclusively) and save the result to "d_bitScan"
		// Goi ham kernel scan
		scanBlkKernel << <gridSizeScan, blockSizes[1], smem_size >> > (d_bit, n, d_bitScan, d_blkSums);
		CHECK(cudaGetLastError());
		
		// Chep du lieu tu device sang host
		CHECK(cudaMemcpy(blkSums, d_blkSums, gridSizeScan.x * sizeof(int), cudaMemcpyDeviceToHost));

		// Goi ham scan tai host cho mang blkSums
		scanBlockSum(blkSums, scan_blkSums, gridSizeScan.x);

		// Chep du lieu tu host sang device
		CHECK(cudaMemcpy(d_blkSums, scan_blkSums, gridSizeScan.x * sizeof(int), cudaMemcpyHostToDevice));

		// Goi ham kernel de tinh tong
		scanSumKernel << <gridSizeScan, blockSizes[1] >> > (d_bitScan, d_blkSums, n);
		CHECK(cudaGetLastError());

		// Chep du lieu tu device sang host
		CHECK(cudaMemcpy(bitScan, d_bitScan, n * sizeof(int), cudaMemcpyDeviceToHost));

		// Tao mang exclusive
		createHistExclusiveScan(bitScan, bitScanTemp, n);

		// Chep du lieu tu "bitScan" sang "d_bitScan"
		CHECK(cudaMemcpy(d_bitScan, bitScan, n * sizeof(int), cudaMemcpyHostToDevice));


		// TODO: Scatter
		uint32_t *end = (uint32_t*)malloc(1 * sizeof(uint32_t));
		CHECK(cudaMemcpy(end, &d_in[n - 1], 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		int nZeros = n - bitScan[n - 1] - ((end[0] >> bit) & 1);
		scatter << <gridSizeScatter, blockSizes[2] >> > (d_in, n, d_out, d_bitScan, bit, nZeros);
		CHECK(cudaGetLastError());

		// TODO: Swap "d_in" and "d_out"
		swapMemories(d_in, d_out);
	}

	// TODO: Copy result to "out"
	CHECK(cudaMemcpy(out, d_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	// Free memories
	free(bitScan);
	free(blkSums);
	free(scan_blkSums);
	free(bitScanTemp);
	CHECK(cudaFree(d_in));
	CHECK(cudaFree(d_out));
	CHECK(cudaFree(d_bit));
	CHECK(cudaFree(d_bitScan));
	CHECK(cudaFree(d_blkSums));
}

// Radix sort
void sort(const uint32_t * in, int n,
	uint32_t * out,
	int nBits,
	bool useDevice = false, int * blockSizes = NULL)
{
	GpuTimer timer;
	timer.Start();

	if (useDevice == false)
	{
		printf("\nRadix sort by host\n");
		sortByHost(in, n, out, nBits);
	}
	else // use device
	{
		printf("\nRadix sort by device\n");
		sortByDevice(in, n, out, blockSizes);
	}

	timer.Stop();
	printf("Time: %.3f ms\n", timer.Elapsed());
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

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
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

void printArray(uint32_t * a, int n)
{
	for (int i = 0; i < n; i++)
		printf("%i ", a[i]);
	printf("\n");
}

int main(int argc, char ** argv)
{
	// PRINT OUT DEVICE INFO
	printDeviceInfo();

	// SET UP INPUT SIZE
	int n = (1 << 24) + 1;
	//n = 12;
	printf("\nInput size: %d\n", n);

	// ALLOCATE MEMORIES
	size_t bytes = n * sizeof(uint32_t);
	uint32_t * in = (uint32_t *)malloc(bytes);
	uint32_t * out = (uint32_t *)malloc(bytes); // Device result
	uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

	// SET UP INPUT DATA
	for (int i = 0; i < n; i++)
		in[i] = rand();
	//printArray(in, n);
	//uint32_t temp[12] = { 0,1,1,1,0,1,0,1,0,0,1,0 };
	//memcpy(in, temp, 12 * sizeof(uint32_t));

	// SET UP NBITS
	int nBits = 4; // Default
	if (argc > 1)
		nBits = atoi(argv[1]);
	printf("\nNum bits per digit: %d\n", nBits);

	// DETERMINE BLOCK SIZES
	int blockSizes[3] = { 512, 512, 512 }; // One for histogram, one for scan
	if (argc == 5)
	{
		blockSizes[0] = atoi(argv[2]);
		blockSizes[1] = atoi(argv[3]);
		blockSizes[2] = atoi(argv[4]);
	}
	printf("\nHist block size: %d, scan block size: %d, scatter block size: %d\n", blockSizes[0], blockSizes[1], blockSizes[2]);

	// SORT BY HOST
	sort(in, n, correctOut, nBits);
	//printArray(correctOut, n);

	// SORT BY DEVICE
	sort(in, n, out, nBits, true, blockSizes);
	checkCorrectness(out, correctOut, n);

	// FREE MEMORIES 
	free(in);
	free(out);
	free(correctOut);

	return EXIT_SUCCESS;
}