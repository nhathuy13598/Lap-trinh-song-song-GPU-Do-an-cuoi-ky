#include <stdint.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#define LIBRARY 0
#define HOST 1
#define DEVICE 2
#define CHECK(call)                                            \
{                                                              \
	const cudaError_t error = call;                            \
	if (error != cudaSuccess)                                  \
	{                                                          \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
		fprintf(stderr, "code: %d, reason: %s\n", error,       \
				cudaGetErrorString(error));                    \
		exit(1);                                               \
	}                                                          \
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

	void Stop() { cudaEventRecord(stop, 0); }

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
// Ta sẽ sử dụng ý tưởng từ thuật toán sắp xếp tuần tự mới chứ không phải
// thuật toán mà ta đã sử dụng trong bài tập số 3
/**
    Sắp xếp tuần tự trên host
    @blockSize kích thước một block mà ta sẽ duyệt (Ta vẫn duyệt tuần tự)
*/
void sortByHost(const uint32_t *in, int n, uint32_t *out, int nBits, int blockSize)
{
	int nBins = 1 << nBits; // Số lượng bin là 2^nBits

	uint32_t *src = (uint32_t *)malloc(n * sizeof(uint32_t));			   // Biến tạm để lưu trữ dữ liệu input
	memcpy(src, in, n * sizeof(uint32_t));		// Sao chép dữ liệu từ in vào src
	uint32_t *originalSrc = src;		   		// Use originalSrc to free memory later
	uint32_t *dst = out;				   		// Mảng kết quả

	// [TODO]: Khởi tạo các mảng cần thiết khi chạy
	int sizeHist = nBins * ((n - 1) / blockSize +1);	// Tính kích thước của mảng listLocalHist sizeHist = Số bin x Số lượng block
	int *listLocalHist = (int *)malloc(sizeHist * sizeof(int)); 			// Mảng chứa các localHist
	int *listLocalHistConvert = (int *)malloc(sizeHist * sizeof(int)); 		// Mảng chuyển đổi của listLocalHistConvert

	int *histScan = (int *)malloc(sizeHist * sizeof(int));		// Mảng exclusive scan của listLocalHistConvert

	int *eleBefore =(int *)malloc(n * sizeof(int)); 			// Mảng chứa chỉ số phân tử đứng trước và bằng nó trong từng block

	int numBlock = (n - 1) / blockSize + 1; // Số lượng các block cần thiết
	for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
	{
		// TODO: Mỗi block tính local histogram của digit-đang-xét trên phần
		// dữ liệu của mình và chép vào mảng listLocalHist
		memset(listLocalHist, 0, sizeHist * sizeof(int)); // Gán mảng listLocalHist bằng 0
		for (int blkIdx = 0; blkIdx < numBlock; blkIdx++)
		{
			int start = blkIdx * blockSize; 								// Chỉ số bắt đầu của block
			int end = (blkIdx == numBlock - 1)? n : start + blockSize; 		// Chỉ số kết thúc của block
			for (int index = start; index < end; index++)
			{ // Duyệt tất cả phần tử của block
				int bin = (src[index] >> bit) & (nBins - 1);
				listLocalHist[blkIdx * nBins + bin]++;
			}
		}

		// [DEBUG]: In ra mảng listLocalHist
		/*printf("Mang listLocalHist: ");
    	for(int i=0; i < sizeHist; i++){
        	printf("%d ", listLocalHist[i]);
    	}
    	printf("\n");*/

		// TODO: Với mảng 2 chiều mà mỗi dòng là local hist của một block,
		// thực hiện exclusive scan trên mảng một chiều gồm các cột
		// nối lại với nhau (Xem slide để hiểu rõ)
		int indexLLHC = 0; // Chỉ số trong mảng listLocalHistConvert
		for (int i = 0; i < nBins; i++)
		{ // Duyệt tất cả các phần tử trong một localHist
			for (int j = 0; j < numBlock; j++)
			{ // Duyệt tất cả các localHist
				listLocalHistConvert[indexLLHC++] =listLocalHist[i + j * nBins]; 	// i là chỉ số bin trong localHist
												  									// j * nBins là chỉ số của block
			}
		}
		// [DEBUG]: In ra mảng listLocalHistConvert
		/*printf("Mang listLocalHistConvert: ");
		for(int i=0; i<sizeHist; i++){
			printf("%d ", listLocalHistConvert[i]);
		}
    	printf("\n");*/

		// Tính histScan (exculusive scan) cho mảng listLocalHistConvert
		histScan[0] = 0;
		for (int i = 1; i < sizeHist; i++)
		{
			histScan[i] = histScan[i - 1] + listLocalHistConvert[i - 1];
		}

		// [DEBUG]: In ra mảng histScan
		/*printf("Mang histScan: ");
		for(int i=0; i<sizeHist; i++){
			printf("%d ", histScan[i]);
		}
		printf("\n");*/

		// TODO: Mỗi block thực hiện scatter phần dữ liệu của mình xuống
		// mảng output dựa vào kết quả scan ở trên
		//      ▪ Mỗi block sắp xếp cục bộ phần dữ liệu của mình theo digit đang
		//          xét (dùng Radix Sort với k=1 bit và làm trên SMEM)
		//      ▪ Mỗi block tính chỉ số bắt đầu (xét cục bộ trong block) của mỗi giá
		//          trị digit
		//      ▪ Mỗi thread trong block tính số lượng phần tử đứng trước mình
		//          trong block có digit-đang-xét bằng digit-đang-xét của phần tử mà
		//          mình phụ trách
		//      ▪ Mỗi thread trong block tính rank và thực hiện scatter

		// Sắp xếp các data trong block tăng dần theo Bubble Sort
		// Ta sẽ thực hiện luôn trên mảng src để tiết kiệm bộ nhớ
		for (int blkIdx = 0; blkIdx < numBlock; blkIdx++)
		{	// Duyệt từng block
			int start = blkIdx * blockSize; 								// Chỉ số bắt đầu của block
			int end = (blkIdx == numBlock - 1) ? n : start + blockSize; 	// Chỉ số kết thúc của block
			for (int x = end - start; x >= 1; x--)
			{ // Ta tưởng tượng đây là sắp xếp mảng có end - start phần tử
				for (int y = 0; y < x - 1; y++)
				{
					int first = (src[blkIdx * blockSize + y] >> bit) & (nBins - 1);
					int second = (src[blkIdx * blockSize + y + 1] >> bit) & (nBins - 1);
					if (first > second)
					{
						uint32_t temp = src[blkIdx * blockSize + y];
						src[blkIdx * blockSize + y] = src[blkIdx * blockSize + y + 1];
						src[blkIdx * blockSize + y + 1] = temp;
					}
				}
			}
		}

		// [DEBUG]: In ra mảng sortBlockData
		/*printf("Mang da duoc sap xep theo block: ");
		for(int i=0; i<n; i++){
			printf("%d ", src[i]);
		}
		printf("\n");*/

		// Tính chỉ số bắt đầu trong block và tính luôn số lượng
		// phần tử giống nó và đứng trước nó
		memset(eleBefore, 0, n * sizeof(int)); // Khởi tạo mảng chứa các phần tử đứng trước bằng 0
		for (int blkIdx = 0; blkIdx < numBlock; blkIdx++)
		{
			int start = blkIdx * blockSize; // Chỉ số bắt đầu của block
			int end = (blkIdx == numBlock - 1) ? n : start + blockSize; // Chỉ số kết thúc của block
			for (int index = 1; index < end - start; index++)
			{
				int first = (src[blkIdx * blockSize + index - 1] >> bit) & (nBins - 1);
				int second = (src[blkIdx * blockSize + index] >> bit) & (nBins - 1);
				if (first == second)
				{
					eleBefore[blkIdx * blockSize + index] = eleBefore[blkIdx * blockSize + index - 1] + 1;
				}
			}
		}
		// [DEBUG]: In ra mảng eleBefore
		/*printf("Mang eleBefore: ");
		for(int index=0; index<n; index++){
			printf("%d ", eleBefore[index]);
		}
		printf("\n");*/

		// Tính rank và scatter
		for (int index = 0; index < n; index++)
		{
			int blIdx = index / blockSize;
			int bin = (src[index] >> bit) & (nBins - 1);
			int rank = histScan[bin * numBlock + blIdx] + eleBefore[index];
			dst[rank] = src[index];
		}

		// [DEBUG]: Mang dst
		/*printf("Mang dst: ");
		for (int index = 0; index < n; index++) {
			printf("%d ", dst[index]);
		}
		printf("\n");*/

		// TODO: Swap "src" and "dst"
		uint32_t *temp = src;
		src = dst;
		dst = temp;
	}
	// [DEBUG]: In mảng src
	/*printf("\nMang ket qua la: ");
	for(int index=0; index<n; index++){
		printf("%d ", src[index]);
	}
	printf("\n");*/

	// TODO: Copy result to "out"
	memcpy(out, src, n * sizeof(uint32_t));

	// Free memories
	free(listLocalHist);
	free(histScan);
	free(originalSrc);
	free(listLocalHistConvert);
	free(eleBefore);
}

void sortByLibrary(const uint32_t *in, int n, uint32_t *out, int nBits)
{
	// TODO
	thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}

#define getBin(num) (((num) >> (bit)) & ((nBins)-1))

__global__ void histogramKernel(uint32_t *in, int n, int *histArr, int nBits, int bit)
{
	extern __shared__ uint32_t s_in[];
	int nBins = 1 << nBits; // Số lượng bin
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Gán các bin của Hist là 0
	for (int stride = threadIdx.x; stride < nBins; stride += blockDim.x){ // Số lượng bin có thể lớn hơn số thread
		s_in[stride] = 0;
	}
	//s_in[nBins + threadIdx.x] = (idx < n) ? in[idx] : 0;
	__syncthreads();
	
	if (idx < n){
		atomicAdd(&s_in[getBin(in[idx])], 1);
		//atomicAdd(&s_in[getBin(s_in[nBins + threadIdx.x])], 1);
	}
	__syncthreads();

	for (int stride = threadIdx.x; stride < nBins; stride += blockDim.x){ // Số lượng bin có thể lớn hơn số thread
		histArr[blockIdx.x + gridDim.x * stride] = s_in[stride];
	}
}

__global__ void scanBlkKernel(int *in, int n, int *out, int *blkSums)
{
	extern __shared__ int temp[];

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	// load input into shared memory. this is exclusive scan, so shift right by
	// one and set first element to 0
	temp[threadIdx.x] = (threadIdx.x > 0) ? in[idx - 1] : 0;
	__syncthreads();

	if (idx >= n)
	{
		return;
	}

	out[idx] = temp[threadIdx.x];
	__syncthreads();

	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if (threadIdx.x >= stride)
		{
			out[idx] += temp[threadIdx.x - stride];
		}
		__syncthreads();
		temp[threadIdx.x] = out[idx];
		__syncthreads();
	}
	if (blkSums != NULL && (idx == n - 1 || threadIdx.x == blockDim.x - 1))
	{
		blkSums[blockIdx.x] = out[idx] + in[idx];
	}
}

__global__ void addBlkKernel(int *in, int n, int *blkSums)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIdx.x == 0 || idx >= n)
	{
		return;
	}
	in[idx] = in[idx] + blkSums[blockIdx.x - 1];
}

__global__ void transposeKernel(int *in, int n, int width, int *out)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= n)
	{
		return;
	}
	int x = idx % width;
	int y = idx / width;
	out[x * (n / width) + y] = in[idx];
}

__global__ void scatterKernel(uint32_t *in, int n, uint32_t *out,
								int *scanHistogramArrayTranspose, 
								int nBits, int bit)
{
	/*
	Smem sẽ gồm có ? phần dữ liệu
		1. blockDim.x phần tử (dữ liệu input)
		2. phần tử dummy có 1 phần tử
		3. blockDim.x phần tử (Chuỗi nhị phân)
		4. 2 ^ nBits phần tử (chứa chỉ số bắt đầu)
		5. 2 ^ nBits phần tử (chứa scanHistogramArrayTranspose cho từng block)
	*/
	int nBins = 1 << nBits; // Số lượng bin
	int size = blockDim.x; //  Số lượng phần tử trong block
	if (blockIdx.x == gridDim.x - 1){
		size = n - (gridDim.x - 1) * blockDim.x;
	}
	// Load dữ liệu từ in vào smem
	extern __shared__ uint32_t tp[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	tp[threadIdx.x] = (idx < n) ? in[idx] : 0;
	// Load scanHistogramArrayTranspose vào smem
	// Số thread có thể ít hơn số bin
	int startHistArr = 2 * blockDim.x + 1 + nBins;
	// if (threadIdx.x < nBins){
	// 	tp[startHistArr + threadIdx.x] = scanHistogramArrayTranspose[blockIdx.x * nBins + threadIdx.x];
	// }
	for (int i = 0; i < nBins; i += blockDim.x){
		if (threadIdx.x + i < nBins){
			tp[startHistArr + threadIdx.x + i] = scanHistogramArrayTranspose[blockIdx.x * nBins + threadIdx.x + i];
		}
	}
	__syncthreads();
	// FIXME: Debug
	/*if (threadIdx.x == 0){
		printf("Mang scan Hist: \n");
		for (int i = 0; i < nBins; i++){
			printf("%d %d\n", tp[startHistArr + i], scanHistogramArrayTranspose[blockIdx.x * nBins + threadIdx.x]);
			if (i == nBins - 1)
				printf("\n");
		}
	}
	__syncthreads();*/
	// Lấy ra nBits (bit) của các phần tử trong block với chỉ số bit đầu tiên là bit

	// Sắp xếp các phần tử trong block bằng nBits (bit) này
	int startBitArr = blockDim.x + 1; // Chỉ số bắt đầu chuỗi nhị phân
	int startBitScan = blockDim.x; // Chỉ số bắt đầu của mảng scan-chuỗi-nhị-phân
	int nZeros = 0; // Số lượng số 0
	for (int i = 0; i < nBits; i++){

		// Lấy chuỗi bit
		if (threadIdx.x < size){
			uint32_t oneBit = (getBin(tp[threadIdx.x]) >> i) & 1;
			tp[startBitArr + threadIdx.x] = oneBit;
		}
		//__syncthreads();

		// FIXME: Debug
		/*if (threadIdx.x == 0){
			printf("Chuoi bit: ");
			for (int i = 0; i < size; i++){
				printf("%d ", tp[startBitArr + i]);
				if (i == size - 1)
					printf("\n");
			}
		}
		__syncthreads();*/

		// Set giá trị cho chuỗi bit scan
		tp[blockDim.x] = 0;
		__syncthreads();
		// Scan chuỗi bit
		for (int stride = 1; stride < size; stride *= 2) {
			int temp = 0;
			if (threadIdx.x >= stride && threadIdx.x < size) {
				temp = tp[startBitScan + threadIdx.x - stride];
			}
			__syncthreads();
			if (threadIdx.x >= stride && threadIdx.x < size) {
				tp[startBitScan + threadIdx.x] += temp;
			}
			__syncthreads();
		}
		//__syncthreads();
		
		// FIXME: Debug
		/*if (threadIdx.x == 0){
			printf("Chuoi bit scan: ");
			for (int j = 0; j < size; j++){
				printf("%d ", tp[startBitScan + j]);
				if (j == size - 1)
					printf("\n");
			}
		}
		__syncthreads();*/

		// Scatter
		nZeros = size - tp[startBitScan + size - 1] - tp[startBitArr + size - 1];
		
		// Lấy phần tử trong mảng ra lưu lại
		uint32_t ele;
		if (threadIdx.x < size){
			ele = tp[threadIdx.x];
		}
		__syncthreads();
		if (threadIdx.x < size){
			uint32_t oneBit = (getBin(ele) >> i) & 1;
			if (oneBit == 0){
				int rank = threadIdx.x - tp[startBitScan + threadIdx.x];
				tp[rank] = ele;
			}
			else{
				int rank = nZeros + tp[startBitScan + threadIdx.x];
				tp[rank] = ele;
			}
		}
		__syncthreads();
		
		// FIXME: Debug
		/*if (threadIdx.x == 0){
			printf("So luong zero la: %d\n",nZeros);
			printf("Chuoi bit sau khi sap xep: ");
			for (int j = 0; j < size; j++){
				printf("%d ", tp[j]);
				if (j == size - 1)
					printf("\n");
			}
		}
		__syncthreads(); return;*/
	}

	// FIXME: Debug
	/*if (threadIdx.x == 0){
		printf("Mang da sap xep: ");
		for (int i = 0; i < size; i++){
			printf("%d ", tp[i]);
			if (i == size - 1)
				printf("\n");
		}
	}
	__syncthreads();*/

	// Tính chỉ số bắt đầu của từng bộ nBits (bit) trong block
	int startArrIdx = 2 * blockDim.x + 1; // Chỉ số bắt đầu của mảng chứa chỉ-số-bắt-đầu-của-từng-bộ-nBits
	if (threadIdx.x == 0){
		int bin = getBin(tp[threadIdx.x]);
		tp[startArrIdx + bin] = 0;
	}
	else if (threadIdx.x < size){
		if (getBin(tp[threadIdx.x]) != getBin(tp[threadIdx.x - 1])){
			tp[startArrIdx + getBin(tp[threadIdx.x])] = threadIdx.x;
		}
	}
	__syncthreads();

	// FIXME: Debug
	/*if (threadIdx.x == 0){
		printf("Mang chi so bat dau: ");
		for (int i=0; i < nBins; i++){
			printf("%d ", tp[startArrIdx + i]);
			if (i == nBins - 1)
				printf("\n");
		}
	}
	__syncthreads();*/

	// Tính số phần tử đứng trước nó theo từng bộ nBits (bit) của các phần tử trong block
	int startArrEleBef = blockDim.x; // Chỉ số bắt đầu của mảng chứa số-phần-tử-đứng-trước-nó
	int bin = getBin(tp[threadIdx.x]);
	tp[startArrEleBef + threadIdx.x] = threadIdx.x - tp[startArrIdx + bin];
	//__syncthreads(); // !CHÚ Ý: Đoạn này chưa hiểu tại sao không bị lỗi

	// FIXME: Debug
	/*if (threadIdx.x == 0){
		printf("Mang so luong phan tu bat dau: ");
		for (int i = 0; i < size; i++){
			printf("%d ", tp[startArrEleBef + i]);
			if (i == size - 1)
				printf("\n");
		}
	}
	__syncthreads();*/

	// FIXME: Debug
	/*if (threadIdx.x == 0){
		printf("Mang scan hist: ");
		for(int k = 0; k < nBins; k++){
			printf("%d ", scanHistogramArrayTranspose[k]);
			if(k == nBins - 1)
				printf("\n");
		}
	}*/

	// Scatter
	if (threadIdx.x < size){
		//int rank = scanHistogramArrayTranspose[blockIdx.x * nBins + bin] + tp[startArrEleBef + threadIdx.x];
		int rank = tp[startHistArr + bin] + tp[startArrEleBef + threadIdx.x]; 
		out[rank] = tp[threadIdx.x];
	}

	// FIXME: Debug
	/*__syncthreads();
	if (threadIdx.x == 0){
		printf("Mang sau khi da sap xep: ");
		for (int i = 0; i < size; i++){
			printf("%d ", out[i]);
			if (i == size - 1)
				printf("\n");
		}
	}*/
}


void sortByDevice(const uint32_t *in, int n, uint32_t *out, int nBits, int *blockSizes)
{

	// initialize data
	int nBins = 1 << nBits; // 2 ^ nBits
	int gridSizeHist = (n - 1) / blockSizes[0] + 1;
	int gridSizeScan = (gridSizeHist * nBins - 1) / blockSizes[1] + 1;

	// Allocate data on device
	int in_size = n * sizeof(uint32_t);
	int out_size = in_size;
	uint32_t *d_src, *d_dst;
	//uint32_t *dst = (uint32_t*)malloc(out_size);
	CHECK(cudaMalloc(&d_src, in_size));
	CHECK(cudaMalloc(&d_dst, out_size));
	CHECK(cudaMemcpy(d_src, in, in_size, cudaMemcpyHostToDevice));

	// Allocate another data on device
	size_t histArr_size = gridSizeHist * nBins * sizeof(int);
	size_t size_blksum = gridSizeScan * sizeof(int);
	//int *histArr = (int *)malloc(histArr_size);
	//int *scanHistArr = (int *)malloc(histArr_size);
	int *blkSums = (int *)malloc(size_blksum);
	int *d_histArr, *d_scanHistArr, *d_blkSums;
	CHECK(cudaMalloc(&d_histArr, histArr_size));
	CHECK(cudaMalloc(&d_scanHistArr, histArr_size));
	CHECK(cudaMalloc(&d_blkSums, size_blksum));
	

	// Set time
	GpuTimer timer;
	float histTime, scanTime, addTime, transposeTime, scatterTime;
	histTime = scanTime = addTime = transposeTime = scatterTime = 0;
	for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
	{
		CHECK(cudaMemset(d_histArr, 0, histArr_size));

		// TODO: Do histogram
		timer.Start();
		histogramKernel<<<gridSizeHist, blockSizes[0], 
							(/*blockSizes[0] +*/ nBins) * sizeof(uint32_t)>>>(d_src, n, d_histArr, nBits, bit);
		CHECK(cudaGetLastError());
		timer.Stop();
		histTime += timer.Elapsed();

		// FIXME: Debug
		/*CHECK(cudaMemcpy(histArr, d_histArr, histArr_size, cudaMemcpyDeviceToHost));
		for (int i = 0; i < gridSizeHist * nBins; ++i) {
		  	printf("%d\t", histArr[i]);
			if (i == gridSizeHist * nBins - 1)
				printf("\n===|===|===|===|===|===|===|===|===|===\n");
		}*/
		

		// TODO: Scan histogram
		timer.Start();
		scanBlkKernel<<<gridSizeScan, blockSizes[1], blockSizes[1] * sizeof(int)>>>(d_histArr, gridSizeHist * nBins, d_scanHistArr, d_blkSums);
		CHECK(cudaGetLastError());

		// copy result to host
		CHECK(cudaMemcpy(blkSums, d_blkSums, size_blksum, cudaMemcpyDeviceToHost));

		// scan vector blkSums
		for (int i = 1; i < gridSizeScan; ++i)
		{
			blkSums[i] += blkSums[i - 1];
		}
		timer.Stop();
		scanTime += timer.Elapsed();

		// copy data to device
		CHECK(cudaMemcpy(d_blkSums, blkSums, size_blksum, cudaMemcpyHostToDevice));

		// TODO: Add after scan
		timer.Start();
		addBlkKernel<<<gridSizeScan, blockSizes[1]>>>(d_scanHistArr, gridSizeHist * nBins, d_blkSums);
		CHECK(cudaGetLastError());
		timer.Stop();
		addTime += timer.Elapsed();

		// FIXME: Debug
		/*CHECK(cudaMemcpy(scanHistArr, d_scanHistArr, histArr_size, cudaMemcpyDeviceToHost));
		for (int i = 0; i < gridSizeHist * nBins; ++i) {
		  	printf("%d\t", scanHistArr[i]);
		  	if(i == gridSizeHist * nBins - 1)
		  		printf("\n===|===|===|===|===|===|===|===|===|===\n");
		}*/
		

		// TODO: Transpose
		timer.Start();
		transposeKernel<<<gridSizeScan, blockSizes[1]>>>(d_scanHistArr, gridSizeHist * nBins, gridSizeHist, d_histArr);
		CHECK(cudaGetLastError());
		timer.Stop();
		transposeTime += timer.Elapsed();

		// FIXME: Debug
		/*CHECK(cudaMemcpy(scanHistArr, d_scanHistArrTranpose, histArr_size, cudaMemcpyDeviceToHost));
		for (int i = 0; i < gridSizeHist * nBins; ++i) {
		  	printf("%d ", scanHistArr[i]);
			if(i == gridSizeHist * nBins - 1)
				printf("\n===|===|===|===|===|===|===|===|===|===\n");
		}*/
		

		// TODO: Scatter
		timer.Start();
		scatterKernel<<<gridSizeHist, blockSizes[0], 
						(2 * blockSizes[0] + 1 + 2 * nBins)* sizeof(uint32_t)>>>(d_src, n, d_dst, d_histArr, nBits, bit);
		CHECK(cudaGetLastError());
		timer.Stop();
		scatterTime += timer.Elapsed();

		// FIXME: Debug
		/*CHECK(cudaMemcpy(dst, d_dst, out_size, cudaMemcpyDeviceToHost));
		for (int i = 0; i < n; ++i) {
		  	printf("%d\t", dst[i]);
			if (i == n - 1)
				printf("\n===|===|===|===|===|===|===|===|===|===\n");
		}*/
		//break;
		// Swap "src" and "dst"
		uint32_t *tp = d_src;
		d_src = d_dst;
		d_dst = tp;
	}
	// Print runtime
	printf("Hist Time: %.3f\n", histTime);
	printf("Scan Time: %.3f\n", scanTime);
	printf("Add Time: %.3f\n", addTime);
	printf("Transpose Time: %.3f\n", transposeTime);
	printf("Scatter Time: %.3f\n", scatterTime);

	// DONE: Copy result from "d_src" to "out"
	CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	// free memories
	CHECK(cudaFree(d_src));
	CHECK(cudaFree(d_dst));
	CHECK(cudaFree(d_histArr));
	CHECK(cudaFree(d_scanHistArr));
	CHECK(cudaFree(d_blkSums))
	free(blkSums);
	//free(histArr);
	//free(scanHistArr);
	//free(dst);
}

// Radix sort
/*
        @type 	0 Sử dụng thư viện
                1 Sử dụng Host
                2 Sử dụng Device
*/
void sort(const uint32_t *in, int n, uint32_t *out, int nBits, int type,
		  int *blockSizes = NULL)
{
	GpuTimer timer;
	timer.Start();
	if (type == 0)
	{
		printf("\nRadix sort by library\n");
		sortByLibrary(in, n, out, nBits);
	}
	else if (type == 1)
	{
		printf("\nRadix sort by host\n");
		//sortByHost(in, n, out, nBits, 32);
	}
	else // use device
	{
		printf("\nRadix sort by device\n");
		sortByDevice(in, n, out, nBits, blockSizes);
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

void checkCorrectness(uint32_t *out, uint32_t *correctOut, int n)
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

void printArray(uint32_t *a, int n)
{
	for (int i = 0; i < n; i++)
		printf("%i ", a[i]);
	printf("\n");
}

int main(int argc, char **argv)
{
	// PRINT OUT DEVICE INFO
	printDeviceInfo();

	// SET UP INPUT SIZE
	int n = (1 << 24) + 1;
	//n = 1000000;
	//n = 10;
	printf("\nInput size: %d\n", n);

	// ALLOCATE MEMORIES
	size_t bytes = n * sizeof(uint32_t);
	uint32_t *in = (uint32_t *)malloc(bytes);
	uint32_t *out = (uint32_t *)malloc(bytes);		  // Device result
	uint32_t *correctOut = (uint32_t *)malloc(bytes); // Host result

	// SET UP INPUT DATA
	for (int i = 0; i < n; i++)
		in[i] = rand();
	// in[i] = rand() % 8;
	//uint32_t temp[10] = {3,2,5,7,9,9,8,8,1,1}; 
	//memcpy(in, temp, n * sizeof(uint32_t)); 
	//printArray(in, n);

	// SET UP NBITS
	int nBits = 4; // Default
	//nBits = 1;
	if (argc > 1)
		nBits = atoi(argv[1]);
	printf("\nNum bits per digit: %d\n", nBits);

	// DETERMINE BLOCK SIZES
	int blockSizes[2] = {512, 512}; // One for histogram, one for scan
	// cudaDeviceProp devProv;
	// CHECK(cudaGetDeviceProperties(&devProv, 0));
	// if (devProv.major <= 3 && devProv.minor <= 7)
	// {
	// 	blockSizes[0] = blockSizes[1] = devProv.maxThreadsPerMultiProcessor / 16;
	// }
	// else if (devProv.major <= 7 && devProv.minor <= 3)
	// {
	// 	blockSizes[0] = blockSizes[1] = devProv.maxThreadsPerMultiProcessor / 32;
	// }
	// else {
	// 	blockSizes[0] = blockSizes[1] = devProv.maxThreadsPerMultiProcessor / 16;
	// }
	if (argc == 4)
	{
		blockSizes[0] = atoi(argv[2]);
		blockSizes[1] = atoi(argv[3]);
	}
	printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0],
		   blockSizes[1]);

	// SORT BY LIBRARY
	sort(in, n, correctOut, nBits, LIBRARY);

	// SORT BY HOST
	sort(in, n, out, nBits, HOST);
	checkCorrectness(out, correctOut, n);

	// SORT BY DEVICE
	out[0] = 1; // Sửa lại mảng out để output sẽ khác
	sort(in, n, out, nBits, DEVICE, blockSizes);
	checkCorrectness(out, correctOut, n);

	// FREE MEMORIES
	free(in);
	free(out);
	free(correctOut);

	return EXIT_SUCCESS;
}
