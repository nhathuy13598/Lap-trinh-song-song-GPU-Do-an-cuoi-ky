#include <stdio.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
<<<<<<< HEAD
=======

>>>>>>> 6a2efb4e6a1c9cc246b8bdc76aa62b4154036860
#define LIBRARY 0
#define HOST 1
#define DEVICE 2
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
// Ta sẽ sử dụng ý tưởng từ thuật toán sắp xếp tuần tự mới chứ không phải
// thuật toán mà ta đã sử dụng trong bài tập số 3
/**
    Sắp xếp tuần tự trên host
    @blockSize kích thước một block mà ta sẽ duyệt (Ta vẫn duyệt tuần tự)
*/
void sortByHost(const uint32_t * in, int n,
                uint32_t * out,
                int nBits,
                int blockSize)
{
    int nBins = 1 << nBits;												// Số lượng bin là 2^nBits
	
	

	uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));			// Biến tạm để lưu trữ dữ liệu input
	memcpy(src, in, n * sizeof(uint32_t));								// Sao chép dữ liệu từ in vào src
	uint32_t * originalSrc = src;										// Use originalSrc to free memory later
	uint32_t * dst = out;												// Mảng kết quả

	// [TODO]: Khởi tạo các mảng cần thiết khi chạy
	int sizeHist = nBins * ((n - 1) / blockSize + 1);					// Tính kích thước của mảng listLocalHist sizeHist = Số bin x Số lượng block
	int *listLocalHist = (int *)malloc(sizeHist * sizeof(int));			// Mảng chứa các localHist
	int *listLocalHistConvert = (int*)malloc(sizeHist * sizeof(int));	// Mảng chuyển đổi của listLocalHistConvert

	int * histScan = (int*)malloc(sizeHist * sizeof(int));				// Mảng exclusive scan của listLocalHistConvert

	int *eleBefore = (int*)malloc(n * sizeof(int));						// Mảng chứa chỉ số phân tử đứng trước và bằng nó trong từng block

	int numBlock = (n - 1) / blockSize + 1;								// Số lượng các block cần thiết
	for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
	{
		// TODO: Mỗi block tính local histogram của digit-đang-xét trên phần 
		// dữ liệu của mình và chép vào mảng listLocalHist
		memset(listLocalHist, 0, sizeHist * sizeof(int));					// Gán mảng listLocalHist bằng 0
		for (int blkIdx = 0; blkIdx < numBlock; blkIdx++) {
			int start = blkIdx * blockSize;									// Chỉ số bắt đầu của block
			int end = (blkIdx == numBlock - 1) ? n : start + blockSize;		// Chỉ số kết thúc của block
			for (int index = start; index < end; index++) {					// Duyệt tất cả phần tử của block
				int bin = (src[index] >> bit) & (nBins - 1);
				listLocalHist[blkIdx*nBins + bin]++;
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
		int indexLLHC = 0;                                                                  // Chỉ số trong mảng listLocalHistConvert
		for (int i = 0; i < nBins; i++) {                                                   // Duyệt tất cả các phần tử trong một localHist
			for (int j = 0; j < numBlock; j++) {											// Duyệt tất cả các localHist
				listLocalHistConvert[indexLLHC++] = listLocalHist[i + j * nBins];           // i là chỉ số bin trong localHist, j * nBins là chỉ số của block
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
		for (int i = 1; i < sizeHist; i++) {
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
		for (int blkIdx = 0; blkIdx < numBlock; blkIdx++) {					// Duyệt từng block
			int start = blkIdx * blockSize;									// Chỉ số bắt đầu của block
			int end = (blkIdx == numBlock - 1) ? n : start + blockSize;		// Chỉ số kết thúc của block
			for (int x = end - start; x >= 1; x--) {						// Ta tưởng tượng đây là sắp xếp mảng có end - start phần tử
				for (int y = 0; y < x - 1; y++) {
					int first = (src[blkIdx*blockSize + y] >> bit) & (nBins - 1);
					int second = (src[blkIdx*blockSize + y + 1] >> bit) & (nBins - 1);
					if (first > second) {
						uint32_t temp = src[blkIdx*blockSize + y];
						src[blkIdx*blockSize + y] = src[blkIdx*blockSize + y + 1];
						src[blkIdx*blockSize + y + 1] = temp;
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
		memset(eleBefore, 0, n * sizeof(int));								// Khởi tạo mảng chứa các phần tử đứng trước bằng 0
		for (int blkIdx = 0; blkIdx < numBlock; blkIdx++) {
			int start = blkIdx * blockSize;									// Chỉ số bắt đầu của block
			int end = (blkIdx == numBlock - 1) ? n : start + blockSize;		// Chỉ số kết thúc của block
			for (int index = 1; index < end - start; index++) {
				int first = (src[blkIdx * blockSize + index - 1] >> bit) & (nBins - 1);
				int second = (src[blkIdx * blockSize + index] >> bit) & (nBins - 1);
				if (first == second) {
					eleBefore[blkIdx*blockSize + index] = eleBefore[blkIdx*blockSize + index - 1] + 1;
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
		for (int index = 0; index < n; index++) {
			int blIdx = index / blockSize;
			int bin = (src[index] >> bit) & (nBins - 1);
			int rank = histScan[bin*numBlock + blIdx] + eleBefore[index];
			dst[rank] = src[index];
		}

		// [DEBUG]: Mang dst
		/*printf("Mang dst: ");
		for (int index = 0; index < n; index++) {
			printf("%d ", dst[index]);
		}
		printf("\n");*/

		// TODO: Swap "src" and "dst"
		uint32_t * temp = src;
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

void sortByLibrary(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits)
{
    // TODO
	thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}

// Radix sort
/*
	@type 	0 Sử dụng thư viện
			1 Sử dụng Host
			2 Sử dụng Device
*/
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits,
        int type, int * blockSizes=NULL)
{
    GpuTimer timer; 
    timer.Start();
	if (type == 0){
		printf("\nRadix sort by library\n");
        sortByLibrary(in, n, out, nBits);
	}
    if (type == 1)
    {
    	printf("\nRadix sort by host\n");
        sortByHost(in, n, out, nBits, 32);
    }
    else // use device
    {
    	printf("\nRadix sort by device\n");
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
    n = 1000;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes);             // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes);      // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();
        //in[i] = rand() % 8;
    //uint32_t temp[11] = {41,18467,6334,26500,19169,15724,11478,29358,26962,24464,5705};
    //memcpy(in, temp, n * sizeof(uint32_t));
    //printArray(in, n);

    // SET UP NBITS
    int nBits = 4; // Default
    if (argc > 1)
        nBits = atoi(argv[1]);
    printf("\nNum bits per digit: %d\n", nBits);

    // DETERMINE BLOCK SIZES
    int blockSizes[2] = {512, 512}; // One for histogram, one for scan
    if (argc == 4)
    {
        blockSizes[0] = atoi(argv[2]);
        blockSizes[1] = atoi(argv[3]);
    }
    printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0], blockSizes[1]);

    // SORT BY LIBRARY
    sort(in, n, correctOut, nBits, LIBRARY);

    // SORT BY HOST
    sort(in, n, out, nBits, HOST);
    checkCorrectness(out, correctOut, n);
    
	// SORT BY DEVICE
	out[0] = 1; // Sửa lại mảng out để output sẽ khác
    sort(in, n, correctOut, nBits, DEVICE, blockSizes);
	checkCorrectness(out, correctOut, n);

    // FREE MEMORIES 
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
