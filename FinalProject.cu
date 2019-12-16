#include <stdio.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

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
    int nBins = 1 << nBits; // 2^nBits
    int * localHist = (int *)malloc(nBins * sizeof(int));           // Khởi tạo Local Histogram cho từng block

    // Tạo mảng chứa các localHist
    int sizeHist = nBins * ((n - 1) / blockSize + 1);               // Tính kích thước của mảng listLocalHist
                                                                    // sizeHist = Số bin x Số lượng block
    //printf("Kich thuoc cua mang listLocalHist: %d\n", sizeHist);
    int * listLocalHist = (int *)malloc(sizeHist * sizeof(int));    // Khởi tạo listLocalHist

    // In each counting sort, we sort data in "src" and write result to "dst"
    // Then, we swap these 2 pointers and go to the next counting sort
    // At first, we assign "src = in" and "dest = out"
    // However, the data pointed by "in" is read-only 
    // --> we create a copy of this data and assign "src" to the address of this copy
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    /*uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;*/

    
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        // TODO: Mỗi block tính local histogram của digit-đang-xét trên phần 
        // dữ liệu của mình và chép vào mảng listLocalHist

        //printf("So luong phan tu mang la: %d\n", n);
        //printf("Kich thuoc blockSize la: %d\n", blockSize);
        memset(localHist, 0, nBins * sizeof(int));
        int count = 0;
        for(int i=0; i<n; i++){
            int bin = (src[i] >> bit) & (nBins - 1);
            localHist[bin]++;
            count++;
            if (count == blockSize || i == n - 1){                    // Kiểm tra xem ta đã duyệt được blockSize phần tử hay chưa
                count = 0;
                
                // TODO: Ta in ra mảng localHist
                /*printf("Mang localHist: ");
                for(int j=0; j < nBins; j++){
                    printf("%d ", localHist[j]);
                }
                printf("\n");*/

                // TODO: Chép dữ liệu vào listLocalHist
                int blockIndex = (i == n - 1)? (n - 1) / blockSize : (i + 1) / blockSize - 1;       // Tính xem đây là block thứ mấy
                //printf("BlockIndex la: %d\n", blockIndex);
                int index = blockIndex * nBins;                                                     // Tính chỉ số bắt đầu trong mảng listLocalHist
                //printf("Index trong mang listLocalHist: %d\n", index);
                for (int j = 0; j < nBins; j++ ){
                    listLocalHist[index++] =  localHist[j];
                }

                // TODO: Set lại mảng localHist
                memset(localHist, 0, nBins * sizeof(int));
            }
        }

        // TODO: Với mảng 2 chiều mà mỗi dòng là local hist của một block,
        // thực hiện exclusive scan trên mảng một chiều gồm các cột
        // nối lại với nhau (Xem slide để hiểu rõ)
       


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
        

    	// TODO: Swap "src" and "dst"
        /*uint32_t * temp = src;
        src = dst;
        dst = temp;*/ 

        break;  // Xóa dòng này sau khi test
    }

    // TODO: Copy result to "out"
    /*memcpy(out, src, n * sizeof(uint32_t));*/
    // TODO: In ra mảng listLocalHist
    for(int i=0; i < sizeHist; i++){
        printf("%d ", listLocalHist[i]);
    }
    // Free memories
    free(localHist);
    free(listLocalHist);
    free(src);
}

// (Partially) Parallel radix sort: implement parallel histogram and parallel scan in counting sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
// Why "int * blockSizes"? 
// Because we may want different block sizes for diffrent kernels:
//   blockSizes[0] for the histogram kernel
//   blockSizes[1] for the scan kernel
void sortByDevice(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits, int * blockSizes)
{
    // TODO
	thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}

// Radix sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits,
        bool useDevice=false, int * blockSizes=NULL)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix sort by host\n");
        sortByHost(in, n, out, nBits, 4);
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
    n = 8;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        //in[i] = rand();
        in[i] = rand() % 8;
    printf("Mang input la: ");
    printArray(in, n);

    // SET UP NBITS
    //int nBits = 4; // Default
    int nBits = 1;
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
