#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <stdio.h>

int main()
{
	int n = 10;
	thrust::host_vector<int> hv(n);
	thrust::generate(hv.begin(), hv.end(), rand);
	for (int i = 0; i < n; i++)
		printf("%d\n", hv[i]);

	thrust::device_vector<int> dv = hv;

	thrust::sort(dv.begin(), dv.end());

	hv = dv;
	
	printf("\n");
	for (int i = 0; i < n; i++)
		printf("%d\n", hv[i]);
}