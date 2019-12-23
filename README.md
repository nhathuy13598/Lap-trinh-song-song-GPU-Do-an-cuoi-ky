# Lập trình song song GPU: Đồ án cuối kỳ
Lập trình song song GPU: Đồ án thuật toán sắp xếp Radix Sort
## 1. Thông tin nhóm
|**STT**|**Họ và tên**|**MSSV**|
|:---:|:---:|:---:|
|1|Trần Nhật Huy|1612272|
|2|Nguyễn Thành Đạt|1612088|


## 2. Yêu cầu của đồ án
Đồ án sẽ gồm có 4 phiên bản baseline:
1. Cài đặt tuần tự thuật toán Radix Sort tuần tự (đã làm ở BT4)
2. Cài đặt song song 2 bước histogram và scan của thuật toán Radix Sort tuần tự (đã làm ở BT4)
3. Cài đặt song song thuật toán Radix Sort với k = 1 bit
4. Cài đặt tuần tự ý tưởng của thuật toán Radix Sort song song trong bài báo của NVIDIA (đây là phiên bản mà lúc sau ta sẽ tập trung song song hóa và tối ưu hóa).

**Đích đến:** Tối ưu gần bằng thuật toán sắp xếp có sẵn trong thư viện Thrust
## 3. Kế hoạch thực hiện (dự kiến)
**Ngày vấn đáp:** `10/01/2020`

|**Công việc**|**Ngày bắt đầu**|**Ngày kết thúc**|**Ngày hoàn thành thực tế**|
|---|:---:|:---:|:---:|
|Code thuật toán radix sort tuần tự|13/12/2019|16/12/2019|17/12/2019|
|Code thuật toán radix sort song song|17/12/2019|22/12/2019|22/12/2019|
|Viết baseline số 3|24/12/2019|24/12/2019||
|Tối ưu thuật toán radix sort|25/12/2019|08/01/2020||
|Viết báo cáo|09/01/2020|09/01/2020||
|Viết slide|09/01/2020|09/01/2020||
|Dự phòng|10/01/2020|10/01/2020||

## 4. Quy trình tối ưu thuật toán Radix Sort
Việc tối ưu thuật toán sẽ chia làm 3 bước: `Phân tích`, `Thiết kế` và `Cài đặt`  
Vì thầy sẽ đánh giá điểm dựa vào quy trình tối ưu thuật toán là có hợp lý hay không hơn là đánh giá vào thời gian chạy của thuật toán nên ***tất cả các bước tối ưu hóa sau khi cài đặt được thuật toán radix sort song song sẽ được ghi tiếp ở file này theo các bước trên***  
> Việc làm như vậy sẽ dễ dàng trong việc viết báo cáo về sau 
### 4.1. Lần tối ưu hóa 1
#### Phân tích
....
#### Thiết kế
....
#### Cài đặt 
....
### 4.2. Lần tối ưu hóa 2
#### Phân tích
....
#### Thiết kế
....
#### Cài đặt 
....
## 5. Kết quả + Hình ảnh chụp
#### Baseline 1
File:

Image:
#### Baseline 2
File: 

Image:
#### Baseline 3
File: 

Image:
#### Baseline 4
File: 

Image:
|**Version**|**histogramKernel**|**scanBlkKernel**|**addBlkKernel**|**transposeKernel**|**scatterKernel**|**Total**|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Baseline 1||||||?ms|
|Baseline 2||||||?ms|
|Baseline 3||||||?ms|
|Baseline 4||||||?ms|
|Thrust||||||?ms|