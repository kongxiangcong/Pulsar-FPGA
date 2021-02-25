# 一 Pulsar-FPGA

- XUP 2021 project：

  **Algorithm-hardware co-design for pulsar coherent de-dispersion algorithm on FPGA.**

## 1 项目介绍

### 1.1 脉冲星消色散背景

**脉冲星(Pulsar)** 信号在星际空间传播的过程中，由于星际介质的存在造成观测到的脉冲星信号发生色散效应，因此需要对接收的脉冲星信号进行 **消色散（de-dispersion）**，以获得原始的脉冲星信号。

- 标准的脉冲星搜寻方法

![标准的脉冲星搜寻方法](https://github.com/kongxiangcong/Pulsar-FPGA/blob/main/pic/pulsar_detect_flow.png)

### 1.2 消色散算法

星际介质的色散效应相当于移相器的作用，因此接收到的脉冲星信号相当于原始信号加上移相器得到的结果，移相器可以用传递函数H(f) 表示：
![消色散公式](https://github.com/kongxiangcong/Pulsar-FPGA/blob/main/pic/type.jpg)

其中，f0为本振频率; f1为中频频率，D为色散常量，DM为色散值。由**传递函数H(f)** 可知，原始信号可以通过乘以其复**共轭H(-f)** 进行完全消色散，消除整个观测带宽内的色散效应。

- 相干消色散步骤：

  （1）FFT：将模拟的基带数据进行FFT，得到频域数据

  （2）计算色散函数：根据基带信号频率信息计算色散函数H(-f)

  （2）频域消色散：将频域数据乘以消色散函数H(-f) 得到消色散后的频域数据

  （3）IFFT：把消色散后的频域数据进行逆傅里叶变换得到原始序列长度的消色散后的时
  域数据

> 黄玉祥. 脉冲星相干消色散与非相干消色散的比较研究[D].中国科学院大学(中国科学院云南天文台), 2018.

![消色散算法流程](https://github.com/kongxiangcong/Pulsar-FPGA/blob/main/pic/algorithm_flow.png)

**在整个脉冲星搜寻过程中，基带数据的消色散过程，是获得清晰完整原始信号的关键，也是最耗费计算资源部分**。

- 项目目的：

  结合FPGA高带宽和低功耗以及电路可重构的优点，本项目将基于xilinx FPGA开发套件，针对消色散算法包含的大量**FFT/IFFT**，**色散函数计算**，**矩阵乘法**等操作进行并行加速，从而实现大规模数据的实时消色散处理。

## 2 项目规划

### 2.2 消色散算法计算瓶颈

- **计算瓶颈**：

1. 脉冲大数据限制了FFT规模
2. 内存访问成为了算法加速的瓶颈
3. 多色散并行问题，大矩阵相乘并行问题

- **预期技术路线**

（1）采用**分段卷积**解决大数据FFT规模
（2）代码软硬件重优化，设计定制电路来降低内存访问消耗
（3）流水线设计提高吞吐率

![流水线](https://github.com/kongxiangcong/Pulsar-FPGA/blob/main/pic/pipeline.png)

> Kishalay De,Yashwant Gupta. A real-time coherent dedispersion pipeline for the giant metrewave radio telescope[J]. Experimental Astronomy,2016,41(1-2).

### 2.3 项目规划

总体目标是在U280等Alveo加速卡上实现面向大数据消色散算法的加速计算，实现脉冲星基带数据的实时消色散处理。。

- 初步规划：**实现小规模脉冲星数据消色散加速**。
  单个色散量（1个DM）条件下，基于HLS实现FFT/IFFT的并行加速，色散函数并行计算，脉冲星频域数据与色散函数相乘。

- 进阶规划：**实现大规模脉冲星数据消色散加速**。
  单个色散量（1个DM）条件下，基于分段卷积实现大规模FFT/IFFT的并行计算。

- 最终规划：**超大规模相干消色散算法异构并行加速**。
  多个色散量（多DM）条件下，通过多个FPGA，或者FPGA+GPU+CPU异构方法，实现超大规模相干消色散的实时处理

由于完成时间有限以及开发经验不足，**本项目打算由两位学员共同完成**，两人分工明确：

（1）经过本项目，初步上手Alveo加速卡，熟悉VITIS软件开发流程；

（2）体验HLS自动综合的电路如何实现多任务流水线并行操作，初步实现小规模的定制电路；

（3）截止日期前完成项目规划中的初步规划

# 二 项目进展
## 3.1 MATLAB模拟数据
MATLAB生成基带数据，写入文件pulsar_S0.raw，float32类型（单精度浮点型）

## 3.2 Python读取基带数据，生成FFT后的golden data
python读取数据，调用scipy.fft，FFT处理后作为golden data文件(scipy_fft.dat)在testbench中验证算法C代码

## 3.3 HLS仿真验证
### 3.1 S1_Baseline
三个for循环，没有代码重构，没有dataflow，cos,sin调用DSP资源实现
![S1_精度](https://github.com/kongxiangcong/Pulsar-FPGA/blob/main/pic/error_0.07.png)
- 由于采用单精度浮点，与golden data对比，平均误差达到7.6%

### 3.2 S2_Unroll
- 优化方法：
 展开第一层循环（展开每一级蝶形运算），方便dataflow；
 cos,sin调用DSP资源实现，精度方面与S1_Baseline一样
 
### 3.3 S4_DATAFLOW
- 优化方法：
 在S_2的基础上，采用查找表方法实现cos,sin计算，平均误差下降到0.1%
 ![S4_精度](https://github.com/kongxiangcong/Pulsar-FPGA/blob/main/pic/error_0.001.png)
 
### compare report
- 经过优化加速，latency降低到2.458ms
![latency](https://github.com/kongxiangcong/Pulsar-FPGA/blob/main/pic/latency.png)

- 牺牲少部分资源，优化了代码运算效率
![resource](https://github.com/kongxiangcong/Pulsar-FPGA/blob/main/pic/resource.png)
