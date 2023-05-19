# 深度学习框架开发 (C++, CUDA, Python)
背景: CMU-10714 课程，实现一个完整的深度学习框架 (包括框架的架构设计、自动微分机制以及底层算法的实现)

* 实现支持 CPU 和 GPU 环境下的 NDArray 和 Tensor，并基于此实现常用算子的前向和反向计算
* 实现神经网络模块 (如 Linear 和 BatchNorm)，优化器 (如 SGD 和 Adam) 以及混合网络 (如 MLP 和 ResNet)
* 实现稀疏 Tensor 的三种存储方式 (COO, CSR, CSC) 和矩阵乘法运算