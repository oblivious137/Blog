---
title: Machine Learning Notes Ⅱ
date: 2016-10-30 09:52:52
tags:
- Machine learning
- Stanford
categories:
- Machine learning
description: 监督学习的应用：梯度下降
---
# 监督学习的应用：梯度下降

样例问题：由房屋大小、房间个数来估计房屋价格。

### 一些定义：
1. $m$：样本数量。
2. $x$：输入变量（又叫特征），用$x_{j}^{(i)}$来表示第$i$个样本的第$j$个特征量。
3. $y$：输出变量（又叫目标），用$y_{(i)}$来表示第$i$个样本的标准答案。
4. $(x,y)$：样本点。

对于该样例题目来说，样本表示方式如下：

| 房屋大小($x_{1}$) | 卧室个数($x_{2}$) | 房屋价格($y$) |
| :-----------: | :-----------: | :-------: |
|   $\cdots$    |   $\cdots$    | $\cdots$  |


### 机器学习要实现的功能：
输入样本数据 $\rightarrow$ 运行机器学习算法 $\rightarrow$ 得到假设函数h $\rightarrow$ 通过假设函数对新数据进行预测。

### 解决步骤：
1. 决定假设函数h的表示方式。针对样例问题，我们采取线性表示方法，即$h(x)=h_{\theta}(x)=\theta _{0}+\theta _{1} \times x _{1} + \theta _{2} \times x _{2}$。
   不妨定义$x _{0}=1$，$n$为特征数目，则$h(x)=\sum _{i=0}^{n}{\theta _{i} \times x _{i}}$。
2. 决定“优化判据”，即对拟合结果优劣的评价，一般数值越小拟合结果越好。在这里，我们定义$J(\theta)=\frac{1}{2} \sum_{i=1}^{m}{(h_{\theta}(x^{(i)})-y^{(i)})^2}$为优化判据，通过最小化$J(\theta)$的值得到最优的假设函数$h_{\theta}$。
3. 设计算法得到较优或最优的假设函数。对于样例，我们可以采取搜索算法或针对上面的判据而推导出的最小二乘法公式。

### 算法讲解:

首先讲解搜索算法，这里我们采用梯度下降算法来求得最优的一组$\theta$。

算法的核心就是假定一个起始解，不断地向$J(\theta)$减小的地方移动，最终到达一个局部极小值。对于这种二次函数，可以证明只存在一个全局最优解，没有其他局部最优解。

对于在一个解处找到如何移动才能使$J$函数减小，我们采取求偏导的方法来解决。
对于$\theta_{i}$来说，每一次变化为

$$
\begin{align*}

\theta_{i} &:= \theta_{i} - \alpha \times (\frac{\partial J(\theta)}{\partial \theta_{i}}) \\

&:= \theta_{i} - \sum_{j=1}^{m}{(\alpha \times (h_{\theta}(x^{(j)})-y^{(j)}) \times x_{i}^{(j)})} \\

\end{align*}
$$
注意到在上面的过程中，我们每次更新一个$\theta_{i}$需要做$n \times m$次运算，当样本量较大使，运行速度缓慢。这时，我们可以改造一下梯度下降算法，每次只用**一组样本**去更新$\theta$，这样可以大大减小运行时间，但得到的$\theta$可能不是最优方案。

### 对步骤的简化：

对于一个矩阵到实数的映射$y=f(A)$，$A$为一个$p\times q$的矩阵，我们定义该映射的梯度矩阵为
$$
\begin{bmatrix}

\frac{\partial y}{\partial x_{11}} & \frac{\partial y}{\partial x_{12}} & \cdots & \frac{\partial y}{\partial x_{1q}} \\

\frac{\partial y}{\partial x_{21}} & \frac{\partial y}{\partial x_{22}} & \cdots & \frac{\partial y}{\partial x_{2q}} \\

\vdots & \vdots & \ddots & \vdots \\

\frac{\partial y}{\partial x_{p1}} & \frac{\partial y}{\partial x_{p2}} & \cdots & \frac{\partial y}{\partial x_{pq}} \\

\end{bmatrix}
$$
这样，如果我们把一个解$\theta$写成一个$1 \times n$的矩阵，$J(\theta)$就是一个矩阵到实数的映射。那么梯度下降时，每次的操作就变成了$\theta\,:=\theta - \alpha \times \nabla _{\theta}J$。


### 最小二乘法公式推导

**一些定义：**

定义$n \times n$方阵$A$的迹为$tr(A)=\sum_{i=1}^{n}{A_{ii}}$

有以下性质：
1. $tr(A \times B) = tr(B \times A)$，$tr(x_{1} \times x_{2} \times \cdots \times x_{n})=tr(x_{n} \times x_{1} \times \cdots \times x_{n-1})$。
2. 令$y=tr(A \times B)$，有$\nabla_{A}y=B^{T}$。
3. $tr(A)=tr(A^{T})$。
4. 若$a \in \mathbb{R}$，则$tr(a)=a$。
5. 令$y=tr(A \times B \times A^{T} \times  C)$，有$\nabla _{A}y = C \times  A \times B + C^{T} \times  A \times B^{T}$
   **（证明在最后）**

**推导过程：**

定义设计矩阵$X$，由所有样本数据构成，
$$
X=\begin{bmatrix}

\cdots & (x^{(1)})^{T} & \cdots \\

  & \vdots &   \\

\cdots & (x^{(m)})^{T} & \cdots \\

\end{bmatrix}
$$
