---
title: Machine Learnling Notes Ⅱ
date: 2016-10-30 09:52:52
tags:
- Machine learning
- Stanford
categories:
- Machine learning
description: 监督学习的应用：梯度下降。
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
|:-----------:|:-----------:|:-----------:|
| $\cdots$ | $\cdots$ | $\cdots$ |


### 机器学习要实现的功能：
输入样本数据 $\rightarrow$ 运行机器学习算法 $\rightarrow$ 得到假设函数h $\rightarrow$ 通过假设函数对新数据进行预测。

### 解决步骤：
1. 决定假设函数h的表示方式。针对样例问题，我们采取线性表示方法，即$h(x)=h\_{\theta}(x)=\theta \_{0}+\theta \_{1} \times x \_{1} + \theta \_{2} \times x \_{2}$。
不妨定义$x \_{0}=1$，$n$为特征数目，则$h(x)=\sum \_{i=0}^{n}{\theta \_{i} \* x \_{i}}$。
2. 决定“优化判据”，即对拟合结果优劣的评价，一般数值越小拟合结果越好。在这里，我们定义$J(\theta)=\frac{1}{2}*\sum\_{i=1}^{m}{(h\_{\theta}(x^{(i)})-y^{(i)})^2}$为优化判据，通过最小化$J(\theta)$的值得到最优的假设函数$h\_{\theta}$。
3. 设计算法得到较优或最优的假设函数。对于样例，我们可以采取搜索算法或针对上面的判据而推导出的最小二乘法公式。

### 算法讲解:

首先讲解搜索算法，这里我们采用梯度下降算法来求得最优的一组$\theta$。

算法的核心就是假定一个起始解，不断地向$J(\theta)$减小的地方移动，最终到达一个局部极小值。对于这种二次函数，可以证明只存在一个全局最优解，没有其他局部最优解。

对于在一个解处找到如何移动才能使$J$函数减小，我们采取求偏导的方法来解决。
对于$\theta\_{i}$来说，每一次变化为
$$\begin{align\*}
\theta\_{i} &:= \theta\_{i} - \alpha \* (\frac{\partial J(\theta)}{\partial \theta\_{i}}) \\\\
&:= \theta\_{i} - \sum\_{j=1}^{m}{(\alpha \* (h\_{\theta}(x^{(j)})-y^{[j]}) \* x\_{i}^{(j)})} \\\\
\end{align\*}$$
其中$\frac{\partial J(\theta)}{\partial \theta\_{i}}$表示函数$J(\theta)$对于$\theta\_{i}$的偏导，每次依据偏导向使$J(\theta)$的值减小的方向移动。而$\alpha$则是我们自己给定的一个学习速率，用于控制每次移动的步长，$\alpha$过大会导致跨过最优解，$\alpha$过小会导致收敛缓慢。

注意到在上面的过程中，我们每次更新一个$\theta\_{i}$需要做$n \* m$次运算，当样本量较大使，运行速度缓慢。这时，我们可以改造一下梯度下降算法，每次只用**一组样本**去更新$\theta$，这样可以大大减小运行时间，但得到的$\theta$可能不是最优方案。

### 对步骤的简化：

对于一个矩阵到实数的映射$y=f(A)$，$A$为一个$p\*q$的矩阵，我们定义该映射的梯度矩阵为
$$\begin{bmatrix}
\frac{\partial y}{\partial x\_{11}} & \frac{\partial y}{\partial x\_{12}} & \cdots & \frac{\partial y}{\partial x\_{1q}} \\\\
\frac{\partial y}{\partial x\_{21}} & \frac{\partial y}{\partial x\_{22}} & \cdots & \frac{\partial y}{\partial x\_{2q}} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
\frac{\partial y}{\partial x\_{p1}} & \frac{\partial y}{\partial x\_{p2}} & \cdots & \frac{\partial y}{\partial x\_{pq}} \\\\
\end{bmatrix}$$

这样，如果我们把一个解$\theta$写成一个$1 \* n$的矩阵，$J(\theta)$就是一个矩阵到实数的映射。那么梯度下降时，每次的操作就变成了$\theta\,:=\theta - \alpha \* \nabla \_{\theta}J$。


### 最小二乘法公式推导

**一些定义：**

定义$n \* n$方阵$A$的迹为$tr(A)=\sum\_{i=1}^{n}{A\_{ii}}$

有以下性质：
1. $tr(A \* B) = tr(B \* A)$，$tr(x\_{1} \* x\_{2} \* \cdots \* x\_{n})=tr(x\_{n} \* x\_{1} \* \cdots \* x\_{n-1})$。
2. 令$y=tr(A \* B)$，有$\nabla\_{A}y=B^{T}$。
3. $tr(A)=tr(A^{T})$。
4. 若$a \in R$，则$tr(a)=a$。
5. 令$y=tr(A \* B \* A^{T} \* C)$，有$\nabla \_{A}y = C \* A \* B + C^{T} \* A \* B^{T}$

**推导过程：**

定义设计矩阵$X$，由所有样本数据构成，$X=\begin{bmatrix}
\cdots & (x^{(1)})^{T} & \cdots \\\\
  & \vdots &   \\\\
\cdots & (x^{(m)})^{T} & \cdots \\\\
\end{bmatrix}$。定义$\theta$为列向量，$\theta=\begin{bmatrix}
\theta\_{1} \\\\
\vdots \\\\
\theta\_{n} \\\\
\end{bmatrix}$。
则$X \* \theta = \begin{bmatrix}
(x^{(1)})^{T} \*  \theta \\\\
\vdots \\\\
(x^{(m)})^{T} \*  \theta \\\\
\end{bmatrix}
=\begin{bmatrix}
h\_{\theta}(x^{(1)}) \\\\
\vdots \\\\
h\_{\theta}(x^{(m)}) \\\\
\end{bmatrix}$
接下来定义$y=\begin{bmatrix}
y^{(1)} \\\\
\vdots \\\\
y^{(m)} \\\\
\end{bmatrix}$。
此时，$(X \* \theta-y)=\begin{bmatrix}
h(x^{(1)})-y^{(1)} \\\\
\vdots \\\\
h(x^{(m)})-y^{(m)} \\\\
\end{bmatrix}$。
所以$\frac{1}{2} \* (X \* \theta - y)^{T} \* (X \* \theta - y) = \frac{1}{2} \* \sum\_{i=1}^{m}{(h\_{\theta}(x^{(i)})-y^{(i)})^2} = J(\theta)$。
在最小值处，有
$$\begin{align\*}
\nabla \_{\theta} J(\theta) &= \vec 0 \\\\
\nabla \_{\theta} \frac{1}{2} \* (X \* \theta -y)^{T} \* (X \* \theta -y) &= \vec 0 \\\\
\frac{1}{2} \nabla\_{\theta}(\theta^T X X \theta - \theta^TX^Ty - y^TX\theta + y^Ty) &= \vec 0 \\\\
\frac{1}{2} \nabla\_{\theta}tr(\theta^T X X \theta - \theta^TX^Ty - y^TX\theta + y^Ty) &= \vec 0 \\\\
\frac{1}{2} \nabla\_{\theta}tr(\theta \theta^T X^TX) -\nabla\_{\theta}tr(\theta y^TX) -\nabla\_{\theta}tr(\theta y^TX) &= \vec 0 \tag{1} \\\\
\end{align\*}$$

$(1)$处的推出，用到了括号内各项乘积结果为是一个$1 \* 1$的矩阵，可以把它们看成一个实数，所以它们之和的迹等于他们的迹之和。然后我们利用性质1把第一项中最后一个矩阵提前，把第二项转置后调整顺序，第三项调整顺序，第四项与$\theta$无关求梯度后为$0$。
然后$\nabla\_{\theta}tr(\theta \theta^T X^TX)$可以在中间加入一个单位矩阵变为$\nabla\_{\theta}tr(\theta I \theta^T X^TX)$，根据性质5可以得出该项等于$X^TX\theta I + X^TX\theta I$，忽略单位矩阵后等于$2\*X^TX\theta$。
对于$\nabla\_{\theta}tr(\theta y^TX)$，应用性质2可以得到该项等于$X^Ty$。
所以$$\begin{align\*}
J(\theta)&=\frac{1}{2} (X^TX\theta + X^TX\theta -X^Ty -X^Ty) \\\\
&=X^TX\theta - X^Ty \\\\
\end{align\*}$$
所以$$\begin{align\*}
X^TX\theta - X^Ty &=\vec 0 \\\\
X^TX\theta &=X^Ty \\\\
\theta &=(X^TX)^{-1}X^Ty \\\\
\end{align\*}$$

------------------------
** 对于迹的性质的证明：**
1. 设$A$为$n \* m$的矩阵，$B$为$m \* n$的矩阵，则$$\begin{align\*}
tr(AB)&=\sum\_{i=1}^{n}{(AB)\_{(i)(i)}} \\\\
&=\sum\_{i=1}^{n}{\sum\_{k=1}^{m}{A\_{(i)(k)} \* B\_{(k)(i)}}} \\\\
&=\sum\_{k=1}^{m}{\sum\_{i=1}^{n}{B\_{(k)(i)} \* A\_{(i)(k)}}} \\\\
&=\sum\_{k=1}^{m}{(BA)\_{(k)(k)}} \\\\
&=tr(BA) \\\\
\end{align\*}$$
2. 由刚才推出的式子有 $tr(AB)=\sum\_{i=1}^{n}{\sum\_{k=1}^{m}{A\_{(i)(k)} \* B\_{(k)(i)}}}$。对于$A\_{(i)(k)}$来说，它只出现了一次，系数是$B\_{(k)(i)}$，所以$\nabla\_{A}tr(AB)=B^T$。