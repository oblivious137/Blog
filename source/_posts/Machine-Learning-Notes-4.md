---
title: Machine Learning Notes Ⅳ
date: 2016-11-10 17:32:00
tags:
- Machine learning
- Stanford
categories:
- Machine learning
description: 牛顿方法与广义线性模型
---
## Newton Method
牛顿方法主要用于求函数零点。
对于$y=f(\theta)$来说，基本思想就是先假设一个初始解，每次求出解所在位置的切线，将解移动到切线与$\theta$轴的交点处。
![图片加载失败](Machine-Learning-Notes-4\1.png)
由斜率的定义我们可以推出$$\begin{align\*}
f'(\theta)&=\frac{f(\theta)}{\Delta} \\\\
\Delta&=\frac{f(\theta)}{f'(\theta)} \\\\
\end{align\*}$$
又因为$\theta:=\theta-\Delta$，所以有$\theta:=\theta - \frac{f(\theta)}{f'(\theta)}$。
对于一部分回归模型，我们可以把最大化对数似然率变成求对数似然率的导数的零点，从而使用牛顿方法。迭代方式就是$\theta:=\theta-\frac{f'(\theta)}{f''(\theta)}$。
在一般化的情况下，$\theta$会是一个向量。所以$\theta:=\theta-H^{-1}\nabla\_{\theta}I$
其中$I$是目标函数；$H$为Hessian矩阵，满足$H\_{ij}=\frac{\partial^2 I}{\partial \theta\_i \partial \theta\_j}$，可以把它看作是二阶偏导的矩阵。可以直接看作$\theta$减去一阶偏导除以二阶偏导。
牛顿方法在逼近解时可以达到二次收敛，也就是每次迭代可以使解的有效数字加倍。
牛顿方法的优点就是收敛快。但当特征数目较多时，求Hessian矩阵的逆会比较耗费时间。

## Generalized Linear Models
目前已有的两种建模思路：
1. $y\in \mathbb{R}$，我们假设$y$满足高斯分布，从而得到了基于最小二乘的线性回归。
2. $y\in \lbrace 0,1 \rbrace$，我们假设$y$满足伯努利分布，从而得到了Logistic Regression。

伯努利分布与高斯分布的概率函数为：
$$\begin{align\*}
&Bernoulli(\phi):\;P(y=1;\phi)=\phi \\\\
&N(\mu,\sigma^2):\;P(x;\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2}) \\\\
\end{align\*}$$
其实他们都属于同一类分布，这类分布叫指数分布族。如果一种分布的概率函数可以写成以下形式，那么他就属于指数分布族。
$$P(y;\eta)=b(y)\*exp(\eta^TT(y)-a(\eta))$$
- $\eta$是自然参数，与分布的具体情况有关。例如高斯分布中的均值$\mu$。
- $T(y)$是充分统计量，只与分布的种类有关，它用来充分的描述一个随机变量。例如对于$\phi$值任意的所有伯努利分布都有$T(y)=y$。因为对于伯努利分布来说，它的随机变量取值只能为$0$或$1$，对于一种$y$的取值的充分统计就是它等于$0$还是$1$，所以$T(1)=1,\,T(0)=0$，也就是$T(y)=y$。在大部分指数分布族中，$T(y)=y$。

**验证伯努利分布与高斯分布属于指数分布族**
对于$Ber(\phi)$： $$\begin{align\*}P(y;\phi)&=\phi^y(1-\phi)^{1-y} \\\\
&=exp(\ln{(\phi^y(1-\phi)^{1-y})}) \\\\
&=exp(y\ln{\phi}+(1-y)\ln{(1-\phi)}) \\\\
&=exp(y(\ln{\phi}-\ln{(1-\phi)})+\ln{()1-\phi}) \\\\
&=exp(y\ln{\frac{\phi}{1-\phi}}+\ln{(1-\phi)}) \\\\
\end{align\*}$$可以发现$b(y)=1,\,\eta=\ln{\frac{\phi}{1-\phi}},\,a(\eta)=\ln{(1-\phi)}$。可以推出$\phi=\frac{1}{1+e^{-\eta}}$，所以$a(\eta)=\ln{(1+e^{-\eta})}$。

对于$N(\mu,\sigma^2)$，因为我们要求的与随机变量的期望有关，而$\sigma$的取值并不影响期望，所以在这里我们只考虑$\mu$为参数，将$\sigma$视为1。$$\begin{align\*}
P(y;\mu)&=\frac{1}{\sqrt{2\pi}}exp(-\frac{(y-\mu)^2)}{2}) \\\\
&=\frac{1}{\sqrt{2\pi}}exp(-\frac{1}{2}y^2-\frac{1}{2}\mu^2+y\mu) \\\\
&=\frac{1}{\sqrt{2\pi}}exp(-\frac{1}{2}y^2)exp(\mu y-\frac{1}{2}\mu^2) \\\\
\end{align\*}$$
可以发现$b(y)=\frac{1}{\sqrt{2\pi}}exp(-\frac{1}{2}y^2),\,\eta=\mu,\,T(y)=y,\,a(\eta)=-\frac12\mu^2$。可以推出$\mu=\eta$，所以$a(\eta)=-\frac12\eta^2$

**广义线性模型的一般方法**
1. 假设$y\mid x;\theta \sim ExpFamily(\eta)$
2. 目的是：给定$x$，输出$E[T(y)\mid x]$，也就是在参数为$x$时，$y$的充分统计量的期望。
3. 设计决策：假设输入特征与$\eta$的关系，$\eta=\theta^Tx$。（这里$\eta$是一个实数，一般化的情况下，$\eta\_i=\theta\_i^Tx$，可以把$\eta$看作一个向量，$\theta$就是一种将$x$变成$\eta$的线性变换）

**伯努利分布的推导**
首先假设$y\mid x;\theta \sim ExpFamily(\eta)$。
对于给定的$x,\theta$，我们的输出为$$\begin{align\*}
h\_\theta(x)&=E[T(y)\mid x;\theta] \\\\
&=E[y\mid x;\theta] \\\\
&=\phi \\\\
&=\cfrac{1}{1+e^{-\eta}} \\\\
&=\cfrac{1}{1+e^{-\theta^Tx}} \\\\
\end{align\*}$$
这样，我们就得到了Logistic回归的函数形式。

**正则响应函数与正则关联函数**
定义$g(\eta)=E[y;\eta]$，将$g(\eta)$称为正则响应函数，它将$\eta$与$y$的期望联系了起来。
另外将${g(\eta)}^{-1}$称为正则关联函数。

**多项式分布的推导**
多项式分布是伯努利分布的推广，在多项式分布中$y\in \lbrace 1,2,\ldots,k \rbrace$。
多项式分布的参数应该有$\phi\_1,\phi\_2,\ldots,\phi\_k$，其中$P(y=i)=\phi\_i$。
但是，由概率和为1可得：$\phi\_k=1-(\phi\_1+\phi\_2+\cdots+\phi\_{k-1})$。
所以一个$y$有$k$个取值的多项式分布，它的实际参数应该有$k-1$个，分别是$\phi\_1,\ldots,\phi\_{k-1}$。
考虑对于$y$的一个取值来说，它的充分统计量应该是什么。显然应该是$y$的取值，考虑到如果令$T(y)=y$，$T(y)$的期望没有实际意义，所以我们令$$T(1)=\begin{bmatrix}
1 \\\\
0 \\\\
\vdots \\\\
0 \\\\
\end{bmatrix}\;
T(2)=\begin{bmatrix}
0 \\\\
1 \\\\
\vdots \\\\
0 \\\\
\end{bmatrix}\;
T(k-1)=\begin{bmatrix}
0 \\\\
0 \\\\
\vdots \\\\
1 \\\\
\end{bmatrix}\;
T(k)=\begin{bmatrix}
0 \\\\
0 \\\\
\vdots \\\\
0 \\\\
\end{bmatrix}\;\; \in \mathbb{R}^{k-1}$$
这时$T(y)$的期望可以代表$y$分别取$k-1$种值的概率。
我们再定义指示函数$1\\{state\\}$。其中若$state$为真，则取值为1；若$state$为假，则取值为0。
令$T(y)\_i$为$T(y)$第$i$维的取值，那么$T(y)\_i=1\\{y=i\\}$。

所以有$$\begin{align\*}
P(y)&=\phi\_1^{1\lbrace y=1 \rbrace} \phi\_2^{1\lbrace y=2 \rbrace} \cdots \phi\_k^{1\lbrace y=k \rbrace} \\\\
&=\phi\_1^{T(y)\_1} \phi\_2^{T(y)2} \cdots \phi\_k^{1-\sum{i=1}^{k-1}{T(y)\_i}} \\\\
&=exp(T(y)\_1\ln{\phi\_1}+T(y)\_2\ln{\phi\_2}+\cdots+(1-\sum\_{i=1}^{k-1}{T(y)\_i})\ln{\phi\_k}) \\\\
&=exp(T(y)\_1\ln{(\phi\_1-\phi\_k)}+T(y)\_2\ln{(\phi\_2-\phi\_k)}+\cdots+\ln{\phi\_k}) \\\\
&=exp(T(y)\_1\ln{\frac{\phi\_1}{\phi\_k}}+T(y)\_2\ln{\frac{\phi\_2}{\phi\_k}}+\cdots+\ln{\phi\_k}) \\\\
\end{align\*}$$
可以看出：$\eta=[\ln{\frac{\phi\_1}{\phi\_k}},\ln{\frac{\phi\_2}{\phi\_k}},\cdots,\ln{\frac{\phi\_{k-1}}{\phi\_k}}]\;$,$\;a(\eta)=\ln{\phi\_k}$。
由第一个等式可推出$\phi\_i=\cfrac{e^{\eta\_i}}{1+\sum\_{j=1}^{k-1}{e^{\eta\_j}}}$。

所以我们的输出函数$$
h\_\theta(x)=E[T(y)\mid x;\theta]=\begin{bmatrix}
\phi\_1 \\\\
\phi\_2 \\\\
\vdots \\\\
\phi\_{k-1} \\\\
\end{bmatrix}=\begin{bmatrix}
\\cfrac{e^{\eta\_1}}{1+\sum\_{j=1}^{k-1}{e^{\eta\_j}}} \\\\
\\cfrac{e^{\eta\_2}}{1+\sum\_{j=1}^{k-1}{e^{\eta\_j}}} \\\\
\vdots \\\\
\\cfrac{e^{\eta\_{k-1}}}{1+\sum\_{j=1}^{k-1}{e^{\eta\_j}}} \\\\
\end{bmatrix}=\begin{bmatrix}
\\cfrac{e^{\theta\_1^Tx}}{1+\sum\_{j=1}^{k-1}{e^{\theta\_j^Tx}}} \\\\
\\cfrac{e^{\theta\_2^Tx}}{1+\sum\_{j=1}^{k-1}{e^{\theta\_j^Tx}}} \\\\
\vdots \\\\
\\cfrac{e^{\theta\_{k-1}^Tx}}{1+\sum\_{j=1}^{k-1}{e^{\theta\_j^Tx}}} \\\\
\end{bmatrix}$$
这样，我们就得到了当目标满足多项式分布时，应该使用的函数形式。我们把这种回归称为Softmax回归，是Logistic的推广。

同样我们也要对它进行最大似然估计：
$$\begin{align\*}
L(\theta)&=\prod\_{i=1}^{m}{P(y^{(i)}\mid x^{(i)};\theta)} \\\\
&=\prod\_{i=1}^m{ {\left(\cfrac{e^{\theta\_1^Tx^{(i)}}}{1+\sum\_{j=1}^{k-1}{e^{\theta\_j^Tx^{(i)}}}}\right)}^{1\lbrace y^{(i)}=1 \rbrace} \* \cdots \*  {\left( 1- \sum\_{l=1}^{k-1}{\cfrac{e^{\theta\_l^Tx^{(i)}}}{1+\sum\_{j=1}^{k-1}{e^{\theta\_j^Tx^{(i)}}}} } \right)}^{1\lbrace y^{(i)}=k \rbrace} }\\\\
&=\prod\_{i=1}^m{ \cfrac{\prod\_{j=1}^{k-1}{e^{1\lbrace y^{(i)}=j \rbrace \theta\_j^Tx^{(i)}}}}{1+\sum\_{j=1}^{k-1}{e^{\theta\_j^Tx^{(i)}}}} }
\end{align\*}$$
同样令对数似然函数$l(\theta)=\ln{L(\theta)}$，可得$$l(\theta)=\left( \sum\_{i=1}^m{\sum\_{j=1}^{k-1}{1\lbrace y^{(i)}=j \rbrace \theta\_j^Tx^{(i)}}} \right) - \sum\_{i=1}^m{\ln{\left(1+\sum\_{j=1}^{k-1}{e^{\theta\_j^Tx^{(i)}}} \right)}}$$
对$\theta\_r$求偏导可得：$$\cfrac{\partial l(\theta)}{\partial \theta\_r}=\sum\_{i=1}^m{1\lbrace y^{(i)}=j \rbrace x^{(i)}} -    \sum\_{i=1}^m{\cfrac{e^{\theta\_r^Tx^{(i)}}x^{(i)}}{1+\sum\_{j=1}^{k-1}{e^{\theta\_j^Tx^{(i)}}}}}$$
之后我们就可以使用梯度下降来最优化$\theta$了。