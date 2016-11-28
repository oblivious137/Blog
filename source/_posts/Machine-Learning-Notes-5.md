---
title: Machine Learning Notes Ⅴ
date: 2016-11-27 20:08:26
tags:
- Machine learning
- Stanford
categories:
- Machine learning
description: 生成学习算法与贝叶斯平滑
---

# 生成学习算法

我们之前学习的算法都属于判别学习算法，它们都是直接对P(y|x)进行建模。

而生成学习算法则是由贝叶斯公式$P(y=i|x)=\frac{P(y=i,x)}{P(x)}=\frac{P(x|y=i)\times P(y=i)}{\sum\_{j=1}^{k}{P(x|y=j)\times P(y=j)}}$得到。他把$x,y$视为两个有关联的随机变量，通过对$y$的不同取值下的$x$建模来得到$P(x|y=i)$，对$y$的直接建模得到$P(y=i)$，从而计算出计算$P(y=i|x)$。因为我们必须对不同$y$的$x$分别建模，这就要求$y$的取值必须是有限的，所以生成学习算法一般用于分类问题。

## 高斯判别分析
高斯判别分析是一种生成学习算法，它假设了输入特征$x \in \mathbb{R}$在不同的$y$取值下服从高斯分布$N(\vec \mu,\Sigma)$。
上面的高斯分布属于多维高斯分布，其概率密度函数定义为$P(z)=\frac{1}{(2\pi)^{(D/2)} |\Sigma|^{1/2}}exp(-\frac12 {(x-\mu)}^T\Sigma^{-1}(x-\mu))$。其中$\mu$为均值；$\Sigma$为协方差矩阵，有$\Sigma=E[(x-\mu){(x-\mu)}^T]$。
我们假设原问题$y\in \lbrace0,1\rbrace$，那么之前我们会选取Logistic回归中，我们最大化的对数似然函数是$l(\theta)=\log{\prod\_{i=1}^{m}P(y^{(i)}|x^{(i)},\theta)}$。但这里，我们要最大化$l()$