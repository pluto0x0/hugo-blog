+++
date = "2025-12-26T02:14:14.028Z"
draft = false
title = "ELBO，EM和VAE"
description = ""
categories = [ "CV" ]
tags = [ "VAE" ]
+++

# 证据下界 ELBO

在最大似然估计（Maximum Likelihood Estimation,
MLE）中，目标是最大化观测数据 $x$ 的对数似然.

参数 $\theta$ 的对数似然（log likelihood）的定义为

$$\ell(\theta) = \log p_{\theta}(x) = \log\sum_{z}p_{\theta}(x,z),$$

其中 $x$ 是观测变量，$z$ 是潜在变量. 根据 Jensen 不等式，对于任意分布
$q(z)$，有

$$
\begin{aligned}
\ell(\theta) &= \log\sum_{z}p_{\theta}(x,z) = \log\sum_{z}q(z)\frac{p_{\theta}(x,z)}{q(z)} = \log{\mathbb{E}}_{z \sim q}\left\lbrack \frac{p_{\theta}(x,z)}{q(z)} \right\rbrack \\
 &\geq {\mathbb{E}}_{z \sim q}\left\lbrack \log\frac{p_{\theta}(x,z)}{q(z)} \right\rbrack = {\mathbb{E}}_{z \sim q}\left\lbrack \log p_{\theta}(x,z) \right\rbrack - {\mathbb{E}}_{z \sim q}\left\lbrack \log q(z) \right\rbrack
\end{aligned}
$$

由此定义 $\ell(\theta)$ 的证据下界（Evidence Lower BOund, ELBO）$\mathcal{F}$:

$$
\ell(\theta) \geq \mathcal{F}(q,\theta) \equiv {\mathbb{E}}_{z \sim q}\left\lbrack \log p_{\theta}(x,z) \right\rbrack - {\mathbb{E}}_{z \sim q}\left\lbrack \log q(z) \right\rbrack.
$$

## 取等条件

Jensen 不等式能够取等，当且仅当对于所有 $z$,
$\frac{p_{\theta}(x,z)}{q(z)}$
为常数 $c$.
因此

$$
\begin{aligned}
p_{\theta}(x) &= \sum_{z}p_{\theta}(x,z) = c\sum_{z}q(z) = c \\
q(z) &= \frac{p_{\theta}(x,z)}{p_{\theta}(x)} = p_{\theta}\left( z|x \right).
\end{aligned}
$$

这里的 $q(z)$ 可以是任意分布，但使用中常用
$q(z) = q_{\varphi}\left( z|x \right)$，即 $z$
的**近似后验分布**，而近似后验分布越接近**真实后验分布**
$p_{\theta}\left( z|x \right)$，ELBO 越接近对数似然.

# KL散度

任意分布 $p(x)$ 的信息熵定义为

$$H(p) = {\mathbb{E}}_{x \sim p}\left\lbrack - \log p(x) \right\rbrack$$

即对随机事件编码长度的期望值. 如果认为 $x$ 服从另一分布 $q(x)$，在"真实"分布 $p(x)$
下的编码长度期望值定义为交叉熵：

$$H(p,q) = {\mathbb{E}}_{x \sim p}\left\lbrack - \log q(x) \right\rbrack.$$

KL散度即定义为“额外的”编码长度

$$\text{ KL}\left( p\|q \right) = H(p,q) - H(p) = {\mathbb{E}}_{x \sim p}\left\lbrack \log p(x) - \log q(x) \right\rbrack.$$

> KL散度描述了两个分布的差异程度，但是不是距离度量，因为不满足对称性和三角不等式.

## KL 散度的非负性

对于任意分布 $p(x)$ 和 $q(x)$，有 $\text{KL}(p \|q) \ge 0$，因为

$$
\begin{aligned}
\text{ KL}\left( p\|q \right) & = {\mathbb{E}}_{x \sim p}\left\lbrack \log p(x) - \log q(x) \right\rbrack \\
 & = - {\mathbb{E}}_{x \sim p}\left\lbrack \log\frac{q(x)}{p(x)} \right\rbrack \\
\left( \text{Jensen} \right) & \geq - \log{\mathbb{E}}_{x \sim p}\left\lbrack \frac{q(x)}{p(x)} \right\rbrack \\
 & = - \log\int_{x}p(x)\left( \frac{q(x)}{p(x)} \right)dx \\
 & = - \log\int_{x}q(x)dx \\
 & = - \log 1 \\
 & = 0.
\end{aligned}
$$

易知当且仅当 $p = q$ 时，$\text{KL}\left( p\|q \right) = 0$.

## ELBO 等价形式

回到 ELBO，ELBO 和真实对数似然的差距为

$$
\begin{aligned}
\ell(\theta) - \mathcal{F}(q,\theta)
 &= {\mathbb{E}}_{z \sim q}\left\lbrack \log p_{\theta}(x) - \log p_{\theta}(x,z) + \log q(z) \right\rbrack \\
 &= {\mathbb{E}}_{z \sim q}\left\lbrack - \log p_{\theta}\left( z|x \right) + \log q(z) \right\rbrack \\
 &= \text{ KL}\left( q(z)\|p_{\theta}\left( z|x \right) \right) \geq 0.
\end{aligned}
$$

因此，ELBO 等价于

$$\mathcal{F}(q,\theta) = \ell(\theta) - \text{ KL}\left( q(z)\|p_{\theta}\left( z|x \right) \right)$$

显然，取等条件同样为 $q(z) = p_{\theta}\left( z|x \right)$.
而一般的问题中，通常采用以下形式：

$$
\begin{aligned}
 \mathcal{F}(q, \theta)
 &= {\mathbb{E}}_{z \sim q}\left\lbrack \log p_{\theta}(x,z) \right\rbrack - {\mathbb{E}}_{z \sim q}\left\lbrack \log q(z) \right\rbrack \\
 &= {\mathbb{E}}_{z \sim q}\left\lbrack \log\frac{p_{\theta}(x,z)}{p_{\theta}(z)} \right\rbrack - \left( {\mathbb{E}}_{z \sim q}\left\lbrack \log q(z) \right\rbrack - {\mathbb{E}}_{z \sim q}\left\lbrack \log p_{\theta}(z) \right\rbrack \right) \\
 &= {\mathbb{E}}_{z \sim q}\left\lbrack \log p_{\theta}\left( x|z \right) \right\rbrack - \text{ KL}\left( q(z)\|p_{\theta}(z) \right)
\end{aligned}
$$

# EM

EM 算法的目标是最大化参数 $\theta$ 对数似然：

$$\ell(\theta) = \log p\left( x|\theta \right)$$

第 $i$ 步的参数和隐变量分布分别记为 $\theta^{(i)}$ 和 $q^{(i)}(z)$.

## E 步

最大化 $\ell(\theta^{(i)})$ 的 ELBO，即取等条件

$$q^{(i + 1)}(z) = p_{\theta^{(i)}}\left( z|x \right)$$

## M 步

更新参数 $\theta^{(i)}$ 以最大化 ELBO，
$$\theta^{(i + 1)} = \arg\max\limits_{\theta}\mathcal{F}(q^{(i + 1)},\theta)$$

## 收敛性

$$\ell(\theta^{(i + 1)})\underset{\text{ELBO}}{\geq}\mathcal{F}(q^{(i + 1)},\theta^{(i + 1)})\underset{\arg\max}{\geq}\mathcal{F}(q^{(i + 1)},\theta^{(i)})\underset{\text{满足取等条件}}{=}\ell(\theta^{(i)})$$

因此对数似然单调不减.

## 例子：高斯混合模型 GMM

假设观测数据 $x$ 来自 $K$ 个高斯分布的混合：

$$p_{\theta}(x) = \sum_{k = 1}^{K}\pi_{k}\mathcal{N}(x|\mu_{k},\Sigma_{k})$$

其中 $\theta = \left\{ \pi_{k},\mu_{k},\Sigma_{k} \right\}_{k=1}^K$
是模型参数，$\pi_{k}$ 是混合系数，满足 $\sum_{k = 1}^{K}\pi_{k} = 1$.
数据样本为 $\left\{ x^{i} \right\}_{i = 1}^{N}$，引入隐变量 $z$
表示样本属于哪个高斯分布, $z^{i} = k$ 表示样本 $x^{i}$ 来自第 $k$
个高斯分布.

### E 步

计算后验概率（责任度）：

$$
\begin{aligned}
\gamma_{z^{i} = k} & \leftarrow p_{\theta}\left( z^{i} = k|x^{i} \right) \\
 & = \frac{\pi_{k}\mathcal{N}(x^{i}|\mu_{k},\Sigma_{k})}{\sum_{j = 1}^{K}\pi_{j}\mathcal{N}(x^{i}|\mu_{j},\Sigma_{j})}
\end{aligned}
$$

### M 步

更新参数：

$$\theta = \arg\max\limits_{\theta}\mathcal{F}(\gamma,\theta)$$

即在已知责任度 $\gamma_{z^{i} = k}$ 下，最大化似然的参数. 定义
$\begin{aligned}
N_{k} & = \sum_{i = 1}^{N}\gamma_{z^{i} = k}
\end{aligned}$,

$$
\begin{aligned}
\pi_{k} & \leftarrow \frac{N_{k}}{N} \\
\mu_{k} & \leftarrow \left( \frac{1}{N_{k}} \right)\sum_{i = 1}^{N}\gamma_{z^{i} = k}x^{i} \\
\Sigma_{k} & \leftarrow \left( \frac{1}{N_{k}} \right)\sum_{i = 1}^{N}\gamma_{z^{i} = k}\left( x^{i} - \mu_{k} \right)\left( x^{i} - \mu_{k} \right)^{\top}
\end{aligned}
$$

# VAE

普通的自编码器（Autoencoder）就是编码--解码：

1. 编码器把输入数据 $x$ 映射到一个隐空间向量 $z$；

2. 解码器把 $z$ 还原回数据空间，得到重构 $\hat{x}$.

但是，普通自编码器学习到的 $z$ 没有概率解释，不能直接采样用于生成.

VAE 的关键思想是：

给隐变量 $z$ 加一个概率分布的解释，并用变分推断来学习这个分布.

1. 编码器不再输出一个确定向量，而是输出一个 分布参数（均值
   $\mu(x)$、方差 $\sigma^{2}(x)$）

2. 从这个分布中采样 $z \sim q_{\varphi(z|x)}$，再送给解码器生成
   $\hat{x}$.

3. 这样隐空间就被正则化成一个连续、平滑的概率空间，可以用来插值、采样、生成新样本.

## 先验分布 $p(z)$

隐变量 $z$ 的先验分布 $p(z)$ 通常取标准正态分布 $\mathcal{N}(0,I)$.

## 解码器（生成分布） $p_{\theta}\left( x|z \right)$

网络参数 $\theta$ 输入隐变量 $z$，输出数据 $x$
的分布参数，即高斯分布的均值 $\mu(z)$ 和方差 $\Sigma(z)$.

## 编码器（近似后验分布）$q_{\varphi}\left( z|x \right)$

使用 $q_{\varphi}\left( z|x \right)$ 拟合"真实"后验分布
$p_{\theta}\left( z|x \right)$.同样使用神经网络参数 $\varphi$，输入数据
$x$，输出隐变量 $z$ 的分布参数.

### "近似"后验 和 "真实"后验

"真实"的后验分布由贝叶斯公式得出

$$
\begin{aligned}
p_{\theta}\left( x|z \right) &= \frac{p_{\theta}\left( x|z \right)p(z)}{p_{\theta}(x)} \\
p_{\theta}(x) &= \int p_{\theta}\left( x|z \right)p(z)dz
\end{aligned}
$$

因此真实的后验不可解，使用近似后验 $q_{\varphi}\left( z|x \right)$
来代替.

## ELBO 和损失函数

使用 ELBO 替代对数似然：

$$\log p_{\theta}(x) \geq {\mathbb{E}}_{q_{\varphi}\left( z|x \right)}\left\lbrack \log p_{\theta}\left( x|z \right) \right\rbrack - \text{ KL}\left( q_{\varphi}\left( z|x \right)\|p(z) \right)$$

其中

- ${\mathbb{E}}_{q_{\varphi}\left( z|x \right)}\left\lbrack \log p_{\theta}\left( x|z \right) \right\rbrack$
  是重构误差，衡量解码器重构 $\hat{x}$ 与输入 $x$ 的差异；

- $\text{KL}\left( q_{\varphi}\left( z|x \right)\|p(z) \right)$
  是正则化项，使近似后验分布 $q_{\varphi}\left( z|x \right)$
  尽量接近先验分布 $p(z)$（标准正态分布）.

## 训练和重参数化

目前我们有

1. 编码器 $q_{\varphi}\left( z|x \right)$，输入 $x$，输出隐变量 $z$
   的高斯分布参数 $\mu(x),\sigma^{2}(x)$；

2. 解码器 $p_{\theta}\left( x|z \right)$，输入隐变量 $z$，输出重构
   $\hat{x}$ 的分布；

3. 先验分布 $p(z)$，通常取标准正态分布 $\mathcal{N}(0,I)$.

4. 损失函数 ELBO
   $$\mathcal{L} = - {\mathbb{E}}_{q_{\varphi}\left( z|x \right)}\left\lbrack \log p_{\theta}\left( x|z \right) \right\rbrack + \text{ KL}\left( q_{\varphi}\left( z|x \right)\|p(z) \right)$$

训练时，期望项（重构误差）没有解析解，使用 $z$ 的采样来近似.
为了保证采样 $z \sim \mathcal{N}(\mu(x),\sigma^{2}(x))$
可导，使用重参数化技巧：

$$
\begin{array}{r}
z = \mu(x) + \sigma(x) \odot \varepsilon, \\
\end{array}
$$

其中 $\varepsilon \sim \mathcal{N}(0,I)$ 是独立噪声.

## 展开 KL 散度

记 $q_{\text{agg }}(z) = {\mathbb{E}}_{x \sim P(x)}q\left( z|x \right)$,

$$
\begin{aligned}
\text{ELBO}
&= \mathbb{E}_{q_{\varphi}(z|x)}\!\left[\log p_{\theta}(x|z)\right] - \mathrm{KL}\!\left(q_{\text{agg}}(z)\|p(z)\right) \\
&= \underbrace{\mathbb{E}_{q_{\varphi}(z|x)}\!\left[\log p_{\theta}(x|z)\right]}_{\text{重构项}} - \underbrace{H\!\left(q_{\text{agg}}(z),p(z)\right)}_{\text{交叉熵}} + \underbrace{H\!\left(p(z)\right)}_{\text{熵}}.
\end{aligned}
$$

1. 重构项：鼓励隐变量分布远离先验分布 $p(z)$，提高重构质量
2. 交叉熵：把隐变量分布拉向先验分布 $p(z)$ 中心，会同时减小均值和方差
3. 熵：鼓励隐变量分布增大方差，分布更扁平

## VAE 的问题

### Prior Hole

VAE 的隐空间分布 $q_{\text{agg }}(z)$ 往往只占据先验分布 $p(z)$
的一小部分，导致从先验分布采样的 $z$
很可能落在"空洞"区域，解码器无法生成有效样本.

### Posterior Collapse

如果解码器足够强大，例如 autoregressive 模型，可以不依赖 $z$, 或者仅从
$z$ 的一部分维度中重构输入 $x$. 即 $z$
的某些维度对重构没有贡献，导致这些维度的近似后验分布
$q_{\varphi}\left( z|x \right)$ 退化为先验分布
$p(z)$，从而无法学习到有效的隐表示.

$$\exists i\ s.t.\ \forall x,q_{\varphi}\left( z_{i}|x \right) = p\left( z_{i} \right)$$

## Vector Quantized VAE(VQ VAE)

让隐变量 $z$ 取离散值，而不是连续值.
定义一个有限的码本（codebook），编码器输出一个向量 $e_{i}$，然后将
$e_{i}$ 映射到码本中距离最近的离散向量 $e_{k}$.

解决了 Prior Hole 和 Posterior Collapse
问题，但训练时需要使用特殊的技巧（如直通估计器）来处理离散变量的不可导问题.
