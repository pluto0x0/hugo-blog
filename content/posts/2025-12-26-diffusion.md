---
title: Diffusion Model
description: DDPM, Langevin dynamics, CFG
date: 2025-12-26T11:46:28.632Z
preview: ""
draft: false
tags:
  - Diffusion
categories: []
---

# Diffusion Model

## 加噪过程

定义一个**马尔可夫链**，从数据 $x_{0} \sim q(x)$
开始，每一步加一点噪声：

$$q\left( x_{t}~|~x_{t - 1} \right) = \mathcal{N}(x_{t};\sqrt{1 - \beta_{t}}x_{t - 1},\beta_{t}I)$$

使用重参数化，等价于

$$x_{t} = \sqrt{1 - \beta_{t}}x_{t - 1} + \sqrt{\beta_{t}}\varepsilon_{t},\quad\varepsilon_{t} \sim N(0,I),$$

其中 $\beta$ 是一系列预定义的参数.

对于两个高斯分布 $x_{1} \sim \mathcal{N}(\mu_{1},\sigma_{1}^{2})$ 和
$x_{2} \sim \mathcal{N}(\mu_{2},\sigma_{2}^{2})$，我们有：

$$x_{1} + x_{2} \sim \mathcal{N}(\mu_{1} + \mu_{2},\sigma_{1}^{2} + \sigma_{2}^{2})$$

于是定义 $\alpha_{t} = 1 - \beta_{t}$，则

$$
\begin{aligned}
x_{t} & = \sqrt{\alpha_{t}}x_{t - 1} + \sqrt{1 - \alpha_{t}}\varepsilon_{t} \\
 & = \sqrt{\alpha_{t}\alpha_{t - 1}}x_{t - 2} + \sqrt{\alpha_{t}\left( 1 - \alpha_{t - 1} \right)}\varepsilon_{t} + \sqrt{1 - \alpha_{t}}\varepsilon_{t} \\
 & = \sqrt{\alpha_{t}\alpha_{t - 1}}x_{t - 2} + \sqrt{1 - \alpha_{t}\alpha_{t - 1}}\varepsilon \\
 & = \sqrt{\alpha_{t}\alpha_{t - 1}\alpha_{t - 2}}x_{t - 3} + \sqrt{1 - \alpha_{t}\alpha_{t - 1}\alpha_{t - 2}}\varepsilon \\
 & = \ldots
\end{aligned}
$$

定义 ${\overline{\alpha}}_{t} = \prod_{s = 1}^{\top}\alpha_{s}$, 得到
$x_{t}$ 的封闭形式

$$x_{t} = \sqrt{{\overline{\alpha}}_{t}}x_{0} + \sqrt{1 - {\overline{\alpha}}_{t}}\varepsilon,\quad\varepsilon \sim N(0,I).$$

即

$$q\left( x_{t}~|~x_{0} \right) = \mathcal{N}(x_{t};\sqrt{{\overline{\alpha}}_{t}}x_{0},\left( 1 - {\overline{\alpha}}_{t} \right)I)$$

因为 $\alpha_{t} < 1$, 因此
${\overline{\alpha}}_{\infty} \rightarrow 0$，此时
$q\left( x_{t}~|~x_{0} \right) \rightarrow \mathcal{N}(0,I)$. 即随着 $t$
增大，$x_{t}$ 趋近于纯噪声.

## 去噪过程

我们的目标是找到加噪过程 $q\left( x_{t}~|~x_{t - 1} \right)$
的逆过程，即逐步去噪的条件高斯分布

$$q\left( x_{t - 1}~|~x_{t} \right) = \mathcal{N}(x_{t - 1};\mu_{t}\left( x_{t} \right),\Sigma_{t}\left( x_{t} \right))$$

$q\left( x_{t - 1}~|~x_{t} \right)$不可解，因此使用一个模型 $\theta$
近似，以 $x_{t}$ 和 $t$ 作为输入，输出分布参数

$$p_{\theta}\left( x_{t - 1}~|~x_{t} \right) = \mathcal{N}(x_{t - 1};\mu_{\theta}\left( x_{t},t \right),\Sigma_{\theta}\left( x_{t},t \right))$$

## 损失函数：最大似然估计 MLE

目标是最小化去噪模型 $p_{\theta}$ 对初始数据 $x_{0}$ 的负对数似然：

$$\mathcal{L}(\theta) = - \log p_{\theta}\left( x_{0} \right)$$

同样地，使用变分下界 ELBO 优化，其中 $x_{0}$ 是已知变量，$x_{1:T}$
是隐变量：

$$
\begin{array}{r}
\mathcal{F}(q,\theta) = {\mathbb{E}}_{q\left( x_{1:T}|x_{0} \right)}\left\lbrack \log p_{\theta}\left( x_{0:T} \right) - \log q\left( x_{1:T}|x_{0} \right) \right\rbrack \\
\mathcal{L}(\theta) \leq - \mathcal{F}(q,\theta)
\end{array}
$$

根据马尔可夫性质，

$$
\begin{array}{r}
q\left( x_{1:T}|x_{0} \right) = q\left( x_{T}~|~x_{0} \right)\prod_{t = 2}^{\top}q\left( x_{t - 1}~|~x_{t},x_{0} \right) \\
p_{\theta}\left( x_{0:T} \right) = p_{\theta}\left( x_{T} \right)\prod_{t = 1}^{\top}p_{\theta}\left( x_{t - 1}~|~x_{t} \right)
\end{array}
$$

代入得到 $\mathcal{F}(q,\theta)$

$$
\begin{aligned}
 = & {\mathbb{E}}_{q\left( x_{1}:x_{T}|x_{0} \right)}\left\lbrack \log p_{\theta}\left( x_{T} \right) + \sum_{t = 1}^{\top}\log p_{\theta}\left( x_{t - 1}~|~x_{t} \right) - \log q\left( x_{T}~|~x_{0} \right) - \sum_{t = 2}^{\top}q\left( x_{t - 1}~|~x_{t},x_{0} \right) \right\rbrack \\
 = & {\mathbb{E}}_{q\left( x_{1}|x_{0} \right)}\left\lbrack \log p_{\theta}\left( x_{0}|x_{1} \right) \right\rbrack - \sum_{t = 2}^{\top}{\mathbb{E}}_{q\left( x_{t},x_{t - 1}|x_{0} \right)}\left\lbrack \log q\left( x_{t - 1}~|~x_{t},x_{0} \right) - \log p_{\theta}\left( x_{t - 1}~|~x_{t} \right) \right\rbrack \\
 & + {\mathbb{E}}_{q\left( x_{T}|x_{0} \right)}\left\lbrack \log p_{\theta}\left( x_{T} \right) - \log q\left( x_{T}~|~x_{0} \right) \right\rbrack \\
 = & {\mathbb{E}}_{q\left( x_{1}|x_{0} \right)}\left\lbrack \log p_{\theta}\left( x_{0}|x_{1} \right) \right\rbrack - \sum_{t = 2}^{\top}{\mathbb{E}}_{q\left( x_{t}|x_{0} \right)}{\mathbb{E}}_{q\left( x_{t - 1}|x_{t},x_{0} \right)}\left\lbrack \log q\left( x_{t - 1}~|~x_{t},x_{0} \right) - \log p_{\theta}\left( x_{t - 1}~|~x_{t} \right) \right\rbrack \\
 & + {\mathbb{E}}_{q\left( x_{T}|x_{0} \right)}\left\lbrack \log p_{\theta}\left( x_{T} \right) - \log q\left( x_{T}~|~x_{0} \right) \right\rbrack \\
 = & {\mathbb{E}}_{q\left( x_{1}|x_{0} \right)}\left\lbrack \log p_{\theta}\left( x_{0}|x_{1} \right) \right\rbrack - \sum_{t = 2}^{\top}{\mathbb{E}}_{q\left( x_{t}|x_{0} \right)}\text{ KL}\left( q\left( x_{t - 1}~|~x_{t},x_{0} \right)\| p_{\theta}\left( x_{t - 1}~|~x_{t} \right) \right) \\
 & - \text{ KL}\left( \log q\left( x_{T}~|~x_{0} \right)\|\log p_{\theta}\left( x_{T} \right) \right)
\end{aligned}
$$

分别记作

$$\mathcal{L} ≔ - \mathcal{F}(q,\theta) = \mathcal{L}_{0} + \sum_{t = 2}^{\top}\mathcal{L}_{t - 1} + \mathcal{L}_{T}.$$

其中

- 初始项
  $\mathcal{L}_{0} = - {\mathbb{E}}_{q\left( x_{1}|x_{0} \right)}\left\lbrack \log p_{\theta}\left( x_{0}|x_{1} \right) \right\rbrack$

  - 最小化最后一步 $x_{1}$ 到 $x_{0}$ 的重建误差

  - 当 $p_{\theta}$ 取高斯分布时，相当于最小化均方误差 MSE:
    ${\mathbb{E}}\left\lbrack \left\| {x_{1} - x_{0}} \right\|^{2} \right\rbrack$

- 中间项
  $\mathcal{L}_{t - 1} = {\mathbb{E}}_{q\left( x_{t}|x_{0} \right)}\text{ KL}\left( q\left( x_{t - 1}~|~x_{t},x_{0} \right)\| p_{\theta}\left( x_{t - 1}~|~x_{t} \right) \right)$

  - 让模型 $p_{\theta}$ 学习去噪过程，缩小和真实后验
    $q\left( x_{t - 1}~|~x_{t},x_{0} \right)$ 的差距。

  - $$q\left( x_{t - 1}|x_{t},x_{0} \right) = \frac{q\left( x_{t - 1}|x_{0} \right)q\left( x_{t}|x_{t - 1} \right)}{q\left( x_{t}|x_{0} \right)} \propto q\left( x_{t - 1}|x_{0} \right)q\left( x_{t}|x_{t - 1} \right)$$

- 终端项
  $\mathcal{L}_{T} = \text{ KL}\left( \log q\left( x_{T}~|~x_{0} \right)\|\log p_{\theta}\left( x_{T} \right) \right)$

  - 让模型 $p_{\theta}$ 的先验分布接近 $q\left( x_{T}~|~x_{0} \right)$

  - 对于足够大的
    $T$，$q\left( x_{T}~|~x_{0} \right) \approx \mathcal{N}(0,I)$，而先验
    $p_{\theta}\left( x_{T} \right)$ 被定义为 $\mathcal{N}(0,I)$

  - 因此该项可以**忽略不计**.

## 单步去噪损失

> 两个高斯分布的密度的乘积满足：
$$
\mathcal{N}(x;\mu_{1},\Sigma_{1})\mathcal{N}(x;\mu_{2},\Sigma_{2}) \propto \mathcal{N}(x;\mu,\Sigma)
$$
其中
$\Sigma = \left( \Sigma_{1}^{- 1} + \Sigma_{2}^{- 1} \right)^{- 1}$,
$\mu = \Sigma\left( \Sigma_{1}^{- 1}\mu_{1} + \Sigma_{2}^{- 1}\mu_{2} \right)$

考虑单步去噪过程：

$$
\tag{a}
\begin{aligned}
  q\left( x_{t - 1}|x_{t},x_{0} \right)
  &= \frac{q\left( x_{t}~|~x_{t - 1},x_{0} \right)q\left( x_{t - 1}|x_{0} \right)}{q\left( x_{t}|x_{0} \right)} \\
  &\propto \mathcal{N}(x_{t};\sqrt{\alpha_{t}}x_{t - 1},\left( 1 - \alpha_{t} \right)I)\mathcal{N}(x_{t - 1};\sqrt{{\overline{\alpha}}_{t - 1}}x_{0},\left( 1 - {\overline{\alpha}}_{t - 1} \right)I) \\ 
  &\propto \mathcal{N}\left(x_{t - 1};\frac{\sqrt{\alpha_{t}}(1 - {\overline{\alpha}}_{t - 1})x_{t} + \sqrt{{\overline{\alpha}}_{t - 1}}\beta_{t}x_{0}}{1 - {\overline{\alpha}}_{t}},\frac{\left( 1 - \alpha_{t} \right)\left( 1 - {\overline{\alpha}}_{t - 1} \right)}{1 - {\overline{\alpha}}_{t}}I\right) \\
  &=: \mathcal{N}(x_{t-1};\; \mu_q(x_t, x_0),\; \sigma_q(t)^2 I)
\end{aligned}
$$

其中
$\mathcal{N}(x_{t};\sqrt{\alpha_{t}}x_{t - 1},\beta_{t}I) \propto \mathcal{N}(x_{t - 1};\left( \frac{1}{\sqrt{\alpha_{t}}} \right)x_{t},\left( \frac{\beta_{t}}{\alpha_{t}} \right)I)$
，由多元高斯分布定义
$\mathcal{N}(x,\mu,\Sigma) = \frac{1}{\sqrt{(2\pi)^{d}|\Sigma|}}\exp( - \frac{1}{2}(x - \mu)^{\top}\Sigma^{- 1}(x - \mu))$
带入可得。

(a) 中的真实条件分布
$q\left( x_{t - 1}|x_{t},x_{0} \right)$
就是去噪模型的
$p_{\theta}\left( x_{t - 1}~|~x_{t} \right)$
的目标。二者差异在 $\mathcal{L}_{t - 1}$ 中通过KL散度描述.

观察 (a) 可知条件分布的方差独立于 $x$，因此只需要学习分布的均值。

> 两个高斯分布之间的KL散度为：
$$
\begin{aligned}
  &{\text{KL}\left( \mathcal{N}(\mu_{1},\Sigma_{1})\|\mathcal{N}(\mu_{2},\Sigma_{2}) \right)} \\
  =& \frac{1}{2}\left\lbrack \log\left( \frac{\left| \Sigma_{2} \right|}{\left| \Sigma_{1} \right|} \right) - d + \operatorname{tr}(\Sigma_{2}^{- 1}\Sigma_{1}) + \left( \mu_{2} - \mu_{1} \right)^{\top}\Sigma_{2}^{- 1}\left( \mu_{2} - \mu_{1} \right) \right\rbrack
\end{aligned}
$$

又因为两个分布的方差是确定的, 最小化 $\mathcal{L}_{t - 1}$ 等价于最小化

$$
\tag{b}
{\mathbb{E}}_{q\left( x_{t}|x_{0} \right)}\left\lbrack \frac{1}{2\sigma_{q}^{2}(t)}\left\| {\mu_{q}\left( x_{t},x_{0},t \right) - \mu_{\theta}\left( x_{t},t \right)} \right\|_{2}^{2} \right\rbrack
$$

模型预测的**原始数据 $x_0$** 记为 $\hat{x_{\theta}}(x_{t},t)$,
则预测的均值为
$\mu_{\theta}\left( x_{t},t \right) = \mu_{q}\left( x_{t},\hat{x_{\theta}}(x_{t},t),t \right)$.
带入得到 (b) 等价于

$$
\tag{b.1}
{\mathbb{E}}_{q\left( x_{t}|x_{0} \right)}\left\lbrack \frac{1}{2\sigma_{q}^{2}(t)}\frac{{\overline{\alpha}}_{t - 1}\beta_{t}^{2}}{\left( 1 - {\overline{\alpha}}_{t} \right)^{2}}\left\| {\hat{x_{\theta}}(x_{t},t) - x_{0}} \right\|_{2}^{2} \right\rbrack
$$

## 训练步骤

到现在为止，可以很容易得到完整的训练步骤：

1. 采样：数据 $x_{0} \sim q(x)$，时间步 $t \sim \text{ Unif}\left( \left\{ 1,\ldots,T \right\} \right)$，标准高斯采样 $\varepsilon \sim \mathcal{N}(0,I)$
2. 计算 $x_{t} = \sqrt{{\overline{\alpha}}_{t}}x_{0} + \sqrt{1 - {\overline{\alpha}}_{t}}\varepsilon$;
3. 预测 $\hat{x_{\theta}}(x_{t},t)$
4. 计算 loss $\mathcal{L}_{t} = w(t)\left\| {\hat{x_{\theta}}(x_{t},t) - x_{0}} \right\|_{2}^{2}$ 其中 $w(t) = \frac{1}{2\sigma_{q}^{2}(t)}\frac{{\overline{\alpha}}_{t - 1}\beta_{t}^{2}}{\left( 1 - {\overline{\alpha}}_{t} \right)^{2}}$
5. 最小化 $\mathcal{L}_{t}$, 更新参数 $\theta$.

## 预测噪声和v-parameterization等价

目前为止，对去噪步骤的预测是通过预测原始数据 $x_0$ 得到的。而
$x_{t} = \sqrt{{\overline{\alpha}}_{t}}x_{0} + \sqrt{1 - {\overline{\alpha}}_{t}}\varepsilon$
，或者记为
$x_t = \gamma_t x_0 + \eta_t \varepsilon$
，则有

$$
x_{0}
= \frac{x_t - \eta_t \varepsilon}{\gamma_t}
= \frac{x_{t} - \sqrt{\left( 1 - {\overline{\alpha}}_{t} \right)}\varepsilon}{\sqrt{{\overline{\alpha}}_{t}}}
,\quad \varepsilon \sim \mathcal{N}(0,I)
$$

所以可以改为预测 $\varepsilon$ 而非 $x_{0}$。则loss变为

相比预测原始数据，噪声 $\varepsilon$ 的分布更稳定（标准高斯分布），因此更容易学习。但是，预测噪声和预测原始数据是等价的，两者可以通过简单的线性变换互相转换.

显然，只要预测目标和$x_t$是“线性可逆”的，就都是(b)合法的等价形式。在Stable Diffusion等现代Diffusion Model中，更常见的做法是使用v-parameterization，即预测一个线性组合：

$$
v_t := \gamma_t \varepsilon - \eta_t x_{0}
$$

注意到 $\gamma_t^2 + \eta_t^2 = 1$，因此在 $x_0$-$\varepsilon$ 空间中，$x_t$ 对应一个1/4单位圆上的点，(1,0) 为数据 $x_0$，对应 $t=0$，(0,1) 为噪声 $\varepsilon$，对应 $t=T$。$v_t$ 就是这个旋转过程中的角速度，描述了从噪声流向数据的方向。

# Score-based Diffusion Model & Langevin dynamics

## Langevin dynamics

### Boltzmann distribution

统计力学的玻尔兹曼分布（Boltzmann distribution）指出一个系统的状态分布为

$$p_{i} \propto \exp( - \frac{\varepsilon_{i}}{kT})$$ []{#eq:boltzmann}

其中 $\varepsilon_{i}$ 是状态 $i$ 的能量，$k$ 是玻尔兹曼常数，$T$
是温度.

### Langevin equation

而 Langevin equation 描述的是粒子在（一维）势能场中的布朗运动，

$$dx_{t} = - \frac{1}{\gamma}\nabla_{x}U\left( x_{t} \right)dt + \sqrt{\frac{2kT}{\gamma}}dW_{t}$$
[]{#eq:langevin}

其中

- $x$ 是粒子位置

- $U(x)$ 是势能函数

- $\gamma$ 是阻尼系数

- $W_{t}$ 是标准 Wiener
  过程：$W_{t + \Delta} = W_{t} + \mathcal{N}(0,\Delta)$

根据 Boltzmann distribution

$$U(x) = - kT\log p(x) + \text{ constant},$$

代入得

$$
\begin{aligned}
dx_{t} & = \frac{kT}{\gamma}\nabla_{x}\log p\left( x_{t} \right)dt + \sqrt{\frac{2kT}{\gamma}}dW_{t} \\
 & = \frac{kT}{\gamma}\nabla_{x}\log p\left( x_{t} \right)dt + \sqrt{\frac{2kT}{\gamma}}dW_{t}
\end{aligned}
$$

方程在离散时间 $x_{k} ≔ x(k\tau)$ 下的形式为

$$
\begin{array}{rlr}
x_{k + 1} - x_{k} & = - \frac{kT}{\gamma}\tau\nabla_{x}\log p\left( x_{t} \right) + \sqrt{\frac{2kT}{\gamma}\tau}\xi\quad & ,\xi \sim \mathcal{N}(0,I) \\
 & = - \eta\nabla_{x}\log p\left( x_{t} \right) + \sqrt{2\eta}\xi & ,\xi \sim \mathcal{N}(0,I)
\end{array}
$$

其中 $\eta = \frac{kT}{\gamma}\tau$ 是步长. 回忆 $x_{t}$
描述的是粒子的随机位置，因此

$$x_{k} \sim p(x)$$

即已知对数梯度 $\nabla\log p(x)$ 时，Langevin dynamics
迭代的过程可以对分布 $p(x)$ 采样，而不需要显式地知道 $p(x)$ 的形式。

观察迭代形式，这实际上是一个带随机扰动的对数梯度上升过程，梯度项使得粒子趋向于高概率区域，而随机扰动则保证了采样的多样性.

### Simulated Annealing

实际上，如果令玻尔兹曼分布 ([@eq]:boltzmann) 中势能不变，$T$ 逐渐收敛至
$0$, 则分布收敛到单点分布 $p(x) = \delta(x - x^{\ast})$，其中
$x^{\ast} = \arg\min\limits_{x}U(x)$ 是势能的最小值点, 此时 Langevin
equation ([@eq]:langevin) 中的随机扰动项收敛至 $0$，Langevin dynamics
退化为势能的对数梯度下降. 而温度 $T$
的收敛速度控制了采样过程的随机性，快速的降温会导致探索不足，陷入局部势能极小值，反之则有更大概率达到势能全局最小值。这种模拟降温过程的方法即为**模拟退火**
Simulated Annealing.

## Score-based Diffusion Model

定义 score 函数，即**数据分布**的对数梯度：

$$\text{ score}(x) = \nabla_{x}\log p(x)$$

根据 Langevin dynamics ([@eq]:langevin)，只要能估计出数据分布的 score
函数，即可通过迭代采样得到数据分布的样本.

## 学习 score

在 Score-based Diffusion Model 中，唯一需要学习的就是一个
noise-conditioned score network:

$$s_{\theta}\left( x_{t},t \right) \approx \nabla_{x_{t}}\log p\left( x_{t} \right)$$

采取 DSM（Denoising Score Matching）损失：

$$\mathcal{L} = {\mathbb{E}}_{x,z,t}\left\lbrack \lambda(t)\left\| {s_{\theta}\left( x + \sigma_{t}z,t \right) + \frac{z}{\sigma_{t}}} \right\|^{2} \right\rbrack$$

即 $s_{\theta}\left( x + \sigma_{t}z,t \right)$ 的真实值为

$$\nabla\log\mathcal{N}(u;x,\sigma_{t}^{2}) = \nabla\frac{{- (u - x)}^{2}}{2\sigma_{t}^{2}} = - \frac{u - x}{\sigma_{t}^{2}} = - \frac{z}{\sigma_{t}}$$

而 $\lambda(t)$ 是一个权重函数，用于平衡不同噪声水平下的损失贡献.
常见的选择是

$$\lambda(t) = \sigma_{t}^{2}$$

## 采样/去噪

# 条件 Diffusion Model

给定条件 $c$, 需要

$$\nabla_{x}\log p\left( x|c \right)$$

Bayes 公式：

$$\nabla_{x}\log p\left( x|c \right) = \nabla_{x}\log p(x) + \nabla_{x}\log p\left( c|x \right)$$

如何得到 $\nabla_{x}\log p\left( c|x \right)$:

classifier guidance：基于 $x_{t}$ 的分类器，通过反向传播获取梯度。
通过给 $\nabla_{x}\log p\left( c|x \right)$ 乘以一个系数
$s > 1$，可以增强条件信息的影响，得到更符合条件的样本生成结果.

# Classifier-Free Diffusion Guidance (CFG)

不使用分类器，而是定义两种去噪模型的 score：

- 条件模型
  $\nabla_{x}\log p_{\theta}\left( x_{t},c \right) = \frac{1}{\sigma^{2}}\left( D_{\theta}\left( x_{t},\sigma,c \right) - x_{0} \right)$

- 无条件模型
  $\nabla_{x}\log p_{\theta}\left( x_{t} \right) = \frac{1}{\sigma^{2}}\left( D_{\theta}\left( x_{t},\sigma \right) - x_{0} \right)$

其中 $D_{\theta}$ 是去噪器, 使用一个 null
条件（如全零向量）来表示无条件模型. 则上式的等价形式

$$
\begin{aligned}
\nabla_{x}\log p\left( x|c \right) & = \nabla_{x}\log p(x) + S\nabla_{x}\log p\left( c|x \right) \\
 & = \nabla_{x}\log p_{\theta}\left( x_{t} \right) + S\left( \nabla_{x}\log p_{\theta}\left( x_{t},c \right) - \nabla_{x}\log p_{\theta}\left( x_{t} \right) \right) \\
 & = S\nabla_{x}\log p_{\theta}\left( x_{t},c \right) + (1 - S)\nabla_{x}\log p_{\theta}\left( x_{t} \right) \\
 & = \frac{1}{\sigma^{2}}\left( SD_{\theta}\left( x_{t},\sigma,c \right) + (1 - S)D_{\theta}\left( x_{t},\sigma \right) - x_{0} \right)
\end{aligned}
$$

即用二者的凸组合来进行去噪采样.
