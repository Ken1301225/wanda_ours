# 基于局部输出扰动的 MoE-SwiGLU 版 Wanda 形式化整理

## 1. 问题设定

考虑一个带有 top-k 路由的 MoE 层。对任意 token 隐状态 $x\in\mathbb{R}^d$，记其 MoE 层输出为

$$
y_{\mathrm{moe}}(x)
=
\sum_{e\in\mathcal{T}(x)} g_e(x)\,f_e(x),
$$

其中：

- $\mathcal{T}(x)$ 表示路由器选出的 top-k expert 集合；
- $g_e(x)$ 表示 expert $e$ 的路由权重；
- $f_e(x)$ 表示 expert $e$ 的输出。

对一个 SwiGLU expert，记

$$
f_e(x)=W_{\mathrm{down},e}h_e(x),
$$

其中

$$
h_e(x)=\phi(a_e(x))\odot b_e(x),
$$

并有

$$
a_e(x)=W_{\mathrm{gate},e}x,\qquad b_e(x)=W_{\mathrm{up},e}x.
$$

这里：

- $W_{\mathrm{gate},e}$ 为门控投影；
- $W_{\mathrm{up},e}$ 为上投影；
- $W_{\mathrm{down},e}$ 为下投影；
- $\phi$ 通常取 SiLU；
- $\odot$ 表示逐元素乘法。

本文讨论如下局部问题：在校准样本附近，若仅剪去某个 expert 内部的部分权重，则当前 MoE 层输出的扰动如何近似，并如何据此构造 Wanda 风格剪枝分数。

为此，采用如下局部近似设定：

1. 路由器决策在校准样本上冻结；
2. top-k expert 集合在校准样本上冻结；
3. 路由权重在校准样本上冻结；
4. 只考察当前 MoE 层输出扰动，不向更深层传播联合误差；
5. 对非线性项在当前展开点附近采用一阶近似。

最终目标是近似

$$
\|\Delta y_{\mathrm{moe}}(x)\|_2^2.
$$

## 2. 线性层上的 Wanda 原理

先回顾线性层情形。设

$$
y=Wx.
$$

若将权重 $w_{ij}$ 剪为零，则输出第 $i$ 个坐标的变化为

$$
\Delta y_i=-w_{ij}x_j.
$$

于是有

$$
\mathbb{E}\left[(\Delta y_i)^2\right]
=
w_{ij}^2\mathbb{E}[x_j^2].
$$

因此，对线性层中单个权重的自然重要性代理量为

$$
|w_{ij}|\sqrt{\mathbb{E}[x_j^2]}.
$$

这就是原始 Wanda 的基本形式：权重幅值乘以输入通道的二阶激活尺度。

## 3. 冻结路由下的基本化简

**定理 1**  
在路由器决策、top-k expert 集合以及路由权重均冻结的条件下，若仅改变 expert $e$ 内部参数，则当前 MoE 层输出扰动满足

$$
\Delta y_{\mathrm{moe}}(x)=g_e(x)\,\Delta f_e(x),
$$

从而

$$
\|\Delta y_{\mathrm{moe}}(x)\|_2^2
=
g_e(x)^2\|\Delta f_e(x)\|_2^2.
$$

**证明**  
在冻结路由条件下，MoE 输出可写为

$$
y_{\mathrm{moe}}(x)
=
\sum_{r\in\mathcal{T}(x)} g_r(x)\,f_r(x).
$$

若仅改变 expert $e$ 内部参数，则对于任意 $r\neq e$，$f_r(x)$ 不变，而 $g_r(x)$ 与 $\mathcal{T}(x)$ 亦不变，因此

$$
\Delta y_{\mathrm{moe}}(x)
=
g_e(x)\Delta f_e(x).
$$

两边取欧几里得范数平方即可得到

$$
\|\Delta y_{\mathrm{moe}}(x)\|_2^2
=
g_e(x)^2\|\Delta f_e(x)\|_2^2.
$$

证毕。

**推论 1**  
在上述设定下，MoE-Wanda 分数的构造问题可化归为：先度量 expert 内部的局部输出敏感度，再乘上路由暴露因子 $g_e(x)^2$。

## 4. 下投影分支的精确结果

### 4.1 单权重扰动

设 $(W_{\mathrm{down},e})_{ij}$ 为下投影中的一个权重，则

$$
f_e(x)_i=\sum_{j=1}^m (W_{\mathrm{down},e})_{ij}h_e(x)_j.
$$

**定理 2**  
若仅将权重 $(W_{\mathrm{down},e})_{ij}$ 剪为零，则有

$$
\Delta f_e(x)_i=-(W_{\mathrm{down},e})_{ij}h_e(x)_j,
$$

且

$$
\|\Delta f_e(x)\|_2^2
=
(W_{\mathrm{down},e})_{ij}^2h_e(x)_j^2.
$$

进而

$$
\|\Delta y_{\mathrm{moe}}(x)\|_2^2
=
g_e(x)^2 (W_{\mathrm{down},e})_{ij}^2 h_e(x)_j^2.
$$

**证明**  
剪去 $(W_{\mathrm{down},e})_{ij}$ 仅会改变输出向量的第 $i$ 个坐标，且对应变化量恰为该项在线性和中的贡献相反数，因此

$$
\Delta f_e(x)_i=-(W_{\mathrm{down},e})_{ij}h_e(x)_j.
$$

由于其余坐标不变，故

$$
\|\Delta f_e(x)\|_2^2=(W_{\mathrm{down},e})_{ij}^2h_e(x)_j^2.
$$

再由定理 1 得到

$$
\|\Delta y_{\mathrm{moe}}(x)\|_2^2
=
g_e(x)^2\|\Delta f_e(x)\|_2^2
=
g_e(x)^2 (W_{\mathrm{down},e})_{ij}^2 h_e(x)_j^2.
$$

证毕。

### 4.2 对应的 Wanda 分数

对校准样本取期望，可得

$$
\mathbb{E}\left[\|\Delta y_{\mathrm{moe}}(x)\|_2^2\right]
=
(W_{\mathrm{down},e})_{ij}^2
\mathbb{E}\left[g_e(x)^2h_e(x)_j^2\right].
$$

因此定义下投影的 MoE-Wanda 分数为

$$
M^{(\mathrm{down})}_{e,ij}
=
|(W_{\mathrm{down},e})_{ij}|
\sqrt{\mathbb{E}[g_e(x)^2h_e(x)_j^2]}.
$$

**命题 1**  
分数 $M^{(\mathrm{down})}_{e,ij}$ 是当前设定下最接近原始 Wanda 且最精确的 MoE 扩展形式。

**说明**  
该分数同时编码：

- expert 被选中的频率；
- 路由权重的强弱；
- expert 内对应隐通道的活跃程度。

因此它比原始线性层 Wanda 额外体现了 MoE 的路由暴露信息。

## 5. 上投影分支的条件化结果

### 5.1 局部传播公式

考虑上投影中的权重 $(W_{\mathrm{up},e})_{kj}$。由

$$
b_e(x)_k=\sum_j (W_{\mathrm{up},e})_{kj}x_j,\qquad
h_e(x)_k=\phi(a_e(x)_k)b_e(x)_k
$$

可知，剪去该权重会引起

$$
\Delta b_e(x)_k=-(W_{\mathrm{up},e})_{kj}x_j.
$$

若将 $a_e(x)_k$ 与 $W_{\mathrm{down},e}$ 视为固定，则

$$
\Delta h_e(x)_k
=
\phi(a_e(x)_k)\Delta b_e(x)_k
=
-\phi(a_e(x)_k)(W_{\mathrm{up},e})_{kj}x_j.
$$

于是

$$
\Delta f_e(x)
=
-W_{\mathrm{down},e}[:,k]\phi(a_e(x)_k)(W_{\mathrm{up},e})_{kj}x_j.
$$

**定理 3**  
在固定 $a_e(x)_k$ 与 $W_{\mathrm{down},e}$ 的条件下，上投影单权重 $(W_{\mathrm{up},e})_{kj}$ 的局部扰动满足

$$
\|\Delta f_e(x)\|_2^2
=
(W_{\mathrm{up},e})_{kj}^2x_j^2\phi(a_e(x)_k)^2\|W_{\mathrm{down},e}[:,k]\|_2^2,
$$

从而

$$
\|\Delta y_{\mathrm{moe}}(x)\|_2^2
=
g_e(x)^2
(W_{\mathrm{up},e})_{kj}^2x_j^2\phi(a_e(x)_k)^2\|W_{\mathrm{down},e}[:,k]\|_2^2.
$$

**证明**  
在 $a_e(x)_k$ 固定时，映射

$$
h_k=\phi(a_k)b_k
$$

对 $b_k$ 是线性的，因此

$$
\Delta h_k=\phi(a_k)\Delta b_k
$$

为严格成立的关系。代入

$$
\Delta b_e(x)_k=-(W_{\mathrm{up},e})_{kj}x_j
$$

以及

$$
\Delta f_e(x)=W_{\mathrm{down},e}[:,k]\Delta h_e(x)_k
$$

即可得到所述公式。再由定理 1 乘上 $g_e(x)^2$ 即得结论。

证毕。

### 5.2 对应分数

由上式可定义

$$
M^{(\mathrm{up})}_{e,kj}
=
|(W_{\mathrm{up},e})_{kj}|
\|W_{\mathrm{down},e}[:,k]\|_2
\sqrt{\mathbb{E}[g_e(x)^2\phi(a_e(x)_k)^2x_j^2]}.
$$

若进一步采用对角解耦近似

$$
\mathbb{E}[g_e(x)^2\phi(a_e(x)_k)^2x_j^2]
\approx
\mathbb{E}[g_e(x)^2\phi(a_e(x)_k)^2]\,
\mathbb{E}[x_j^2],
$$

则得到更便宜的近似分数

$$
M^{(\mathrm{up})}_{e,kj}
\approx
|(W_{\mathrm{up},e})_{kj}|
\|W_{\mathrm{down},e}[:,k]\|_2
\sqrt{\mathbb{E}[g_e(x)^2\phi(a_e(x)_k)^2]}
\sqrt{\mathbb{E}[x_j^2]}.
$$

**讨论 1**  
上投影分支的结论并非联合最优结论，而是在“门控分支固定、下投影固定”条件下的边际扰动结论。其近似误差主要来自：

1. 未考虑后续门控分支变化；
2. 未考虑后续下投影变化；
3. 多权重同时剪枝时的交叉项。

## 6. 门控投影分支的一阶近似结果

考虑门控投影中的权重 $(W_{\mathrm{gate},e})_{kj}$。有

$$
a_e(x)_k=\sum_j (W_{\mathrm{gate},e})_{kj}x_j,
$$

剪去该权重后

$$
\Delta a_e(x)_k=-(W_{\mathrm{gate},e})_{kj}x_j.
$$

由于

$$
h_e(x)_k=\phi(a_e(x)_k)b_e(x)_k,
$$

需对 $\phi$ 做一阶泰勒展开：

$$
\Delta \phi(a_e(x)_k)
\approx
\phi'(a_e(x)_k)\Delta a_e(x)_k.
$$

故

$$
\Delta h_e(x)_k
\approx
b_e(x)_k\phi'(a_e(x)_k)\Delta a_e(x)_k
=
-b_e(x)_k\phi'(a_e(x)_k)(W_{\mathrm{gate},e})_{kj}x_j.
$$

进一步有

$$
\Delta f_e(x)=W_{\mathrm{down},e}[:,k]\Delta h_e(x)_k.
$$

**定理 4**  
在固定 $b_e(x)_k$ 与 $W_{\mathrm{down},e}$ 的条件下，门控投影单权重 $(W_{\mathrm{gate},e})_{kj}$ 的一阶局部扰动满足

$$
\|\Delta f_e(x)\|_2^2
\approx
(W_{\mathrm{gate},e})_{kj}^2
x_j^2
b_e(x)_k^2
\phi'(a_e(x)_k)^2
\|W_{\mathrm{down},e}[:,k]\|_2^2,
$$

从而

$$
\|\Delta y_{\mathrm{moe}}(x)\|_2^2
\approx
g_e(x)^2
(W_{\mathrm{gate},e})_{kj}^2
x_j^2
b_e(x)_k^2
\phi'(a_e(x)_k)^2
\|W_{\mathrm{down},e}[:,k]\|_2^2.
$$

**证明**  
由一阶泰勒展开，

$$
\phi(a_k+\Delta a_k)-\phi(a_k)
\approx
\phi'(a_k)\Delta a_k.
$$

再由

$$
h_k=\phi(a_k)b_k
$$

可得

$$
\Delta h_k
\approx
b_k\phi'(a_k)\Delta a_k.
$$

将

$$
\Delta a_k=-(W_{\mathrm{gate},e})_{kj}x_j
$$

代入，并通过线性下投影传播到输出，即得结论。最后再由定理 1 乘上 $g_e(x)^2$。

证毕。

由此定义门控投影的分数

$$
M^{(\mathrm{gate})}_{e,kj}
=
|(W_{\mathrm{gate},e})_{kj}|
\|W_{\mathrm{down},e}[:,k]\|_2
\sqrt{\mathbb{E}[g_e(x)^2 b_e(x)_k^2\phi'(a_e(x)_k)^2x_j^2]}.
$$

若采用对角解耦近似，可写成

$$
M^{(\mathrm{gate})}_{e,kj}
\approx
|(W_{\mathrm{gate},e})_{kj}|
\|W_{\mathrm{down},e}[:,k]\|_2
\sqrt{\mathbb{E}[g_e(x)^2 b_e(x)_k^2\phi'(a_e(x)_k)^2]}
\sqrt{\mathbb{E}[x_j^2]}.
$$

**讨论 2**  
门控投影分支比上投影和下投影更困难，原因在于：

1. 它必须显式依赖非线性局部斜率 $\phi'(a_k)$；
2. 它对展开点位置更敏感；
3. 当一次剪枝过多时，一阶近似更容易失真。

## 7. 统一的局部线性化表达

上面三类分支可以通过同一个局部公式统一起来。

对任意 hidden channel $k$，在当前展开点附近有

$$
\Delta h_k
\approx
b_k\phi'(a_k)\Delta a_k
+
\phi(a_k)\Delta b_k.
$$

若再引入下投影变化，则有

$$
\Delta f_e
\approx
\Delta W_{\mathrm{down},e}[:,k]\,h_k
+
W_{\mathrm{down},e}[:,k]\Big(
b_k\phi'(a_k)(\Delta W_{\mathrm{gate},e}[k,:]x)
+
\phi(a_k)(\Delta W_{\mathrm{up},e}[k,:]x)
\Big).
$$

再由定理 1，得到对应的 MoE 层局部扰动：

$$
\Delta y_{\mathrm{moe}}
\approx
g_e(x)\Delta f_e.
$$

**定理 5**  
上述公式给出了一个统一的 channel 级局部近似框架，其中：

- 下投影通过 $h_k$ 直接作用于输出；
- 上投影通过 $\phi(a_k)$ 调制后的 $\Delta b_k$ 作用于输出；
- 门控投影通过 $b_k\phi'(a_k)$ 调制后的 $\Delta a_k$ 作用于输出。

**证明**  
由

$$
h_k=\phi(a_k)b_k
$$

对 $(a_k,b_k)$ 做一阶展开，即得

$$
\Delta h_k
\approx
b_k\phi'(a_k)\Delta a_k+\phi(a_k)\Delta b_k.
$$

再将其代入

$$
f_e=W_{\mathrm{down},e}h_e
$$

的微扰表达式，并加入 $\Delta W_{\mathrm{down},e}$ 的线性项即可得到统一公式。最后由定理 1 推广到 MoE 层输出。

证毕。


## 9. 静态展开点与当前方法的局限性

在 one-shot Wanda 风格的方法中，通常先用未剪枝模型在校准集上做一次前向传播，收集各层输入统计量，再基于这些旧统计量一次性完成当前层的剪枝。因此：

- 下投影打分时，所使用的 $h_e$ 通常是旧的中间激活；
- 而不是在先剪完上投影或门控投影之后重新得到的新 $h_e$；
- 这意味着所有分数都围绕同一个静态展开点定义。

**命题 2**  
在静态展开点方法中，

- 下投影打分默认固定 $h_e$；
- 上投影打分默认固定 $a_e$ 与 $W_{\mathrm{down},e}$；
- 门控投影打分默认固定 $b_e$ 与 $W_{\mathrm{down},e}$。

因此三类分数本质上都是条件化的局部边际分数，而不是联合最优分数。

**说明**  
由此可知，当前静态分数没有显式建模以下因素：

1. 上投影、门控投影、下投影的联合变化；
2. 多个参数块同时剪枝产生的交叉项；
3. 剪枝后展开点本身的漂移；
4. 重新路由或重新校准后统计量的变化。
