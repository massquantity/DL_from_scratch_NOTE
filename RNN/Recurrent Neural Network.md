==key==  key^1^  key~1~        



### Vanilla RNN

![](https://raw.githubusercontent.com/massquantity/DL_from_scratch_NOTE/master/pic/RNN/1.png)



上图中 $\bf{X, h, y}$ 都是向量，公式如下：
$$
% <![CDATA[
\begin{align}
\textbf{h}_{t} &= f_{\textbf{W}}\left(\textbf{h}_{t-1}, \textbf{x}_{t} \right) \tag{1} \\
\textbf{h}_{t} &= f\left(\textbf{W}_{hx}\textbf{x}_{t} + \textbf{W}_{hh}\textbf{h}_{t-1} + \textbf{b}_{h}\right) \tag{2a} \\
\textbf{h}_{t} &= \textbf{tanh}\left(\textbf{W}_{hx}\textbf{x}_{t} +  \textbf{W}_{hh}\textbf{h}_{t-1} + \textbf{b}_{h}\right) \tag{2b} \\
\hat{\textbf{y}}_{t} &= \textbf{softmax}\left(\textbf{W}_{yh}\textbf{h}_{t} + \textbf{b}_{y}\right) \tag{3}
\end{align} %]]>
$$
其中 $\textbf{W}_{hx} \in \mathbb{R}^{h \times x}, \; \textbf{W}_{hh} \in \mathbb{R}^{h \times h},  \; \textbf{W}_{yh} \in \mathbb{R}^{y \times h}, \; \textbf{b}_{h} \in \mathbb{R}^{h}, \; \textbf{b}_{h} \in \mathbb{R}^{h}$

$(2a)$ 式中的两个矩阵 $\mathbf{W}$ 可以合并：
$$
\begin{align}
\textbf{h}_{t} &= f\left(\textbf{W}_{hx}\textbf{x}_{t} + \textbf{W}_{hh}\textbf{h}_{t-1} + \textbf{b}_{h}\right) \\
& = f\left(\left(\textbf{W}_{hx}, \textbf{W}_{hh}\right) 
\begin{pmatrix}
\textbf{x}_t \\
\textbf{h}_{t-1}
\end{pmatrix}
+ \textbf{b}_{h}\right) \\
& =  f\left(\textbf{W}
\begin{pmatrix}
\textbf{x}_t \\
\textbf{h}_{t-1}
\end{pmatrix}
+ \textbf{b}_{h}\right)
\end{align}
$$


注意到在计算时，每一步使用的参数 $\textbf{W}, \; \textbf{b}$ 都是一样的，也就是说每个步骤的参数都是共享的，这是RNN的重要特点。

==TODO： RNN有两个输入， 李宏毅==

和普通的全连接层相比，RNN 除了输入 $\textbf{x}_t$ 外，还有输入隐藏层上一节点 $\mathbf{h}_{t-1}$ ，RNN 每一层的值就是这两个输入用矩阵 $\textbf{W}_{hx}$，$\textbf{W}_{hh}$和激活函数进行组合。从 $(2a)$ 式可以看出 $\textbf{x}_t$ 和 $\mathbf{h}_{t-1}$  都是与 $\textbf{h}_h$ 全连接的，下图形象展示了各个时间节点 RNN 隐藏层记忆的变化：(https://blog.csdn.net/zzukun/article/details/49968129)，随着时间流逝，蓝色结点保留地越来越少，这意味着RNN对于长时记忆的困难。

![](/home/massquantity/Documents/DL_from_scratch_NOTE/pic/RNN/recurrence_gif.gif)





#### Vanishing & Exploding Gradient Problems

RNN 中 Loss 的计算图示例：

![](https://raw.githubusercontent.com/massquantity/DL_from_scratch_NOTE/master/pic/RNN/2.png)



总的 Loss 是每个时间结点的加和 ： $\mathcal{\large{L}} (\hat{\textbf{y}}, \textbf{y}) = \sum_{t = 1}^{T} \mathcal{ \large{L} }(\hat{\textbf{y}_t}, \textbf{y}_{t})$



**backpropagation through time (BPTT)** 算法：
$$
\frac{\partial \textbf{E}}{\partial \textbf{W}} = \sum_{t=1}^{T} \frac{\partial \textbf{E}_{t}}{\partial \textbf{W}} = \sum_{t=1}^{T} \frac{\partial \textbf{E}_t}{\partial \textbf{y}_{t}} \frac{\partial \textbf{y}_{t}}{\partial \textbf{h}_{t}} \overbrace{\frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{k}}}^{ \bigstar } \frac{\partial \textbf{h}_{k}}{\partial \textbf{W}}
$$
其中 $\frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{k}}$ 包含一系列 $\text{Jacobian}$ 矩阵，
$$
\frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{k}} = \frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{t-1}} \frac{\partial \textbf{h}_{t-1}}{\partial \textbf{h}_{t-2}} \cdots \frac{\partial \textbf{h}_{k+1}}{\partial \textbf{h}_{k}} 
= \prod_{i=k+1}^{t} \frac{\partial \textbf{h}_{i}}{\partial \textbf{h}_{i-1}}
$$
由于 RNN 中每个 time step 都是用相同的 $\textbf{W}$ ，所以由 $(2a)$ 式可得：
$$
\prod_{i=k+1}^{t} \frac{\partial \textbf{h}_{i}}{\partial \textbf{h}_{i-1}} = \prod_{i=k+1}^{t} \textbf{W}^\top \text{diag} \left[ f'\left(\textbf{h}_{i-1}\right) \right]
$$


由于 $\textbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 为方阵，对其进行特征值分解：
$$
\mathbf{W} = \mathbf{V} \, \text{diag}(\boldsymbol{\lambda}) \, \mathbf{V}^{-1}
$$
由于上式是连乘 $\text{t}$ 次 $\mathbf{W}$ :
$$
\mathbf{W}^t = (\mathbf{V} \, \text{diag}(\boldsymbol{\lambda}) \, \mathbf{V}^{-1})^t = \mathbf{V} \, \text{diag}(\boldsymbol{\lambda})^t \, \mathbf{V}^{-1}
$$
连乘的次数多了之后，则若最大的特征值 $\lambda >1$ ，会产生梯度爆炸； $\lambda < 1$ ，则会产生梯度消失 。



#### 梯度爆炸的解决办法：

(1)  Truncated Backpropagation through time：每次只 BP 固定的 time step 数，类似于 mini-batch SGD。缺点是丧失了长距离记忆的能力。

![](https://raw.githubusercontent.com/massquantity/DL_from_scratch_NOTE/master/pic/RNN/3.png)

(2)  Clipping Gradients： 当梯度超过一定的 threshold 后，就进行 element-wise 的裁剪，该方法的缺点是又引入了一个新的参数 threshold。该方法可被视为一种基于瞬时梯度大小来自适应 learning rate 的方法：
$$
\text{if} \quad \lVert \textbf{g} \rVert \ge \text{threshold} \\[1ex]
\textbf{g} \leftarrow \frac{\text{threshold}}{\lVert \textbf{g} \rVert} \textbf{g}
$$
![](https://raw.githubusercontent.com/massquantity/DL_from_scratch_NOTE/master/pic/RNN/4.png)



#### 梯度消失的解决办法

(1)  使用 LSTM、GRU等升级版 RNN，使用各种 gates 控制信息的流通。

(2)  [将权重矩阵 $\textbf{W}$ 初始化为正交矩阵](https://arxiv.org/pdf/1602.06662.pdf)。正交矩阵有如下性质：$A^T A =A A^T =  I, \; A^T = A^{-1}$， 正交矩阵的特征值的绝对值为 $\text{1}$ 。证明：  对矩阵 $A$ 有，
$$
\begin{align*}
& A \mathbf{v} = \lambda \mathbf{v} \\[1ex]
 ||A \mathbf{v}||^2& = (A \mathbf{v})^\text{T} (A \mathbf{v}) \\
&= \mathbf{v}^\text{T}A ^{\text{T}}A \mathbf{v} \\
& = \mathbf{v}^{\text{T}}\mathbf{v} \\ &
= ||\mathbf{v}||^2 \\ &
= |\lambda|^2 ||\mathbf{v}||^2
\end{align*}
$$
由于 $\mathbf{v}$ 为特征向量，$\mathbf{v} \neq 0$ ，所以 $|\lambda| = 1$ ，这样连乘之后 $\lambda^t$ 不会出现越来越小的情况。





### LSTM

> 1、原始的 LSTM 是没有 forget gate 的，或者说相当于 forget gate 恒为 1，所有不存在梯度消失问题；
>
> 2、现在的 LSTM 被引入了 forget gate，但是 LSTM 的一个初始化技巧就是将 forget gate 的 bias 置为正数（例如 1 或者 5，这点可以查看各大框架源码），这样一来模型刚开始训练时 forget gate 的值都接近 1，不会发生梯度消失；
>
> 3、随着训练过程的进行，forget gate 就不再恒为 1 了。不过，一个训好的模型里各个 gate 值往往不是在 [0, 1] 这个区间里，而是要么 0 要么 1，很少有类似 0.5 这样的中间值，其实相当于一个二元的开关。假如在某个序列里，forget gate 全是 1，那么梯度不会消失；否则，若某一个 forget gate 是 0，这时候虽然会导致梯度消失，但这是 feature 不是 bug，体现了模型的选择性（有些任务里是需要选择性的，比如情感分析里”这部电影很好看，但是票价有点儿贵“，读到”但是“的时候就应该忘掉前半句的内容，模型不想让梯度流回去）；
>
> 4、基于第 3 点，我不喜欢从梯度消失/爆炸的角度来谈论 LSTM/GRU 等现代门控 RNN 单元，更喜欢从选择性的角度来解释，模型选择记住（或遗忘）它想要记住（或遗忘）的部分，从而更有效地利用其隐层单元。
>
> https://www.zhihu.com/question/34878706

















