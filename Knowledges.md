# Mathematical operation

## outer

*"outer" refers to operations that involve pairwise combinations of elements from two input sets or arrays, producing a new set or array.*

In the case of "**outer addition,**" the term describes an operation where you take two input arrays and compute the pairwise addition of their elements, resulting in a new array with a shape equal to the concatenation of the input shapes. The term "outer" is used here because the operation involves combining elements from the "outer" dimensions of the input arrays.

Similarly, you may have heard of the "**outer product**" in the context of vectors and matrices. The outer product of two vectors creates a matrix, where each element at position (i, j) is the product of the i-th element of the first vector and the j-th element of the second vector. This operation is also called the "tensor product" in some contexts.

## 求和符号

- 对于有限项求和，求和符号顺序可以交换

- 对于无穷求和：
  $$
  如果，\sum_{i=1}^\infty\sum_{j=1}^\infty\lvert a_{i.j}\rvert<\infty，则:
  $$

  $$
  \sum_{i=1}^\infty\sum_{j=1}^\infty a_{ij}=\sum_{j=1}^\infty\sum_{i=1}^\infty a_{ij}
  $$

# Convolution

## 1-D Filters

$$
(f*g)[n]=\sum_{m=-\infty}^{\infty}f[m]g[n-m]\\
f\longrightarrow \text{the data}
\\g\longrightarrow\text{the filter(kernel)}
$$

第二种形式更容易理解:
$$
(f*g)[n]=\sum_{m=-\infty}^{\infty}f[n+m]g[-m]
$$
**if the length of the kernel is 3:**
$$
\begin{align}
(f*g)[n]&=f[n-1]g[1]+f[n]g[0]+f[n+1]g[-1]\\
&=\left[f[n-1],\ f[n],\ f[n+1]\right]\cdot[g[1],\ g[0],\ g[-1]]
\\
&=\sum_{m=n-1}^{n+1}f[m]g[n-m]
\\&=\sum_{m=-1}^{m=1}f[n+m]g[-m]
\end{align}
$$
the center of inner product is f[n]




$$
\text{Derivative Theorem: }
\\\frac{d}{dx}(f*g)=f*\frac{d}{dx}g
$$

## 2-D Filters

$$
\begin{align}
(f*h)(m,n)&=\sum_{k,l}f(k,l)h(m-k,n-l)
\\&=\sum_{k,l}f(m+k,n+l)h(-k,-l)\\\text{if the kernel size is (3,3)}
\\&=\sum_{k=-1}^{1}\sum_{l=-1}^{1}f(m+k,n+l)h(-k,-l)

\end{align}
$$

the matrix **h** is rotated by 180 degrees



## Backpropagation for a Linear Layer

http://cs231n.stanford.edu/handouts/linear-backprop.pdf

Given：
$$ {*}
\begin{gather*}
input: X_{N\times D}\\
weight\_matrix : W_{D\times M}
\\Y_{N\times M} = XW
\end{gather*}
$$


- $\frac{\part L}{\part Y}$ will be a matrix with the same shape as $Y$
  - $\frac{\partialＬ}{\partial y_{i,j}}$ must at the position of $y_{i,j}$



By the chain rule, we know that:
$$
\frac{\part{L}}{\part X} = \frac{\part L}{\part Y}\frac{\part Y}{\part X}\\
\frac{\part{L}}{\part W} = \frac{\part L}{\part Y}\frac{\part Y }{\part W}
$$
the items $\frac{\part Y}{\part X}\ \ \frac{\part Y}{\part W}$ are really huge Jacobian matrices

> $\frac{\part Y}{\part X}$ is $N \times M \times N\times D$
>
> $\frac{\part Y}{\part W}$ is $N\times M \times D\times M$

so it's infeasible to conduct the chain rule directly



Then how to compute the $\frac{\part L}{\part X}$:
$$
X = \begin{pmatrix}x_{1,1}&x_{1,2}\\x_{2,1}&x_{2,2}\end{pmatrix}\Longrightarrow\frac{\part L}{\part X} = \begin{pmatrix}\frac{\part L }{\part x_{1,1}}& \frac{\part L }{\part x_{1,2}}\\\frac{\part L }{\part x_{2,1}}&\frac{\part L }{\part x_{2,2}}\end{pmatrix}
$$
 **thinking one element at a time** 
$$
\begin{align}
\left(\frac{\part L}{\part X}\right)_{N\times D} &= \left(\frac{\part L}{\part Y}\right)_{N\times M}(W_{D\times M})^T\\
\left(\frac{\part L }{\part W}\right)_{D\times M}&=(X_{N\times D})^T\left(\frac{\part L}{\part Y}\right)_{N \times M}
\end{align}
$$


# Rotation

Geometry provides us with four types of transformations, namely, rotation, reflection, translation, and resizing.



## Rotation Matrices

- rotate a vector in the counterclockwise direction by an angle θ.

- [orthogonal matrices](https://www.cuemath.com/algebra/orthogonal-matrix/) with a determinant equal to 1



> basic/elementary rotation

```python
##### row
# counterclockwise rotation of \theta about the x axis
R_x(θ) = | 1    0         0   |
         | 0  cos(θ)  -sin(θ) |
         | 0  sin(θ)   cos(θ) |
        
##### pitch
# counterclockwise rotation of \theta about the y axis
R_y(θ) = |  cos(θ)  0  sin(θ) |
         |    0     1    0    |
         | -sin(θ)  0  cos(θ) |
        
##### yaw
# counterclockwise rotation of \theta about the z axis
R_z(θ) = | cos(θ)  -sin(θ)  0 |
         | sin(θ)   cos(θ)  0 |
         |   0        0     1 |
```

**The columns of R represent the transformed basis vectors of the rotated coordinate system**

> 旋转的本质是，将原来的基 $(e_1, e_2, ... , e_n)$ 转化成了一组新的正交基
>
> column vectors ---> new basis ---> new axises

**the rows represent the original coordinate system's basis vectors projected onto the rotated coordinate system.**



> 对于正交矩阵（旋转矩阵） 可以这样理解：
>
> 
>
> **列向量是一组新的正交基在原坐标系下的坐标**
>
> $\begin{pmatrix} 3 \\ 2\end{pmatrix}$是在这组新的基下的坐标，$\begin{pmatrix} \frac{5}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}\end{pmatrix}$是线性组合成的向量在原坐标系下的坐标
>
> **行向量是坐标系x,y轴在新的坐标系中的坐标**
>
> 设$\begin{pmatrix}x\\y\end{pmatrix}$是x轴在新基下的坐标：
> $$
> \begin{align}
> A\begin{pmatrix}x\\y\end{pmatrix}&=\begin{pmatrix}1\\0\end{pmatrix}\\
> \begin{pmatrix}x\\y\end{pmatrix}&=A^T\begin{pmatrix}1\\0\end{pmatrix}
> \end{align}
> $$

旋转的发生相当于将原来的基换成了一组旋转后的基，然后计算所有向量作为旋转后基的系数，这样算出的坐标就是在原来基下旋转的坐标







## Euler angles

Euler angles are a set of three angles (α, β, γ) that describe a rotation in 3D space using three consecutive rotations about different axes. There are multiple conventions for Euler angles, such as ZYX (yaw-pitch-roll), XYZ (roll-pitch-yaw), and others. Euler angles are compact and intuitive but suffer from gimbal lock, a phenomenon that can cause loss of a degree of freedom in certain configurations.



**Problems**:

- gimbal lock



> In summary, the correspondence between rotation matrices and Euler angles is one-to-one for a specific order, except in cases of gimbal lock, where multiple Euler angle combinations can represent the same rotation matrix.





## Axis-angle representation

$$
\mathbf{\theta} = \theta e
$$

Axis-angle representation uses a unit vector (k_x, k_y, k_z) to define the rotation axis and an angle θ to define the rotation magnitude around that axis. The axis-angle representation avoids gimbal lock and has a more compact representation than rotation matrices but requires conversion to other representations for certain operations, such as interpolation or composition.



## Quaternions

Quaternions are a four-dimensional extension of complex numbers and can represent rotations in 3D space using a scalar part (w) and a vector part (x, y, z). Quaternions avoid gimbal lock, have a compact representation, and can be easily interpolated (e.g., using SLERP for smooth rotation interpolation). However, they are less intuitive than Euler angles and require conversions for certain operations.

Each representation has its own specific use cases, advantages, and drawbacks. In general, SO(3) transformations are crucial for representing and manipulating rotations in 3D space across various applications.

#### Stereographic Projection

**help us to figure out what's a quaternion be like in a 4-d space**

通过这种投影，可以将一个n维的东西投影到n-1维上

- 复平面(2-d)上的单位圆 ---> vertical line (1-d)
- a uni-sphere (3-d) ---> a plane (2-d)
- a uni-hypersphere (4-d) ---> a whole 3-d space





在投影时，

- real-part --> omitted --> 改变了在投影空间中的位置（scale）
- virtual-part --> be projected on --> 改变了在投影空间中的方向（orientation）



2-d rotation --> 1-degree freedom --> uni-complex

3-d rotation --> 3-degree freedom --> uni-quaternion



#### What is quaternion

- Definition:

$$
q = w +x\boldsymbol{i}+y\boldsymbol{j}+k\boldsymbol{k}
$$



- how to multiply?

  - compute the production

  $$
  \begin{align}
  For\ i,j,k:\\
  ij&= -ji=k\\
  jk&=-kj=i\\
  ki&=-ik=j
  \end{align}
  $$

  - scale or squish

$$
q_1 \cdot q_2 = \underbrace{\frac{q_1}{\Vert{q_1}\Vert}}_{Apply\ special\ 4d\ rotation} \overbrace{\Vert q_1\Vert \cdot q_2}^{Scale\ q_2}
$$

#### quaternion and 3-d rotation

- **double cover**

  each rotation in 3-d corresponds to 2 quaternions



## SO(3)

SO(3) stands for the "Special Orthogonal group of order 3" and represents the set of all 3x3 **orthogonal matrices with determinant equal to 1**. In simpler terms, SO(3) transformations describe rotations in three-dimensional space. **These transformations preserve the distance between points and the angles between vectors**, which makes them particularly important in fields such as robotics, computer vision, and computer graphics.

*If the determinant were -1, it would mean that the transformation includes a reflection, which would change the orientation (handedness) of the coordinate system.*







# SE（3）

The Special Euclidean group SE(3) represents rigid body transformations in three-dimensional space. **These transformations include both rotations and translations.** *The elements of SE(3) are 4x4 homogeneous transformation matrices that can be decomposed into a 3x3 rotation matrix (from SO(3)) and a 3x1 translation vector.* SE(3) is a subgroup of the more general Euclidean group E(3), which includes not only rigid body transformations but also scaling and other affine transformations.













#　数据处理

![img](https://pic3.zhimg.com/80/56079895270388df370b8643f3537dee_1440w.webp)



- VQ的约束是要求H的每一列只有一个元素为1，其它为0，因此相当于将m个数据归纳成了k个代表，原数据映射过去就是取k个基当中与原向量距最小的来重新表示。所以VQ的基都是一张张完整正常的脸，它们都是最具代表性的脸。 用于 GIF

- PCA大家用得比较多，就是求一组标准正交基，第一个基的方向取原数据方差最大的方向，然后第二个基在与第一个基正交的所有方向里再取方差最大的，这样在跟前面的基都正交的方向上一直取到k个基。所以PCA的基没有直观的物理意义，而且W和H里面的元素都是可正可负的，这就意味着还原时是用W的基加加减减得到的。

- NMF约束了非负，一些部件 的的线性组合

  - 左侧列向量 --> 部件向量
  - 右侧列向量 ---> 系数向量

  

- SVD
  $$
  A_{mn}= U_{mm}\Sigma_{mn}V_{nn}^T
  $$
  



## PCA （Principal Component Analysis）

### 一组数据可以在不同基的表示下有不同的形态

设$n$是数据的维度 $m$是数据的数量 （每个列向量是一个数据）
$$
Y_{nm} = \begin{pmatrix}\alpha_1^T\\ \alpha_2^T\\
\vdots \\ \alpha_n^T\end{pmatrix}X_{nm}
$$
可以认为：

- X的列向量实在原基下的坐标

- Y的列向量是在新基下的坐标



### 衡量基选取好坏的标准

希望投影后的投影值尽可能分散，因为如果重叠就会有样本消失。当然这个也可以从熵的角度进行理解，熵越大所含信息越多。

在一维空间中我们可以用方差来表示数据的分散程度。而对于高维数据，我们用协方差进行约束，协方差可以表示两个变量的相关性。为了让两个变量尽可能表示更多的原始信息，我们希望它们之间不存在线性相关性，因为相关性意味着两个变量不是完全独立，必然存在重复表示的信息。

至此，我们得到了降维问题的优化目标：**将一组 N 维向量降为 K 维，其目标是选择 K 个单位正交基，使得原始数据变换到这组基上后，各变量两两间协方差为 0，而变量方差则尽可能大（在正交的约束下，取最大的 K 个方差）**

### **协方差矩阵**

当x的每个**列向量**是一个数据时：
$$
XX^T
$$


#### 如何统一表示方差和协方差

最终要达到的目的与**变量内方差及变量间协方差**有密切关系。因此我们希望能将两者统一表示，仔细观察发现，两者均可以表示为内积的形式，而内积又与矩阵相乘密切相关

>  假设我们每个向量是2维，总共有m个变量，则有：

$$
X = \begin{pmatrix}a_1 & a_2 & \cdots &a_m\\b_1&b_2&\cdots &b_m\end{pmatrix}
$$

$$
\frac{1}{m}XX^\mathsf{T}= \begin{pmatrix}  \frac{1}{m}\sum_{i=1}^m{a_i^2} & \frac{1}{m}\sum_{i=1}^m{a_ib_i} \\ \frac{1}{m}\sum_{i=1}^m{a_ib_i} & \frac{1}{m}\sum_{i=1}^m{b_i^2}  \end{pmatrix} = \begin{pmatrix}  Cov(a,a) & Cov(a,b) \\  Cov(b,a) & Cov(b,b) \end{pmatrix} \\
$$



因此，**我们需要将除对角线外的其它元素化为 0，并且在对角线上将元素按大小从上到下排列（变量方差尽可能大）**



### 如何寻找一组合适的基

- 原始数据矩阵 X 对应的协方差矩阵为 C

- P 由选择的基**按行组成**的矩阵

- Y 为 X 对 P 做基变换后的数据

$$
Y_{nm}=P_{nn}X_{nm}
$$



- 设 Y 的协方差矩阵为 D

$$
\begin{aligned}  D & =  \frac{1}{m}YY^T \\  & = \frac{1}{m}(PX)(PX)^T \\ & = P(\frac{1}{m}XX^T)P^T \\  & = PCP^T  \end{aligned}  \\
$$
优化目的:使得D除对角线外的其它元素化为 0，并且在对角线上将元素按大小从上到下排列（变量方差尽可能大

**所以将C做特征值分解即可**

将特征向量按对应特征值大小从上到下按行排列成矩阵，取前 k 行组成矩阵 P
$$
Y_{nm} = \begin{pmatrix}\alpha_1^T\\ \alpha_2^T\\
\vdots \\ \alpha_n^T\end{pmatrix}X_{nm}
\Longrightarrow Y_{km}=\underbrace{\begin{pmatrix}\alpha_1\\\alpha_2\\\vdots\\\alpha_k\end{pmatrix}
X_{km}}_{\text{select\ k\ dims}}
$$


## SVD

$$
A_{nm}=U_{nn}\Sigma_{nm}V_{mm}^T
$$

$$
A^TA=(U \Lambda V^T)^TU \Lambda V^T =V \Lambda^T U^TU \Lambda V^T  = V\Lambda^2 V^T \\ AA^T=U \Lambda V^T(U \Lambda V^T)^T =U \Lambda V^TV \Lambda^T U^T = U\Lambda^2 U^T  \\
$$

- $U$是左奇异矩阵 从PCA可见，代表其**列向量空间**中的“最优”正交基 是$AA^T$的特征向量
- $V$是右奇异矩阵 从PCA可见，代表其**行向量空间**中的“最优”正交基 是$A^TA$的特征向量
- $\Lambda$中的奇异值是二者特征值的平方根

![img](https://images2015.cnblogs.com/blog/1042406/201701/1042406-20170105140822191-1774139119.png)







# Divergence



## KL Divergence

A natural measurement of **distance between probability distributions** motivated by looking at how likely the second distribution will be able to generate samples from the first distribution.

Actually, it's:

![Screenshot from 2023-04-08 15-10-57](/home/wangqx/Pictures/Screenshot from 2023-04-08 15-10-57.png)



**Defination:**

- For two discrete probability distributions P(x) and Q(x):
  $$
  KL(P||Q)=\sum\left[P(x)*\log\left(\frac{P(x)}{Q(x)}\right)\right]
  $$

- For two continuous probability distributions P(x) and Q(x):
  $$
  KL(P||Q)=\int P(x)*\log\left(\frac{P(x)}{Q(x)}\right)dx
  $$



A few important properties of KL divergence:

1. **Non-negativity**: KL divergence is always non-negative (KL(P || Q) ≥ 0). It is equal to 0 if and only if P and Q are the same distribution.
2. **Not symmetric**: KL(P || Q) ≠ KL(Q || P) in general. This means that KL divergence is not a true metric or distance measure, as it does not satisfy the symmetry property.
3. **Not defined for non-overlapping distributions**: If there are events x where P(x) > 0 and Q(x) = 0, the KL divergence is not defined, since the logarithm term becomes infinite.



The formula for KL divergence is derived from the concept of **entropy and cross-entropy** in information theory. 

KL divergence (KL(P || Q)) is defined as the difference between the cross-entropy of P and Q, and the entropy of P:
$$
KL(P||Q)=H(P,Q) - H(P)
$$

## Entropy

- **Entropy (H(P)) of a probability distribution P is a measure of the uncertainty or randomness associated with that distribution.** 
  $$
  H(p)= -\sum\left[P(x)*\log(P(x))\right]\\
  H(p)= -\int P(x)*\log(P(x))dx
  $$

  > **Entropy reaches its maximum value of log(N) when the distribution is uniform**
  >
  > 1. 当所有概率均等时，不确定性最大
  >
  > 2. 从数学角度
  >
  >    because $-\log x$ is a convex function ---> Jensen inequality
  >
  >    只有当$x_1=x_2=\cdots=x_n$时等号成立

  $$
  t_1f(x_1)+t_2f(x_2)+\cdots+t_nf(x_n)\le f(t_1x_1+t_2x_2+\cdots+t_nx_n)
  $$

  

  > **Why the entroy is defined like this?**
  >
  > **已知概率分布的情况下，一个信息的价值**
  >
  > 信息熵其实是 已知信息的概率分布，获得确定信息所需要的“问题数”
  >
  > 这等价与一个高尔顿板模型，每个信息所需要的“bounces”也就是深度，所以取log

## Cross-entropy

- **Cross-entropy (H(P, Q)) is a measure of the average number of bits needed to encode events from distribution P when we use an encoding scheme optimized for distribution Q.**

![Screenshot from 2023-04-08 15-18-42](/home/wangqx/Pictures/Screenshot from 2023-04-08 15-18-42.png)

## JS Divergence

M(x) = (P(x) + Q(x)) / 2

The JS divergence is then defined as:

JS(P || Q) = (1/2) * KL(P || M) + (1/2) * KL(Q || M)



1. Symmetry: JS divergence is symmetric, meaning that JS(P || Q) = JS(Q || P). This is in contrast to the KL divergence, which is not symmetric (KL(P || Q) ≠ KL(Q || P)).
2. Non-negativity: JS divergence is always non-negative (JS(P || Q) ≥ 0), and it is equal to 0 if and only if P and Q are the same distribution.
3. Bounded: JS divergence is bounded between 0 and log(2) (for distributions with the same support) when using logarithm base 2 (which corresponds to using bits as the unit of information). The upper bound is achieved when the two distributions are completely disjoint.
4. Smoothing: JS divergence is less sensitive to zero-probability events compared to KL divergence. It is defined even when the supports of P and Q do not overlap completely, as the intermediate distribution M(x) smooths out the probabilities.

# Imitation Learning

Imitation Learning (IL) is a machine learning approach where an agent learns to perform a task by observing and imitating an expert's behavior.



##  Behavior Cloning (BC) 



1. Data collection: The first step in BC is to collect a dataset of expert demonstrations. These demonstrations usually consist of **state-action pairs**, where the state represents the current environment or situation, and the action represents the expert's response or decision in that state. The dataset should ideally contain diverse examples covering various scenarios that the agent might encounter during its own execution.
2. Preprocessing: Depending on the problem domain and the collected data, preprocessing might be necessary to clean, normalize or transform the data into a suitable format for the learning algorithm. This step might also involve splitting the dataset into training and validation sets to evaluate the performance of the learned model during training.
3. Model selection: Choose an appropriate model for learning the expert's behavior. **This model can be a simple linear regressor, a neural network, or any other machine learning model that can capture the mapping between the state and the action.** The choice of the model depends on the complexity of the problem and the data.
4. Training: Train the selected model using the preprocessed dataset. **The objective of training is to minimize the difference between the expert's actions and the model's predicted actions given the same states.** During this process, the model learns to approximate the expert's policy, which is a mapping from states to actions. The training process usually involves multiple iterations, updating the model parameters to minimize a loss function that quantifies the difference between expert and predicted actions.
5. Evaluation: After training, evaluate the performance of the learned model on a separate validation set or through simulation in the environment. This step helps to determine if the model has learned the expert's policy effectively and generalizes well to new situations.
6. Deployment: Once the learned model's performance is deemed satisfactory, deploy the model as the agent's policy in the target environment. The agent will now use the learned model to make decisions in various states, imitating the expert's behavior.
7. Fine-tuning (optional): In some cases, the model's performance may not be adequate for the task, or the agent may encounter situations that were not present in the training data. In such cases, it may be necessary to fine-tune the model using additional data, reinforcement learning, or other techniques to improve its performance and adapt to new situations.



### compounding errors

A major shortcoming of BC is compounding errors, where errors from previous timesteps accumulate and cause the robot to drift off of its training distribution, leading to hard-to-recover states



## DAgger

DAgger (Dataset Aggregation) is an algorithm for imitation learning that aims to address the distribution mismatch problem that arises when the learned policy deviates from the expert's policy. It was introduced by Stefano Ross et al. in 2011. The main idea of DAgger is to iteratively train the model using both expert demonstrations and the agent's own experiences, refining the policy in the process. Here is a step-by-step description of how the DAgger algorithm is constructed:

1. Initialization: First, initialize the dataset D with a set of expert demonstrations. These demonstrations consist of **state-action pairs**, where the state represents the current environment or situation, and the action represents the expert's response or decision in that state. Also, **initialize the policy π, which can be a simple linear model or a complex neural network**, depending on the problem.

2. Training iteration: Repeat the following steps for a fixed number of iterations or until a performance criterion is met:

   a. Train the **policy π** on the dataset D. The objective of training is to minimize the difference between the expert's actions and the policy's predicted actions, given the same states. *this is the same as BC*

   > a simple linear model of a complex neural network can just be *a policy*
   >
   > SO *policy need not be in RL*

   b. **Execute** the policy π in the environment to generate a new set of trajectories. During this process, the agent interacts with the environment and collects new state-action pairs based on its current policy.

   c. **For each state visited by the agent during the execution, query the expert for the best action in that state**. Add the new state-action pairs (consisting of the state visited by the agent and the expert's action) to the dataset D. *This step is crucial because it helps **align** the agent's policy with the expert's policy in states that the agent is likely to visit.*

   d. Optionally, update the policy mixing parameter β, which determines the ratio of actions chosen by the current policy and the expert's actions during the execution. This parameter can be decreased over time, allowing the agent to rely more on its own policy and less on the expert's actions as it gains experience.

3. Evaluation: After the training iterations are completed, evaluate the performance of the final policy π in the environment. This step helps to determine if the policy has learned the expert's behavior effectively and generalizes well to new situations.

4. Deployment: Once the learned policy's performance is deemed satisfactory, deploy the policy in the target environment. The agent will now use the learned policy to make decisions in various states, imitating the expert's behavior.

In summary, DAgger addresses the distribution mismatch problem in imitation learning by iteratively training the policy on a combination of expert demonstrations and the agent's own experiences. *This process refines the policy and helps it better imitate the expert's behavior, even in situations that were not covered by the initial expert demonstrations.*



## GAIL



# NeRF

[NeRF paper](https://www.matthewtancik.com/nerf)

https://dellaert.github.io/NeRF/



[Volume Rendering](https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/intro-volume-rendering.html)

## Volume Rendering

a 3D scalar field (such as density, temperature, or other quantities) can be represented by a combination of emission and absorption properties along a ray path through the volume.

**The light go straight , don't get stronger or weaker without emission and absorption**

![img](https://pic4.zhimg.com/80/v2-29113a762fad08067a9871d67e95735f_1440w.webp)

There are four kind of interaction between light and particles:

- 吸收 (absorption)：光子被粒子吸收，会导致入射光的辐射强度**减弱**

  **Define**: The Variable $s$ represents the distance along the ray path through the volume. 

  $\tau_a(s) = \rho(s)A$ is the *absorption coefficient*, which is determined by the projected area $A$ and density $ \rho(s)$ of the particles. $A$ equals to $\pi r^2$
  $$
  \frac{dI}{ds}=-\rho(s)AI(s)=-\tau_a(s)I(s)\\
  I(s)=I_0e^{-\int_0^s\tau_a(t)dt}
  $$


  This means that if the medium (particle swarm) is homogeneous, that is, $\tau_a(t)$equals everywhere. So after the incident light passes through the medium, the radiation intensity will exhibit exponential decay. This is known as the *Beer-Lambert Law*:

  ![img](https://pic4.zhimg.com/80/v2-2bb0df83e58e031a3d57cc3f8556b293_1440w.webp)

  We can then define the **internal transmittance**:
$$
  T(s) = \frac{I_i}{I_0}=e^{-\int\tau_a(t)dt}
$$
  this is only determined by $\tau_a$

  **the probability that the ray travels from $t_n$ to $t$ without hitting any other particles.

  

- 放射 (emission)：粒子本身可能发光，比如气体加热到一定程度就会离子化，变成发光的「火焰」。这会进一步**增大**辐射强度
  $$
  \frac{dI}{ds}=I_e(s)\rho(s)A=I_e(s)\tau_a(s)
  $$
  **Attention:** 

  This formula describes the **changes** in light from the emission perspective, only considering the newly emitted light at the 'ds point'

- 外散射 (out-scattering)：光子在撞击到粒子后，可能会发生弹射，导致方向发生偏移，会**减弱**入射光强度
  $$
  \frac{dI}{ds}=-\tau_s(s)I(s)
  $$

- 内散射 (in-scattering)：其他方向的光子在撞到粒子后，可能和当前方向上的光子重合，从而**增强**当前光路上的辐射强度
  $$
  \frac{dI}{ds}= \tau_s(s)I_s(s)
  $$
  

**体渲染方程**

Define: $\tau_t=\tau_a+\tau_s$ 
$$
I(s) = \underbrace{I_0 e^{-\int_0^s \tau_t(t) dt}}_{\text{absorption \& out-scattering}} + \overbrace{\int_0^s e^{-\int_0^t  \tau_t(u) du}[ \tau_a(t)I_e(t)+\tau_s(t)I_s(t)] dt}^{\text{emission \& in-scattering}}
$$


Assume $\sigma = \tau_t=\tau_a=\tau_s$ ----> the optical depth

Let $C = I_e+I_s$ ----> the Intensity
$$
\begin{align} I(s)=&\int_0^s e^{-\int_0^t \sigma(u)du}\sigma(t)C(t)dt+ \notag I_0\exp(-\int_0^s\sigma(t)dt)  \notag \\ 
=&\underbrace{\int_0^s T(t)\sigma(t)C(t)dt}_{\text{emmision \& in-scattering}}+\underbrace{T(s)I_0}_{\text{background\ light}} \end{align}
$$
**If we know every point's $T(\text{determined by } \sigma ),\ \sigma,\ C$, we can figure out the Intensity change for every radience.**



NeRF paper:

![img](https://pic4.zhimg.com/80/v2-9a8517a2820f203a0081dbd21d207c67_1440w.webp)

discretization:

![img](https://pic1.zhimg.com/80/v2-3f925ebd004e6a2d6f047dc2931d5898_1440w.webp)



## *Neural rendering*

*“deep image or video generation approaches that enable explicit or implicit control of scene properties such as illumination, camera parameters, pose, geometry, appearance, and semantic structure.”*

### The Prelude: Neural Implicit Surfaces

The immediate precursors to neural volume rendering are the approaches that use a neural network to define an **implicit** surface representation.

> **"implicit surface representation"** refers to a method of defining the surface of a 3D object indirectly through a mathematical function, rather than explicitly defining the surface using a set of vertices or other geometric primitives like triangles, points, or voxels.
>
> In an implicit surface representation, the surface is defined as the set of points (x, y, z) where the mathematical function F(x, y, z) takes on a specific value, usually zero. For example, if F(x, y, z) is a signed distance function (SDF), the surface is defined as the set of points where F(x, y, z) = 0, with negative values inside the surface and positive values outside.

- [Occupancy networks](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks) 

- [IM-NET](https://github.com/czq142857/implicit-decoder)

- [DeepSDF](https://github.com/facebookresearch/DeepSDF)

  directly regresses a **signed distance function** or SDF, rather than binary occupancy, from a 3D coordinate and optionally a latent code.

- [PIFu](https://shunsukesaito.github.io/PIFu/) 



#### Signed Distance Function (SDF)

The distance is positive outside the shape, negative inside, and zero exactly on the surface. In other words, an SDF gives you **the shortest distance from a given point to the surface** of a shape, with the sign indicating whether the point is inside or outside the shape

- Smoothness
- Differentiability
- Robustness: SDFs can represent complex geometries, including sharp features and thin structures,

### NeRF itself

In essence, they take the DeepSDF architecture but regress not a signed distance function, but density and color. They then use an (easily differentiable) numerical integration method to approximate a true volumetric rendering step



# unbuntu上C++在vscode中如何编译运行？

当在 Ubuntu 上使用 Visual Studio Code 编译 C++ 文件时，以下步骤将被执行：

1. 首先，VS Code 使用一个名为 `tasks.json` 的配置文件来设置和配置任务。该文件包含任务配置，如编译命令、命令参数、选项等。
2. 当你执行一个任务（例如编译 C++ 文件），VS Code 会根据 `tasks.json` 中的配置在终端中执行相应的命令。在这种情况下，它会执行 `g++` 命令来编译您的 C++ 文件。
3. 编译完成后，`g++` 会生成一个可执行文件（如 `a.out`），然后可以运行该可执行文件以运行您的 C++ 程序。

examples for `tasks.json`:

```json
{
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: g++ build active file",
            "command": "/usr/bin/g++",
            "args": [
                "-fdiagnostics-color=always",
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "options": {
                "cwd": "${fileDirname}"
            },
            "problemMatcher": [
                "$gcc"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "detail": "Task generated by Debugger."
        }
    ],
    "version": "2.0.0"
}
```



对于C++文件的所有相关配置将会被放在一个.vscode文件夹，其中含有：

- `tasks.json`：如前所述，这个文件用于配置 VS Code 中的任务，如构建、编译和运行。在您的情况下，它包含一个用于编译 C++ 文件的任务配置。
- `c_cpp_properties.json`：这个文件是用于配置 C++ 扩展的属性，以便为您的项目提供智能感知、代码导航、自动完成等功能。此文件包含了包括编译器路径、标准库路径、C++ 标准版本等在内的各种设置。

`c_cpp_properties.json` 文件的主要目的是帮助 VS Code 理解您的项目环境和配置，以便为您提供更好的编程体验。`.vscode` 文件夹是一个隐藏文件夹，用于存储针对特定项目的 VS Code 配置文件，包括 `tasks.json` 和 `c_cpp_properties.json`。



# Embedding

> 将离散的变量进行连续编码的一种方式

## A not that cool friend: *one-hot*

1. 对于具有非常多类型的类别变量，变换后的向量维数过于巨大，且过于稀疏。

2. 不同类别---> tensor的不同维度

   而不同的维度是完全独立的，并不能表示出不同类别之间的联系

```python
# One Hot Encoding Categoricals
books = ["War and Peace", "Anna Karenina", 
          "The Hitchhiker's Guide to the Galaxy"]
books_encoded = [[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
Similarity (dot product) between First and Second = 0
Similarity (dot product) between Second and Third = 0
Similarity (dot product) between First and Third = 0

# Idealized Representation of Embedding
books = ["War and Peace", "Anna Karenina", 
          "The Hitchhiker's Guide to the Galaxy"]
books_encoded_ideal = [[0.53,  0.85],
                       [0.60,  0.80],
                       [-0.78, -0.62]]
Similarity (dot product) between First and Second = 0.99
Similarity (dot product) between Second and Third = -0.94
Similarity (dot product) between First and Third = -0.97
```



## How can we proform enbedding

embedding 实际上就是vectorize, 是将离散数据（如 word） 映射到向量空间的过程











