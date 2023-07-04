# L1

  任务：机器人抓取物体

supervised learning: 需要一个数据集，成功的(x,y,z) 抓取位置 ---> 难

RL：不需要正确抓取位置，but good and bad experence

## what is RL

- Mathematical formalism for learing-based decision making
- Approach for learning decision making and control from experience



## how is this different from other machine learning topics

- standard(supercvised) machine learning ---> **rely on truth**

  Given $D ={(x_i,y_i)}$, learning to predict $y$ from $x$

  - i.i.d. data ($x,y$ pairs don't influent each other)

  - known ground truth outputs in training

  - **监督学习可以从分布角度理解**

    > 在监督学习的过程中，我们关注的是条件概率分布P(Y|X)，即给定输入特征X时，输出标签Y的概率分布。我们希望找到一个模型f(X)，它可以近似地表示这个条件概率分布。通过优化损失函数，我们使得模型学到的条件概率分布尽量接近真实的分布P(Y|X)。

    >在监督学习中，从联合分布的角度来理解，我们可以认为输入特征（X）和输出标签（Y）都是随机变量，它们遵循某种未知的联合概率分布P(X, Y)。我们的训练数据是从这个未知分布中独立同分布地抽取出来的。我们的目标是根据这些训练数据学习到这个未知分布，从而能够对新的输入数据进行预测。

    > 可是，我们无论是训练模型还是使用模型，都往往是从条件概率分布P(Y|X)的角度进行，那么从联合概率分布P(X, Y)的角度理解还有什么意义呢？：
    >
    > 1. 训练数据来源：训练数据集是从某个未知的联合分布P(X, Y)中独立同分布地抽取的。因此，了解联合分布有助于我们理解训练数据的来源和潜在结构。这对于选择合适的模型和特征工程具有指导意义。
    > 2. 数据分布的变化：在现实问题中，我们可能会遇到训练数据和测试数据分布不一致的情况。从联合分布的角度来看，这意味着训练数据和测试数据来自不同的联合分布。理解这种差异有助于我们应对概念漂移（concept drift）和协变量偏移（covariate shift）等问题。
    > 3. 模型评估：虽然我们关注条件概率分布P(Y|X)，但在评估模型性能时，我们还需要考虑输入特征X的分布。通过联合分布P(X, Y)，我们可以计算期望风险（expected risk）或者泛化误差（generalization error），从而更全面地评估模型的性能。
    > 4. 概率图模型：在一些复杂的监督学习问题中，我们可能需要利用概率图模型（如贝叶斯网络和马尔可夫随机场）来表示变量之间的依赖关系。在这些情况下，理解联合概率分布P(X, Y)对于构建和推理概率图模型是至关重要的。

    

- RL ---> **rely on feedback**

  - Data is not i.i.d. (previous outputs influence futuren inputs)
  - not know GT
    - succeed / fail
    - the reward



## Why we need Deep RL

- Intelligence Robot needs ot adapt

- Deep Learning helps us handle unstructured environments

- RL provides a formalism for behavior

  ![Screenshot from 2023-05-04 21-44-51](/home/wangqx/Pictures/Screenshot from 2023-05-04 21-44-51.png)



# L2 Imitation Learning

 the future is conditionally independent of the past given the  present ---> **markov property**

![Screenshot from 2023-05-04 23-07-38](/home/wangqx/Pictures/Screenshot from 2023-05-04 23-07-38.png)

## Behavioral Cloning

just train $\pi_\theta(a_t|o_t)$as a supervised learning

![Screenshot from 2023-05-04 23-07-46](/home/wangqx/Pictures/Screenshot from 2023-05-04 23-07-46.png)

每次学习的时候，**是在学在$o_t$下$a_t$的分布。**

supervised学习时，是希望将$\pi_\theta(a_t|o_t)$与$\pi_{data}(a_t,o_t)$尽可能靠近

### **Why BC always go wrong?**

#### shifting

![Screenshot from 2023-05-04 23-34-09](/home/wangqx/Pictures/Screenshot from 2023-05-04 23-34-09.png)

**math version:**

Given the data, we have the distribution of $o$: $p_{data}(o_t)$. From the *supervised learning theory*, if we want the $\pi_\theta(a_t，o_t)$ works, we should make the $p_{\pi_\theta}(o_t)$ be the same as the $p_{data}(o_t)$.
$$
\text{Do well in }s\in D_{train}\Longleftrightarrow\text{Do well in j  }s\sim p_{train}(s)
$$


> 原因：
>
> 1. 一致性假设：supervised learning 中，我们假设训练数据和测试数据来自同一个数据生成过程。这意味着它们的分布应该是一致的。如果它们的分布不同，那么在训练数据上学到的规律可能不适用于测试数据，导致泛化能力下降。**也就是说，数据之间在学习过程中是互相影响的**
> 2. 泛化能力：当训练数据和测试数据的分布相同时，我们可以期望在训练数据上学到的知识能够很好地泛化到测试数据。否则训练表现不佳可能是没见过类似样本

However, the $p_{\pi_\theta}(o_t)$ is affected by our policy $\pi_\theta$.

**solution**

- change the $p_{\pi_\theta}(o_t)$: make no mistakes --> impossible!

- change the $p_{data}(o_t)$:

  **DAgger**(Dataset Aggregation)

  goal: collect training data from $p_{\pi_\theta}(o_t)$ instead of $p_{data}(o_t)$.

  ![Screenshot from 2023-05-04 23-34-23](/home/wangqx/Pictures/Screenshot from 2023-05-04 23-34-23.png)

#### Non-Markovian behavior

*experts are not Markovian expers*

Driving: people is not consistent, even there is an optimal Marcov Decision actully, we cannot find it because people show inconsistent

**solution**

use RNN with LSTM as the backbone

#### Causal confusion

#### Multimodal behavior

fly left to avoid the tree vs. fly right to avoid the tree  

**solution**

do not use *single Gaussian output distribution*, output mixture of Gaussians

etc.

## what's the problem of Imitation Learning

- human's data is finite
- human sometims not expert
- how can learn autonomously

![Screenshot from 2023-05-05 01-35-14](/home/wangqx/Pictures/Screenshot from 2023-05-05 01-35-14.png)
$$
\min_{\theta}E_{s_1:T,a_1:T}\left[\sum_t c(s_t,a_t)\right]
$$

$$
reward(r)= -cost(c)
$$

#### Some other analysis for shifting in Imitation learning

<img src="/home/wangqx/Pictures/Screenshot from 2023-05-05 11-08-51.png" width="300px">

<img src="/home/wangqx/Pictures/Screenshot from 2023-05-05 11-09-01.png">

## other way to do Imitation Learning

**Goal conditioned behavior Cloning**



**go beyond just Imitation**

![Screenshot from 2023-05-05 11-29-03](/home/wangqx/Pictures/Screenshot from 2023-05-05 11-29-03.png)



# L4 intro 2 RL

## Markov chain

![Screenshot from 2023-05-05 13-29-15](/home/wangqx/Pictures/Screenshot from 2023-05-05 13-29-15.png)

there is no place for control



## Markov decision process

![Screenshot from 2023-05-05 13-37-40](/home/wangqx/Pictures/Screenshot from 2023-05-05 13-37-40.png)

![Screenshot from 2023-05-05 13-38-13](/home/wangqx/Pictures/Screenshot from 2023-05-05 13-38-13.png)

**and we can add the $\mathcal{O}$ into the process**

![Screenshot from 2023-05-05 13-38-33](/home/wangqx/Pictures/Screenshot from 2023-05-05 13-38-33.png)

## the goal of RL

$$
\underbrace{p(s_1,a_1,\cdots, s_T,a_T)}_{p(\tau)} = p(s_1)\prod_{t=1}^{T}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)\\
\\\textbf{the\ goal \ of\ reinforcement\ learning:} 
\\\arg\max_\theta E_{\tau\sim p_\theta(\tau)}\left[\sum_{t}r(s_t,a_t)\right]
\\\theta^* =\begin{cases}
 \arg\max_\theta\sum_{t=1}^{T} E_{(s_t,a_t)\sim p_\theta(s_t,a_t)}\left[r(s_t,a_t)\right] & \text{finite horizon case}\\
 \arg\max_\theta E_{(s,a)\sim p_\theta(s,a)}\left[r(s,a)\right] & \text{infinite horizon case}
\end{cases}
$$

interesting:

- expected values can be continuous in the parameters of the corresponding distribution, even when the function that we are takiing the expectation of is itself highly discontinuous.

  that's why RL algorithm can use smooth optimization methods like gradient decent to optimize objectives that are seemingly non-differentiable like binary rewards for winning or losing a game.

  **In RL, we almost always care about *expectations***

  **policy的本质是一种在reward空间上的分布** 我们希望尽可能的分布在reward较高的地方

  ![Screenshot from 2023-05-05 14-37-53](/home/wangqx/Pictures/Screenshot from 2023-05-05 14-37-53.png)

## The anatomy of a reinforcement learning algorithm

**All RL algorithm composed with this three modules**

![Screenshot from 2023-05-06 13-12-56](/home/wangqx/Pictures/Screenshot from 2023-05-06 13-12-56.png)

## Value Functions

为什么要引入 *Q-funtion* 和 *Value-funtion*

为了处理*RL Objective*中复杂并且嵌套存在的*Expectation*, 把其中一部分作为一个整体定义

![Screenshot from 2023-05-06 13-44-53](/home/wangqx/Pictures/Screenshot from 2023-05-06 13-44-53.png)

***定义***

![Screenshot from 2023-05-06 13-44-59](/home/wangqx/Pictures/Screenshot from 2023-05-06 13-44-59.png)

![Screenshot from 2023-05-06 13-54-24](/home/wangqx/Pictures/Screenshot from 2023-05-06 13-54-24.png)





## Tradeoffs Between Algorithms



# L5: Policy Decent

Given:
$$
J(\theta)= E_{\tau\sim p_\theta(\tau)}\left[\sum_t r(s_t,a_t)\right]\\
\theta^\star = \arg \max_\theta J(\theta)
$$

## Estimate $j(\theta)$ without knowing $p(s_1),\ p(s_{t+1}|s_t)$ 

$$
J(\theta)= E_{\tau\sim p_\theta(\tau)}\left[\sum_t r(s_t,a_t)\right]\approx\frac{1}{N}\sum_{i(i^{th} simple)}\sum_{t(t^{th} step)}r(s_{i,t}|a_{i,t})
$$

we use sample to estimate the value

***Expections* can be evaluated using *samples***

## Update $j(\theta)$ without knowing $p(s_1),\ p(s_{t+1}|s_t)$ 

   ![Screenshot from 2023-05-05 23-11-33](/home/wangqx/Pictures/Screenshot from 2023-05-05 23-11-33.png)

![Screenshot from 2023-05-05 23-11-53](/home/wangqx/Pictures/Screenshot from 2023-05-05 23-11-53.png)





SO, **the final protocol for the policy gradient**:

![Screenshot from 2023-05-05 23-12-01](/home/wangqx/Pictures/Screenshot from 2023-05-05 23-12-01.png)





## Understanding Policy Gradients

maximum likelihood（supervised learning) 在不停提升$(a_{i,t},s_{i,t})$的联合分布概率。

监督学习可以看成是所有data都是”好“，”reward都很高"。这是因为监督学习所有数据都存在标签，这其实隐含了这样的（data, label) pair 一定是正确的。否则就相当于数据错了。

**在Imitation learning中，(data, label)pair ---> (state, action)pair. 所以 pair一定正确 ---> 专家的经验是正确的**

而RL中则不是，很多数据压根是错误的。所以不可以直接优化所有数据存在的联合分布。所以要加入reward函数。这样才能提升reward高的(state, action)pair的联合概率分布（和条件概率分布）**a weighted version of the gradient for the maximum likelihood objective**

![Screenshot from 2023-05-05 23-28-24](/home/wangqx/Pictures/Screenshot from 2023-05-05 23-28-24.png)

 ## Reducing Variance

**variance**: in finite case, the length of a trags are various, so the rewards' values will vary a lot.







- homework 1
- cs285 L3 L5
- 算分
- 数值分析



- 简量













