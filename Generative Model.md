# Diffusion

1. likelihood-based models

   ,which directly learn the distribution’s probability density (or mass) function via (approximate) maximum likelihood. Typical likelihood-based models include autoregressive models, normalizing flow models, energy-based models (EBMs), and variational auto-encoders (VAEs)

2. implicit generative models

   where the probability distribution is implicitly represented by a model of its sampling process. The most prominent example is generative adversarial networks (GANs)

   , where new samples from the data distribution are synthesized by transforming a random Gaussian vector with a neural network.





## Questions

why called energy-based model

why we always use the log likelihood 







### *Problems of Likelihood-based method*

ordinary likelihood-based models model the p.d.f or the p.m.f

we can define a p.d.f. via 
$$
p_\theta(x)=\frac{e^{-f_\theta(x)}}{{Z_\theta}}
$$
we want to trian
$$
\max_\theta \sum_{i=1}^{N}\log p_{\theta}(x_i)
$$
but $Z_\theta$ is ***not tractable***



> Other models' solutions:
>
> Thus to make maximum likelihood training feasible, likelihood-based models must either restrict their model architectures (e.g., causal convolutions in autoregressive models, invertible networks in normalizing flow models) to make $Z_\theta$ tractable, or approximate the normalizing constant (e.g., variational inference in VAEs, or MCMC sampling used in contrastive divergence which may be computationally expensive.

### ***score function solution:***

By modeling the score function instead of the density function, we can sidestep the difficulty of intractable normalizing constants. The **score function** of a distribution $p(x)$ is defined as:
$$
\nabla_x \log p(x)
$$
a model for the score function is called a **score-based model** , which we denote as

$s_\theta(x)$ 
$$
s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_xf_\theta(x)-\underbrace{\nabla_x\log Z_\theta}_{=0} = -\nabla_xf_\theta(x)
$$
***Note that the score-based model $s_\theta(x)$ is independent of the normalizing constant $Z_\theta$!***

we minimizing the ***Fisher Divergence***:
$$
\mathbb{E}_{p(x)}\left[||\nabla_x\log p(x)-s_\theta(x)||_2^2\right]
$$
i.e.
$$
J(\theta)=\frac{1}{2}\int_{x\in\R^n}p(x)||s(x;\theta)-\nabla_x\log p(x)||^2dx\\
\theta ^* = \arg\min_\theta J(\theta)
$$
这是一个典型的非参估计问题

#### *score matching*

$$
J(\theta)=\int_{x\in \R^n}p(x)\sum_{i=1}^n[\part_is_i(x;\theta)+\frac12s_i(s;\theta)^2]dx + \text{constant}
$$

> Fortunately, there exists a family of methods called **score matching** that minimize the Fisher divergence without knowledge of the ground-truth data score. Score matching objectives can directly be estimated on a dataset and optimized with stochastic gradient descent, analogous to the log-likelihood objective for training likelihood-based models (with known normalizing constants). We can train the score-based model by minimizing a score matching objective, **without requiring adversarial optimization**.

#### *Langevin dynamics*

use an iterative procedure called [**Langevin dynamics**](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) to draw samples from it.







# cv作业

