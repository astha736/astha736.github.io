---
layout: post
title: DDPM learns generation by reversing Gaussian noise
date: 2026-08-02 22:00:00+0200
description: A theory-first walkthrough of Denoising Diffusion Probabilistic Models. The post builds from the fixed forward corruption process to the learned reverse process, derives the noise-prediction objective, explains its connection to score matching and Langevin-like sampling, and shows why a timestep-conditioned U-Net can generate images from noise.
tags: ddpm diffusion generative-models u-net score-matching
categories: paper-notes
thumbnail: assets/img/blog/ddpm/ddpm_infographic.png
toc:
  beginning: true
related_posts: false
mermaid:
  enabled: true
  zoomable: false
---

**Denoising Diffusion Probabilistic Models**, or DDPMs, are latent-variable generative models that learn to reverse a fixed Gaussian corruption process {% cite ho2020denoising --file blogs_ddpm %}. The model begins with a process that destroys structure: a clean image becomes slightly noisy, then noisier, and eventually resembles Gaussian noise. It then learns a second process in the opposite direction: Gaussian noise becomes slightly more structured, then increasingly image-like, and finally a generated image.

At first, this looks like a complicated way to generate data. Why deliberately destroy images and then learn to reconstruct them? Directly mapping Gaussian noise to a complete image in one step is difficult because there is no simple rule for how each noisy pattern should become a coherent image. DDPM instead creates noisy training examples from real images, for which the added noise is known, and learns to reverse this corruption. It turns one difficult generation problem into a sequence of smaller conditional problems, each of which moves from $(x_t)$ to a slightly cleaner $(x_{t-1})$.

Ho et al. show that the reverse process can be trained by asking a neural network to predict the Gaussian noise present in a corrupted image. Their central result is that a model trained using a relatively simple noise-prediction objective can generate high-quality images. 

> For a code-first walkthrough, see my [Python notebook](https://github.com/astha736/research-os-labs/blob/main/modern_robotics_stack/ddpm_mnist.ipynb). Since this post is theory-heavy, I recommend starting with the notebook and then returning to the blog for the full derivation.

## Applying DDPM to image generation

The central idea of DDPM is to replace one difficult generation problem—from noise to an image—with many easier denoising problems. Instead of learning a direct map from pure noise to a complete image, DDPM defines a known forward process, based on Gaussian noise in the original work, that gradually corrupts data. It then learns a reverse process that undoes this corruption step by step.

The forward process is denoted by $(q)$. It starts from real data $(x_0)$ and gradually adds Gaussian noise:

$$
x_0
\rightarrow
x_1
\rightarrow
\cdots
\rightarrow
x_T.
$$

The reverse process is denoted by $(p_\theta)$. It starts from Gaussian noise and learns to move in the opposite direction:

$$
x_T
\rightarrow
x_{T-1}
\rightarrow
\cdots
\rightarrow
x_0.
$$

The forward process $(q)$ is fixed and known. It is used to create training examples at many noise levels. The reverse process $(p_\theta)$ is learned. This learned reverse process is the generative model.

## A fixed corruption process and a learned reverse process

To ground the notation, imagine that each image contains only `2 × 2` grayscale pixels:

$$
x_t
=
\begin{bmatrix}
x_t^{1} & x_t^{2} \\
x_t^{3} & x_t^{4}
\end{bmatrix}.
$$

Each $(x_t)$ is one complete image tensor, not a map of probabilities. The entries are pixel intensities, usually scaled to $[-1,1]$.

The forward transition is:

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{\alpha_t}\,x_{t-1},
\beta_t I
\right).
$$

This means that, when the previous image $(x_{t-1})$ is given, $(q)$ defines a Gaussian distribution over possible noisier images $(x_t)$. It is not a deterministic function returning one fixed image. To obtain one sample, Gaussian noise with the same shape as the image is drawn:

$$
x_t
=
\sqrt{\alpha_t}\,x_{t-1}
+
\sqrt{\beta_t}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
$$

For the `2 × 2` example, $(\epsilon_t)$ is also a `2 × 2` tensor. All four pixel values are weakened and perturbed to produce one complete noisy image $(x_t)$.

The reverse transition is:

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\sigma_t^2I
\right).
$$

Given the current noisy image $(x_t)$, this distribution describes possible slightly cleaner images $(x_{t-1})$. The neural network does not directly calculate one exact previous image. Instead, it predicts information used to define the centre of the reverse distribution, usually by predicting the noise in $(x_t)$ and converting that prediction into the reverse mean $(\mu_\theta(x_t,t))$. A possible $(x_{t-1})$ is then sampled from this distribution.

The complete reverse model is:

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t).
$$

This assigns a probability density to an entire proposed trajectory:

$$
(x_T,x_{T-1},\ldots,x_1,x_0).
$$

The final quantity $(p_\theta(x_0))$ is not a pixel-wise probability image. It is one scalar density describing how plausible the complete image $(x_0)$ is under the learned model, after accounting for possible intermediate noisy trajectories.

## Forward diffusion: gradually replacing data with noise

Let $(x_0)$ represent a clean sample from the data distribution:

$$
x_0 \sim q(x_0).
$$

For image generation, $(x_0)$ is an image tensor. The forward process introduces latent variables:

$$
x_1,x_2,\ldots,x_T,
$$

with the same dimensionality as $(x_0)$.

The complete forward process is:

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}
q(x_t\mid x_{t-1}).
$$

This is a Markov chain because each state depends only on the previous state:

$$
q(x_t\mid x_{0:t-1})
=
q(x_t\mid x_{t-1}).
$$

Each transition is Gaussian:

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{1-\beta_t}\,x_{t-1},
\beta_t I
\right).
$$

The scalar $(\beta_t)$ is the forward-process variance at timestep $(t)$. It determines how much new Gaussian noise is introduced in that transition. An equivalent sampling equation is:

$$
x_t
=
\sqrt{1-\beta_t}\,x_{t-1}
+
\sqrt{\beta_t}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
$$

### Why the signal and noise coefficients are chosen this way

If we only added noise at every step,

$$
x_t=x_{t-1}+\sqrt{\beta_t}\epsilon_t,
$$

the variance of $(x_t)$ would grow throughout the chain. DDPM instead weakens the previous sample while adding noise, so that signal is gradually replaced by noise while the overall scale stays controlled.

If both $(x_{t-1})$ and $(\epsilon_t)$ have approximately unit coordinate-wise variance, then:

$$
\mathrm{Var}(x_t)
\approx
(1-\beta_t)\mathrm{Var}(x_{t-1})
+
\beta_t\mathrm{Var}(\epsilon_t).
$$

For unit variance:

$$
\mathrm{Var}(x_t)
\approx
(1-\beta_t)+\beta_t
=
1.
$$

The forward process therefore replaces signal with noise while keeping the coordinate-wise variance approximately stable.

### The noise schedule

The sequence:

$$
\beta_1,\beta_2,\ldots,\beta_T
$$

is called the **noise schedule**.

Ho et al. use:

$$
T=1000,
$$

with a linear schedule from:

$$
\beta_1=10^{-4}
\qquad\text{to}\qquad
\beta_T=0.02.
$$

The values are small enough that each forward transition makes only a small change. At the same time, the cumulative effect of all transitions makes $(x_T)$ almost independent of $(x_0)$ and close to a standard Gaussian.

This endpoint matters because generation begins from:

$$
p(x_T)=\mathcal N(0,I).
$$

_The forward process progressively reduces the contribution of the original image and increases the contribution of Gaussian noise._

{% include figure.liquid loading="eager" path="assets/img/blog/ddpm/ddpm_infographic.png" class="img-fluid rounded z-depth-1" zoomable=true %}

### Closed-form noising: sampling $(x_t)$ directly from $(x_0)$

The forward process is designed so that a noisy sample $(x_t)$ can be generated directly from $(x_0)$ for any chosen timestep, without rolling out every intermediate image.


This is achieved by defining:
$$
\alpha_t=1-\beta_t.
$$

The cumulative signal-retention coefficient is:

$$
\bar{\alpha_t}
=
\prod_{s=1}^{t}\alpha_s.
$$

Using these coefficients, $(x_t)$ can be sampled directly as:

$$
x_t
=
\sqrt{\bar{\alpha_t}}\,x_0
+
\sqrt{1-\bar{\alpha_t}}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
$$

The schedules $(\beta_t)$, $(\alpha_t)$, and $(\bar{\alpha}_t)$ can be computed and stored in advance. Constructing $(x_t)$ then requires only sampling a noise tensor and scaling the image and noise by the coefficients for the chosen timestep.


The result above is sufficient for implementation. The derivation below shows why the cumulative noise remains Gaussian; readers interested only in the final sampling equation can skip to the next section.

---
**Derivation**

The paper defines:

$$
\alpha_t=1-\beta_t.
$$

Then the one-step transition becomes:

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{\alpha_t}\,x_{t-1},
(1-\alpha_t)I
\right),
$$

with sampling equation:

$$
x_t
=
\sqrt{\alpha_t}\,x_{t-1}
+
\sqrt{1-\alpha_t}\,\epsilon_t.
$$

To express $(x_t)$ directly in terms of the original image $(x_0)$, we repeatedly substitute the previous forward transitions.

Starting from:

$$
x_t
=
\sqrt{\alpha_t}\,x_{t-1}
+
\sqrt{1-\alpha_t}\,\epsilon_t,
$$

a one-step substitution of $(x_{t-1})$ gives:

$$
x_t
=
\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}
+
\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1}
+
\sqrt{1-\alpha_t}\,\epsilon_t.
$$

The last two terms are a weighted sum of independent Gaussian variables. A weighted sum of independent Gaussians is also Gaussian, with variance:

$$
\alpha_t(1-\alpha_{t-1})
+
(1-\alpha_t)
=
1-\alpha_t\alpha_{t-1}.
$$

Therefore, the two noise terms can be replaced by one standard Gaussian variable $(\epsilon')$:

$$
x_t
=
\sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}
+
\sqrt{1-\alpha_t\alpha_{t-1}}\,\epsilon',
\qquad
\epsilon'\sim\mathcal N(0,I).
$$

Continuing this recursive substitution back to $(x_0)$ gives:

$$
x_t
=
\sqrt{\bar{\alpha_t}}\,x_0
+
\sqrt{1-\bar{\alpha_t}}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
$$

where:

$$
\bar{\alpha_t}
=
\prod_{s=1}^{t}\alpha_s.
$$

Thus, although the forward process is defined step by step, the cumulative effect of all previous Gaussian noise terms can be represented by one equivalent Gaussian variable. This allows $(x_t)$ to be sampled directly from $(x_0)$ at any timestep $(t)$.

The closed-form forward distribution is:

$$
q(x_t\mid x_0)
=
\mathcal N
\left(
x_t;
\sqrt{\bar{\alpha_t}}\,x_0,
(1-\bar{\alpha_t})I
\right).
$$

Equivalently:

$$
x_t
=
\sqrt{\bar{\alpha_t}}\,x_0
+
\sqrt{1-\bar{\alpha_t}}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

The variable $(\epsilon)$ in this equation represents the combined effect of all individual forward noises $(\epsilon_1,\epsilon_2,\ldots,\epsilon_t)$. It is not one particular transition noise $(\epsilon_t)$.

This closed form matters because training on a randomly selected timestep $(t)$ does not require simulating the entire chain $(x_0\rightarrow x_1\rightarrow\cdots\rightarrow x_t)$. The forward model remains a Markov chain conceptually, but training can construct $(x_t)$ from $(x_0)$ in one step.

## The reverse process: learning to denoise one step at a time

The reverse process starts from a standard Gaussian prior:

$$
p(x_T)=\mathcal N(x_T;0,I).
$$

It then applies learned reverse transitions:

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t).
$$

Each reverse transition is parameterized as a Gaussian:

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

The paper simplifies the reverse covariance to:

$$
\Sigma_\theta(x_t,t)
=
\sigma_t^2I.
$$

Thus, sampling one reverse step has the form:

$$
x_{t-1}
=
\mu_\theta(x_t,t)
+
\sigma_t z,
\qquad
z \sim \mathcal N(0,I).
$$

In the implementation, the neural network predicts the noise term $(\epsilon_\theta(x_t,t))$, which is converted into the reverse mean $(\mu_\theta(x_t,t))$ through the known diffusion equations. The variable $(z)$ is different: it is fresh Gaussian noise sampled during the reverse step and added as $(\sigma_t z)$ to preserve the stochasticity of the transition.

Ho et al. evaluate two fixed choices, $(\sigma_t^2=\beta_t)$ and $ (\sigma_t^2=\tilde{\beta_t})$. Here, $(\beta_t)$ is the forward-process variance, while $(\tilde{\beta_t})$ is the posterior variance of the tractable forward posterior $(q(x_{t-1}\mid x_t,x_0))$. Both choices gave similar experimental results in the paper. The main quantity to learn to predict is therefore the reverse mean:

$$
\mu_\theta(x_t,t).
$$


## Deriving the DDPM training objective

This section works through the equations used to derive the DDPM training objective and clarifies what the algorithm optimizes.

The derivation begins with the goal of assigning high probability to real clean samples $(x_0)$ by maximizing $(p_\theta(x_0))$. This quantity is difficult to evaluate directly because it requires accounting for every possible latent trajectory through which $(x_0)$ could be generated. The authors therefore introduce the tractable forward process $(q(x_{1:T}\mid x_0))$ and rewrite the objective as:

$$
p_\theta(x_0)
=
\int
p_\theta(x_{0:T})
\,dx_{1:T}.
$$

By multiplying and dividing by $(q(x_{1:T}\mid x_0))$, they express the log-likelihood as an expectation under the known forward process:

$$
\log p_\theta(x_0)
=
\log
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
$$

This expectation makes the forward process available as a tractable distribution over latent trajectories. Jensen’s inequality then gives a lower bound on the log-likelihood. Negating this bound produces a loss to minimize. After further substitutions, the loss becomes a sum of divergences between tractable forward posteriors and learned reverse transitions. With fixed reverse variance, these terms reduce to mean matching and, under the noise parameterization, to a weighted noise-prediction error.

The full derivation is given below. Readers interested only in the final training objective can skip to **The loss objective of DDPM is further simplified**.

---

**DDPM Training Objective**

The goal of DDPM training is to make the model assign high probability to real clean data samples $(x_0)$. In principle, we would like to maximize the log-likelihood:

$$
\log p_\theta(x_0).
$$

This means that the model should believe that a real image $(x_0)$ is a likely sample under its learned generative process.

The difficulty is that DDPM does not generate $(x_0)$ in one step. It generates a whole reverse trajectory:

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0.
$$

Therefore, the likelihood of $(x_0)$ requires integrating over all possible hidden noisy trajectories $(x_{1:T})$ that could have produced it:

$$
p_\theta(x_0)
=
\int
p_\theta(x_{0:T})
\,dx_{1:T}.
$$

Here, $(x_{1:T})$ are latent variables. They are called latent because during likelihood evaluation we only observe the clean image $(x_0)$, not the full reverse path that generated it.

---
**Use the tractable forward process $(q)$**

The key idea is to introduce the known forward noising process:

$$
q(x_{1:T}\mid x_0)
$$

as a tractable distribution over latent noisy trajectories. We want to evaluate:

$$
p_\theta(x_0)
=
\int
p_\theta(x_{0:T})
\,dx_{1:T}.
$$

Now multiply and divide the integrand by the same forward-process distribution:

$$
p_\theta(x_0)
=
\int
q(x_{1:T}\mid x_0)
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\,dx_{1:T}.
$$

The change is an exact algebraic identity. We multiply and divide the integrand by the known forward distribution $(q)$.


Rearrange the terms:

$$
p_\theta(x_0)
=

\int
q(x_{1:T}\mid x_0)
\left[
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right]
dx_{1:T}.
$$

By definition, an expectation under a distribution $(q(z))$ is:

$$
\mathbb E_{q(z)}[f(z)]
=
\int q(z)f(z),dz.
$$

Using:

$$
z=x_{1:T},
$$

and:

$$
f(z)
=
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
},
$$

we obtain:

$$
\boxed{
p_\theta(x_0)
=

\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right]
}.
$$

What does the expectation mean?

We hold the clean image $(x_0)$ fixed and sample complete noisy trajectories from the forward process:

$$
x_{1:T}
\sim
q(x_{1:T}\mid x_0).
$$

For each sampled trajectory, calculate the ratio:

$$
w(x_{1:T})
=
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}.
$$

This ratio is an **importance weight**.

$(q(x_{1:T}\mid x_0))$ indicates how likely the forward process is to propose the trajectory, while $(p_\theta(x_{0:T}))$ indicates how much probability density the current reverse model assigns to the complete trajectory. Their ratio corrects for how frequently $(q)$ proposes that trajectory.

A trajectory with a high $(q)$ value is sampled frequently, so dividing by $(q)$ prevents it from being overcounted. Conversely, a trajectory with a low $(q)$ value is sampled rarely, so the ratio amplifies its contribution.

---
**Rewrite the log-likelihood of $(p(x_0))$ as a lower bound**

Taking the logarithm gives:

$$
\log p_\theta(x_0)
=
\log
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
$$

Because the logarithm is concave, Jensen’s inequality gives:

$$
\log \mathbb{E}[Z]
\geq
\mathbb{E}[\log Z].
\tag{Jensen’s inequality}
$$

This gives the evidence lower bound, or ELBO:

$$
\boxed{
\log p_\theta(x_0)
\geq
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\log
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
}
\tag{ELBO}
$$

So $(q)$ is used as a tractable proposal distribution over hidden noisy paths. It lets us replace an intractable integral over all trajectories with an expectation over noisy trajectories that we can sample.

---
**Simplifying the loss $(L)$ in terms of KL divergence**

If we multiply by $(-1)$, the inequality flips:

$$
-\log p_\theta(x_0)
\leq
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
-\log
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
$$

The loss function $L$ is therefore:
$$
L
=
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
-\log
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
$$

The learned reverse process factorizes as:

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t),
$$

while the fixed forward process factorizes as:

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}
q(x_t\mid x_{t-1}).
$$

Substituting these into the ELBO gives:

$$
L
=
\mathbb{E}_q
\left[
-\log p(x_T)
-
\sum_{t=1}^{T}
\log p_\theta(x_{t-1}\mid x_t)
+
\sum_{t=1}^{T}
\log q(x_t\mid x_{t-1})
\right].
$$

At this point, the loss is written as a sum over the full diffusion trajectory, but it is not yet in the useful DDPM form. The useful form compares the learned reverse transition $(p_\theta(x_{t-1}\mid x_t))$ with the tractable posterior of the forward process, $(q(x_{t-1}\mid x_t,x_0))$.

For $(t>1)$, Bayes’ rule gives:

$$
q(x_t\mid x_{t-1})
=
\frac{
q(x_{t-1}\mid x_t,x_0)\,q(x_t\mid x_0)
}{
q(x_{t-1}\mid x_0)
}.
$$

This follows from:

$$
q(x_{t-1}\mid x_t,x_0)
=
\frac{
q(x_t\mid x_{t-1},x_0)\,q(x_{t-1}\mid x_0)
}{
q(x_t\mid x_0)
}.
$$

Because the forward process is Markov, once $(x_{t-1})$ is known, $(x_t)$ does not need $(x_0)$ directly:

$$
q(x_t\mid x_{t-1},x_0)
=
q(x_t\mid x_{t-1}).
$$

Taking logarithms gives:

$$
\log q(x_t\mid x_{t-1})
=
\log q(x_{t-1}\mid x_t,x_0)
+
\log q(x_t\mid x_0)
-
\log q(x_{t-1}\mid x_0).
$$

Now substitute this only for $(t=2,\dots,T)$. The forward-chain term separates as:

$$
\sum_{t=1}^{T}
\log q(x_t\mid x_{t-1})
=
\log q(x_1\mid x_0)
+
\sum_{t=2}^{T}
\log q(x_t\mid x_{t-1}).
$$

After substitution:

$$
\sum_{t=2}^{T}
\log q(x_t\mid x_{t-1})
=
\sum_{t=2}^{T}
\log q(x_{t-1}\mid x_t,x_0)
+
\sum_{t=2}^{T}
\left[
\log q(x_t\mid x_0)
-
\log q(x_{t-1}\mid x_0)
\right].
$$

The second sum telescopes:

$$
\sum_{t=2}^{T}
\left[
\log q(x_t\mid x_0)
-
\log q(x_{t-1}\mid x_0)
\right]
=
\log q(x_T\mid x_0)
-
\log q(x_1\mid x_0).
$$

The separated term $(\log q(x_1\mid x_0))$ then cancels the leftover negative term. Therefore, the full forward-chain contribution becomes:

$$
\sum_{t=1}^{T}
\log q(x_t\mid x_{t-1})
=
\sum_{t=2}^{T}
\log q(x_{t-1}\mid x_t,x_0)
+
\log q(x_T\mid x_0).
$$

The forward transitions $(q(x_t\mid x_{t-1}))$ have now been rewritten as posterior reverse terms $(q(x_{t-1}\mid x_t,x_0))$ plus one endpoint term $(q(x_T\mid x_0))$.

Substituting this back into the ELBO and separating the learned reverse term at $(t=1)$ gives:

$$
L
=
\mathbb{E}_q
\left[
\log q(x_T\mid x_0)
-
\log p(x_T)
+
\sum_{t=2}^{T}
\left(
\log q(x_{t-1}\mid x_t,x_0)
-
\log p_\theta(x_{t-1}\mid x_t)
\right)
-
\log p_\theta(x_0\mid x_1)
\right].
$$

Each pair of log terms now becomes a KL divergence. The endpoint pair becomes:

$$
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)
\;\|\;
p(x_T)
\right),
$$

and each intermediate reverse-step pair becomes:

$$
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\;\|\;
p_\theta(x_{t-1}\mid x_t)
\right).
$$

So the final timestep-wise decomposition is:

$$
\boxed{
L
=
\mathbb{E}_q
\left[
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)
\;\|\;
p(x_T)
\right)
+
\sum_{t>1}
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\;\|\;
p_\theta(x_{t-1}\mid x_t)
\right)
-
\log p_\theta(x_0\mid x_1)
\right].
}
$$

The cancellation happens only among the marginal forward terms $(\log q(x_t\mid x_0))$. The reverse posterior terms $(\log q(x_{t-1}\mid x_t,x_0))$ do not cancel; they remain and become the KL terms that train the learned reverse process.

The first term checks whether the final noised-image distribution is close to the prior. Because the forward noise schedule is fixed, this term is unaffected by neural-network training. The middle term is the main learned denoising term: at each timestep, the learned reverse step should match the tractable posterior of the forward process. The last term is the final reconstruction, or decoder, term from $(x_1)$ to $(x_0)$.

---
**Loss function is reduced to reverse-mean matching**

Consider one of the middle terms in the variational bound:

$$
L_{t-1}
=
\mathbb{E}_q
\left[
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\;\|\;
p_\theta(x_{t-1}\mid x_t)
\right)
\right].
$$

During training, the clean image $(x_0)$ is known. This makes the posterior of the forward process tractable:

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde{\mu}_t(x_t,x_0),
\tilde{\beta_t} I
\right).
$$

Its mean and variance are:

$$
\tilde{\mu}_t(x_t,x_0)
=
\frac{
\sqrt{\bar{\alpha}_{t-1}}\beta_t
}{
1-\bar{\alpha_t}
}
x_0
+
\frac{
\sqrt{\alpha_t}
\left(
1-\bar{\alpha}_{t-1}
\right)
}{
1-\bar{\alpha_t}
}
x_t,
$$

and:

$$
\tilde{\beta_t}
=
\frac{
1-\bar{\alpha}_{t-1}
}{
1-\bar{\alpha_t}
}
\beta_t.
$$

The learned reverse transition is also chosen to be Gaussian:

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t),
\sigma_t^2 I
\right).
$$

For two $D$-dimensional Gaussian distributions,

$$
q=\mathcal{N}(m_q,\Sigma_q),
\qquad
p=\mathcal{N}(m_p,\Sigma_p),
$$

their KL divergence is:

$$
D_{\mathrm{KL}}(q\|p)
=
\frac{1}{2}
\left[
\operatorname{tr}
\left(
\Sigma_p^{-1}\Sigma_q
\right)
+
(m_p-m_q)^\top
\Sigma_p^{-1}
(m_p-m_q)
-
D
+
\log
\frac{
\det\Sigma_p
}{
\det\Sigma_q
}
\right].
$$

Substituting:

$$
\Sigma_q=\tilde{\beta_t} I,
\qquad
\Sigma_p=\sigma_t^2 I,
$$

gives:

$$
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\;\|\;
p_\theta(x_{t-1}\mid x_t)
\right)
=
\frac{1}{2\sigma_t^2}
\left\|
\tilde{\mu}_t(x_t,x_0)
-
\mu_\theta(x_t,t)
\right\|^2
+
C_t,
$$

where:

$$
C_t
=
\frac{D}{2}
\left[
\frac{\tilde{\beta_t}}{\sigma_t^2}
-
1
+
\log
\frac{\sigma_t^2}{\tilde{\beta_t}}
\right].
$$

If $(\sigma_t^2)$ is fixed, then $(C_t)$ does not depend on the neural-network parameters $(\theta)$. Therefore, it does not affect the gradient used to train the reverse mean.

The trainable part of the reverse-process loss is consequently:

$$
\boxed{
L_{t-1}
=
\mathbb{E}_q
\left[
\frac{1}{2\sigma_t^2}
\left\|
\tilde{\mu}_t(x_t,x_0)
-
\mu_\theta(x_t,t)
\right\|^2
\right]
+
C_t
}
$$

Thus, once both reverse distributions are chosen to be Gaussian and the reverse variance is fixed, minimizing the KL divergence reduces to matching their means.

$(C_t)$ contains terms that do not depend on the neural-network parameters $(\theta)$. Therefore, the model only needs to learn the reverse mean $(\mu_\theta(x_t,t))$.

This is the **conceptual bridge from variational inference to denoising**: a likelihood-maximization problem becomes a tractable mean-matching objective.

### Why predicting noise is enough

One option would be to train the neural network to predict the reverse mean directly. Ho et al. instead parameterize the reverse mean through noise prediction. This works because there is a clear relationship between $(x_t)$, $(x_0)$, and the sampled noise $(\epsilon)$:

$$
x_t
=
\sqrt{\bar{\alpha_t}}x_0
+
\sqrt{1-\bar{\alpha_t}}\epsilon.
$$

Solving this equation for $(x_0)$ gives:

$$
x_0
=
\frac{
x_t-\sqrt{1-\bar{\alpha_t}}\epsilon
}{
\sqrt{\bar{\alpha_t}}
}.
$$

Therefore, if the neural network can predict the noise,

$$
\epsilon_\theta(x_t,t)\approx \epsilon,
$$

then it can implicitly estimate the clean image because $(\bar{\alpha}_t)$ is known:

$$
\hat{x}_0
=
\frac{
x_t-\sqrt{1-\bar{\alpha_t}}\epsilon_\theta(x_t,t)
}{
\sqrt{\bar{\alpha_t}}
}.
$$

The same predicted noise can also be converted into the reverse mean:

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{
\beta_t
}{
\sqrt{1-\bar{\alpha_t}}
}
\epsilon_\theta(x_t,t)
\right).
$$

Thus, the network predicts the effective noise pattern in $(x_t)$, and the known diffusion equations convert that prediction into the reverse Gaussian mean. The model is not learning the whole sampling equation from scratch; it learns the unknown denoising direction contained in $(\epsilon_\theta(x_t,t))$.

### The loss term is evaluated as weighted noise-prediction error

Substituting the noise-prediction parameterization into the reverse mean-matching loss derived above gives a weighted noise-prediction error:

$$
L_{t-1}
=
\mathbb{E}_{x_0,\epsilon}
\left[
\frac{
\beta_t^2
}{
2\sigma_t^2\alpha_t(1-\bar{\alpha_t})
}
\left\|
\epsilon
-
\epsilon_\theta(x_t,t)
\right\|^2
\right]
+
C.
$$

where:

$$
x_t
=
\sqrt{\bar{\alpha_t}}x_0
+
\sqrt{1-\bar{\alpha_t}}\epsilon.
$$

Define the timestep-dependent weight:

$$
\lambda_t
=
\frac{
\beta_t^2
}{
2\sigma_t^2\alpha_t(1-\bar{\alpha_t})
}.
$$

Then the loss can be implemented as:

$$
\boxed{
L_{t-1}
=
\mathbb{E}
\left[
\lambda_t
\left\|
\epsilon
-
\epsilon_\theta(x_t,t)
\right\|^2
\right]
+
C
}
$$

With the noise parameterization, mean matching becomes a weighted MSE between the true noise $(\epsilon)$ and the predicted noise $(\epsilon_\theta(x_t,t))$.


### The loss objective of DDPM is further simplified

Ho et al. simplify the objective by removing the timestep-dependent weight $(\lambda_t)$. The simplified objective is:

$$
\mathcal{L}_{\mathrm{simple}}(\theta)
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon
-
\epsilon_\theta
\left(
\sqrt{\bar{\alpha_t}}x_0
+
\sqrt{1-\bar{\alpha_t}}\epsilon,
t
\right)
\right\|^2
\right].
$$

Here, $(t)$ is sampled uniformly from the diffusion timesteps, $(x_0)$ is sampled from the data distribution, and $(\epsilon)$ is sampled from a standard Gaussian during the forward process calculation. 

In practice, the objective is evaluated by taking a clean image, sampling a timestep, adding known Gaussian noise, and training the neural network to predict the exact noise that was added.

The simplified objective is not exactly equal to the original variational bound. It is an **unweighted version** of the denoising terms that arise from the variational derivation. Ho et al. report that this simplified objective improves sample quality, while the full variational objective gives better likelihood.

## Sampling after training

Training and sampling use the model differently. During training, we start from a real $(x_0)$, sample one timestep, construct $(x_t)$ directly, and predict the known effective noise. During sampling, we start from a random $(x_T)$, run through every reverse timestep, predict noise at each state, and sample a new $(x_{t-1})$. Training does not need to execute the full reverse chain for every update, but sampling does. This is one reason DDPM training is straightforward while generation can be comparatively slow.


After training, generation does not require a real input image. The initial sample is pure Gaussian noise:

$$
x_T\sim\mathcal N(0,I).
$$

Then, for $(t=T,T-1,\ldots,1)$, compute:

$$
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{
\beta_t
}{
\sqrt{1-\bar{\alpha_t}}
}
\epsilon_\theta(x_t,t)
\right)
+
\sigma_tz.
$$

The result is a full reverse chain:

$$
x_T
\rightarrow
x_{T-1}
\rightarrow
\cdots
\rightarrow
x_1
\rightarrow
x_0.
$$

{% include figure.liquid loading="eager" path="assets/img/blog/ddpm/ddpm_reverse_sampling.png" class="img-fluid rounded z-depth-1" zoomable=true %}

The reverse chain begins with Gaussian noise. Broad image structure appears first, while finer details emerge during later, lower-noise steps.

## Score learning and Langevin dynamics

The training objective tells us that the network learns to predict the noise inside a corrupted sample:

$$
\epsilon_\theta(x_t,t)\approx \epsilon.
$$

This is the most direct implementation view. Given a noisy image $(x_t)$ and timestep $(t)$, the network estimates the noise pattern that should be removed.

There is also a deeper interpretation. For a probability density $(p(x))$, the **score** is:

$$
\nabla_x \log p(x).
$$

*The score points in the local direction where the log probability increases fastest*. In other words, it tells us how to move a sample toward a region that the model considers more likely.

For the forward noising distribution,

$$
q(x_t\mid x_0)
=
\mathcal N
\left(
x_t;
\sqrt{\bar{\alpha_t}}x_0,
(1-\bar{\alpha_t})I
\right),
$$

the conditional score is:

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-
\frac{
x_t-\sqrt{\bar{\alpha_t}}x_0
}{
1-\bar{\alpha_t}
}.
$$

Using the noising equation,

$$
x_t-\sqrt{\bar{\alpha_t}}x_0
=
\sqrt{1-\bar{\alpha_t}}\epsilon,
$$

we get:

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-
\frac{
\epsilon
}{
\sqrt{1-\bar{\alpha_t}}
}.
$$

So the noise $(\epsilon)$ and the score contain the same directional information, up to a known scale and sign. If the network predicts noise, then it can also be interpreted as predicting a score:

$$
s_\theta(x_t,t)
\approx
-
\frac{
\epsilon_\theta(x_t,t)
}{
\sqrt{1-\bar{\alpha_t}}
}.
$$

This means the denoising network is not only estimating which noise was added. It is also learning a vector field over noisy samples that points them back toward higher-density regions of the data distribution.

The reverse update can be written using the score-like quantity:

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t+\beta_t s_\theta(x_t,t)
\right).
$$

Then sampling becomes:

$$
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t+\beta_t s_\theta(x_t,t)
\right)
+
\sigma_t z.
$$

This reverse update resembles Langevin dynamics because it combines two effects. The predicted score provides a local direction in image space toward configurations that are more probable under the noisy-data distribution at timestep $(t)$. The additional Gaussian term preserves the stochastic nature of the reverse transition, allowing several plausible cleaner images to emerge from the same noisy state rather than forcing one deterministic reconstruction.

However, DDPM sampling is more accurately described as **Langevin-like**. A generic Langevin sampler uses a chosen step size to move through one fixed probability distribution. DDPM instead moves through a sequence of distributions with decreasing noise levels, and the coefficients of each update are derived from the predefined forward diffusion schedule $(\alpha_t,\beta_t,\bar{\alpha_t})$. Thus, it has the same basic pattern—score-directed movement plus stochastic noise—but its update rule comes specifically from reversing the discrete diffusion process.

The practical importance is that the same learned function has three interpretations. As a noise predictor, it estimates the corruption inside $(x_t)$. As a reverse-process parameterization, it defines the mean of $(p_\theta(x_{t-1}\mid x_t))$. As a score field, it gives a local direction toward more realistic data.

## Why a timestep-conditioned U-Net works

The same network parameters are shared across all reverse steps, but the required denoising behavior changes with the noise level. At small $(t)$, $(x_t)$ is close to a real image and the network should make fine corrections. At large $(t)$, $(x_t)$ is close to Gaussian noise and the network must infer broader global structure. The network therefore receives both the current noisy sample $(x_t)$ and the timestep $(t)$ that identifies the current noise level.

Ho et al. encode $(t)$ using a sinusoidal embedding based on the Transformer positional encoding {% cite vaswani2017attention --file blogs_attention %}.

The model predicts:

$$
\epsilon_\theta(x_t,t),
$$

with the same spatial shape as $(x_t)$. Thus, the task is a dense spatial prediction problem. The network must decide what noise estimate to produce at every location and channel.

Denoising requires local information, such as edges, textures, and small structures, as well as global information, such as object shape, pose, and consistency between distant regions. A U-Net is well suited to this combination because it combines high-resolution skip connections with lower-resolution global features.

For a more detailed architectural explanation, see [Understanding U-Net as a “what → where” machine]({% post_url 2026-06-27-understanding-unet %}).

Ho et al. use GroupNorm throughout the U-Net. GroupNorm treats every batch element separately. It divides the channels of each sample into groups and computes normalization statistics over the channels and spatial locations inside each group. Unlike BatchNorm, it does not rely on batch-wide statistics, which is useful when batch sizes are small or variable and avoids mixing normalization statistics between images corrupted at different timesteps.

The DDPM architecture also inserts self-attention blocks at the `16 × 16` feature-map resolution. Convolution naturally models local neighborhoods, while attention lets one spatial location directly combine information from distant locations. This can help the denoiser maintain consistency across different parts of an object, repeated textures, global pose, symmetry, and foreground-background structure.

For the attention mechanism itself, see [Attention is an information-routing operator]({% post_url 2026-06-29-understanding-attention %}).

The model contains many equations, but not every quantity is learned. The forward transition $q(x_t\mid x_{t-1})$, the noise schedule $\beta_t$, $\alpha_t$, $(\bar{\alpha_t})$, the prior $(p(x_T)=\mathcal N(0,I))$, the form of the reverse Gaussian, and the equations converting $(\epsilon_\theta)$ into $(\mu_\theta)$ are fixed. The learned part is the neural network parameterized by $(\theta)$, which predicts $(\epsilon_\theta(x_t,t))$.

## Bridge to Diffusion Policy: from images to action sequences

The same idea can be extended from image generation to action generation. In image DDPMs, the learned score-like denoising field moves noisy images toward realistic images. In Diffusion Policy, the generated object is a future robot action sequence, and denoising is conditioned on observations.

The conceptual bridge is:

$$
\underbrace{x_0}_{\text{image}}
\rightarrow
\underbrace{A_t^0}_{\text{future action sequence}}
$$

and:

$$
\underbrace{\epsilon_\theta(x_t,t)}_{\text{image noise predictor}}
\rightarrow
\underbrace{\epsilon_\theta(O_t,A_t^k,k)}_{\text{action-sequence noise predictor}}.
$$

Here, $(t)$ is robot/control time, while $(k)$ is the diffusion denoising index. In DDPM, the model denoises images. In Diffusion Policy, the model denoises action sequences conditioned on observations.

For example, a robot may move around an obstacle on the left or on the right. Direct regression can average these incompatible strategies. A diffusion model can instead represent a distribution over coherent trajectories and sample one mode.

## Bringing the complete model together

DDPM begins with a fixed forward Markov chain that gradually replaces data with Gaussian noise. The noise schedule determines how much signal remains at each step, while the cumulative coefficient $(\bar{\alpha}_t)$ makes it possible to sample any corrupted state $(x_t)$ directly from $(x_0)$. During training, each reverse transition is learned from a randomly selected noise level by predicting the effective noise present in $(x_t)$.

The reverse model uses a timestep-conditioned neural network that receives both $(x_t)$ and an encoding of $(t)$. The simplified objective trains this network to predict the sampled noise directly. Through the known diffusion equations, that prediction defines the reverse mean and also corresponds to a score-like direction toward higher-density image configurations.

During generation, the model starts from $(x_T\sim\mathcal N(0,I))$ and repeatedly applies these learned reverse transitions until it produces an image-like $(x_0)$.

## Retrieval summary

DDPM learns generation by reversing a known Gaussian corruption process. The forward process $(q)$ maps clean data to Gaussian noise through a fixed Markov chain, while the reverse process $(p_\theta)$ maps Gaussian noise back to data through a learned Markov chain.

The forward transition is:

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
(1-\alpha_t)I
\right).
$$

The closed-form corruption equation is:

$$
x_t
=
\sqrt{\bar{\alpha_t}}x_0
+
\sqrt{1-\bar{\alpha_t}}\epsilon.
$$

The reverse transition is:

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\sigma_t^2I
\right).
$$

The noise-prediction reverse mean is:

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar{\alpha_t}}}
\epsilon_\theta(x_t,t)
\right).
$$

The exact reverse-transition objective gives a weighted noise-prediction error:

$$
\lambda_t
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2.
$$

The simplified objective removes this timestep-dependent weight:

$$
\mathcal L_{\mathrm{simple}}
=
\mathbb E
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
\right].
$$

Noise prediction is also connected to score learning:

$$
s_\theta(x_t,t)
=
-
\frac{
\epsilon_\theta(x_t,t)
}{
\sqrt{1-\bar{\alpha_t}}
}
\approx
\nabla_{x_t}\log q_t(x_t).
$$

Sampling starts from $(x_T\sim\mathcal N(0,I))$ and repeatedly predicts noise, computes the reverse mean, adds reverse-process noise, and obtains $(x_0)$. The U-Net is used because denoising is a dense spatial prediction problem that benefits from both local detail and global context. The timestep embedding tells the network which noise level it is denoising. GroupNorm normalizes within each sample, and self-attention supports long-range spatial consistency.

The key idea is that DDPM does not learn one direct jump from noise to data. It learns how to reverse a known corruption process, one small probabilistic transition at a time.

---

<h2>References</h2>

---

{% bibliography --file blogs_ddpm --cited_in_order --template bib_blog --group_by none %}
