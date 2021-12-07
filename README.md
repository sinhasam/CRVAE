# CR-VAE
Official code release for Consistency Regularization for VAEs, NeurIPS 2021.

[Samarth Sinha](https://www.samsinha.me/), [Adji B. Dieng](https://adjidieng.github.io/)

[Arxiv](https://arxiv.org/abs/2105.14859),  [Proceedings](https://papers.nips.cc/paper/2021/hash/6c19e0a6da12dc02239312f151072ddd-Abstract.html)

# Installation

```
  git clone https://github.com/sinhasam/CRVAE.git
  cd CRVAE
  pip3 install -e .
```
  
# Usage

Basic usage of the CR-VAE API, that can be added to your favorite VAE variant and training:

```python
  from CRVAE import CRVAE
  
  ... data loading
  
  crvae = CRVAE(gamma=self.gamma, beta_1=self.beta_1, beta_2=self.beta_2)
  loss, log = crvae.calculate_loss(images, augmented_images)
  loss.backward()
  
  ... optimizer step
  
```

# TODO

Make some basic augmentation policies built into CRVAE, as well as some VAE architectures.
