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
  loss, log = crvae.calculate_loss(model, images, augmented_images)
  loss.backward()
  
  ... optimizer step
  
```

To use base hyperparameters, simply use

```python
  from CRVAE import CRVAE
  
  ... data loading
  
  loss, logs = CRVAE().calculate_loss(model, images, augmented_images)
  loss.backward()
  
  ... optimizer step
  
```

There are two simple VAE architechtures implemented but can be easily extended.

To use the architectures:

```python
from CRVAE.models import CNNVAE

cnn_model = CNNVAE()
```


There are few image augmenatation policies implemented which can be used as:

```python
from CRVAE.augmentations import get_augmentation

simple_augmentation = get_augmentation('simple')
large_augmentation_normalize = get_augmentation('large', normalize=True)
large_color_jitter_augmentation = get_augmentation('large_jitter', normalize=True)
large_vertical_flip = get_augmentation('large_vertical_flip', normalize=True)
...
```
