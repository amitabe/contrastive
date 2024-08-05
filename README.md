# Contrastive Loss C++ PyTorch Extension

A simple PyTorch extension for computing the Soft-Nearest Neighbors loss function ([Salakhutdinov & Hinton 2007](https://proceedings.mlr.press/v2/salakhutdinov07a.html), [Frosst et al. 2019](https://arxiv.org/abs/1902.01889)). The loss function is used in contrastive learning, where the goal is to learn representations that are invariant to certain transformations. The loss function is defined as follows or a single vector $z_i$ in a batch of vectors:

```math
{\mathcal{L}}_{\textit{SNN}}(z_i) = -\log\left(\frac{\sum_{p \in \mathbb{P}(i)} \exp(\text{sim}(z_i, z_p)/\tau)}{\sum_{n \in \mathbb{N}(i)} \exp(\text{sim}(z_i, z_n)/\tau)}\right)
```

Where $\mathbb{P}(i)$ is the set of positive examples for $z_i$, $\mathbb{N}(i)$ is the set of negative examples for $z_i$, $\text{sim}$ is a similarity function, and $\tau$ is the temperature.
Then the final loss is the expectation of the above loss function over all the vectors in the batch:
```math
 \mathcal{L}_{\textit{SNN}} = \mathop{\mathbb{E}}_{z} \left[ \mathcal{L}_{\textit{SNN}} (z) \right]
```

## Installation

contrastive needs to be installed first before use. The code requires `python>=3.10`, as well as `torch`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install  PyTorch dependencies. You can install contrastive using:

```bash
git clone https://github.com/amitabe/contrastive.git
cd contrastive && pip install .
```

or simply by
```bash
pip install git+https://github.com/amitabe/contrastive.git
```

## Usage
To use the package, you must first import PyTorch, and then you can call the Soft-Nearest Neighbors loss function. The signature of the function is as follows:
```python
def snnl(tensor, labels, temperature=1, reduce="mean", use_cosine=False)
```
### Parameters
* **tensor** (Tensor) – input tensor of shape `Bx...xD` (it must have at least 2 dimensions)
* **labels** (Tensor) - labels tensor of shape `B`. It represents the indices of the positive examples for each vector in the batch. The rest of the examples are considered negative examples. It must have the same size as the first dimension of `tensor`.
* **temperature** (float, _optional_) - the temperature of the loss. Default: `1`.
* **reduce** (str, _optional_) - Specifies the reduction to apply to the output: `'none' | 'mean'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output. Default: `'mean'`.
* **use_cosine** (bool, _optional_) - By default, the loss is computed using negative L2 distance between the tensors in the batch. When `use_cosine` is `True`, the loss is computed using cosine distances. Default: `False`.


### Example

There are 2 examples in the [`notebooks`](notebooks/) directory. One of them shows the use of the negative $L_2$ distance with points, and the other shows the use of cosine distance with vectors.

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
