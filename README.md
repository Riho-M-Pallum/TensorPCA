# TensorPCA
**TensorPCA** or **TPCA** is shot for *Tensor Principal Component Analysis*, it is an extension of PCA to tensor (higher dimensional) datasets. TensorPCA is a Python package that implements the algorithm introduced in the paper [*Tensor Principal Component Analysis*](https://arxiv.org/abs/2212.12981).

## Installation

## TensorPCA Guide

Functionality of **TensorPCA** includes 
1. *tensor matricization (unfolding)*, 
1. *tensor factor model estimation*, 
1. *hypothesis test for selecting number of factors*, 
1. *generating random tensor data*.

For a quick start, we can first randomly generate a 3-dim tensor factor model by the following code:
```
R = 2 # rank, number of factors
# tensor size TxNxJ
T = 100
N = 30
J = 20

Y, s, M = DGP((T,N,J),R)
```
The above code assigns the randomly generated tensor to variable `Y`, the scale components to variable `s`, and the vector components to variable `M`. Note that, `s` and `M` are the true parameters that generate the tensor `Y`.

Then following code inputs the tensor `Y` into a class of functions:
```
Z = TensorPCA(Y)
```
The class has method *tensor matricization (unfolding)*:
```
Z.unfold(0) # to unfold the tensor in its first dimension
Z.unfold(1) # to unfold the tensor in its second dimension
Z.unfold(2) # to unfold the tensor in its third dimension
```
We can also estimate the tensor factor model by the code:
```
s_hat, M_hat = Z.t_pca(2) # to estimate a 2-factor model
```
Lastly, for the hypothesis
| H<sub>0</sub>  | H<sub>1</sub> |
| ------------- |:-------------:|
| $\leq$ k factors      | the number of factors > k, but $\leq$ K     |

the following code implements the above test in each dimension:
```
Stat, pValue = Z.ranktest(1, 3)
```
where k=1, K=3.

The output `pValue` would reject the null hypothesis because tensor `Y` has two 2 factors.

## Citing

If you use TensorPCA in an academic paper, please cite 

Babii, Andrii, Eric Ghysels, and Junsu Pan. **"Tensor Principal Component Analysis."** *arXiv preprint arXiv:2212.12981* (2022). 

If you have any question, please contact junsupan1994@gmail.com