# Physics-induced Graph Neural Network (PGNN)

An official 're'-implementation of 
[Physics-induced graph neural network: An application to wind-farm power estimation](https://www.sciencedirect.com/science/article/pii/S0360544219315555) (PGNN).

## Notice
We lost the asset for regenerating the journal version of PGNN and also the lost asset is
considered to be obsolete due to the updates of dependencies. Here, we provide re-implementation 
of PGNN based on `DGL`. 
The original implementation were based on the older version of `pytorch` and 
`floris`. However, since we submitted the journal paper, considerable changes has been
made on the dependencies, especially `floris`.

We provide slightly different hyperparameter choices of PGNN so that 
the new implementation shows similar (or better) predictive performances as compared to the journal 
version.

## Difference of re-implementation from the journal version.
- Use Euclidean coordinates of wind turbines as additional node features.
- Use [BatchNorm](https://arxiv.org/abs/1502.03167) in front of the `edge_model`,`node_model` 
  and `global_model` of the PGN layers.
  We experimentally confirmed that using BatchNorm greatly improve the training speed 
  and also slightly improve the asymptotic performance of PGNNs when augmenting Euclidean coordinates
  as inputs.
  
## Dependencies
- pytorch (1.7.1)
- dgl (0.5.x)
- [GraphFloris](https://github.com/Junyoungpark/GraphFloris) - For data generation
- [Adamp](https://github.com/clovaai/AdamP) - An optimizer
- [python-box](https://github.com/cdgriffith/Box) - For handling experiment configurations
- [Ray](https://github.com/ray-project/ray) - For parallel training data generation
- [wandb](https://github.com/wandb/client) - For logging

## Physics-induced Graph Network (PGN) layer

```python
class PGN(nn.Module):
    """
    Pytorch-DGL implementation of the physics-induced Graph Network Layer
    "https://www.sciencedirect.com/science/article/pii/S0360544219315555"
    """

    def __init__(self,
                 edge_model: nn.Module,
                 node_model: nn.Module,
                 global_model: nn.Module,
                 residual: bool,
                 use_attention: bool,
                 edge_aggregator: str = 'mean',
                 global_node_aggr: str = 'mean',
                 global_edge_aggr: str = 'mean'):
```

`edge_model`, `node_model` and `global_model` are any differentiable functions 
that computes the updated edge, node, and global embedding respectively. In the examples,
we used multi-layer perceptron as the `edge_model`,`node_model` and `global_model`.

## Physics-induced Attention
```python
class PhysicsInducedAttention(nn.Module):

    def __init__(self, 
                 input_dim=3, 
                 use_approx=True, 
                 degree=5):
```
`PhysicsInducedAttention` computes the attention score based on the steady-state wake
deficit factor proposed by [J. Park et al](https://www.sciencedirect.com/science/article/pii/S0306261915004560).


## Quick start
### Training new PGNN model from scratch
Using `wandb` logger and `ray`-based parallel training data generation
```console
python train_pgnn.py -device cuda:0 (or 'cpu')
```

or simply
```console
python train_pgnn_naive.py -device cuda:0 (or 'cpu')
```

### Test trained model
We also provide pre-trained PGNN models. The hyperparameter and trained parameters of 
PGNN are saved in `config.yaml` and `model.pt` respectively. The following code loads the 
pre-trained PGNN from disk and evaluates the model.

```console
python test_model.py
```