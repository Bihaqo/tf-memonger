A TensorFlow implementation of the idea from the paper [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)

The code is messy and doesn't really work, but can be a starting point for someone who want to properly reimplement the idea in TensorFlow.

`gradients` function from TensorFlow gets as input a variable `V` and builds a computational graph that computes the gradient of `V` w.r.t. the parameters. The backward pass computational graph uses the values of all nodes of the forward computational graph. `try.ipynb` contains my reimplementation of `gradients` function. It also builds a graph to compute the gradient, but it only uses the values from the nodes listed in `store_activations_set` on the backward pass, all other necessary values are recomputed on the fly. Right now `store_activations_set` is hardcoded, but `MemoryOptimizer.py` is a still not working attempt to smartly choose what to put into `store_activations_set` based on the analysis of the forward computation graph.

`try.ipynb` builds a simple MNIST network and makes a few iterations of optimization. Use TensorBoard to explore the constructed graph.


