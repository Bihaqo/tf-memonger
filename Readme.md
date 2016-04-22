A TensorFlow implementation of the idea from the paper [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)

The code is messy and doesn't really work, but can be a good starting point for someone who want to properly reimplement the idea in TensorFlow.

'gradients' function from TensorFlow get as input a variable 'V' and builds a computational graph that computes the gradient of 'V' w.r.t. parameters. 'try.ipynb' contains my reimplementation of 'gradients' function, which also builds a graph to compute the gradient, but this graph only uses the values from 'store_activations_set' on the backward pass, all other necessary values are recomputed on the fly. Right now 'store_activations_set' is hardcoded, but 'MemoryOptimizer.py' is a still not working attempt to smartly choose what to put into 'store_activations_set' based on the forward computation graph.

'try.ipynb' builds a simple MNIST network and makes a few iterations of optimization. Use TensorBoard to look at the constructed graph.


