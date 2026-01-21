PyTorch implementation of backprop without storing activations algorithm from [The Reversible Residual Network: Backpropagation Without Storing Activations](https://arxiv.org/abs/1707.04585) paper. All credit goes to original authors.

~0.5x GPU memory usage during training while using RevBlocks instead of (Bottleneck) ResBlocks, depending on the amount of reversibile blocks you have in your network.

Relevant block of code is the RevBlockFunction class. You can derive from that.
