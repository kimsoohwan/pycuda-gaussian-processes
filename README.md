# Gaussian Process Regression on the GPU

Gaussian Process Regression is a state-of-the-art non-parametric machine learning method. It's a fancy interpolation technique that lets the data "speak for itself", in the sense that it uses more degrees of freedom in regions where there is more variation in the data.

It has a flexible and expressive way of incorporating priors, or problem-specific beliefs, into the model. This is done by specifying covariance kernels, functions that specify how correlated (similar) we think two points will be. The form of the covariance function determines the kinds of models that will be generated.

## All this power comes for a price

There's a big expensive O(N^3) step during model training that has limited the use of Gaussian Process models to problems with about 1000 training points. That's not enough data to learn interesting and complicated functions.

## Good news: we can use GPUs

The good news is that all the expensive steps in training GPs can be parallelized effectively using GPUs. Those steps are: computing covariance matrices (N^2), computing Cholesky factors of positive-definite matrices (N^3), solving triangular systems using back-substitution (N^2), and optimizing hyperparameters using some kind of parallel global optimization or sampling algorithm.

## Python and CUDA

When building machine-learning models we want to play with data at interactive speeds, but GPs are computationally expensive. That's why this implementation of GPs uses GPUs for all the expensive bits but interfaces with Python so you can mess around.
