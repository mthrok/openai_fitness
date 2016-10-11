input.h5 was created from [MNIST database](http://deeplearning.net/data/mnist/mnist.pkl.gz) and includes the first sample of each label from test data set.

parameter.h5 was created with

- weight: "np.random.randn(3, 1, 7, 5)"
- bias: "np.random.randn(3,)"

where
- n_input_channels==1
- n_filters==3
- filter_height==7
- filter_width==5
