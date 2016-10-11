The followings are the command with which `parameter.h5` and `input.h5` for `conv2d_valid` test are created.

- input: `python tool/create_h5_data_from_mnist.py --channel 4 --key input tests/integration/data/layer/conv2d_valid/input.h5`
- weight: `python tool/create_h5_data.py "np.random.randn(3, 4, 7, 5)" tests/integration/data/layer/conv2d_valid/parameter.h5 --key "weight"`
- bias: `python tool/create_h5_data.py "np.random.randn(3,)" tests/integration/data/layer/conv2d_valid/parameter.h5 --key "bias"`

where

- 4 => #Input Channels
- 3 => #Filters
- 7 => Filter Height
- 5 => FilterWidth

`input.h5` is created from [MNIST database](http://deeplearning.net/data/mnist/mnist.pkl.gz) and includes the first 4 samples of each label from test data set.
