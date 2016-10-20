This directory contains configuration and test data used for layer IO compatibility test.

Each YAML file contains the test setting and `run_layer_numerical_compatibility_tests.sh` script automatically fetches them.
When new layer is added, you can add IO compatibility test just by adding those configuration and value data.

The followings are how the parameter and input data were created.

--
For `relu`, `dense*` layer tests

* `input_randn_5x3.h5`

`python tool/create_h5_data.py --key input "np.random.randn(5, 3)" tests/integration/data/layer/input_randn_5x3.h5`


* `parameter_randn_3x7.h5`

- weight: `python tool/create_h5_data.py --key weight "np.random.randn(3, 7)" tests/integration/data/layer/parameter_randn_3x7.h5`
- bias:   `python tool/create_h5_data.py --key bias   "np.random.randn(7,)"   tests/integration/data/layer/parameter_randn_3x7.h5`

--

For `true_div` layer tests

* `input_randint_1x3x5x7.h5`

`python tool/create_h5_data.py --key input "np.random.randint(255, size=(1, 3, 5, 7), dtype=np.uint8)" tests/integration/data/layer/input_randint_1x3x5x7.h5`

--

For `flatten`, `conv2d_*` layer tests

* `input_mnist_10x4x28x27.h5`

`python tool/create_h5_data_from_mnist.py --key input --channel 4 tests/integration/data/layer/input_mnist_10x4x28x27.h5`

* `parameter_randn_3x4x7x5.h5`

- weight: `python tool/create_h5_data.py --key weight "np.random.randn(3, 4, 7, 5)" tests/integration/data/layer/parameter_randn_3x4x7x5.h5`
- bias:   `python tool/create_h5_data.py --key bias   "np.random.randn(3,)"         tests/integration/data/layer/parameter_randn_3x4x7x5.h5`

where 
- 4 => #Input Channels
- 3 => #Filters
- 7 => Filter Height
- 5 => FilterWidth

--

For `batch_normalization_*` layer tests

* `input_randn_3x5_offset_3.h5`

`python tool/create_h5_data.py --key input "3 + np.random.randn(32, 5)" tests/integration/data/layer/input_randn_3x5_offset_3.h5`

* `parameter_bn.h5`

- mean:   `python tool/create_h5_data.py --key mean   "np.zeros((5,))"    tests/integration/data/layer/parameter_bn.h5`
- var:    `python tool/create_h5_data.py --key var    "np.ones((5,))"     tests/integration/data/layer/parameter_bn.h5`
- scale:  `python tool/create_h5_data.py --key scale  "3 * np.ones((5,))" tests/integration/data/layer/parameter_bn.h5`
- offset: `python tool/create_h5_data.py --key offset "5 * np.ones((5,))" tests/integration/data/layer/parameter_bn.h5`
