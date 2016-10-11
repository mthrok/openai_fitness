The followings are the command with which `parameter.h5` and `input.h5` for `dense` test are created.

- input:  `python tool/create_h5_data.py --key input  "np.random.randn(32, 8)" tests/integration/data/layer/dense/input.h5`
- weight: `python tool/create_h5_data.py --key weight "np.random.randn(8, 12)" tests/integration/data/layer/dense/parameter.h5`
- bias:   `python tool/create_h5_data.py --key bias   "np.random.randn(12,)"   tests/integration/data/layer/dense/parameter.h5`
