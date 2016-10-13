The followings are the command with which `parameter.h5` and `input.h5` for `dense` test are created.

- input:  `python tool/create_h5_data.py --key input   "3 + np.random.randn(32, 16)" tests/integration/data/layer/batch_normalization_2d_learn/input.h5`
- weight: `python tool/create_h5_data.py --key mean    "np.zeros((16,))"             tests/integration/data/layer/batch_normalization_2d_learn/parameter.h5`
- bias:   `python tool/create_h5_data.py --key inv_std "np.ones((16,))"              tests/integration/data/layer/batch_normalization_2d_learn/parameter.h5`
