The followings are the command with which `parameter.h5` and `input.h5` for `dense` test are created.

- input:  `python tool/create_h5_data.py --key input   "3 + np.random.randn(32, 16)" tests/integration/data/layer/batch_normalization_2d_learn/input.h5`

- mean:    `python tool/create_h5_data.py --key mean    "np.zeros((16,))" tests/integration/data/layer/batch_normalization_2d_learn/parameter.h5`
- inv_std: `python tool/create_h5_data.py --key inv_std "np.ones((16,))"  tests/integration/data/layer/batch_normalization_2d_learn/parameter.h5`
- scale:   `python tool/create_h5_data.py --key scale   "3.0"             tests/integration/data/layer/batch_normalization_2d_learn/parameter.h5`
- center:  `python tool/create_h5_data.py --key center  "0.5"             tests/integration/data/layer/batch_normalization_2d_learn/parameter.h5`
