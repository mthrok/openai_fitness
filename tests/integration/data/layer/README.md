This directory contains configuration and test data used for layer IO compatibility test.

Each subdirestory must contain
- `config.yml`: Describes the configuration of layer beign test
- `input.h5`: Input data to the layer

and optionally
- `parameter.h5`: Layer parameter file.

`run_layer_numerical_compatibility_tests.sh` automatically fetches data in this directory and run test against them.

When new layer is added, you can add IO compatibility test just by adding those configuration and value data.
