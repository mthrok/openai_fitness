This directory contains the list of intergation tests.

* `run_dqn.sh`

This test builds and run DQN against ALEEnvironment so as to verify that it is not broken.

* `run_initializer_compatibility_tests.sh`

This test runs initializers and check if the distribution is correct.

* `run_layer_numerical_compatibility_tests.sh`

This test compares the outputs from fixed layer configuration/parameter and input so as to ensure layers' behavior is same across backends.

* `run_optimizer_numerical_compatibility_tests.sh`

This test runs optimizers on a set of simple curve and compares the results across backends so as to ensure optimizers' behavior is same across backends.

* `run_serialization_tests.sh`

This test builds model and optimization operations, serialize their parameters, then check if they can be deserialized.
