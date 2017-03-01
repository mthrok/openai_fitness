import numpy as np

import luchador.nn as nn

input_shape = (5, 4)
exp = '2 * x'

model_config = nn.get_model_config('debug/anonymous.yml', input_shape=input_shape, exp=exp)
model = nn.make_model(model_config)

in_val = np.ones(input_shape)

session = nn.Session()
out_val = session.run(
    outputs=model.output,
    inputs={model.input: in_val}
)

print in_val
print out_val
