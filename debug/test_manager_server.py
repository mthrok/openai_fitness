from __future__ import print_function
from __future__ import absolute_import

import requests

import luchador.env


res = requests.post(
    'http://localhost:5000/create', json={
        'environment': {
            'name': 'ALEEnvironment',
            'args': {
                'rom': 'breakout.bin',
                'display_screen': True,
            }
        },
        'port': 5001,
        'host': '0.0.0.0',
    }
)
print(res)
