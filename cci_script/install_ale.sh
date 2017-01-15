#!/bin/bash
set -e

if [ ! -d "Arcade-Learning-Environment" ]; then
    git clone https://github.com/mgbellemare/Arcade-Learning-Environment
fi
cd Arcade-Learning-Environment
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF .
make
pip install --upgrade .
