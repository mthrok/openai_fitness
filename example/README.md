## How to prepare dataset

To run examples which use external dataset, you need to prepare dataset manually.


### MNIST dataset

1. Download pickled MNIST data from [here](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz)

2. Provide the local path to examples you want to run.


### CelebA Faces dataset

1. Download aligned & cropped face images from [CelebA project](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

2. Use [this script](./dataset/create_celeba_face_dataset.py) to preprocess the images and pickle them.

3. Provide the local path to examples you want to run.
