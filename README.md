# Brazillian License Plate Recognition with Keras

This is a port of Matthew Earl's [work](https://github.com/matthewearl/deep-anpr) to Keras, allowing easier experimentation of new architectures and quick change in training methods.

## Getting Started

./extractbgs.py SUN397.tar.gz: Extract ~3GB of background images from the SUN database into bgs/. (bgs/ must not already exist.) The tar file (36GB) can be downloaded here. This step may take a while as it will extract 108,634 images.

./gen.py 1000: Generate 1000 test set images in test/. (test/ must not already exist.) This step requires a .ttf font to be in the fonts/ directory.

./train.py: Train the model. A GPU is recommended for this step.

## Prerequisites

The project has the following dependencies:

```
Numpy
Tensorflow
Keras
OpenCV

```
