# License Plate Recognition with Keras

This is a port of Matthew Earl's [deep ANPR CNN](https://matthewearl.github.io/2016/05/06/cnn-anpr/) to Keras, allowing easier experimentation with new architectures and training methods.

## Getting Started

```./train.py```: Train the model. A GPU is recommended for this step. You can uncomment [line 63](https://github.com/dizidio/anpr_keras/blob/master/train.py#L63) if you want to display the predictions for each test image while training.

### Additional Info
Background images to be used for generating training/test images are included in this repository in the ```bgs/``` folder. However, you can extract ~3GB of background images from the SUN database into bgs/ with ```./extractbgs.py SUN397.tar.gz``` (bgs/ must not already exist.) The tar file (36GB) can be [downloaded here](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz). This step may take a while as it will extract 108,634 images.

```./gen.py 1000```: Generate 1000 test set images in test/. (test/ must not already exist.) This step requires a .ttf font to be in the fonts/ directory. The [Mandatory Font](https://www.dafont.com/pt/mandatory.font) is already included, but you may add other fonts.

## Prerequisites

The project has the following dependencies:

```
Numpy
Tensorflow
Keras
OpenCV

```
