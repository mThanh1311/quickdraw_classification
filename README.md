# QuickDraw
## Introduction :
Hello there, I am Thanh Minh Truong, this is one of my beginnings in the process of learning AI

Here is my python source code about Deep Learning - Image Classification with Quick Draw dataset - an online game developed by Google.

I used **Pytorch** framework for my project, this includes:
1. Preprocessing and validating input images and labels
2.  Build CNN architecture
3.  Model training
4.  Model evaluation
5.  Testing

*In the future, i will use pen (or any object) with color red. When we are drawing some object (like star, moon,...), model will give a prediction of the object we draw*
## Dataset:
The dataset used for training my model could be found at [Quick Draw dataset] https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap. Here I only picked up 20 files for 20 categories
## Categories:
|       |         |            |           |        |
|:-----:|:-------:|:----------:|:---------:|:------:|
| apple | banana  | basketball | book      | clock  |
| cloud | eye     | flip flops | flower    | hat    |
| key   | moon    | pillow     | pizza     | star   |
| sun   | t-shirt | table      | underwear | zigzag |
## Model:
This is my model parameters ( use ```summary``` from ```torchsummary```):

![Model](https://github.com/mThanh1311/quickdraw_classification/blob/main/experiments/model.png)
## Training :
I downloaded .npy file corresponding to 20 classes and store them in folder [**Dataset**](https://github.com/mThanh1311/quickdraw_classification/tree/main/Dataset)

After that, I will run this scripts in the ```terminal```:

```
python trian.py -r 0.8 -s 1000 -b 8 -e 10 -o Adam -l 0.001
 ```
## Checkpoint:
I save the checkpoint to the [**checkpoint/best.pth folder**](https://github.com/mThanh1311/quickdraw_classification/blob/main/checkpoint/best.pth) after the training process
## Experiments :
For each class, I take the first 1000 images, and then split them to training and test sets with ratio **8:2**.

This is result:

![Train/Loss](https://github.com/mThanh1311/quickdraw_classification/blob/main/experiments/tensorboard-train-loss.png)
![Valid](https://github.com/mThanh1311/quickdraw_classification/blob/main/experiments/tensorboard-val.png)
## Confunsion matrix:
![Conf_matrix](https://github.com/mThanh1311/quickdraw_classification/blob/main/experiments/tensorboard-conf-matrix.png)
## Test
Run ```
test.py ``` with images for testing in folder **test_imaged**

![Test](https://github.com/mThanh1311/quickdraw_classification/blob/main/experiments/test.png)
## Requirements :
* python 3.10
* torch 2.1.1
* opencv-python 4.8.1
* matplotlib 3.8.2
* tqdm 4.66.1
* numpy 1.26.2
* scikit-learn 1.3.2
* tensorboard 2.15.1
