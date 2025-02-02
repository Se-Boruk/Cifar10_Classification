
# Cifar10 Human-like classification
### Project is based on library NeuroUtils
[![Static Badge](https://img.shields.io/badge/NeuroUtils_GitHub-0.2.3-purple)](https://github.com/Ciapser/NeuroUtils) [![Static Badge](https://img.shields.io/badge/NeuroUtils_PyPi-0.2.3-blue)](https://pypi.org/project/NeuroUtils/)

This project aims to achieve human like accuracy of prediction on Cifar10 dataset (94%), using library NeuroUtils. 

## Description
Cifar10 classification task is widely known and recognizable as simple benchmark for many neural network architectures. It contains 10 classes of 32x32 RGB  images, and each of them contains 6k of samples:
airplane - automobile - bird - cat - deer - dog - frog - horse - ship - truck
## Results
- **F scores analysis plots**

![alt text](https://github.com/Ciapser/Cifar10_Classification/blob/master/ReadMe_files/Test%20F%20scores.jpg)

- **Train_history**

![alt text](https://github.com/Ciapser/Cifar10_Classification/blob/master/ReadMe_files/Overall%20train%20history.jpg)

- **Confusion matrix** 

![alt text](https://github.com/Ciapser/Cifar10_Classification/blob/master/ReadMe_files/Conf_matrix%20Test.jpg)

- **Model PDF report** 

![alt text](https://github.com/Ciapser/Cifar10_Classification/blob/master/ReadMe_files/Model_preview.jpg)

## Fast usage
If you want to use the model just go into the **Actual_best_result** folder and download pretrained model with all metadata including training and specification of the model.

You can load and use model by:
```
#Tested version of tensorflow: 2.10, it can work on previous ones, but does not have to
import tensorflow as tf
model = tf.keras.models.load_model("Model_best.keras")
predictions = model.predict(Dataset_to_predict)
```
## If you want to make it on your own
If you want to recreate the project on your own, the **main.py** file is your friend. 
You will find there step by step instructions and my last configuration of the model.

In fact, all files you need is **main** file and if you want to use architecture I used, you will find it in **architectures.py**, but you can your own as well. 
All necessary folders and files will be created after you run the project so make sure to put it in some separate folder. 
This project's goal was to use NeuroUtils library, so I highly recommend to use it's tutorial on main page, if you want to experiment.


## Feedback

If you have any feedback about the project, you can reach me on my github profile

