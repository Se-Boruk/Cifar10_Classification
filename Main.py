from NeuroUtils import Core
from NeuroUtils import ML_assets as ml
import Architectures
import tensorflow as tf
import numpy as np
#Data preparation
##############################################################
#1
#Creating Class of the project, putting parameters from Config file
Cifar = Core.Project.Classification_Project()

#2
#Initializating data from main database folder to project folder. 
#Parameters of this data like resolution and crop ratio are set in Config
Cifar.Initialize_data(Database_Directory = "D:\Bazy_Danych\Cifar10",
                      Img_Height = 32,
                      Img_Width = 32,
                      Grayscale = False
                      )


#3
#Loading and merging data to trainable dataset.
#Optional reduction of the size class
x, y, dictionary = Cifar.Load_and_merge_data(Reduced_class_size= None)




#4
#Processing data by splitting it to train,val and test set and data normalization
x_train, y_train , x_val , y_val , x_test , y_test = Cifar.Process_data(X = x,
                                                                        Y = y,
                                                                        Val_split = 0.1,
                                                                        Test_split = 0.075,
                                                                        Normalization = 'Z_score_channel',
                                                                        DataType = np.float32
                                                                        )

#5
#Data Injection
Cifar.Save_Data(x_train = x_train,
               y_train = y_train,
               x_val = x_val,
               y_val = y_val,
               dictionary = dictionary,
               x_test = x_test,
               y_test = y_test,
               new_process_mark = None,
               new_normalization_mark = None
               )


#Model training
##############################################################
#6
#Architecture declaration, model loading

model = Architectures.NineNet(shape = (Cifar.IMG_H,Cifar.IMG_W,Cifar.CHANNELS), n_classes = Cifar.N_CLASSES)


#7
# Compile the model
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



#7.1 Optional 
#Define lr_scheduler
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    if epoch > 150:
        lrate = 0.0001
    return lrate

#7/2 Define augmentation 
aug_config = ml.DataSets.Augmentation_Config(
    apply_flip_left_right = True,
    apply_rotation = True,
    apply_width_shift = True,
    apply_height_shift = True)



#
#Training of the model. It can load previously saved data from project folder
#or train from scratch.


model = Cifar.Initialize_weights_and_training(Model = model,
                                             Architecture_name = 'NineNet',
                                             Epochs= 500,
                                             Batch_size = 32,
                                             Train = True,
                                             Learning_rate_scheduler = None,
                                             Augmentation_config = aug_config,
                                             Patience = 50,
                                             Min_delta_to_save = 0.001,
                                             Device = "GPU",
                                             Checkpoint_monitor = "val_accuracy",
                                             Checkpoint_mode = "max",
                                             add_config_info = ""
                                             )


#8
#Performing high detail analysis of all models trained
Core.Utils.Models_analysis(show_plots = False,
                           save_plots = True
                           )

#print(Cifar)




