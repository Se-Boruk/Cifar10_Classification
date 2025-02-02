import tensorflow as tf





def NineNet(shape, n_classes):
    def crop_batch_images(tensor):
        # tensor is of shape [batch_size, height, width, channels]
        batch_size, height, width, channels = tensor.shape
        
        # Calculate the new height and width after cropping
        new_height, new_width = height - 2, width - 2
        
        # Crop each image in the batch
        cropped_tensor = tf.image.crop_to_bounding_box(
            tensor, 
            offset_height=1, 
            offset_width=1, 
            target_height=new_height, 
            target_width=new_width
        )
        
        return cropped_tensor
    

    img_H , img_W , channels = shape
    #Inputs
    inputs = tf.keras.layers.Input((img_H, img_W, channels)) 
    
    def cnn_block(input_layer, filters, kernel = 3, dropout = 0.2, maxpooling = False):
        cropped_inputs = crop_batch_images(input_layer)
        #32x32x3
        
        x = tf.keras.layers.Conv2D(filters, (kernel,kernel), padding='valid')(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)
        
        x = tf.keras.layers.Conv2D(filters, (kernel,kernel), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)
        
        x = tf.keras.layers.Conv2D(filters, (kernel,kernel), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)

        x = tf.keras.layers.concatenate([x,cropped_inputs])
        if maxpooling:
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
        
        return x
    

    
    x = cnn_block(input_layer = inputs, filters = 64, kernel = 3, dropout = 0.2, maxpooling = False) 
    x = cnn_block(input_layer = x, filters = 128, kernel = 3, dropout = 0.2, maxpooling = True) 
    x = cnn_block(input_layer = x, filters = 192, kernel = 3, dropout = 0.3, maxpooling = False) 
    x = cnn_block(input_layer = x, filters = 384, kernel = 3, dropout = 0.3, maxpooling = True) 
    x = cnn_block(input_layer = x, filters = 512, kernel = 3, dropout = 0.35, maxpooling = False) 

    #Epilog
    e = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    e = tf.keras.layers.Dense(2048)(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.Activation(tf.keras.activations.elu)(e)
    e = tf.keras.layers.Dropout(0.5)(e)
    
    e = tf.keras.layers.Dense(1024)(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.Activation(tf.keras.activations.elu)(e)
    e = tf.keras.layers.Dropout(0.4)(e)
    
    e = tf.keras.layers.Dense(512)(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.Activation(tf.keras.activations.elu)(e)
    e = tf.keras.layers.Dropout(0.3)(e)

    #########################################################
    #########################################################
    #Outputs 
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(e)
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    #Return model
    return model




def NineNet_Reduced(shape, n_classes):
    def crop_batch_images(tensor):
        # tensor is of shape [batch_size, height, width, channels]
        batch_size, height, width, channels = tensor.shape
        
        # Calculate the new height and width after cropping
        new_height, new_width = height - 2, width - 2
        
        # Crop each image in the batch
        cropped_tensor = tf.image.crop_to_bounding_box(
            tensor, 
            offset_height=1, 
            offset_width=1, 
            target_height=new_height, 
            target_width=new_width
        )
        
        return cropped_tensor
    

    img_H , img_W , channels = shape
    #Inputs
    inputs = tf.keras.layers.Input((img_H, img_W, channels)) 
    
    def cnn_block(input_layer, filters, kernel = 3, dropout = 0.2, maxpooling = False):
        cropped_inputs = crop_batch_images(input_layer)
        #32x32x3
        
        x = tf.keras.layers.Conv2D(filters, (kernel,kernel), padding='valid')(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)
        
        x = tf.keras.layers.Conv2D(filters, (kernel,kernel), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)
        
        x = tf.keras.layers.Conv2D(filters, (kernel,kernel), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.elu)(x)

        x = tf.keras.layers.concatenate([x,cropped_inputs])
        if maxpooling:
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
        
        return x
    

    
    x = cnn_block(input_layer = inputs, filters = 32, kernel = 3, dropout = 0.2, maxpooling = False) 
    x = cnn_block(input_layer = x, filters = 64, kernel = 3, dropout = 0.2, maxpooling = True) 
    x = cnn_block(input_layer = x, filters = 96, kernel = 3, dropout = 0.3, maxpooling = False) 
    x = cnn_block(input_layer = x, filters = 192, kernel = 3, dropout = 0.3, maxpooling = True) 
    x = cnn_block(input_layer = x, filters = 256, kernel = 3, dropout = 0.35, maxpooling = False) 

    #Epilog
    e = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    e = tf.keras.layers.Dense(1024)(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.Activation(tf.keras.activations.elu)(e)
    e = tf.keras.layers.Dropout(0.5)(e)
    
    e = tf.keras.layers.Dense(512)(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.Activation(tf.keras.activations.elu)(e)
    e = tf.keras.layers.Dropout(0.4)(e)
    
    e = tf.keras.layers.Dense(256)(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.Activation(tf.keras.activations.elu)(e)
    e = tf.keras.layers.Dropout(0.3)(e)

    #########################################################
    #########################################################
    #Outputs 
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(e)
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    #Return model
    return model