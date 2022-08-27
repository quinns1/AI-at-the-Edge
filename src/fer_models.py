# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:12:43 2022

@author: Shane
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, AveragePooling2D, GaussianDropout 
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

import tensorflow as tf



def model_1(input_shape = (48, 48, 1), target_classes=7, weights_path=None):
    """
    Baseline model 1 for FER classification
    
    Returns
    -------
    model : Sequential Model
        Model.
    """


    model_str = 'model1'
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(GaussianDropout(0.2))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',  padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(GaussianDropout(0.2))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(GaussianDropout(0.2))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(GaussianDropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(target_classes, activation='softmax'))

    
        
    return model, model_str



    

def alt_pruned_model1(Train_data_gen, Test_data_gen, params):
    """
    Allows pruning densely connected and convolutional layers to a different sparsity. 
    params['conv_spar'] & params['dense_spar'] specify sparsity.

    Parameters
    ----------
    Train_data_gen : IMAGE DATA GEN
        Train
    Test_data_gen : IMAGE DATA GEN
        Test.
    params : DICT
        Parameters used, see main program for definitions

    Returns
    -------
    stripped_model : Sequential Model
        Strip pruned model
    model_str : STRIN
        Name of model for logging
    pruned_model_accuracy : FLOAT
    """
    
    
    model_str = 'model1_alt_pruned'
    

    pruning_params_sparsit_conv = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=params['conv_spar'] ,
                                                              begin_step=0,
                                                              frequency=100)}
    
    
    pruning_params_sparsity_dens = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=params['dense_spar'] ,
                                                              begin_step=0,
                                                              frequency=100)}
    
    train_count, val_count = params['t_v_count'] 
    input_shape = (params['img_size'][0], params['img_size'][1], 1)
    
    model = tf.keras.Sequential([
        prune_low_magnitude(
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same', name="conv2d_1a_pruned"),
            **pruning_params_sparsit_conv),
        prune_low_magnitude(
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_1b_pruned"),
            **pruning_params_sparsit_conv),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        GaussianDropout(0.2),

        prune_low_magnitude(
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_2a_pruned"),
            **pruning_params_sparsit_conv),
        prune_low_magnitude(
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_2b_pruned"),
            **pruning_params_sparsit_conv),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        GaussianDropout(0.2),
        
        
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_3a_pruned"),
            **pruning_params_sparsit_conv),
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_3b_pruned"),
            **pruning_params_sparsit_conv),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2), padding='same'),
        GaussianDropout(0.2),
        
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_4a_pruned"),
            **pruning_params_sparsit_conv),
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_4b_pruned"),
            **pruning_params_sparsit_conv),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2), padding='same'),
        GaussianDropout(0.2),
        
        Flatten(),
        prune_low_magnitude(
            Dense(1024, activation='relu', name="structural_pruning_dense"),
            **pruning_params_sparsity_dens),
        Dropout(0.5),
        Dense(len(params['target_classes']), activation='softmax')])
    

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr'], decay=params['d']),
                  metrics=['accuracy'])
    
    model.fit(
            Train_data_gen,
            steps_per_epoch= train_count // params['batch_size'], 
            epochs=params['epochs'] ,
            validation_data=Test_data_gen,
            validation_steps= val_count // params['batch_size'],
            callbacks=tfmot.sparsity.keras.UpdatePruningStep())
    
    _, pruned_model_accuracy = model.evaluate(Test_data_gen)
    stripped_model = tfmot.sparsity.keras.strip_pruning(model)    


    return stripped_model, model_str, pruned_model_accuracy



def structured_pruned_model1(Train_data_gen, Test_data_gen, params):
    """
    Encorporates structured pruning, in every 4 elements, at least 2 with the lowest magnitude are pruned (set to 0)

    Parameters
    ----------
    Train_data_gen : IMAGE DATA GEN
        Train
    Test_data_gen : IMAGE DATA GEN
        Test.
    params : DICT
        Parameters used, see main program for definitions

    Returns
    -------
    stripped_model : Sequential Model
        Strip pruned model
    model_str : STRIN
        Name of model for logging
    pruned_model_accuracy : FLOAT
    """
    
    model_str = 'model1_struc_pruned'
    
    
    pruning_params_2_by_4 = {
    'sparsity_m_by_n': (2, 4),}

    train_count, val_count = params['t_v_count'] 
    input_shape = (params['img_size'][0], params['img_size'][1], 1)
    
    model = tf.keras.Sequential([
        prune_low_magnitude(
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same', name="conv2d_1a_pruned"),
            **pruning_params_2_by_4),
        prune_low_magnitude(
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_1b_pruned"),
            **pruning_params_2_by_4),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        GaussianDropout(0.2),

        prune_low_magnitude(
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_2a_pruned"),
            **pruning_params_2_by_4),
        prune_low_magnitude(
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_2b_pruned"),
            **pruning_params_2_by_4),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        GaussianDropout(0.2),
        
        
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_3a_pruned"),
            **pruning_params_2_by_4),
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_3b_pruned"),
            **pruning_params_2_by_4),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2), padding='same'),
        GaussianDropout(0.2),
        
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_4a_pruned"),
            **pruning_params_2_by_4),
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_4b_pruned"),
            **pruning_params_2_by_4),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2), padding='same'),
        GaussianDropout(0.2),
        
        Flatten(),
        prune_low_magnitude(
            Dense(1024, activation='relu', name="structural_pruning_dense"),
            **pruning_params_2_by_4),
        Dropout(0.5),
        Dense(len(params['target_classes']), activation='softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr'], decay=params['d']),
                  metrics=['accuracy'])
    
    model.fit(
            Train_data_gen,
            steps_per_epoch= train_count // params['batch_size'], 
            epochs=params['epochs'] ,
            validation_data=Test_data_gen,
            validation_steps= val_count // params['batch_size'],
            callbacks=tfmot.sparsity.keras.UpdatePruningStep())
    
    
    _, pruned_model_accuracy = model.evaluate(Test_data_gen) 
    stripped_model = tfmot.sparsity.keras.strip_pruning(model)
    
    return stripped_model, model_str, pruned_model_accuracy



def model_2(input_shape = (96, 96, 1), target_classes=7, weights_path=None):


    model_str = 'model2'
    
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(GaussianDropout(0.2))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(GaussianDropout(0.2))
    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(GaussianDropout(0.2))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(target_classes, activation='softmax'))


        
    return model, model_str



def structured_pruned_model2(Train_data_gen, Test_data_gen, params):


    model_str = 'model2_struc_pruned'
    
    
    pruning_params_2_by_4 = {
    'sparsity_m_by_n': (2, 4),}

    train_count, val_count = params['t_v_count'] 
    input_shape = (params['img_size'][0], params['img_size'][1], 1)
    
    
    model = tf.keras.Sequential([
        prune_low_magnitude(
            Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding='same', name="conv2d_1_pruned"),
            **pruning_params_2_by_4),
        GaussianDropout(0.2),
        prune_low_magnitude(
            Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name="conv2d_2_pruned"),
            **pruning_params_2_by_4),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        GaussianDropout(0.2),
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_3_pruned"),
            **pruning_params_2_by_4),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        GaussianDropout(0.2),        
        Flatten(),
        prune_low_magnitude(
            Dense(300, activation='relu', name="structural_pruning_dense"),
            **pruning_params_2_by_4),
        Dropout(0.5),
        Dense(len(params['target_classes']), activation='softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr'], decay=params['d']),
                  metrics=['accuracy'])
    
    model.fit(
            Train_data_gen,
            steps_per_epoch= train_count // params['batch_size'], 
            epochs=params['epochs'] ,
            validation_data=Test_data_gen,
            validation_steps= val_count // params['batch_size'],
            callbacks=tfmot.sparsity.keras.UpdatePruningStep())
    
    
    _, pruned_model_accuracy = model.evaluate(Test_data_gen) 
    stripped_model = tfmot.sparsity.keras.strip_pruning(model)
    
    return stripped_model, model_str, pruned_model_accuracy
        



def alt_pruned_model2(Train_data_gen, Test_data_gen, params):
    """
    Allows pruning densely connected and convolutional layers to a different sparsity. 
    params['conv_spar'] & params['dense_spar'] specify sparsity.

    Parameters
    ----------
    Train_data_gen : IMAGE DATA GEN
        Train
    Test_data_gen : IMAGE DATA GEN
        Test.
    params : DICT
        Parameters used, see main program for definitions

    Returns
    -------
    stripped_model : Sequential Model
        Strip pruned model
    model_str : STRIN
        Name of model for logging
    pruned_model_accuracy : FLOAT
    """
    
    
    model_str = 'model2_alt_pruned'
    
    pruning_params_sparsit_conv = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=params['conv_spar'] ,
                                                              begin_step=0,
                                                              frequency=100)}
    
    
    pruning_params_sparsity_dens = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=params['dense_spar'] ,
                                                              begin_step=0,
                                                              frequency=100)}
    
    train_count, val_count = params['t_v_count'] 
    input_shape = (params['img_size'][0], params['img_size'][1], 1)
    
    model = tf.keras.Sequential([
        prune_low_magnitude(
            Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding='same', name="conv2d_1_pruned"),
            **pruning_params_sparsit_conv),
        GaussianDropout(0.2),
        prune_low_magnitude(
            Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same', name="conv2d_2_pruned"),
            **pruning_params_sparsit_conv),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        GaussianDropout(0.2),
        prune_low_magnitude(
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name="conv2d_3_pruned"),
            **pruning_params_sparsit_conv),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),
        GaussianDropout(0.2),        
        Flatten(),
        prune_low_magnitude(
            Dense(300, activation='relu', name="structural_pruning_dense"),
            **pruning_params_sparsity_dens),
        Dropout(0.5),
        Dense(len(params['target_classes']), activation='softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr'], decay=params['d']),
                  metrics=['accuracy'])
    
    model.fit(
            Train_data_gen,
            steps_per_epoch= train_count // params['batch_size'], 
            epochs=params['epochs'] ,
            validation_data=Test_data_gen,
            validation_steps= val_count // params['batch_size'],
            callbacks=tfmot.sparsity.keras.UpdatePruningStep())
    
    
    _, pruned_model_accuracy = model.evaluate(Test_data_gen) 
    stripped_model = tfmot.sparsity.keras.strip_pruning(model)
    
    return stripped_model, model_str, pruned_model_accuracy
        




