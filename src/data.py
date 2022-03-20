# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:41:14 2022

@author: Shane
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import math




logger = logging.getLogger(__name__) 



def get_fer_2013_data_gens(data_path = r'../data', batch_size = 64, img_size = (64, 64)):
    """
    Ensure FER-2013 dataset is stored in data subdirectory, if not modify train_dir and val_dir paths
    

    Parameters
    ----------
    batch_size : INT, optional
        Batch Size. The default is 64.

    Returns
    -------
    X_train_gen : IMAGE DATA GENERATOR 
        Training data generator
    y_test_gen : IMAGE DATA GENERATOR 
        Test data generator.
    """
    

    train_dir = data_path + r'/train'
    val_dir = data_path + r'/test'
    
    X_gen = ImageDataGenerator(rescale=1./255, rotation_range=45, brightness_range=(0.5,1.5),  
                                horizontal_flip=True, zoom_range = [0.8, 1.2])
    

    # X_gen = ImageDataGenerator(rescale=1./255)
    
    y_gen = ImageDataGenerator(rescale=1./255)
    
    X_train_gen = X_gen.flow_from_directory(
            train_dir,
            target_size=(img_size[0],img_size[1]),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')
    
    y_test_gen = y_gen.flow_from_directory(
            val_dir,
            target_size=(img_size[0],img_size[1]),
            batch_size=batch_size,
            color_mode="grayscale",
            shuffle=False,
            class_mode='categorical')
    
    return X_train_gen, y_test_gen





def get_ckplus_data_gens(data_path = r'../data', batch_size = 64, img_size = (64, 64)):
    

    
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=45, brightness_range=(0.5,1.5),  horizontal_flip=True, validation_split=0.2)
    
    
    X_train_gen = train_datagen.flow_from_directory(
            data_path,
            target_size=(img_size[0],img_size[1]),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical',
            subset='training')
    
    y_test_gen = train_datagen.flow_from_directory(
            data_path,
            target_size=(img_size[0],img_size[1]),
            batch_size=batch_size,
            color_mode="grayscale",
            shuffle=False,
            class_mode='categorical',
            subset='validation')
    
    
    return X_train_gen, y_test_gen




  
    
def get_train_val_count(directory, dataset):
    
    
    train_count = val_count = 0
    
    if dataset == 'FER_2013' or dataset == 'FER_2013+':
        for _, _, files in os.walk(directory + r'/train'):
            for Files in files:
                train_count += 1
    
        for _, _, files in os.walk(directory + r'/test'):
            for Files in files:
                val_count += 1   
                
                
    elif dataset == 'CK+':
        
        for _, _, files in os.walk(directory):
            for Files in files:
                train_count += 1
                
        
        val_count = math.floor(20*train_count/100)
        train_count = math.floor(80*train_count/100)
        
        
    return (train_count, val_count)

