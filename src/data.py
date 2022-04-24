# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:41:14 2022

@author: Shane
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import math
import shutil




logger = logging.getLogger(__name__) 



def get_pre_split_data_gens(data_path = r'../data', batch_size = 64, img_size = (64, 64)):
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
    # logger = logging.getLogger(__name__) 
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





def get_non_split_data_gens(data_path = r'../data', batch_size = 64, img_size = (64, 64)):
    

    
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
                
                
    elif dataset == 'CK+' or dataset == 'JAFFE':
        
        for _, _, files in os.walk(directory):
            for Files in files:
                train_count += 1
                
        
        val_count = math.floor(20*train_count/100)
        train_count = math.floor(80*train_count/100)
        
        
    return (train_count, val_count)



def prep_jaffe(data_path):
    
    'Create output folder'
    
    new_path = data_path+'/prepped_jaffe'
    
    try:
        os.makedirs(new_path)
    except FileExistsError:
        pass
    
    output_dirs = ['Disgust', 'Anger', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Happy']
    
    
    mapping = dict()
    mapping['DI'] = 'Disgust'
    mapping['AN'] = 'Anger'
    mapping['SA'] = 'Sad'
    mapping['SU'] = 'Surprise'
    mapping['FE'] = 'Fear'
    mapping['NE'] = 'Neutral'
    mapping['HA'] = 'Happy'
    
    
    dir_dict = dict()
    for d in output_dirs:
        new_dir = new_path + '/' + d
        try:
            os.makedirs(new_dir)
        except FileExistsError:
            pass
        
        dir_dict[d] = new_dir
        


    
    for img in os.listdir(data_path):
        splt = img.split('.')
        if splt[-1] == 'tiff':
            feeling = splt[1][0:2]
            new_file_path = new_path + '/' + mapping[feeling] + '/' + img
            print(new_file_path)
            shutil.copyfile(data_path + '/' + img, new_file_path)
            




    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        