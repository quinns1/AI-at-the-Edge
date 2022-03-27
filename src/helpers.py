# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:20:42 2022

@author: Shane
"""


import matplotlib.pyplot as plt
import tensorflow as tf
import enum
import time
import logging
import os
import copy
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2




logger = logging.getLogger(__name__) 

def plot_history(model_history, plot_name='ModelHistory.png'):
    """
    Plot Accuracy and Loss

    Parameters
    ----------
    model_history : TYPE
        DESCRIPTION.
    plot_name : TYPE, optional
        DESCRIPTION. The default is 'plot.png'.

    Returns
    -------
    None.

    """


    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig(plot_name)
    plt.show()
        





def gen_confusion_matrix(model, val_data_gen, target_names, batch_size=64, num_of_test_samples = 7178):
    

    val_data_gen.reset()
    Y_pred = model.predict(val_data_gen, num_of_test_samples // batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    return confusion_matrix(val_data_gen.classes,y_pred), classification_report(val_data_gen.classes, y_pred, target_names=target_names)

  
  
def convert_keras_tflite(model):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # with open(r'../trained_models/model.tflite', 'wb') as f:
    #   f.write(tflite_model)
    
    return tflite_model


  




def evaluate_model(model_interpretor, Test_data_gen, tv_count, img_size = (48, 48), batch_size = 64):
    """
    Evaluate model performance on Test_data_gen. Record inference times and return accuracy & avg inference time

    Parameters
    ----------
    model_interpretor : tf.model or tflite interpretor
        model to be evaluated
    Test_data_gen : tf.imagedatagenerator
        Validation images
    tv_count : TUPLE
        (test images count, validation images count)
    img_size : TUPLE, optional
        Input image size. The default is (48, 48).
    batch_size : INT, optional
        Training batch size. The default is 64.

    Returns
    -------
    accuracy : FLOAT
        correct predictions / total predictions.
    average_inference : FLOAT
        Average inference time in ms.

    """

    _, val_count = tv_count
    images = list()
    labels = list()
    predicted_labels = list()  
    inference_times = list()          
    prediction_place_holder = np.zeros(Test_data_gen[0][1].shape[1], dtype=float)           #Shaped dependant on number of classes
    count = 0
    
    try:
        'Discern if model_interpretor is keras model or tflite interpretor'
        model_interpretor.summary()
        model = model_interpretor
        interpreter = False
    except:
        interpreter = model_interpretor
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        interpreter.allocate_tensors()

    
    if interpreter:

        for tup in Test_data_gen:
            'Iterate through batches'
            if count >= val_count:
                break
            imgs = tup[0]
            labes = tup[1]
            
            for i in range(imgs.shape[0]):
                'Iterate through each image in batch'
                start_time = time.time()
                img = np.squeeze(imgs[i])
                label = labes[i]
                images.append(img)
                labels.append(label)
                input_tensor = np.expand_dims(np.expand_dims(cv2.resize(img , img_size), -1), 0).astype(np.float32)
                interpreter.set_tensor(input_index, input_tensor)                
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                results = np.squeeze(output_data)             
                predicted_i = np.argmax(results)
                predicted_class = copy.deepcopy(prediction_place_holder)
                predicted_class[predicted_i] = 1
                predicted_labels.append(predicted_class)
                inference_time = (time.time() - start_time)*1000         #Inference time in ms
                inference_times.append(inference_time)
                
            count += batch_size
            print('Evaluating interpretor on Validation Images, Percent Complete: {}%'.format(round(100*count/val_count, 2)))
            
            
    
    else:

        for tup in Test_data_gen:
            'Iterate through batches'
            if count >= val_count:
                break
            imgs = tup[0]
            labes = tup[1]

            for i in range(imgs.shape[0]):
                'Iterate through each image in batch'
                start_time = time.time()
                img = np.squeeze(imgs[i])
                label = labes[i]
                images.append(img)
                labels.append(label)
                prepped_image = np.expand_dims(np.expand_dims(cv2.resize(img , img_size), -1), 0)
                output = model.predict(prepped_image)
                predicted_i = np.argmax(output)
                predicted_class = copy.deepcopy(prediction_place_holder)
                predicted_class[predicted_i] = 1
                predicted_labels.append(predicted_class)
                inference_time = (time.time() - start_time)*1000         #Inference time in ms
                inference_times.append(inference_time)

            count += batch_size
            print('Evaluating model on Validation Images, Percent Complete: {}%'.format(round(100*count/val_count, 2)))


    average_inference = np.average(np.array(inference_times))

    accurate_count = 0
    for i in range(len(predicted_labels)):
        if np.array_equal(predicted_labels[i], labels[i]):
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(predicted_labels)



    return accuracy, average_inference





class SIZE_UNIT(enum.Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4
    
def convert_unit(size_in_bytes, unit):
    """ Convert the size from bytes to other units like KB, MB or GB"""
    if unit == SIZE_UNIT.KB:
        return size_in_bytes/1024
    elif unit == SIZE_UNIT.MB:
        return size_in_bytes/(1024*1024)
    elif unit == SIZE_UNIT.GB:
        return size_in_bytes/(1024*1024*1024)
    else:
        return size_in_bytes




def get_file_size(file_name, size_type = SIZE_UNIT.BYTES ):
    """ Get file in size in given unit like KB, MB or GB"""
    size = os.path.getsize(file_name)
    return convert_unit(size, size_type)
     

