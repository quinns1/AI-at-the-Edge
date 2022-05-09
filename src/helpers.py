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
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import cv2
import tempfile
import pandas as pd




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
        

def gen_confusion_matrix(model, val_data_gen, target_names, tv_count, batch_size=64):
    """
    Return confusion matrix and classification report for logging

    Parameters
    ----------
    model : KERAS MODEL OBJECT

    val_data_gen : IMAGE DATA GEN

    target_names : LIST
        Target Classes
    tv_count : TUPLE
       Train/validation count
    batch_size : INT, optional
         The default is 64.

    Returns
    -------
    TYPE   TUPLE of STRINGS
        (Confusion Matrix, Classification report)

    """
    
    _, val_count = tv_count
    
    val_data_gen.reset()
    Y_pred = model.predict(val_data_gen, val_count // batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    return confusion_matrix(val_data_gen.classes,y_pred), classification_report(val_data_gen.classes, y_pred, target_names=target_names)

  
  
def convert_keras_tflite(model):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    

    return tflite_model


  




def evaluate_model(model_interpretor, params, Test_data_gen):
    """
    Evaluate model performance on Test_data_gen. Record inference times and return accuracy & avg inference time

    Parameters
    ----------
    model_interpretor : tf.model or tflite interpretor
        model to be evaluated
    Test_data_gen : tf.imagedatagenerator
        Validation images
    params: dict
        Parameters 
        params['img_size'] = (x,y) input image dimensions
        params['batch_size']
        params['t_v_count'] = Training validation count (training images, validations images)


    Returns
    -------
    accuracy : FLOAT
        correct predictions / total predictions.
    average_inference : FLOAT
        Average inference time in ms.

    """
    
    
    Test_data_gen.reset()
    _, val_count = params['t_v_count'] 
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
        output_details = interpreter.get_output_details()
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
                input_tensor = np.expand_dims(np.expand_dims(cv2.resize(img , params['img_size']), -1), 0).astype(np.float32)
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
                
            count += params['batch_size'] 
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
                prepped_image = np.expand_dims(np.expand_dims(cv2.resize(img , params['img_size'] ), -1), 0)
                output = model.predict(prepped_image)
                predicted_i = np.argmax(output)
                predicted_class = copy.deepcopy(prediction_place_holder)
                predicted_class[predicted_i] = 1
                predicted_labels.append(predicted_class)
                inference_time = (time.time() - start_time)*1000         #Inference time in ms
                inference_times.append(inference_time)

            count += params['batch_size'] 
            print('Evaluating model on Validation Images, Percent Complete: {}%'.format(round(100*count/val_count, 2)))
            
    average_inference = np.average(np.array(inference_times))

    accurate_count = 0
    for i in range(len(predicted_labels)):
        if np.array_equal(predicted_labels[i], labels[i]):
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(predicted_labels)


    return accuracy, average_inference



def get_model_size(model):
    """
    Return model size, save first then compute file size

    Parameters
    ----------
    model : Keras or tflite model

    Returns
    -------
    FLOAT32
        Model size

    """
    
    _, file_name = tempfile.mkstemp('.h5')
    
    try:
        with open(file_name, 'wb') as f:
            f.write(model)
    except TypeError:
        model.save(file_name)
    
    return get_file_size(file_name, size_type=SIZE_UNIT.MB)



class SIZE_UNIT(enum.Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4
    
def convert_unit(size_in_bytes, unit):
    """
    Convert the size from bytes to other units like KB, MB or GB

    Parameters
    ----------
    size_in_bytes : FLOAT32
        size
    unit : GB, MB, B, KB
        

    Returns
    -------
    TYPE
        Size in unit.

    """
    if unit == SIZE_UNIT.KB:
        return size_in_bytes/1024
    elif unit == SIZE_UNIT.MB:
        return size_in_bytes/(1024*1024)
    elif unit == SIZE_UNIT.GB:
        return size_in_bytes/(1024*1024*1024)
    else:
        return size_in_bytes




def get_file_size(file_name, size_type = SIZE_UNIT.BYTES ):
    """
    Return file size

    Parameters
    ----------
    file_name : STRING
        File name and path
    size_type : SIZEUNIT, optional
        Bytes, MB, KB. The default is SIZE_UNIT.BYTES.

    Returns
    -------
    TYPE : FLOAT32
        Size of file.

    """
    
    size = os.path.getsize(file_name)
    return convert_unit(size, size_type)
     

def get_model_summary(model):
    """
    Returns model summary string for logging.
    
    Parameters
    ----------
    model : tf Sequential Model

    Returns
    -------
    short_model_summary : STRING
        String representation of model summary for logging.

    """
    
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary


class ResultsRec():
    """
    Class for recording model size, inference time and accuracy. Used for evaluating model compression techniques
    """
    
    def __init__(self, params):
        
        self.model = params['model_str']
        self.out_file = params['results_filename']
        columns = ['Model', 'Quantization', 'Pruning', 'Model Size', 'Average Inference Time', 'Accuracy', 'Notes']
        self.df = pd.DataFrame(columns=columns)
        
    
    
    def add_res(self, quant, prune, ms, inf, acc, notes='NA'):
        
        temp_df = pd.DataFrame({'Model': [self.model],
                            'Quantization': [quant],
                            'Pruning': [prune],
                            'Model Size': [ms],
                            'Average Inference Time': [inf],
                            'Accuracy': [acc],
                            'Notes': [notes]})
        
        self.df  = pd.concat([self.df, temp_df], ignore_index = False, axis = 0)


    def save_res(self):
        
        self.df.to_csv(self.out_file)
        
        print(self.df)





def get_prediction(model_interpretor, img, params):
    """
    Make prediction with model or tflite model

    Parameters
    ----------
    model_interpretor : TYPE
        DESCRIPTION.
    img : Image
        
    params : DICT
        Parameters
    Returns
    -------
    prediction : List
        Softmax prediction 

    """

    try:
        'Discern if model_interpretor is keras model or tflite interpretor'
        model_interpretor.summary()
        model = model_interpretor
        interpreter = False
    except:
        interpreter = model_interpretor
        input_index = interpreter.get_input_details()[0]["index"]
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()


    if interpreter:

        input_tensor = np.expand_dims(np.expand_dims(np.reshape(img , params['img_size']), -1), 0).astype(np.float32)
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.squeeze(output_data)

    else:

        prepped_image = np.expand_dims(np.expand_dims(cv2.resize(img , params['img_size'] ), -1), 0)
        prediction = model.predict(prepped_image)

    return prediction









