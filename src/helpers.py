# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:20:42 2022

@author: Shane
"""


import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import os
import datetime
import logging
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np




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
    
    print()

    return confusion_matrix(val_data_gen.classes,y_pred), classification_report(val_data_gen.classes, y_pred, target_names=target_names)

  
  
def convert_keras_tflite(model):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # with open(r'../trained_models/model.tflite', 'wb') as f:
    #   f.write(tflite_model)
    
    return tflite_model

  

  
  