#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 08:33:58 2022

@author: sq
"""


import tensorflow as tf
import tensorflow_model_optimization as tfmot




class PostTrainingQuantization():
    
    def __init__(self, model):
        self.model = model
    
    
    def dynamic_range_quantization(self):
        'Dynamically quantize float32 weights'
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # converter.experimental_enable_resource_variable = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        
        return quantized_model
    
    def float16_quantization(self):
        'Quantize Float32 weights to Float16 (16-bit)'
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        quantized_model = converter.convert()
        
        return quantized_model
    
    
    def int8_quantization(self):
        'Quantize Float32 weights to integers (8-bit)'
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.uint8]
        
        quantized_model = converter.convert()
        
        return quantized_model

        
   

    

def apply_quantization_to_applicable_layers(layer):
    """
    Annotate densely connected and conv2d layers.
    """
    
    if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer



def get_quantization_aware_model(model):
    """
    Get quantization aware model

    Parameters
    ----------
    model : KERAS MODEL OBJECT
        

    Returns
    -------
    quant_aware_model :  KERAS MODEL OBJECT


    """
    

    annotated_model = tf.keras.models.clone_model(
            model,
            clone_function=apply_quantization_to_applicable_layers,
        )
    
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)   
    
    return quant_aware_model



    
  


