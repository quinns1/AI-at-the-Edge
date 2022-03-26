#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 08:33:58 2022

@author: sq
"""


import tensorflow as tf




class PostTrainingQuantization():
    
    def __init__(self, model):
        self.model = model
    
    
    def dynamic_range_quantization(model):
        'Convert floating point weights to integers (8-bit)'
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        
        return quantized_model
        
    
        
        
        
    
    








