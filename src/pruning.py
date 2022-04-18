#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 21:06:33 2022

@author: sq
"""


import tempfile
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import os
import zipfile




class Pruning():
    
    def __init__(self, model):
        
        self.model = model
       
        
       
        

    def alternate_layer_pruning(self, params, X_data_gen, y_data_gen, epochs = 5, 
                              dense_sparsity=0.90, conv_sparsity=.60):
        
        if not dense_sparsity and not conv_sparsity:
            return self.model
        
        
        train_count = params['t_v_count'][0]
        val_count = params['t_v_count'][1]        
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        logdir = tempfile.mkdtemp()
        
        
        if dense_sparsity:
        
            model_for_pruning = tf.keras.models.clone_model(self.model, clone_function=apply_pruning_to_dense , )
                        
            'Get final step'
            end_step = np.ceil(train_count / params['batch_size']).astype(np.int32) * epochs
            
            # pruning_params = {'block_size': [1, 16]}
            # model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
    
            pruning_params = {
                  'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=dense_sparsity, 
                                                                            begin_step=0, 
                                                                            end_step=end_step)
            }
    
    
            model_for_pruning = prune_low_magnitude(self.model, **pruning_params)
            model_for_pruning.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    
            
            callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), tfmot.sparsity.keras.PruningSummaries(log_dir=logdir), ]
            model_for_pruning.fit(
                    X_data_gen,
                    steps_per_epoch= train_count // params['batch_size'], 
                    epochs=epochs,
                    validation_data=y_data_gen,
                    validation_steps= val_count // params['batch_size'],
                    callbacks=callbacks)
            
            dense_pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
            
        
        else:
            dense_pruned_model = self.model
        
        if conv_sparsity:
            model_for_pruning = tf.keras.models.clone_model(dense_pruned_model, clone_function=apply_pruning_to_conv,)
                        
            'Get final step'
            end_step = np.ceil(train_count / params['batch_size']).astype(np.int32) * epochs
            
            # pruning_params = {'block_size': [1, 16]}
            # model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
    
            pruning_params = {
                  'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=conv_sparsity, 
                                                                            begin_step=0, 
                                                                            end_step=end_step)
            }
    
            
            model_for_pruning = prune_low_magnitude(dense_pruned_model, **pruning_params)
            
            'Fine Tuning'
            epochs = 10
            opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
            model_for_pruning.compile(optimizer=opt,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    
            
            # callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), tfmot.sparsity.keras.PruningSummaries(log_dir=logdir), ]
            model_for_pruning.fit(
                    X_data_gen,
                    steps_per_epoch= train_count // params['batch_size'], 
                    epochs=epochs,
                    validation_data=y_data_gen,
                    validation_steps= val_count // params['batch_size'],
                    callbacks=callbacks)
            
            pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        else:
            pruned_model = dense_pruned_model
            
        

        
        'Strip pruning wrapper'
        stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        sparsity = get_model_weights_sparsity(stripped_model)

        
        return stripped_model, sparsity 
        
    
    

    # 'pruning schedule to be learning rate???'
    
         
        
        
    def low_magnitude_pruning(self, params, X_data_gen, y_data_gen, epochs = 2, 
                              constant_sparsity = False, initial_sparsity=0.50, final_sparsity=.80):
        
        

        train_count = params['t_v_count'][0]
        val_count = params['t_v_count'][1]        
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        
        'Get final step'
        end_step = np.ceil(train_count / params['batch_size']).astype(np.int32) * epochs
        
        # pruning_params = {'block_size': [1, 16]}
        # model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
        
        'Define model for pruning schedule'
        if constant_sparsity:
            pruning_params = {
                  'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=final_sparsity, 
                                                                            begin_step=0, 
                                                                            end_step=end_step)
            }
        else:
            pruning_params = {
                  'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                                           final_sparsity=final_sparsity,
                                                                           begin_step=0,
                                                                           end_step=end_step)
            }
        
        
        'Compile and fit low magnitude pruned model'
        model_for_pruning = prune_low_magnitude(self.model, **pruning_params)
        model_for_pruning.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        logdir = tempfile.mkdtemp()
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), tfmot.sparsity.keras.PruningSummaries(log_dir=logdir), ]
        model_for_pruning.fit(
                X_data_gen,
                steps_per_epoch= train_count // params['batch_size'], 
                epochs=epochs,
                validation_data=y_data_gen,
                validation_steps= val_count // params['batch_size'],
                callbacks=callbacks)
        
        
        # with tfmot.sparsity.keras.prune_scope():
        #     model_for_pruning = tf.keras.models.load_model(model_for_pruning)
                
        'Strip pruning wrapper'
        stripped_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        sparsity = get_model_weights_sparsity(stripped_model)

        
        return stripped_model, sparsity


def get_model_weights_sparsity(model):

    sparsity = dict()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Wrapper):
            weights = layer.trainable_weights
        else:
            weights = layer.weights
        
        for weight in weights:
            # ignore auxiliary quantization weights
            if "quantize_layer" in weight.name:
                continue
            weight_size = weight.numpy().size
            zero_num = np.count_nonzero(weight == 0)
            s = np.round(zero_num/weight_size, 2)
            
            sparsity[weight.name] = "{} sparsity ({}/{})".format(s, zero_num, weight_size)
   
    return sparsity
        
def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

def apply_pruning_to_conv(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer
          