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
    """
    Object for creating low magnitude pruned models
    """
    
    def __init__(self, model):
        
        self.model = model      
        
        
    def low_magnitude_pruning(self, params, X_data_gen, y_data_gen, epochs = 2, 
                              constant_sparsity = False, initial_sparsity=0.50, final_sparsity=.80):
        
        

        train_count = params['t_v_count'][0]
        val_count = params['t_v_count'][1]        
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        
        'Get final step'
        end_step = np.ceil(train_count / params['batch_size']).astype(np.int32) * epochs
        
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
          



def custom_prune(model, params, X_data_gen, y_data_gen, results ):
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)
    
    
    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam()
    log_dir = tempfile.mkdtemp()
    unused_arg = -1
    epochs = 2
    batches = 1 
    
    count = 0
    for tup in X_data_gen:
        'Get training data'
        if count >= 28386/64:
            break
        imgs = tup[0]
        labes = tup[1]
        count +=1
    
    
    # Non-boilerplate.
    model_for_pruning.optimizer = optimizer
    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model_for_pruning)
    log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir) # Log sparsity and other metrics in Tensorboard.
    log_callback.set_model(model_for_pruning)
    
    step_callback.on_train_begin() # run pruning callback
    for _ in range(epochs):
        log_callback.on_epoch_begin(epoch=unused_arg) # run pruning callback
        for _ in range(batches):
            step_callback.on_train_batch_begin(batch=unused_arg) # run pruning callback
            
            with tf.GradientTape() as tape:
                logits = model_for_pruning(imgs, training=True)
                loss_value = loss(labes, logits)
                grads = tape.gradient(loss_value, model_for_pruning.trainable_variables)
                optimizer.apply_gradients(zip(grads, model_for_pruning.trainable_variables))
          
        step_callback.on_epoch_end(batch=unused_arg) # run pruning callback
        
    

    return model_for_pruning



