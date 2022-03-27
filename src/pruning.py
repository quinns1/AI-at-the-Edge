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


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.

    
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(file)
    
    return os.path.getsize(zipped_file)

class Pruning():
    
    def __init__(self, model):
        
        self.model = model
       
        
       

    def strip_pruning(self):
        
        model_for_export = tfmot.sparsity.keras.strip_pruning(self.model)
        
        _, pruned_keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)

        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        pruned_tflite_model = converter.convert()
        _, pruned_tflite_file = tempfile.mkstemp('.tflite')
        
        with open(pruned_tflite_file, 'wb') as f:
            f.write(pruned_tflite_model)
        
        print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
        print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
        print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
                

       
    def low_magnitude_pruning(self, X_data_gen, y_data_gen, t_v_count):
        
        train_count = t_v_count[0]
        val_count = t_v_count[1]
        _, keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(self.model, keras_file, include_optimizer=False)
        print('Saved baseline model to:', keras_file)
        
        
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        
        # Compute end step to finish pruning after 2 epochs.
        batch_size = 64
        epochs = 2
        validation_split = 0.1 # 10% of training set will be used for validation set. 
        
        
        end_step = np.ceil(train_count / batch_size).astype(np.int32) * epochs
        
        # Define model for pruning.
        pruning_params = {
              'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                       final_sparsity=0.80,
                                                                       begin_step=0,
                                                                       end_step=end_step)
        }
        
        model_for_pruning = prune_low_magnitude(self.model, **pruning_params)
        
        # `prune_low_magnitude` requires a recompile.
        model_for_pruning.compile(optimizer='adam',
                      # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        print(model_for_pruning.summary())
        
        logdir = tempfile.mkdtemp()
        
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), tfmot.sparsity.keras.PruningSummaries(log_dir=logdir), ]
        
        # model_for_pruning.fit(X_data_gen, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=callbacks)
        
        model_for_pruning.fit(
                X_data_gen,
                # steps_per_epoch= train_count // batch_size, 
                epochs=epochs,
                validation_data=y_data_gen,
                # validation_steps= val_count // batch_size,
                callbacks=callbacks)
        
        _, model_for_pruning_accuracy = model_for_pruning.evaluate(y_data_gen)
        
        
        
        
        model_for_export = tfmot.sparsity.keras.strip_pruning(self.model)
        
        _, pruned_keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)

        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        pruned_tflite_model = converter.convert()
        _, pruned_tflite_file = tempfile.mkstemp('.tflite')
        
        with open(pruned_tflite_file, 'wb') as f:
            f.write(pruned_tflite_model)
        
        print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
        print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
        print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
                
        
        return model_for_pruning_accuracy, model_for_pruning
