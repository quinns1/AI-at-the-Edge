# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 18:00:27 2022
@author: sq

Train & evaluate various permutations of model compression techniques enabling real-time
Facial Emotion Recognition at the Edge.
"""

'Imports'
import argparse 
import cv2
import datetime
import logging
import os
import sys
import tempfile
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
import tensorflow_model_optimization as tfmot
try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze

'User Defined Imports'
from fer_models import model_1, structured_pruned_model1, structured_pruned_model2, alt_pruned_model2
from helpers import plot_history, gen_confusion_matrix, evaluate_model
from helpers import get_model_size, get_model_summary, ResultsRec
from data import get_pre_split_data_gens, get_train_val_count, get_non_split_data_gens
from quantization import PostTrainingQuantization, get_quantization_aware_model
from pruning import Pruning, get_model_weights_sparsity



logger = logging.getLogger(__name__) 
cv2.ocl.setUseOpenCL(False)



def train_fer_model(model, params, Train_data_gen, Test_data_gen):
    """
    Train model

    Parameters
    ----------
    model : Keras Model
        
    params : DICT
        Parameters.
    Train_data_gen : Image Data Generator

    Test_data_gen : Image Data Generator

    Returns
    -------
    None.

    """
    
    train_count, val_count = params['t_v_count'] 
    lr = params['lr']
    d = params['d']
    logger.info('Train FER Model')
    logger.debug('Target Classes: {}'.format(str(params['target_classes'])))
    logger.debug('Found {} training images and {} validation images'.format(train_count, val_count)) 
    
    'Step 1: Compile and fit model'
    logger.debug('Compiling model')
    logger.debug('Optimiser: Adam')
    logger.debug('Learning rate: {}'.format(str(lr)))
    logger.debug('Decay: {}'.format(str(d)))    
        
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr, decay=d),metrics=['accuracy'])
    logger.debug('Fitting model')
    logger.debug('Epochs: {}'.format(str(params['epochs'])))
    logger.info('Training callbacks printed below in the format:\nEpoch; Training Accuracy; Training Loss; Validation Accuracy; Validation Loss')
    
    csv_logger = CSVLogger(params['logging_filename'], append=True, separator=';')        #Log training metrics
    
    if not params['pruning'] == 'None':
        cbks = [csv_logger, tfmot.sparsity.keras.UpdatePruningStep()]
    else:
        cbks = [csv_logger]
    
    model_info = model.fit(
            Train_data_gen,
            steps_per_epoch= train_count // params['batch_size'], 
            epochs=params['epochs'] ,
            validation_data=Test_data_gen,
            validation_steps= val_count // params['batch_size'],
            callbacks=cbks)
    
    'Step 3: Plot accuracy/loss and save weights'
    base_output_file_str = r'../trained_models/'+params['model_str']+'_'+params['dataset'] +'_'+str(params['epochs'] )+'_'+str(params['batch_size'])+'_'+str(params['img_size'] [0])+'x'+str(params['img_size'] [1])
    plot_name = base_output_file_str+'_model_history.png'
    logger.debug('Plotting Model. Save Plot as {}'.format(plot_name))
    plot_history(model_info, plot_name=plot_name)
    cm, cr = gen_confusion_matrix(model, Test_data_gen, params['target_classes'] , params['t_v_count'])
    
    logger.debug('Confusion Matrix')
    logger.debug(str(cm))
    logger.debug('Classification Report')
    logger.debug(str(cr))

    save_weights_fn = base_output_file_str+'_model_weights.h5'
    save_mode_fn = base_output_file_str+'_model.h5'
    logger.debug('Saving Model weights. Save model weights as {}'.format(save_weights_fn))
    model.save_weights(save_weights_fn)
    logger.debug('Saving Model. Save model as {}'.format(save_mode_fn))
    model.save(save_mode_fn)
    logger.info('Ending.')
      
    
    
    
def evaluate_post_training_quantization(model, params, Test_data_gen, results):
    """
    Evaluate 3 different forms of post training quantization on model. Dynamic range, FLOAT16 & INT8 quantization

    Parameters
    ----------
    model : Keras Model Object
        
    params : DICT
        Parameters.
    Test_data_gen : IMAGE DATA GENERATOR
        
    results : Results Recording Object
        Records accuracy, latency, model size.

    Returns
    -------
    None.

    """
    

    pt_quant = PostTrainingQuantization(model)
    logger.debug('Instatiated Post Training Quantization object')
    
    'Baseline Model Evaluation'
    logger.info('Evaluating Baseline Model performance')
    acc, inf = evaluate_model(model, params, Test_data_gen)
    ms = get_model_size(model)
    params['quantization'] = 'None'
    results.add_res(params['quantization'], params['pruning'], ms, inf, acc,)
    results.save_res()


    'Dynamic Range Quantization'
    dynamic_range_model = pt_quant.dynamic_range_quantization()
    logger.debug('Created dynamic range tflite model')
    _, dynamic_range_model_file = tempfile.mkstemp('.tflite')
    logger.debug('Writing dynamically quantized tflite to file: {}'.format(str(dynamic_range_model_file)))
 
    with open(dynamic_range_model_file, 'wb') as f:
        f.write(dynamic_range_model)
 
    logger.debug('Creating interpretor from {}'.format(str(dynamic_range_model_file)))
    interpreter = tf.lite.Interpreter(model_path=str(dynamic_range_model_file))
    logger.debug('Evaluating Dynamic Range Quantized interpretor performance')
    acc, inf = evaluate_model(interpreter, params, Test_data_gen)
    ms = get_model_size(dynamic_range_model)
    params['quantization'] = 'Dynamic Range'
    results.add_res(params['quantization'], params['pruning'], ms, inf, acc,)
    results.save_res()

    'Float16 Quantization'    
    float16_quant_model = pt_quant.float16_quantization()
    logger.debug('Created float16 quantized tflite model')
    _, float16_quant_model_file = tempfile.mkstemp('.tflite')
    logger.debug('Writing float16 quantized tflite to file: {}'.format(str(float16_quant_model_file)))

    with open(float16_quant_model_file, 'wb') as f:
        f.write(float16_quant_model)
        
    logger.debug('Creating interpretor from {}'.format(str(float16_quant_model_file)))
    interpreter = tf.lite.Interpreter(model_path=str(float16_quant_model_file))
    logger.info('Evaluating FLOAT16 Quantized interpretor performance')
    acc, inf = evaluate_model(interpreter, params, Test_data_gen)
    ms = get_model_size(float16_quant_model)
    params['quantization'] = 'Float 16'
    results.add_res(params['quantization'], params['pruning'], ms, inf, acc,)
    results.save_res()

    'Int8 Quantization'
    int8_quant_model = pt_quant.int8_quantization()
    logger.debug('Created int8 quantized tflite model')
    _, int8_quant_model_file = tempfile.mkstemp('.tflite')
    logger.debug('Writing int8 quantizedt flite to file: {}'.format(str(int8_quant_model_file)))
        
    with open(int8_quant_model_file, 'wb') as f:
        f.write(int8_quant_model)
        
    logger.debug('Creating interpretor from {}'.format(str(int8_quant_model_file)))
    interpreter = tf.lite.Interpreter(model_path=str(int8_quant_model_file))
    logger.info('Evaluating INT8 Quantized interpretor performance')
    acc, inf = evaluate_model(interpreter, params, Test_data_gen)
    ms = get_model_size(int8_quant_model)
    
    params['quantization'] = 'Int8'
    results.add_res(params['quantization'], params['pruning'], ms, inf, acc,)
    results.save_res()

    
    
def evaluate_quantization(model, params, Train_data_gen, Test_data_gen, results):
    """
     Evaluate post training quantization and quantization aware training on model   

    Parameters
    ----------
    model : Keras Model
        
    params : DICT
        Parameters.
    Train_data_gen : Image Data Generator

    Test_data_gen : Image Data Generator

    results : Results Recording Object
        Records accuracy, latency, model size.

    Returns
    -------
    None.

    """
    
    
    lr = params['lr']
    d = params['d']
    epochs = params['epochs']
    train_count, val_count = params['t_v_count']
    logger.info('Evaluating Quantization')
    logger.info('Evaluating quantization using basic model')
    
    'Evaluate post training quantization'
    evaluate_post_training_quantization(model, params, Test_data_gen, results)
    
    'Get quantization aware model'
    quant_aware_model = get_quantization_aware_model(model)
            
    quant_aware_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=Adam(learning_rate=lr, decay=d),metrics=['accuracy'])
    logger.debug('Fitting model')
    logger.debug('Epochs: {}'.format(str(epochs)))

    model_info = quant_aware_model.fit(Train_data_gen,
                            steps_per_epoch= train_count // params['batch_size'],
                            batch_size = params['batch_size'],
                            epochs=epochs,
                            validation_data=Test_data_gen,
                            validation_steps= val_count // params['batch_size'])

    base_output_file_str = '../trained_models/QuantizationAware'

    save_weights_fn = base_output_file_str+'_model_weights.h5'
    save_mode_fn = base_output_file_str+'_model.h5'
    
    logger.debug('Saving Model weights. Save model weights as {}'.format(save_weights_fn))
    model.save_weights(save_weights_fn)
    logger.debug('Saving Model. Save model as {}'.format(save_mode_fn))
    model.save(save_mode_fn)
        
    logger.info('Evaluating quantization on quantization aware model')    
    evaluate_post_training_quantization(quant_aware_model, params, Test_data_gen, results)
    
    

    

    
    
def evaluate_pruning(model, params, Train_data_gen, Test_data_gen, results, pt_quant = False):
    """
    Evaluate various levels of sparsity low magnitude pruning. Constant sparsity and sparsity decay permutations

    Parameters
    ----------
    model : Keras Model
        
    params : DICT
        Parameters.
    Train_data_gen : Image Data Generator

    Test_data_gen : Image Data Generator

    results : Results Recording Object
        Records accuracy, latency, model size.
    pt_quant : BOOL, optional
        If True evaluate post training quantization for all permutations. The default is False.

    Returns
    -------
    None.

    """


    prune = Pruning(model)
    const_sparsity = (True, False)
    init_sparsity = [0, 0.2, 0.5]
    final_sparsity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    logger.info('Evaluating Low Magnitude Pruning')
    for c in const_sparsity:
        for init in init_sparsity:
            if c == True and not init == 0:
                continue
            for final in final_sparsity:
                if not c:
                    if init >= final:
                        continue
                    else:
                        logger.info('Constant: False, Initial sparsity: {}, Final Sparsity: {}'.format(str(init), str(final)))
                        params['pruning'] = str(c)+'_'+str(init)+'_'+str(final)
                else:
                    logger.info('Constant: True, Final Sparsity: {}'.format(str(final)))
                    params['pruning'] = str(c)+'_'+str(final)
                
                params['quantization'] = 'None'
    
                pruned_model, sparsity = prune.low_magnitude_pruning(params, Train_data_gen, Test_data_gen, 
                                                                  constant_sparsity = c, initial_sparsity=init, final_sparsity=final)


                acc, inf = evaluate_model(pruned_model, params, Test_data_gen)
                ms = get_model_size(pruned_model)
                results.add_res(params['quantization'], params['pruning'], ms, inf, acc,)
                results.save_res()
                
                logger.debug('Sparsity per layer')
                for x in sparsity.keys():
                    logger.debug('{}: {}'.format(x, sparsity[x]))

                if pt_quant:
                    evaluate_post_training_quantization(pruned_model, params, Test_data_gen, results)
                    results.save_res()

    





def evaluate_structured_pruning(params, Train_data_gen, Test_data_gen, results):
    """
    Get and evaluate 2/4 structured pruning models with post training quantization and without.

    params : DICT
        Parameters.
    Train_data_gen : Image Data Generator

    Test_data_gen : Image Data Generator

    results : Results Recording Object
        Records accuracy, latency, model size.

    Returns
    -------
    results : Results Recording Object
        Records accuracy, latency, model size.


    """


    'Model 1 Structured Pruning'
    model, params['model_str'], _  = structured_pruned_model1(Train_data_gen, Test_data_gen, params)
    logger.debug('Sparsity per layer. Structured pruned')
    sparsity = get_model_weights_sparsity(model)
    for x in sparsity.keys():
        logger.debug('{}: {}'.format(x, sparsity[x]))
    params['pruning'] = 'structured_pruning'
    save_model_st = '../trained_models/'+params['model_str']+'_'+params['pruning']+'_'+str(params['epochs'])+'.h5'
    model.save(save_model_st)
    
    'Evaluatea with Post Training Quantization'
    evaluate_post_training_quantization(model, params, Test_data_gen, results)
    
    
    'Model 2 Structured Pruning'
    model, params['model_str'], _  = structured_pruned_model2(Train_data_gen, Test_data_gen, params)
    logger.debug('Sparsity per layer. Structured pruned')
    sparsity = get_model_weights_sparsity(model)
    for x in sparsity.keys():
        logger.debug('{}: {}'.format(x, sparsity[x]))
    params['pruning'] = 'structured_pruning'
    save_model_st = '../trained_models/'+params['model_str']+'_'+params['pruning']+'_'+str(params['epochs'])+'.h5'
    model.save(save_model_st)
    
    'Evaluatea with Post Training Quantization'
    evaluate_post_training_quantization(model, params, Test_data_gen, results)
    
    return results
    
            

def evaluate_alternate_sparsity(params, Train_data_gen, Test_data_gen, results, model=False):
    """
    Get and evaluate alternate sparisty models with and without post training quantization.
    Alternate sparisity here means pruning convolutional and densely connected layers to different
    levels of sparsity.

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    Train_data_gen : TYPE
        DESCRIPTION.
    Test_data_gen : TYPE
        DESCRIPTION.
    results : TYPE
        DESCRIPTION.
    model : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    conv_spars = [0.1, 0.3, 0.5]
    dense_spars = [0.5, 0.7, 0.9]

    for conv_spar in conv_spars:
        for dense_spar in dense_spars:
            params['conv_spar'] = conv_spar
            params['dense_spar'] = dense_spar
            model, params['model_str'], _ = alt_pruned_model2(Train_data_gen, Test_data_gen, params)
            
            logger.debug('Sparsity per layer. Convolutional sparsity selected is {}. Dense sparsity is {}.'.format(conv_spar, dense_spar))
            sparsity = get_model_weights_sparsity(model)
            for x in sparsity.keys():
                logger.debug('{}: {}'.format(x, sparsity[x]))
                
            acc, inf = evaluate_model(model, params, Test_data_gen)
            ms = get_model_size(model)
            params['pruning'] = 'alt_'+str(conv_spar)+'_'+str(dense_spar)
            
            results.add_res(params['quantization'], params['pruning'], ms, inf, acc,)
            results.save_res()
            save_model_st = '../trained_models/'+params['model_str']+'_'+params['pruning']+'_'+str(params['epochs'])+'.h5'
            model.save(save_model_st)


        
def main():
     
    'Manual User Inputs'
    load_model_path = False
    params = dict()
    params['img_size'] = (48,48)  
    params['epochs'] = 100
    params['dataset'] = 'FER_2013+'  
    params['batch_size'] = 64
    params['model_str'] = 'model1'
    params['detection_method'] = 'HAAR'
    params['pruning'] = 'None'
    params['quantization'] = 'None'
    params['lr'] = 0.0001
    params['d'] = 1e-6

    load_model_path = '../trained_models/model1_FER_2013+_100_64_48x48_model.h5'
    
    'Parse sys args'
    options_dict = {1: 'Train model', 2:'Evaluate Quantization', 3: 'Evaluate Pruning'}
    ap = argparse.ArgumentParser()
    ap.add_argument("--option",help="1: {}. 2: {}. 3: {}".format(options_dict[1], options_dict[2], options_dict[3]),
                    action='store', type=str, required=False, default='2')
    option = int(ap.parse_args().option)
    
    'Locate dataset'
    if params['dataset'] == 'CK+':
        params['data_path'] = '../data/CK+_Complete'
    elif params['dataset']  == 'FER_2013':
        params['data_path']  = '../data/fer_2013'
    elif params['dataset']  == 'FER_2013+':
        params['data_path']  = '../data/fer_2013+'
    elif params['dataset']  == 'JAFFE':
        params['data_path']  = '../data/prepped_jaffe'
        
    'Init target classes list/dict'
    if params['dataset'] == 'FER_2013':
        params['target_classes_dict'] = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        params['target_classes'] = ["Angry", "Disgusted","Fearful", "Happy", "Neutral","Sad", "Surprised"]
    elif params['dataset'] == 'CK+':
        params['target_classes_dict'] = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Sadness", 6: "Surprise"}
        params['target_classes']  = ["Anger", "Contempt","Disgust", "Fear", "Happy","Sadness", "Surprise"]
    elif params['dataset'] == 'FER_2013+':
        params['target_classes_dict'] = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Neutral", 6: "Sadness", 7: "Surprise"}
        params['target_classes']  = ["Anger", "Contempt","Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]
    elif params['dataset'] == 'JAFFE':
        params['target_classes_dict'] =  {0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
        params['target_classes']  = ["Angry", "Disgust","Fear", "Happy", "Neutral","Sad", "Surprise"]
    
    
    'Get data image data generators'
    if params['dataset']  == 'FER_2013' or params['dataset']  == 'FER_2013+':
        Train_data_gen, Test_data_gen = get_pre_split_data_gens(data_path=params['data_path'] , batch_size=params['batch_size'], img_size=params['img_size']) 
    elif params['dataset']  == 'CK+' or params['dataset']  == 'JAFFE':
        Train_data_gen, Test_data_gen = get_non_split_data_gens(data_path=params['data_path'] , batch_size=params['batch_size'], img_size=params['img_size'] ) 
    
    params['t_v_count'] = get_train_val_count(params['data_path'] , params['dataset'])             # (train_count, val_count)
         
    trained_models_dir = '../trained_models'
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
        
    'Load Model'
    if load_model_path and not option == 1:    
        model = tf.keras.models.load_model(load_model_path)
    else:
        'Models defined in fer_models.py'
        model, params['model_str'] = model_1(input_shape = (params['img_size'][0], params['img_size'][1], 1), target_classes=len(params['target_classes'] ))
        
    model.summary()

    'Configure logging handler'
    logging_directory = '../logging/'
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)
    if option == 1:
        params['logging_filename']  = logging_directory+str(option)+'_'+str(params['epochs'])+'_'+params['dataset'] +'_'+params['model_str']+'_edge_fer_'+str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")+'.log')
    else: 
        params['logging_filename']  = logging_directory+str(option)+'_'+params['dataset'] +'_'+params['model_str']+'_edge_fer_'+str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")+'.log') 
    
    params['results_filename'] = logging_directory+str(option)+'_'+params['dataset'] +'_'+params['model_str']+'_results_'+str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")+'.csv') 
    
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(params['logging_filename'] )
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Running...')
    logger.debug('sys.argv: {}'.format(sys.argv))
    
    'Log Paramters'
    for p in params.keys():
        logger.debug('{}: {}'.format(p, params[p]))
    logger.info('Option {} chosen: {}'.format(option, options_dict[option]))
    logger.debug('Dependancies')                    
    x = freeze.freeze()                                                                                                                                                                              
    for p in x:
        logger.debug(p)
        
    'Convert model summary to string for logging'
    logger.info('Facial Emotion Recognition Model Summary')
    logger.info(get_model_summary(model))
    
    results = ResultsRec(params)
    

    if option == 1:
        train_fer_model(model, params, Train_data_gen, Test_data_gen)
    elif option == 2:
        evaluate_quantization(model, params, Train_data_gen, Test_data_gen, results)
    elif option == 3:
        # Uncomment pruning methods to evaluate
        # evaluate_pruning(model, params, Train_data_gen, Test_data_gen, results)
        # evaluate_alternate_sparsity(params, Train_data_gen, Test_data_gen, results)
        evaluate_structured_pruning(params, Train_data_gen, Test_data_gen, results)
    results.save_res()
        
    
    


    
    
if __name__ == '__main__':
    
    try:
        main()
    finally:
        'Clean up logging handlers'
        x = list(logger.handlers)
        for i in x:
            logger.removeHandler(i)
            i.flush()
            i.close()
        logging.shutdown()
        
    

 