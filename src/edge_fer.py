# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 18:00:27 2022
@author: sq

Real-Time Facial Emotion Recognition application deployable on low-cost edge devices such as a Rasberry Pi

Option 1: Train
Option 2: Single Image FER
Option 3: Real-time FER (Default)
"""

'Imports'


import numpy as np
import time
import cv2
import argparse 
import os
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
import logging
import datetime
import pathlib
import tempfile
try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze

'User Imports'
from fer_models import model_1, model_1_b, model_2, mobile_net, model_3
from helpers import plot_history, convert_keras_tflite, gen_confusion_matrix, evaluate_model, get_file_size, SIZE_UNIT
from face_detection import HAAR, MTCNN, YOLO_v3, HOG, MMOD_CNN, CVLIB_DNN
from data import get_pre_split_data_gens, get_train_val_count, get_non_split_data_gens
from quantization import PostTrainingQuantization, quantization_aware_model
from pruning import Pruning


'Ascertain if running on colab or not'
try:
  import google.colab
  COLAB = True
except:
  COLAB = False
  

logger = logging.getLogger(__name__) 
cv2.ocl.setUseOpenCL(False)



def train_fer_model(model,  target_classes, img_size, dataset, model_str, logging_filename, 
                    Train_data_gen, Test_data_gen,  t_v_count, data_path=r'./data/', 
                    epochs = 50, batch_size = 64, 
                    lr = 0.0001, d = 1e-6):
    """
    Train Facial Emotion Recognition Model
    
    Steps
    ----------
    Step 1: Get training and validation data generators
    Step 2: Compile model
    Step 3: Plot accuracy/loss and save weights
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    weights_filename : STRING, optional
        Save weights to this filename. The default is 'model.h5'.
    epochs : INT, optional
        Number of Epochs to train for. The default is 50.
    batch_size : INT, optional
        Batch Size. The default is 64

    Returns
    -------
    None.

    """
    
    
    
    # train_count = t_v_count[0]
    # val_count = t_v_count[1]


    logger.info('Train FER Model')
    logger.debug('Target Classes: {}'.format(str(target_classes)))
    

    
    train_count, val_count = get_train_val_count(data_path, dataset) 
    logger.debug('Found {} training images and {} validation images'.format(train_count, val_count)) 
    
    'Step 2: Compile and fit model'
    logger.debug('Compiling model')
    logger.debug('Optimiser: Adam')
    logger.debug('Learning rate: {}'.format(str(lr)))
    logger.debug('Decay: {}'.format(str(d)))    
    
    'Convert model summary to string for logging'
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    logger.debug('Facial Emotion Recognition Model Summary')
    logger.debug(short_model_summary)
    
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr, decay=d),metrics=['accuracy'])
    logger.debug('Fitting model')
    logger.debug('Epochs: {}'.format(str(epochs)))
    logger.info('Training callbacks printed below in the format:\nEpoch; Training Accuracy; Training Loss; Validation Accuracy; Validation Loss')
    
    csv_logger = CSVLogger(logging_filename, append=True, separator=';')        #Log training metrics
    model_info = model.fit(
            Train_data_gen,
            steps_per_epoch= train_count // batch_size, 
            epochs=epochs,
            validation_data=Test_data_gen,
            validation_steps= val_count // batch_size,
            callbacks=[csv_logger])
    
    'Step 3: Plot accuracy/loss and save weights'
    base_output_file_str = r'../trained_models/'+model_str+'_'+dataset+'_'+str(epochs)+'_'+str(batch_size)+'_'+str(img_size[0])+'x'+str(img_size[1])
    plot_name = base_output_file_str+'_model_history.png'
    logger.debug('Plotting Model. Save Plot as {}'.format(plot_name))
    plot_history(model_info, plot_name=plot_name)
    cm, cr = gen_confusion_matrix(model, y_data_gen, target_classes)
    logger.debug('Confusion Matrix')
    logger.debug(str(cm))
    logger.debug('Classification Report')
    logger.debug(str(cr))
    print(cm)
    print(cr)
    
    save_weights_fn = base_output_file_str+'_model_weights.h5'
    save_mode_fn = base_output_file_str+'_model.h5'
    
    logger.debug('Saving Model weights. Save model weights as {}'.format(save_weights_fn))
    model.save_weights(save_weights_fn)
    logger.debug('Saving Model weights. Save model weights as {}'.format(save_mode_fn))
    model.save(save_mode_fn)
    logger.info('Ending.')
    
    
    

    
def single_image_fer(model, image_directory, img_size, emotion_dict, detection_method = 'HOG', haar_cascade = '../trained_models/haarcascade_frontalface_default.xml'):
    """
    Run facial emotion recognition on all images in image_directory
    
    Steps
    ----------
    1: Face detection using 'detection_method' 
    2. Facial emotion recognition classification using 'model'
    3. Display image with classification

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    
    
    logger.debug('Instantiating {} face detection classifier'.format(detection_method))
    if detection_method == 'HAAR':
        face_detection_classifier = HAAR(haar_cascade=haar_cascade)
    elif detection_method == 'MTCNN':
        face_detection_classifier = MTCNN()
    elif detection_method == 'YOLO':
        face_detection_classifier = YOLO_v3()
    elif detection_method == 'HOG':
        face_detection_classifier = HOG()
    elif detection_method == 'MMOD_CNN':
        face_detection_classifier = MMOD_CNN()

    
    logger.debug('{} face detection classifier has been instantiated'.format(detection_method))
    logger.debug('Iterating through images in {}'.format(image_directory))
    for filename in os.listdir(image_directory):
        image_file  = os.path.join(image_directory, filename)
        if not os.path.isfile(image_file) or not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.debug('{} not image, skipping'.format(filename))
            continue
        logger.debug('Image: {}'.format(image_file))
        start_time =  time.time()
        img = cv2.imread(image_file)
        
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if detection_method == 'HAAR':
            face_locations = face_detection_classifier.get_face_locations(gray_image)
        elif detection_method == 'MTCNN' or detection_method == 'HOG' or detection_method == 'MMOD_CNN':
            face_locations = face_detection_classifier.get_face_locations(img)

        print(face_locations)
            
     


        
        detection_time = time.time() 
        face_detection_time = detection_time - start_time

        prediction_time = 0
        for (x, y, w, h) in face_locations:
            roi_gray = gray_image[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, img_size), -1), 0)
            prediction = model.predict(cropped_img)
            prediction_time = time.time() - detection_time
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)
            cv2.putText(img, emotion_dict[int(np.argmax(prediction))], (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    
        
        execution_time = time.time() - start_time
        img_name = image_file.split('\\')[-1].split('.')[0]
        
        if prediction_time == 0:
            logger.warning('No faces identified in image')
        else:
            logger.debug('Face Detection Time: {}'.format(round(face_detection_time, 3)))
            logger.debug('Prediction Time: {} '.format(round(prediction_time, 3)))
            logger.debug('Total Inference Time: {}'.format(round(execution_time, 3)))

        
        
        cv2.imshow(img_name, img)  
            
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()





def real_time_fer(model):
    """
    Capture live video footage. Identify faces using HAAR classifier. Classify facial expressions.
    Display bounding box around face with emotion classification.

    Parameters
    ----------
    model : Keras model object
        Model

    Returns
    -------
    None.
    """
    

    

    cap = cv2.VideoCapture(0)
    while True:
        
        

        cv2.imshow('Output', cv2.resize(img,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
   




    
    
    
    
    
def evaluate_post_training_quantization(model, Test_data_gen, t_v_count, load_model_path):
    
    

    pt_quant = PostTrainingQuantization(model)
    logger.debug('Instatiated Post Training Quantization object')
    
    
    'Baseline Model Evaluation'
    logger.info('Evaluating Baseline Model performance')
    baseline_model_accuracy, baseline_model_avg_inference_time = evaluate_model(model, Test_data_gen, t_v_count)
    baseline_model_size = get_file_size(load_model_path, SIZE_UNIT.MB)
    logger.info("Model Size: {}MB".format(round(baseline_model_size, 2)))
    logger.info('Model Accuracy: {}%'.format(round(100*baseline_model_accuracy, 2)))
    logger.info("Average Inference Time: {}ms".format(round(baseline_model_avg_inference_time, 2)))
    
    
    

    'Dynamic Range Quantization'
    dynamic_range_model = pt_quant.dynamic_range_quantization()
    logger.debug('Created dynamic range tflite model')
    _, dynamic_range_model_file = tempfile.mkstemp('.h5')
    logger.debug('Writing dynamically quantized tflite to file: {}'.format(str(dynamic_range_model_file)))
    
    with open(dynamic_range_model_file, 'wb') as f:
        f.write(dynamic_range_model)

    logger.debug('Creating interpretor from {}'.format(str(dynamic_range_model_file)))
    interpreter = tf.lite.Interpreter(model_path=str(dynamic_range_model_file))
    logger.info('Evaluating Dynamic Range Quantized interpretor performance')
    dynamic_range_model_accuracy, dynamic_range_avg_inference_time = evaluate_model(interpreter, Test_data_gen, t_v_count)
    dynamic_range_model_size = get_file_size(dynamic_range_model_file, SIZE_UNIT.MB)
    
    logger.info("Model Size: {}MB".format(round(dynamic_range_model_size, 2)))
    logger.info('Model Accuracy: {}%'.format(round(100*dynamic_range_model_accuracy, 2)))
    logger.info("Average Inference Time: {}ms".format(round(dynamic_range_avg_inference_time, 2)))


    'Float16 Quantization'    
    float16_quant_model = pt_quant.float16_quantization()
    logger.debug('Created float16 quantized tflite model')
    _, float16_quant_model_file = tempfile.mkstemp('.h5')
    logger.debug('Writing float16 quantized tflite to file: {}'.format(str(float16_quant_model_file)))
    
    with open(float16_quant_model_file, 'wb') as f:
        f.write(float16_quant_model)

    logger.debug('Creating interpretor from {}'.format(str(float16_quant_model_file)))
    interpreter = tf.lite.Interpreter(model_path=str(float16_quant_model_file))
    logger.info('Evaluating FLOAT16 Quantized interpretor performance')
    float16_model_accuracy, float16_avg_inference_time = evaluate_model(interpreter, Test_data_gen, t_v_count)
    float16_model_size = get_file_size(float16_quant_model_file, SIZE_UNIT.MB)
    
    logger.info("Model Size: {}MB".format(round(float16_model_size, 2)))
    logger.info('Model Accuracy: {}%'.format(round(100*float16_model_accuracy, 2)))
    logger.info("Average Inference Time: {}ms".format(round(float16_avg_inference_time, 2)))
        
    
    'Int8 Quantization'
    int8_quant_model = pt_quant.int8_quantization()
    logger.debug('Created int8 quantized tflite model')
    _, int8_quant_model_file = tempfile.mkstemp('.h5')
    logger.debug('Writing int8 quantizedt flite to file: {}'.format(str(int8_quant_model_file)))
    
    with open(int8_quant_model_file, 'wb') as f:
        f.write(int8_quant_model)

    logger.debug('Creating interpretor from {}'.format(str(int8_quant_model_file)))
    interpreter = tf.lite.Interpreter(model_path=str(int8_quant_model_file))
    logger.info('Evaluating INT8 Quantized interpretor performance')
    int8_model_accuracy, int8_avg_inference_time = evaluate_model(interpreter, Test_data_gen, t_v_count)
    int8_model_size = get_file_size(int8_quant_model_file, SIZE_UNIT.MB)
    
    logger.info("Model Size: {}MB".format(round(int8_model_size, 2)))
    logger.info('Model Accuracy: {}%'.format(round(100*int8_model_accuracy, 2)))
    logger.info("Average Inference Time: {}ms".format(round(int8_avg_inference_time, 2)))
    
    



    
def evaluate_quantization(model, Train_data_gen, Test_data_gen, t_v_count, load_model_path,
                          lr = 0.0001, d = 1e-6,  epochs = 50, batch_size=64, quant_aware_model = False):
    
    train_count, val_count = t_v_count
    logger.info('Evaluating Quantization')
    
    logger.info('Evaluating quantization using basic model')
    evaluate_post_training_quantization(model, Test_data_gen, t_v_count, load_model_path)
    
    # if not quant_aware_model:
    #     quant_aware_model = quantization_aware_model(model)
    #     # quant_aware_model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr, decay=d),metrics=['accuracy'])
        
    #     print(quant_aware_model.summary())
    #     logger.debug('Fitting model')
    #     logger.debug('Epochs: {}'.format(str(epochs)))
        
    #     model_info = model.fit(Train_data_gen,
    #                             steps_per_epoch= train_count // batch_size,
    #                             batch_size = batch_size,
    #                             epochs=1,
    #                             validation_data=Test_data_gen,
    #                             validation_steps= val_count // batch_size)

    
    
    
    # logger.info('Evaluating quantization on quantization aware model')    
    # evaluate_post_training_quantization(model, Test_data_gen, t_v_count, load_model_path)
    
    
    
    
def evaluate_pruning(model, Train_data_gen, Test_data_gen, t_v_count):
    
    # train_count = t_v_count[0]
    # val_count = t_v_count[1]

    _, baseline_model_accuracy = model.evaluate(
         y_data_gen, verbose=0)

    prune = Pruning(model)
    
    low_magnitude_accuracy, low_magnitude_model = prune.low_magnitude_pruning(Train_data_gen, Test_data_gen, t_v_count)
    
    # prune.strip_pruning()
    

        
def main():
    
    
    
    
    'Manual User Inputs'
    'Required for training'
    dataset = 'FER_2013+'
    epochs = 100
    img_size = (48, 48)             
    batch_size = 64
    
    
    'The following are required for real-time/single image FER'
    # load_model_weights = r'../trained_models/model_1_b_FER_2013+_100_64_48x48_model_weights.h5'  
    load_model_path = '../trained_models/model_1_b_FER_2013+_100_64_48x48_model.h5'    
    img_directory = r'../data/test_images'
    # img_directory = r'C:\Code\GitHub\FER at the Edge\data\fddb\2003\01\16\big'
    detection_method = 'HOG'
        
    
    'Parse sys args'
    options_dict = {1: 'Train FER Model', 2: 'Single Image Facial Emotion Recognition', 3: 'Real-Time Facial Emotion Recognition',
                    4: 'Evaluate Quantization', 5: 'Evaluate Pruning'}
    ap = argparse.ArgumentParser()
    ap.add_argument("--option",help="1: Train. 2: Single Image 3: Real-time facial emotion recognition using webcam 4: Evaluate Quantization 5: Evaluate Pruning",
                    action='store', type=str, required=False, default='4')
    option = int(ap.parse_args().option)
    
    
    'Locate dataset'
    if dataset == 'CK+':
        data_path = '../data/ck+'
    elif dataset == 'FER_2013':
        data_path = '../data/fer_2013'
    elif dataset == 'FER_2013+':
        data_path = '../data/fer_2013+'
    elif dataset == 'JAFFE':
        data_path = '../data/prepped_jaffe'

        
    'Init target classes list/dict'
    if dataset == 'FER_2013':
        target_classes_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        target_classes = ["Angry", "Disgusted","Fearful", "Happy", "Neutral","Sad", "Surprised"]
    elif dataset == 'CK+':
        target_classes_dict = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Sadness", 6: "Surprise"}
        target_classes = ["Anger", "Contempt","Disgust", "Fear", "Happy","Sadness", "Surprise"]
    elif dataset == 'FER_2013+':
        target_classes_dict = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Neutral", 6: "Sadness", 7: "Surprise"}
        target_classes = ["Anger", "Contempt","Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]
    elif dataset == 'JAFFE':
        target_classes_dict =  {0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
        target_classes = ["Angry", "Disgust","Fear", "Happy", "Neutral","Sad", "Surprise"]
    
    
    if dataset == 'FER_2013' or dataset == 'FER_2013+':
        Train_data_gen, Test_data_gen = get_pre_split_data_gens(data_path=data_path, batch_size=batch_size, img_size=img_size) 
    elif dataset == 'CK+' or dataset == 'JAFFE':
        Train_data_gen, Test_data_gen = get_non_split_data_gens(data_path=data_path, batch_size=batch_size, img_size=img_size) 
    
    t_v_count = get_train_val_count(data_path, dataset)             # (train_count, val_count)


    'Loading Model'
    model, model_str = model_1_b(input_shape = (img_size[0], img_size[1], 1), target_classes=len(target_classes))
    try:
        model.load_weights(load_model_weights)
    except:    
        model = tf.keras.models.load_model(load_model_path)
        
    
    'Configure logging handler'
    logging_directory = '../logging/'
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)
    logging_filename = logging_directory+str(option)+'_'+str(epochs)+dataset+model_str+'_edge_fer_'+str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")+'.log') 
    logger.propagate = False
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logging_filename)
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Running...')
    logger.debug('sys.argv: {}'.format(sys.argv))
    logger.debug('Loading image data generators, looking for data in directory: {}'.format(data_path)) 
    logger.debug('Model: {}'.format(model_str)) 
    logger.info('Option {} chosen: {}'.format(option, options_dict[option]))
    logger.debug('Dependancies')                    
    x = freeze.freeze()                                                                                                                                                                              
    for p in x:
        logger.debug(p)
        


        
    if option == 1:
        train_fer_model(model, target_classes, img_size, dataset, model_str, logging_filename, Train_data_gen, Test_data_gen, t_v_count,
                        data_path=data_path, epochs=epochs, batch_size=batch_size)
    elif option == 2:
        single_image_fer(model, img_directory, img_size, target_classes_dict, detection_method=detection_method)
    elif option == 3:
        real_time_fer(logging, model)
    elif option == 4:
        evaluate_quantization(model, Train_data_gen, Test_data_gen, t_v_count, load_model_path)
    elif option == 5:
        evaluate_pruning(model, Train_data_gen, Test_data_gen, t_v_count)
        
    
    


    
    
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
        
    



# model = convert_keras_tflite(model)

 