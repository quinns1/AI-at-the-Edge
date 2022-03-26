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
try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze

'User Imports'
from fer_models import model_1, model_1_b, model_2, mobile_net, model_3
from helpers import plot_history, convert_keras_tflite, gen_confusion_matrix
from face_detection import HAAR, MTCNN, YOLO_v3, HOG, MMOD_CNN, CVLIB_DNN
from data import get_pre_split_data_gens, get_train_val_count, get_non_split_data_gens
from quantization import PostTrainingQuantization


'Ascertain if running on colab or not'
try:
  import google.colab
  COLAB = True
except:
  COLAB = False
  

logger = logging.getLogger(__name__) 
cv2.ocl.setUseOpenCL(False)



def train_fer_model(model,  target_classes, img_size, dataset, model_str, logging_filename, data_path=r'./data/', 
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
    
    
    



    logger.info('Train FER Model')
    logger.debug('Target Classes: {}'.format(str(target_classes)))
    
    'Step 1: Get training and validation data generators' 
    logger.debug('Loading data generators, looking for data in directory: {}'.format(data_path)) 
    if dataset == 'FER_2013' or dataset == 'FER_2013+':
        X_data_gen, y_data_gen = get_pre_split_data_gens(data_path=data_path, batch_size=batch_size, img_size=img_size) 
    elif dataset == 'CK+' or dataset == 'JAFFE':
        X_data_gen, y_data_gen = get_non_split_data_gens(data_path=data_path, batch_size=batch_size, img_size=img_size) 


    
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
            X_data_gen,
            steps_per_epoch= train_count // batch_size, 
            epochs=epochs,
            validation_data=y_data_gen,
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
    
   



def evaluate_quantization(model, model_str, target_classes, img_size, dataset, batch_size = 64, data_path= '../data/'):
    
    logger.info('Evaluating Quantization')
    logger.debug('Target Classes: {}'.format(str(target_classes)))
    
    'Step 1: Get training and validation data generators' 
    logger.debug('Loading data generators, looking for data in directory: {}'.format(data_path)) 
    if dataset == 'FER_2013' or dataset == 'FER_2013+':
        X_data_gen, y_data_gen = get_pre_split_data_gens(data_path=data_path, batch_size=batch_size, img_size=img_size) 
    elif dataset == 'CK+' or dataset == 'JAFFE':
        X_data_gen, y_data_gen = get_non_split_data_gens(data_path=data_path, batch_size=batch_size, img_size=img_size) 



    'Evaluate performance of standard model'
    
    results = model.evaluate(X_data_gen, y_data_gen, batch_size=batch_size)
    
    print(results)
    
    pt_quant = PostTrainingQuantization(model)
    
    'Evaluate performance of dynamic range quantized model'
    
    
    dynamic_range_model = pt_quant.dynamic_range_quantization()


    
def main():
    
    
    
    
    'Manual User Inputs'
    'Required for training'
    dataset = 'FER_2013+'
    epochs = 100
    img_size = (48, 48)             #Resise image, this drastically impacts the size of the model
    
    
    'The following are required for real-time/single image FER'
    load_model_weights = r'../trained_models/model_1_b_FER_2013+_100_64_64x64_model_weights.h5'        
    img_directory = r'../data/test_images'
    # img_directory = r'C:\Code\GitHub\FER at the Edge\data\fddb\2003\01\16\big'
    detection_method = 'HOG'
    
    
    'Parse sys args'
    options_dict = {1: 'Train FER Model', 2: 'Single Image Facial Emotion Recognition', 3: 'Real-Time Facial Emotion Recognition',
                    4: 'Evaluate Quantization'}
    ap = argparse.ArgumentParser()
    ap.add_argument("--option",help="1: Train. 2: Single Image 3: Real-time facial emotion recognition using webcam",
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
        data_path = '/home/sq/code/FER_at_the_Edge/data/prepped_jaffe'

        
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
    
   
    model, model_str = model_1_b(input_shape = (img_size[0], img_size[1], 1), target_classes=len(target_classes))
   

    
    'Configure logging handler'
    logging_directory = '../logging/'
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)
    logging_filename = logging_directory+str(option)+'_'+str(epochs)+dataset+model_str+'_edge_fer_'+str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")+'.log') 
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(logging_filename)
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info('Running...')
    logger.debug('Dependancies')                    
    x = freeze.freeze()                                                                                                                                                                              
    for p in x:
        logger.debug(p)
        
    logger.debug('sys.argv: {}'.format(sys.argv))
    logger.info('Option {} chosen: {}'.format(option, options_dict[option]))
        
    if option == 2 or option == 3 or option == 4:
        logger.info('Loading Model: {}'.format(load_model_weights))
        # model.load_weights(load_model_weights)
        model = tf.keras.models.load_model(r'../trained_models/model_1_b_FER_2013+_100_64_48x48_model.h5')
        
        
        
    if option == 1:
        train_fer_model(model, target_classes, img_size, dataset, model_str, logging_filename, data_path=data_path, epochs=epochs)
    elif option == 2:
        single_image_fer(model, img_directory, img_size, target_classes_dict, detection_method=detection_method)
    elif option == 3:
        real_time_fer(logging, model)
    elif option == 4:
        evaluate_quantization(model, model_str, target_classes, img_size, dataset, data_path=data_path)
        pass
        
    
    
            

    
    
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

 