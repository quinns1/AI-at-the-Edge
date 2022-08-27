#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:06:50 2022

@author: sq
"""


import time
import os
import cv2 
import copy
import numpy as np
import argparse
import tensorflow as tf
from data import get_pre_split_data_gens, get_train_val_count, get_non_split_data_gens
from face_detection import HAAR



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



    
    
def get_prediction(model_interpretor, img, params):
    """
    Return softmax output array of

    Parameters
    ----------
    model_interpretor : TF Model or TFLITE interpretor
        Model/Interpretor to make prediction
    img : NP Array
        Input image of cropped face.
    params: dict
        Parameters 
        params['img_size'] = (x,y) input image dimensions

    Returns
    -------
    prediction : NP Array
        Prediction Softmax Output

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

        input_tensor = np.expand_dims(np.expand_dims(cv2.resize(img , params['img_size']), -1), 0).astype(np.float32)
        interpreter.set_tensor(input_index, input_tensor)                
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.squeeze(output_data)             
    
    else:

        prepped_image = np.expand_dims(np.expand_dims(cv2.resize(img , params['img_size'] ), -1), 0)
        prediction = model.predict(prepped_image)            
            
    return prediction


    
    
def real_time_edge_fer(model, params):
    """
    Run real-time FER using 'model'. Print timings and predicted emotion to console

    Parameters
    ----------
    model : TF Model or TFLITE interpretor
        Model/Interpretor to make prediction
    params: dict
        Parameters 
        params['img_size'] = (x,y) input image dimensions.

    Returns
    -------
    None.

    """
    
    camera = cv2.VideoCapture(0)
    detect_times = list()
    fer_times = list()
    total_times = list()
    if params['detection_method'] == 'HAAR':
        face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 
    
    while True:    
        start_time = time.time()
        ret, frame = camera.read()   
        gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
        t = time.time()
        faces=face_cascade.detectMultiScale(gray_img, scaleFactor=1.3,minNeighbors=5)
        get_face_locations_time = time.time() - t
        t = time.time()
        if len(faces) >= 1:
            detect_times.append(get_face_locations_time)
        try:
            print('Number of faces detected: {} Average time to detect faces: {}'.format(str(len(faces)), sum(detect_times)/len(detect_times)))
        except ZeroDivisionError:
            pass
        
        for x, y, w, h in faces:

            roi_gray = gray_img[y:y + h, x:x + w] 
            cropped_img = cv2.resize(roi_gray, params['img_size'])
            prediction = get_prediction(model, cropped_img, params)
            print(params['target_classes_dict'][int(np.argmax(prediction))])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
            cv2.putText(frame, params['target_classes_dict'][int(np.argmax(prediction))], (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
        fer_time = time.time()-t
        if len(faces) >= 1:
            fer_times.append(fer_time)
        try:
            print('Average FER time: ', sum(fer_times)/len(fer_times))
        except ZeroDivisionError:
            pass
        
        cv2.imshow('Real-time edge-FER', frame)  
        total_time = time.time()-start_time
        if len(faces) >= 1:
            total_times.append(total_time)
        try:
            print('Total time:', sum(total_times)/len(total_times))
        except ZeroDivisionError:
            pass
        print('FER Count: ', str(len(total_times)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    


def main():
    """
    Option 1: Run real-time FER
    Option 2: Evaluate model/interpretor on validation images

    Returns
    -------
    None.

    """
    
    params = dict()
    params['img_size'] = (48,48)  
    params['dataset'] = 'FER_2013+'  
    params['batch_size'] = 64
    params['detection_method'] = 'HAAR'
    
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
    
    'Parse sys args'
    options_dict = {1: 'Real-time edge-FER', 2: 'Evaluate model on edge device'}
    ap = argparse.ArgumentParser()
    ap.add_argument("--option",help="1: {}. 2: {}".format(options_dict[1], options_dict[2]),
                    action='store', type=str, required=False, default='2')
    option = int(ap.parse_args().option)
    
    'Load model'
    model_interpretor = tf.keras.models.load_model('../trained_models/model1_struc_pruned_structured_pruning_100.h5')
    # model_interpretor = tf.lite.Interpreter(model_path='../trained_models/model1_2_4_struct_pruning_float16_quant.tflite')
    
    if params['dataset'] == 'CK+':
        params['data_path'] = '../data/CK+_Complete'
    elif params['dataset']  == 'FER_2013':
        params['data_path']  = '../data/fer_2013'
    elif params['dataset']  == 'FER_2013+':
        params['data_path']  = '../data/fer_2013+'
    elif params['dataset']  == 'JAFFE':
        params['data_path']  = '../data/prepped_jaffe'
        
    params['t_v_count'] = get_train_val_count(params['data_path'] , params['dataset'])
    
    if option == 2:
        'Get data image data generators'
        if params['dataset']  == 'FER_2013' or params['dataset']  == 'FER_2013+':
            Train_data_gen, Test_data_gen = get_pre_split_data_gens(data_path=params['data_path'] , batch_size=params['batch_size'], img_size=params['img_size']) 
        elif params['dataset']  == 'CK+' or params['dataset']  == 'JAFFE':
            Train_data_gen, Test_data_gen = get_non_split_data_gens(data_path=params['data_path'] , batch_size=params['batch_size'], img_size=params['img_size'] ) 
            
    if option == 1:
        'Option 1: (Default) real-time edge-FER'
        real_time_edge_fer(model_interpretor, params)
    elif option == 2:
        'Option 2: Evaluate Validation generator on edge device.'
        evaluate_model(model_interpretor, params, Test_data_gen)
    
    
    




if __name__ == '__main__':
    main()


