
Facial Emotion Recognition at the Edge
---------------------------------------------

High level ML architecture is as follows

![alt text](https://github.com/quinns1/FER-at-the-Edge/blob/main/images/ML_Architecture.png?raw=true)

The goal of this project was to evaluate various permuations of model compression techniques enabling FER at the edge.
Models were evaluated in under model size, top-1 accuracy and latency. Results show 2 by 4 structured pruning
combined with Float16 quantization yield the best reduction in model size and latency with minimal impact to accuracy.


Data
---------------------------------------------
Models were trained with FER-2013+ dataset. CK+, JAFFE & FER-2013 are also supported for evaluation. 
Ensure relevant dataset is available in ../data/ directory.


Directions
---------------------------------------------

To evaluate various permutations of model compression techniques run on edge_fer.py with the following option flags
* 1: Train model (defined in fer_models.py).
* 2: Evaluate quantization (post training and quantization aware training).
* 3: Evaluate pruning (alternate sparsity, 2/4 structured sparsity, sparsity ramp & pruning w/ quantization).

To run on edge device, clone repo and run deployed_edge_fer.py with the follwoing option flags:
* 1: Run on edge-FER model.
* 2: Evaluate model performance on validation images.

