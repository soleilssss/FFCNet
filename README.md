# FFCNet
FFCNet: Fourier Transform-Based Frequency Learning and Complex Convolutional Network for Colon Disease Classification

Our paper has been accepted by MICCAI 2022.

## Prerequisites
Our code is based on python3.6 and pytorch1.1.

## Training the networks 

python train_test.py 

train_dataset-root: Folder to which you downloaded and extracted the training data

val_datapath-root: Folder to which you downloaded and extracted the val data

record_path: The path where the training results are stored

model_path = The path where the model is stored

best_path = The path where the model with the best result on the validation set is stored

First go into the `train_test` and adapt all the paths to match your file system and the download locations of training and test sets.

Then python train_test.py to train your dataset.

## Citation

If you find the code useful for your research, please cite our paper.

Wang, Kai-Ni, et al. "Ffcnet: Fourier transform-based frequency learning and complex convolutional network for colon disease classification." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2022.
