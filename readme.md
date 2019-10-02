# CAPSULE NETWORK FOR HSI SEGMENTATION

This is the code for hyper-spectral image segmentation using capsule network.

## Files in this project

- `caps_model.py` contains the model we used.
- `cl_caps_train.py` is the major file, including the processes of training the model, testing the model and calculate 
the probability map. 
- `HSI_Data_Preparation.py` and `utils.py` process the original HSI data, slice it into patches, label the patches, and 
return the training and test dataset.
- `capslayer` is a package contains necessary modules for the capsNet models.
- `Data` directory stores the original HSI data we use.

## Running Environment

- `python` I use 3.5, but 3.6 or 3.7 should be OK. 3.4 or lower python 3 is not recommended because i don't know 
whether they will cause errors or not. Python 2 isn't compatible.
- `tensorflow` 1.13.1 is recommended but other versions should be OK.
- `numpy` I use 1.16.4, other versions should be OK.
- `pygco` Use the latest version is OK. The team(or person) stopped upgrade the library. I have some problem 
installing it into Windows. So I recommend you to install it into a Linux-based operating system. 
- `scikit-learn` 0.21.2 is recommended but other versions should be OK.
- `scikit-image` 0.15.0 is recommended but other versions should be OK.
- `pandas` 0.24.2 is recommended but other versions should be OK.
- `pillow` 6.1.0 is recommended but other versions should be OK.

## Run the code

1. Download the code from [github](https://github.com/osmium18452/hsi_caps).
2. Run the code with default parameters using `python cl_caps_hsi.py`. Or use `python cl_caps_hsi.py -h` for more 
options.

## Issues and solution

1. There's something abnormal with the data loader. Remember that the dimension of the dataset it returns is 
`(batch_size, channels, patch_size, patch_size)`. But the data dimension the model needs is `(batch_size, patch_size,
patch_size, channels)`. So you have to transpose the dimension to make sure that the dimension of the input data is
correspond to the model's input layer. 
