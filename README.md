# Touch-Dynamics-Research

This repository contains the raw data files and an example file of an extracted file. 
The raw data files and example file are in their respective zip files. 
The reason this only contains the raw data and not the extracted data is due to the restrictions on space in the GitHub repository. 
For this reason we had to restrict the data here to the raw data to fit them in. 

However, the example file is a much shortened version of a real extracted file from all users. 
It takes each users data and appends them into the same file. 
It then goes through and makes it so that each user has the exact same number of rows. 
This way no user is being trained more than others. 

The src file contains all of the code used for training, preprocessing, and sorting. 
The files for the algorithms are x.py for the XGBoost, nn.py for the Neural Network, and s.py for the SVC. 
The preprocess.py is what is used to extract the features from the raw data.

