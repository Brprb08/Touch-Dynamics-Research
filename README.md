# Touch Dynamics Research for Imposter Detection

## Overview

This repository contains code and data used in a research project conducted as part of a college research program. The focus of the research is to analyze **touch dynamics**—the way users interact with a touchscreen device (e.g., smartphones)—to detect imposters. 

### Key Objectives:
1. **Data Collection**: We gathered data from 40 participants playing three different games on an Android phone for 10 minutes each. This raw data includes touch events such as coordinates, time, and pressure.
   
2. **Data Preprocessing**: The raw touch data was cleaned and transformed into meaningful features such as speed, acceleration, jerk, angular velocity, and more.

3. **Model Training**: Using the extracted features, several machine learning algorithms (Neural Network, XGBoost, and Support Vector Machine) were trained to distinguish between legitimate users and imposters.

4. **Performance**: Our models achieved an average of 90% accuracy in detecting imposters, with the Neural Network achieving the highest accuracy at 93%.

## Repository Contents

### Data
- **Raw Data**: The `raw_data.zip` file contains the unprocessed data collected from 40 users.
- **Example Data**: The `example_extracted_data.csv` is a shortened version of the extracted and processed data used for model training. It contains normalized data with a balanced number of rows for each user.
- Due to space limitations, the full preprocessed data is not included here but can be generated using the scripts provided.

### Code
All code related to preprocessing, feature extraction, and machine learning model training is stored in the `src` directory. The following scripts are included:

- **`preprocess.py`**: This script handles the extraction of features from raw touch data.
- **`nn.py`**: Implementation of a Neural Network for user classification.
- **`xgb.py`**: XGBoost algorithm for imposter detection.
- **`svc.py`**: Support Vector Classifier for user classification.

### Algorithms
Three different machine learning models were used to classify users and detect imposters:

1. **Neural Network (`nn.py`)**: Achieved the highest accuracy of 93%.
2. **XGBoost (`xgb.py`)**: Gradient Boosting algorithm that balances performance and speed, with an accuracy of 89%.
3. **Support Vector Classifier (`svc.py`)**: A classification method based on SVM that achieved an accuracy of 88%.

## Instructions

### Requirements

To run the preprocessing and machine learning models, you'll need the following Python packages:
- `pandas`
- `numpy`
- `tensorflow` (for Neural Networks)
- `xgboost`
- `sklearn`
- `matplotlib`

You can install these packages using the following command:
```bash
pip install -r requirements.txt
```

### How to Use

1. **Data Preprocessing**: The raw data must first be converted into usable features. Run the `preprocess.py` script to process the raw data and generate feature files for each user.

```bash
python preprocess.py
```
This script will calculate features such as speed, acceleration, jerk, and angular velocity, and save the processed data to CSV files for each user.

2. **Training a Model**: To train one of the machine learning models (Neural Network, XGBoost, or SVM), use the respective script (`nn.py`, `xgb.py`, or `svc.py`). These scripts will split the data into training and testing sets, train the model, and output performance metrics like accuracy, precision, recall, and ROC-AUC scores.

```bash
python nn.py
```

3. **Custom User Identification**: If you want to test the model’s ability to recognize or reject a specific user, modify the `user_id` variable in the respective scripts and run the model training.

4. **Performance Evaluation**: Each model will output key metrics such as accuracy, F1 score, False Positive Rate (FPR), False Negative Rate (FNR), and confusion matrices. These metrics are useful in evaluating the effectiveness of the models.

## Research Summary

Touch dynamics can provide a lot of information for user authentication. Our study showed that the way users interact with a touchscreen (swiping, tapping, and pressing) can effectively distinguish between legitimate users and imposters. Our research utilized three machine learning models, and the Neural Network performed best with a 93% accuracy. These findings suggest that touch dynamics could be a useful secondary authentication method for securing mobile devices.
