# Touch Dynamics for Imposter Detection

This repo contains the code and data for a college research project that studied how users interact with touchscreen devices to detect imposters. The focus was on analyzing touch behavior such as swipe speed, tap pressure, and motion consistency to classify users using machine learning models.

---

## Project Summary

We collected touchscreen input from 40 users. Each participant interacted with an Android phone for ten minutes while playing a set of simple games. The raw data included timestamped touch events, coordinates, and pressure values.

After preprocessing, we extracted behavioral features including:

- Gesture velocity, acceleration, and jerk  
- Angular velocity and directional stability  
- Pressure variation and timing profiles

These features were used to train classification models that distinguish between legitimate users and imposters. The highest performing model reached 93 percent accuracy.

---

## Machine Learning Models

Three models were trained on the processed feature data:

| Model                  | Accuracy | Description                         |
|-----------------------|----------|-------------------------------------|
| Neural Network         | 93%      | Best overall performance            |
| XGBoost                | 89%      | Fast training and balanced results  |
| Support Vector Machine | 88%      | High precision with some tradeoffs  |

Each model was evaluated using accuracy, F1 score, ROC-AUC, and confusion matrix metrics.

---

## Repo Structure

```
├── src/
│   ├── preprocess.py        # Feature extraction from raw touch logs
│   ├── nn.py                # Neural network training script
│   ├── xgb.py               # XGBoost training script
│   ├── svc.py               # Support vector machine script
├── data/
│   ├── raw_data.zip         # Collected data from 40 users
│   ├── example_extracted_data.csv  # Sample of processed features
├── requirements.txt
├── README.md
```

---

## Research Summary

Touchscreen behavior can be used to distinguish between users with high accuracy. This project showed that behavioral features derived from basic touch interactions can serve as useful inputs for classification models. The neural network performed best, achieving 93 percent accuracy in identifying whether a session belonged to a legitimate user or an imposter. These findings support the idea that touch dynamics could serve as a lightweight behavioral biometric for device security.

