## Overview
This project implements an LSTM (Long Short-Term Memory) model for detecting network intrusions using the CICIDS2017 dataset. The model classifies network traffic as either benign or indicative of an intrusion, leveraging the sequential nature of network data. The primary aim is to achieve high classification accuracy using LSTM's ability to capture long-term dependencies in time-series data.

## Results
The LSTM model performed excellently with the following results:

- **Accuracy:** 99.62%
- **Loss:** 0.0144

These results demonstrate the model’s ability to effectively distinguish between benign and malicious network traffic with high accuracy and confidence.

### Loss Curves:
<img src="https://github.com/leovidith/LSTM-Intrusion-Detection/blob/main/images/image.png" width="600px">

## Features
- **Traffic Attributes:** Flow duration, packet lengths, header lengths, and flag counts.
- **Preprocessing:** Missing and infinite values were handled, and data was scaled using MinMaxScaler.
- **Labeling:** Traffic labels were simplified into two categories: benign and various types of intrusions.

## Sprints
1. **Data Preprocessing:**
   - Cleaned and transformed the dataset, replacing infinite values and handling missing data.
   - Scaled numerical features and reshaped the dataset for LSTM input.
   
2. **Model Architecture:**
   - Built an LSTM model with a single LSTM layer and a dense output layer for binary classification.
   - Used early stopping to prevent overfitting.

3. **Training and Evaluation:**
   - The model was trained on the processed data and evaluated for its performance in terms of accuracy and loss.

## Conclusion
The LSTM model successfully detects intrusions in network traffic with an impressive accuracy of 99.62%. While the model is already highly effective, future improvements may include testing on additional datasets, incorporating more advanced architectures, and expanding the feature set for better generalization and robustness. This project showcases the potential of deep learning in cybersecurity, providing a scalable and effective intrusion detection system.
