# LSTM Model for Intrusion Detection

## **Introduction**
This project implements an LSTM (Long Short-Term Memory) model for detecting network intrusions using the CICIDS2017 dataset. LSTM networks are a type of recurrent neural network (RNN) well-suited for sequential data and time-series analysis. The primary goal of this project is to classify network traffic as either benign or indicative of an intrusion.

## **Preprocessing Steps**
To ensure the dataset is in a suitable format for training the LSTM model, the following preprocessing steps were performed:

### 1. **Loading and Cleaning the Data**
- The dataset was loaded using `pandas` and inspected for missing values and unnecessary columns.
- Infinite values were replaced with `NaN` using:
  ```python
  df.replace([np.inf, -np.inf], np.nan, inplace=True)
  ```
- Rows with `NaN` values were dropped:
  ```python
  df.dropna(inplace=True)
  ```

### 2. **Filtering Relevant Labels**
- Traffic labeled as `DoS` and `BENIGN` was filtered, excluding specific subcategories such as `DoS Slowhttptest`. The final labels were categorized into benign (0) and various intrusion types (1).
- Labels were simplified for multi-class attacks:
  ```python
  df['Label'] = df['Label'].replace(r'.*Patator$', "Brute Force", regex=True)
  ```

### 3. **Feature Scaling**
- Numerical features were scaled to the range [0, 1] using a `MinMaxScaler`:
  ```python
  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  ```

### 4. **Reshaping for LSTM**
- The scaled dataset was reshaped to match the LSTM's expected input format (3D tensor):
  ```python
  X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
  X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
  ```

## **LSTM Model Architecture**
The model was built using Keras with TensorFlow as the backend. The architecture consists of:

- **Input Layer:** Accepts time-series data with dimensions `(timesteps=1, features=78)`.
- **LSTM Layer:** 50 units for sequential pattern recognition.
- **Dense Output Layer:** A single neuron with a sigmoid activation function for binary classification.

### **Model Code:**
```python
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### **Training**
The model was trained for 5 epochs using a batch size of 32, with early stopping to prevent overfitting:
```python
es = tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss", restore_best_weights=True)

history = lstm_model.fit(X_train_lstm, y_train, epochs=5, validation_data=(X_test_lstm, y_test), batch_size=32, callbacks=[es])
```
### **Loss Curves**
<img src="https://github.com/leovidith/LSTM-Intrusion-Detection/blob/main/images/image.png" width="600px">

## **Results**
- **Accuracy:** 99.62%
- **Loss:** 0.0144

The model achieved high accuracy, demonstrating its ability to effectively classify network traffic as benign or malicious. The low loss value further supports the model's confidence in its predictions.

## **Conclusion**
The LSTM model proved to be a robust solution for intrusion detection in network traffic, achieving a near-perfect accuracy score. However, further steps could enhance the model, such as:

1. Testing on additional datasets for better generalization.
2. Expanding the feature set to include temporal and contextual data.
3. Exploring advanced architectures like Bidirectional LSTMs or attention mechanisms.

This implementation highlights the power of deep learning for cybersecurity applications, paving the way for automated and scalable intrusion detection systems.
