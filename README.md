# Deep Learning with LSTM and AutoEncoder architectures

### set_train_test:
Creates batches of train and test data for stateful sequential modelling.
Contains lots of utilities, returning ordinal index, timestamp, scaler used, etc.

### create_ar_lstm_model:
Creates stateful AutoRegressive LSTM model that predicts 1 time-step ahead.

### create_lstm_ae_model:
Creates LSTM AutoEncoder.

### create_lstm_vae_model:
Creates LSTM Variational AutoEncoder.

### create_ae_model:
Creates AutoEncoder model.

### error_to_probability:
Given a threshold and errors, converts the error, that is the difference between the prediction and ground-truth, into anomaly probability score.




# Classification-Evaluation

### plot_roc: 
Given ground truth and probabilities, plots the ROC curve.

### search_probability_threshold:
Given ground truth and probabilities, searchs for the threshold that maximizes the desired metric. (currently F1_score)
  
### all_metrics_together:
Given ground truth, probabilites an predictions, sums up classification metrics accuracy, recall, precision, f1_score and auc in a dataframe.
