# Utility-Functions-for-ML-Classification-Evaluation

- plot_roc(y, probs): 

  given ground truth and probabilities, plots the ROC curve.

- search_probability_threshold(y, probs):

  given ground truth and probabilities, searchs for the threshold that maximizes the desired metric. (currently F1_score)
  
- all_metrics_together(y, probs, preds):

  given ground truth, probabilites an predictions, sums up classification metrics accuracy, recall, precision, f1_score and auc in a dataframe.
