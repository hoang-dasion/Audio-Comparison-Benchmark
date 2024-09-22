Direction:
Training FOR EACH DISEASE
1. We used Stochastic Gradient Descent to optimize for hyperparameters. All the training are enclosed in model_trainer.py
2. The data we trained on are audio_features.csv, nlp_features.csv, and graph_features.csv. We selected the best ML model by this algorithm:
    1. For each single feature and combination of features (audio + nlp, graph + nlp, etc.):
        1. Run all the model models and trained and optimized hyperparameteres to make their best prediction (used Cross Validation then take the mean)
        2. Only filter the model having > 80% test accuracy and from that, pick the one having the minimum difference of test and train accuracy
    2. Now for each ml algorithm filtered from previous step:
        1. Run it across each single file and combination of files to store the test accuracy
    3. After Step 2, we have weighted_accuracies for each feature and combination of features. This represents how the best model in each feature adapt in different feature spaces.
Prediction FOR EACH DISEASE
1. For each file
    1. For each feature and combination of features:
        1. Used the best model from weighted_accuracies with their weights across all features from step 3 in training
        2. Make weighted_prediction = sum of (individual model prediction: 1 or 0) * (model weight in weighted_accuracies) and stored in ensemble_predictions. This represents ensemble voting where we say:
        given the best model founded in the current feature, what does it predict in its best feature (current feature -- current weight) and does it predict the same given different features (other weights)
        3. Find the best weighted_prediction
    2. Given the best weighted_prediction in previous step 1.3, if it passes the given threshold (current 0.6), it means the best working model in this feature space have made the decision in consideration of all possible feature spaces,
        hence we mark "YES". Else, we mark "NO".