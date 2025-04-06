# Prediction-of-Company-Bankruptcies-Using-Machine-Learning-Models
This project involves identifying companies at risk of bankruptcy using machine learning techniques, based on a dataset of 10,716 companies. Each company is described by eight performance indicators, three binary indicators, and its bankruptcy status (healthy or bankrupt). The objective is to construct a model that can effectively classify both healthy and bankrupt companies. Various models were trained and evaluated on both balanced and imbalanced datasets, including Linear Discriminant Analysis, Logistic Regression, Decision Trees, Random Forest, k-Nearest Neighbors, Naïve Bayes, Support Vector Machines, and Gradient Boosting (XGBoost). Performance was optimized through hyperparameter tuning and the use of balanced datasets for better accuracy in predicting bankrupt companies.

Key Achievements:
- Successfully trained and evaluated eight machine learning models to predict bankruptcy: Linear Discriminant Analysis, Logistic Regression, Decision Trees, Random Forest, k-Nearest Neighbors, Naïve Bayes, Support Vector Machines, and Gradient Boosting (XGBoost).
- Employed both balanced and imbalanced datasets and utilized techniques like under-sampling to improve classification accuracy.
- The Random Forest and k-Nearest Neighbors models demonstrated the highest performance, particularly when trained with balanced data.
- Identified optimal hyperparameters for each model using techniques such as RandomSearchCV to achieve improved Recall scores, crucial for identifying bankrupt companies.
- Established that Random Forest and k-Nearest Neighbors models provided superior performance when evaluated on balanced datasets.
