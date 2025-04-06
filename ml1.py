# -*- coding: utf-8 -*-
"""
# **Importing necessary libraries**
"""

# IMPORTING BASIC LIBRARIES
import pandas as pd
import numpy as np
# IMPORTING LIBRARIES FOR DIAGRAMS
import matplotlib.pyplot as plt
import seaborn as sns
import string
# IMPORTING LIBRARIES FOR CONNECTING AND READING FROM GOOGLE DRIVE
from google.colab import drive
from google.colab import files
# IMPORTING LIBRARIES FOR PREPROCESSING
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
# IMPORT LIBRARIES FOR HYPERPARAMETER TUNING
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
# IMPORT LIBRARIES FOR GETTING MODEL SCORES
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
# IMPORT LIBRARIES FOR MODEL BUILDING
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

"""# **Connecting to Google Drive and reading from it**"""

# CONNECTING WITH GOOGLE DRIVE
drive.mount('/content/drive')
# READING FROM GOOGLE DRIVE
file_path = '/content/drive/My Drive/Dataset2Use_Assignment1.xlsx'
df = pd.read_excel(file_path)

"""# **Data Exploration and Visualization**"""

# FIGURE 1: NUMBER OF HEALTHLY AND BANKRUPT COMPANIES EACH YEAR
healthly = df[df['ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)'] == 1].groupby('ΕΤΟΣ').size()
healthly = healthly.reset_index()
unhealthly = df[df['ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)'] == 2].groupby('ΕΤΟΣ').size()
unhealthly = unhealthly.reset_index()
result_df = pd.merge(healthly, unhealthly, on='ΕΤΟΣ')
result_df.columns = ['ΕΤΟΣ', 'ΥΓΙΗΣ', 'ΧΡΕΟΚΟΠΗΜΕΝΕΣ']
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = result_df['ΕΤΟΣ']
bar1 = plt.bar(index - bar_width/2, result_df['ΥΓΙΗΣ'], bar_width, label='ΥΓΙΗΣ')
bar2 = plt.bar(index + bar_width/2, result_df['ΧΡΕΟΚΟΠΗΜΕΝΕΣ'], bar_width, label='ΧΡΕΟΚΟΠΗΜΕΝΕΣ')
plt.ylim(0, 3500)
plt.xlabel('ΈΤΟΣ')
plt.ylabel('ΠΛΗΘΟΣ')
plt.title('Υγιείς και Χρεοκοπημένες επιχειρήσεις, ανά έτος')
plt.xticks(index, result_df['ΕΤΟΣ'])
for i, value in enumerate(result_df['ΥΓΙΗΣ']):
  plt.text(index[i] - bar_width/2, value + 50, str(value), ha='center', va='bottom')
for i, value in enumerate(result_df['ΧΡΕΟΚΟΠΗΜΕΝΕΣ']):
  plt.text(index[i] + bar_width/2, value + 50, str(value), ha='center', va='bottom')
plt.legend()
plt.show()

# FIGURE 2.1
healthly_companies = df[df['ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)'] == 1]
selected_columns = healthly_companies.iloc[:, :8]
min_values = selected_columns.min()
max_values = selected_columns.max()
avg_values = selected_columns.mean()
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(16, 10))
ax1.errorbar(selected_columns.columns, avg_values, yerr=[avg_values - min_values, max_values - avg_values], fmt='o', color='black', capsize=5)
ax2.errorbar(selected_columns.columns, avg_values, yerr=[avg_values - min_values, max_values - avg_values], fmt='o', color='black', capsize=5)
ax3.errorbar(selected_columns.columns, avg_values, yerr=[avg_values - min_values, max_values - avg_values], fmt='o', color='black', capsize=5)
ax4.errorbar(selected_columns.columns, avg_values, yerr=[avg_values - min_values, max_values - avg_values], fmt='o', color='black', capsize=5)
ax1.plot(selected_columns.columns, avg_values, marker='s', color='red', markersize=8, linestyle='None')
ax2.plot(selected_columns.columns, avg_values, marker='s', color='red', markersize=8, linestyle='None')
ax3.plot(selected_columns.columns, avg_values, marker='s', color='red', markersize=8, linestyle='None')
ax4.plot(selected_columns.columns, avg_values, marker='s', color='red', markersize=8, linestyle='None')
for col, min_val, max_val in zip(selected_columns.columns, min_values, max_values):
  ax1.plot(col, min_val, marker='o', color='green', markersize=8, linestyle='None')
  ax1.plot(col, max_val, marker='o', color='blue', markersize=8, linestyle='None')
  ax2.plot(col, min_val, marker='o', color='green', markersize=8, linestyle='None')
  ax2.plot(col, max_val, marker='o', color='blue', markersize=8, linestyle='None')
  ax3.plot(col, min_val, marker='o', color='green', markersize=8, linestyle='None')
  ax3.plot(col, max_val, marker='o', color='blue', markersize=8, linestyle='None')
  ax4.plot(col, min_val, marker='o', color='green', markersize=8, linestyle='None')
  ax4.plot(col, max_val, marker='o', color='blue', markersize=8, linestyle='None')
ax1.set_ylim(100, 1600)
ax2.set_ylim(15, 50)
ax3.set_ylim(1.5, 5.5)
ax4.set_ylim(-1, 1.2)
ax1.legend(['Mean', 'Min', 'Max'])
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)
ax3.spines.top.set_visible(False)
ax3.spines.bottom.set_visible(False)
ax4.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax3.xaxis.tick_bottom()
ax1.tick_params(top=False)
ax2.tick_params(top=False, bottom=False)
ax3.tick_params(top=False, bottom=False)
ax4.tick_params(bottom=False)
d = .5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
ax3.plot([0, 1], [0, 0], transform=ax3.transAxes, **kwargs)
ax4.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
alphabet_labels = list(string.ascii_uppercase)[:len(selected_columns.columns)]
plt.xticks(selected_columns.columns, alphabet_labels)
for ax in [ax1, ax2, ax3, ax4]:
  ax.set_xticks(selected_columns.columns)  # Add this line
  ax.set_xticklabels(alphabet_labels, ha='right', fontsize=10)
  ax.set_xticks(selected_columns.columns)
  ax.set_xticklabels(alphabet_labels, ha='right', fontsize=10)
plt.suptitle('Min Max Mean Values of Performance Indices, Healthy Companies', y=0.9)
plt.show()

# FIGURE 2.2
unhealthly_companies = df[df['ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)'] == 2]
selected_columns = unhealthly_companies.iloc[:, :8]
min_values = selected_columns.min()
max_values = selected_columns.max()
avg_values = selected_columns.mean()
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(16, 10))
ax1.errorbar(selected_columns.columns, avg_values, yerr=[avg_values - min_values, max_values - avg_values], fmt='o', color='black', capsize=5)
ax2.errorbar(selected_columns.columns, avg_values, yerr=[avg_values - min_values, max_values - avg_values], fmt='o', color='black', capsize=5)
ax3.errorbar(selected_columns.columns, avg_values, yerr=[avg_values - min_values, max_values - avg_values], fmt='o', color='black', capsize=5)
ax4.errorbar(selected_columns.columns, avg_values, yerr=[avg_values - min_values, max_values - avg_values], fmt='o', color='black', capsize=5)
ax1.plot(selected_columns.columns, avg_values, marker='s', color='red', markersize=8, linestyle='None')
ax2.plot(selected_columns.columns, avg_values, marker='s', color='red', markersize=8, linestyle='None')
ax3.plot(selected_columns.columns, avg_values, marker='s', color='red', markersize=8, linestyle='None')
ax4.plot(selected_columns.columns, avg_values, marker='s', color='red', markersize=8, linestyle='None')
for col, min_val, max_val in zip(selected_columns.columns, min_values, max_values):
  ax1.plot(col, min_val, marker='o', color='green', markersize=8, linestyle='None')
  ax1.plot(col, max_val, marker='o', color='blue', markersize=8, linestyle='None')
  ax2.plot(col, min_val, marker='o', color='green', markersize=8, linestyle='None')
  ax2.plot(col, max_val, marker='o', color='blue', markersize=8, linestyle='None')
  ax3.plot(col, min_val, marker='o', color='green', markersize=8, linestyle='None')
  ax3.plot(col, max_val, marker='o', color='blue', markersize=8, linestyle='None')
  ax4.plot(col, min_val, marker='o', color='green', markersize=8, linestyle='None')
  ax4.plot(col, max_val, marker='o', color='blue', markersize=8, linestyle='None')
ax1.set_ylim(100, 1600)
ax2.set_ylim(15, 50)
ax3.set_ylim(1.5, 5.5)
ax4.set_ylim(-1, 1.2)
ax1.legend(['Mean', 'Min', 'Max'])
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)
ax3.spines.top.set_visible(False)
ax3.spines.bottom.set_visible(False)
ax4.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax3.xaxis.tick_bottom()
ax1.tick_params(top=False)
ax2.tick_params(top=False, bottom=False)
ax3.tick_params(top=False, bottom=False)
ax4.tick_params(bottom=False)
d = .5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
ax3.plot([0, 1], [0, 0], transform=ax3.transAxes, **kwargs)
ax4.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
alphabet_labels = list(string.ascii_uppercase)[:len(selected_columns.columns)]
plt.xticks(selected_columns.columns, alphabet_labels)
for ax in [ax1, ax2, ax3, ax4]:
  ax.set_xticks(selected_columns.columns)  # Add this line
  ax.set_xticklabels(alphabet_labels, ha='right', fontsize=10)
  ax.set_xticks(selected_columns.columns)
  ax.set_xticklabels(alphabet_labels, ha='right', fontsize=10)
plt.suptitle('Min Max Mean Values of Performance Indices, Unhealthly Companies', y=0.9)
plt.show()

# CHECKING FOR ANY RECORDS WITH NULL VALUES
missing_data = df.isnull().sum()
print("-" * 150)
if missing_data.any():
  print("Υπάρχουν ελλείποντα δεδομένα στο DataFrame. Οι στήλες με ελλείποντα δεδομένα είναι:")
  print(missing_data[missing_data > 0])
else:
  print("Δεν υπάρχουν ελλείποντα δεδομένα στο DataFrame.")
print("-" * 150)

"""# **Data Preparation**"""

# IMPLEMENTING MINI-MAX NORMALIZATION
columns_to_normalize = df.columns
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# IMPLEMENTING STRATIFIEDKFOLD WITH FOURS SPLITS
num_folds = 4
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
X = df.drop(columns=["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"])
y = df["ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)"]
inbl_main_sets = []
test_sets = []
print("-" * 150)
for fold_idx, (main_index, test_index) in enumerate(stratified_kfold.split(X, y)):
  X_main, X_test = X.iloc[main_index], X.iloc[test_index]
  y_main, y_test = y.iloc[main_index], y.iloc[test_index]
  print(f"Fold {fold_idx + 1}")
  print("- Main Set:")
  print("Υγιείς εταιρείες:", sum(y_main == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_main == 1))
  print()
  print("- Test Set:")
  print("Υγιείς εταιρείες:", sum(y_test == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_test == 1))
  print("-" * 150)
  inbl_main_sets.append((X_main, y_main))
  test_sets.append((X_test, y_test))

# IMPLEMENTING UNDER SAMPLING
bl_main_sets = []
print("-" * 150)
for fold_idx, (main_index, test_index) in enumerate(stratified_kfold.split(X, y)):
  X_main, X_test = X.iloc[main_index], X.iloc[test_index]
  y_main, y_test = y.iloc[main_index], y.iloc[test_index]
  print(f"Fold {fold_idx + 1}")
  print("- InBalanced Main Set:")
  print("Υγιείς εταιρείες:", sum(y_main == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_main == 1))
  while sum(y_main==0) > 3*sum(y_main==1):
    random_idx = np.random.choice(y_main[y_main == 0].index)
    X_main = X_main.drop(index=[random_idx])
    y_main = y_main.drop(index=[random_idx])
  print()
  print("- Balanced Main Set:")
  print("Υγιείς εταιρείες:", sum(y_main == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_main == 1))
  print()
  print("- Test Set:")
  print("Υγιείς εταιρείες:", sum(y_test == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_test == 1))
  print("-" * 150)
  bl_main_sets.append((X_main, y_main))

# SPLITTING MAIN SET TO VALIDATION AND TEST SET
bl_train_sets = bl_val_sets = []
inbl_train_sets = inbl_val_sets = []
print("-" * 150)
for fold_idx, ((X_inbl_main, y_inbl_main), (X_bl_main, y_bl_main), (X_test, y_test)) in enumerate(zip(inbl_main_sets, bl_main_sets, test_sets)):
  X_inbl_train, X_inbl_val, y_inbl_train, y_inbl_val = train_test_split(X_inbl_main, y_inbl_main, test_size=0.2, random_state=42, stratify=y_inbl_main)
  X_bl_train, X_bl_val, y_bl_train, y_bl_val = train_test_split(X_bl_main, y_bl_main, test_size=0.2, random_state=42, stratify=y_bl_main)
  print(f"Fold {fold_idx + 1}")
  print("- InBalanced Train Set:")
  print("Υγιείς εταιρείες:", sum(y_inbl_train == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_inbl_train == 1))
  print()
  print("- InBalanced Validation Set:")
  print("Υγιείς εταιρείες:", sum(y_inbl_val == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_inbl_val == 1))
  print()
  print("- Balanced Train Set:")
  print("Υγιείς εταιρείες:", sum(y_bl_train == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_bl_train == 1))
  print()
  print("- Balanced Validation Set:")
  print("Υγιείς εταιρείες:", sum(y_bl_val == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_bl_val == 1))
  print()
  print("- Test Set:")
  print("Υγιείς εταιρείες:", sum(y_test == 0))
  print("Χρεωκοπημένες εταιρείες:", sum(y_test == 1))
  print("-" * 150)
  inbl_train_sets.append((X_inbl_train, y_inbl_train))
  inbl_val_sets.append((X_inbl_val, y_inbl_val))
  bl_train_sets.append((X_bl_train, y_bl_train))
  bl_val_sets.append((X_bl_val, y_bl_val))

"""# **Custom functions**"""

# ADDING ROWS
def add_row(classifier_name, dataset_type, balance_type, num_training_samples, num_non_healthy_samples,
                   true_positives, true_negatives, false_positives, false_negatives, roc_auc):
  global df_outcomes
  new_row = {'Classifier Name': classifier_name,
             'Main or test set': dataset_type, 'Balanced or unbalanced main set': balance_type,
             'Number of main samples': num_training_samples, 'Number of non-healthy companies in main sample': num_non_healthy_samples,
             'True positives TP': true_positives, 'True negatives TN': true_negatives, 'False positives FP': false_positives, 'False negatives FN': false_negatives,
             'ROC-AUC': roc_auc}

  df_outcomes = pd.concat([df_outcomes, pd.DataFrame([new_row])], ignore_index=True)

# FIT AND TEST MODELS
def modeling(given_model, given_param_dist, given_name, balanced):
  model = given_model
  param_dist = given_param_dist
  if balanced:
    main_sets = bl_main_sets
    train_sets = bl_train_sets
    val_sets = bl_val_sets
    isBalanced = "Balanced"
    num_samples = 744
  else:
    main_sets = inbl_main_sets
    train_sets = inbl_train_sets
    val_sets = inbl_val_sets
    isBalanced = "InBalanced"
    num_samples = 8037
  if param_dist: # if our model takes hyperparameter, then
    scorer = make_scorer(recall_score)
    random_search = RandomizedSearchCV(
      model, param_distributions=param_dist, n_iter=10, scoring=scorer, cv=3, random_state=42, n_jobs=-1
    )
    max_recall = -100
    print("Searching for best hyperparameters ..")
    for fold_idx, ((X_train, y_train), (X_val, y_val), (X_main, y_main)) in enumerate(zip(train_sets, val_sets, main_sets)):
      random_search.fit(X_train, y_train)
      good_model = random_search.best_estimator_
      print(f"- For fold {fold_idx + 1}, based on train set, best hyperparamater was {random_search.best_params_}.")
      y_val_pred = good_model.predict(X_val)
      recall_val = recall_score(y_val, y_val_pred)
      if(recall_val > max_recall):
        max_recall = recall_val
        best_model = good_model
    print()
    print(f"Between them, based on validation set, the best hyperparamater was {best_model}.")
    print()
    print(f"So, we test {best_model}, on main and test set ..")
    print()
  else: # else, probably is Naive Bayes
    best_model = given_model
    for fold_idx, (X_main, y_main) in enumerate(main_sets):
      best_model.fit(X_main, y_main)
    print(f"Naive Bayes, do not take hyperparameters, so we just test him on main and test set ..")
    print()
  for fold_idx, ((X_main, y_main), (X_test, y_test)) in enumerate(zip(main_sets, test_sets)):
    y_main_pred = best_model.predict(X_main)
    cm_main = confusion_matrix(y_main, y_main_pred)
    y_test_pred = best_model.predict(X_test)
    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_main, annot=True, fmt="d", cmap="Blues", xticklabels=['Υγιείς', 'Χρεωκοπημένες'], yticklabels=['Υγιείς', 'Χρεωκοπημένες'])
    plt.title("Confusion Matrix - Main Set")
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=['Υγιείς', 'Χρεωκοπημένες'], yticklabels=['Υγιείς', 'Χρεωκοπημένες'])
    plt.title("Confusion Matrix - Test Set")
    plt.show()
    print(f"Fold {fold_idx + 1}")
    print("- Main set:")
    accuracy_main = accuracy_score(y_main, y_main_pred)
    precision_main = precision_score(y_main, y_main_pred)
    recall_main = recall_score(y_main, y_main_pred)
    f1_main = f1_score(y_main, y_main_pred)
    auc_roc_main = roc_auc_score(y_main, best_model.predict_proba(X_main)[:, 1])
    print(f"Accuracy: {accuracy_main:.2f}, Precision: {precision_main:.2f}, Recall: {recall_main:.2f}, F1 Score: {f1_main:.2f}, ROC-AUC: {auc_roc_main:.2f}")
    print("- Test set:")
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)
    auc_roc_test = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f"Accuracy: {accuracy_test:.2f}, Precision: {precision_test:.2f}, Recall: {recall_test:.2f}, F1 Score: {f1_test:.2f}, ROC-AUC: {auc_roc_test:.2f}")
    add_row(given_name, 'Main', isBalanced, num_samples, 186, cm_main[1][1], cm_main[0][0], cm_main[0][1], cm_main[1][0], auc_roc_main)
    add_row(given_name, 'Test', isBalanced, num_samples, 186, cm_test[1][1], cm_test[0][0], cm_test[0][1], cm_test[1][0], auc_roc_test)

"""# **Creating DataFrame for storage**"""

outcomes = {
  'Classifier Name': [],
  'Main or test set': [],
  'Balanced or unbalanced main set': [],
  'Number of main samples': [],
  'Number of non-healthy companies in main sample': [],
  'True positives TP': [],
  'True negatives TN': [],
  'False positives FP': [],
  'False negatives FN': [],
  'ROC-AUC': []
}
df_outcomes = pd.DataFrame(outcomes)

"""# **Linear Discriminant Analysis**"""

model = LinearDiscriminantAnalysis()
param_dist = {
  'solver': ['lsqr', 'eigen'],
  'shrinkage': [None, 'auto', 0.1, 0.5, 0.9],
}
name = 'Linear Discriminant Analysis'
balanced = False
modeling(model, param_dist, name, balanced)

model = LinearDiscriminantAnalysis()
param_dist = {
  'solver': ['lsqr', 'eigen'],
  'shrinkage': [None, 'auto', 0.1, 0.5, 0.9],
}
name = 'Linear Discriminant Analysis'
balanced = True
modeling(model, param_dist, name, balanced)

"""# **Logistic Regression**"""

model = LogisticRegression()
param_dist = {
  'C': [0.001, 0.01, 0.1, 1, 10, 100],
  'penalty': ['l1', 'l2'],
  'solver': ['liblinear', 'saga'],
}
name = 'Logistic Regression'
balanced = False
modeling(model, param_dist, name, balanced)

model = LogisticRegression()
param_dist = {
  'C': [0.001, 0.01, 0.1, 1, 10, 100],
  'penalty': ['l1', 'l2'],
  'solver': ['liblinear', 'saga'],
}
name = 'Logistic Regression'
balanced = True
modeling(model, param_dist, name, balanced)

"""# **Decison Trees**"""

model = DecisionTreeClassifier()
param_dist = {
  'criterion': ['gini', 'entropy'],
  'splitter': ['best', 'random'],
  'max_depth': [None, 10, 20, 30, 40, 50],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4],
  'max_features': ['sqrt', 'log2', None],
}
name = 'Decison Trees'
balanced = False
modeling(model, param_dist, name, balanced)

model = DecisionTreeClassifier()
param_dist = {
  'criterion': ['gini', 'entropy'],
  'splitter': ['best', 'random'],
  'max_depth': [None, 10, 20, 30, 40, 50],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4],
  'max_features': ['sqrt', 'log2', None],
}
name = 'Decison Trees'
balanced = True
modeling(model, param_dist, name, balanced)

"""# **Random Forest**"""

model = RandomForestClassifier()
param_dist = {
  'n_estimators': [50, 100, 200],
  'max_depth': [None, 10, 20, 30],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4]
}
name = 'Random Forest'
balanced = False
modeling(model, param_dist, name, balanced)

model = RandomForestClassifier()
param_dist = {
  'n_estimators': [50, 100, 200],
  'max_depth': [None, 10, 20, 30],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4]
}
name = 'Random Forest'
balanced = True
modeling(model, param_dist, name, balanced)

"""# **k-Nearest Neighbors**"""

model = KNeighborsClassifier()
param_dist = {
  'n_neighbors': [3, 5, 7, 10],
  'weights': ['uniform', 'distance'],
  'p': [1, 2]
}
name = 'k-Nearest Neighbors'
balanced = False
modeling(model, param_dist, name, balanced)

model = KNeighborsClassifier()
param_dist = {
  'n_neighbors': [3, 5, 7, 10],
  'weights': ['uniform', 'distance'],
  'p': [1, 2]
}
name = 'k-Nearest Neighbors'
balanced = True
modeling(model, param_dist, name, balanced)

"""# **Naive Bayes**"""

model = GaussianNB()
param_dist = {}
name = 'Naive Bayes'
balanced = False
modeling(model, param_dist, name, balanced)

model = GaussianNB()
param_dist = {}
name = 'Naive Bayes'
balanced = True
modeling(model, param_dist, name, balanced)

"""# **Support Vector Machines**"""

model = SVC(probability=True)
param_dist = {
  'C': [0.1, 1, 10, 100],
  'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
  'degree': [2, 3, 4],
  'gamma': ['scale', 'auto'],
}
name = 'Support Vector Machines'
balanced = False
modeling(model, param_dist, name, balanced)

model = SVC(probability=True)
param_dist = {
  'C': [0.1, 1, 10, 100],
  'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
  'degree': [2, 3, 4],
  'gamma': ['scale', 'auto'],
}
name = 'Support Vector Machines'
balanced = True
modeling(model, param_dist, name, balanced)

"""# **Gradient Boosting (XGBoost)**"""

model = XGBClassifier()
param_dist = {
  'n_estimators': [50, 100, 200],
  'learning_rate': [0.01, 0.1, 0.2],
  'max_depth': [3, 4, 5],
  'min_child_weight': [1, 2, 3],
  'subsample': [0.8, 0.9, 1.0],
  'colsample_bytree': [0.8, 0.9, 1.0],
  'gamma': [0, 1, 5],
  'objective': ['binary:logistic'],
}
name = 'Gradient Boosting (XGBoost)'
balanced = False
modeling(model, param_dist, name, balanced)

model = XGBClassifier()
param_dist = {
  'n_estimators': [50, 100, 200],
  'learning_rate': [0.01, 0.1, 0.2],
  'max_depth': [3, 4, 5],
  'min_child_weight': [1, 2, 3],
  'subsample': [0.8, 0.9, 1.0],
  'colsample_bytree': [0.8, 0.9, 1.0],
  'gamma': [0, 1, 5],
  'objective': ['binary:logistic'],
}
name = 'Gradient Boosting (XGBoost)'
balanced = True
modeling(model, param_dist, name, balanced)

"""# **Converting DataFrame to .xls**"""

# file_path = '/content/drive/My Drive/balancedDataOutcomes.xlsx'
file_path = '/content/drive/My Drive/test3_balancedDataOutcomes.xlsx'
df_outcomes.to_excel(file_path, index=False)