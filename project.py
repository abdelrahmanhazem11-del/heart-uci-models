#import necessary libraries
import os
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score , accuracy_score, f1_score, precision_score, recall_score , roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GridSearchCV
import pickle
# Download latest version
path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
print(path)
# Load the dataset and cleaning it
data = pd.read_csv(path + "/heart_disease_uci.csv")
data = data.dropna()  # This line removes all rows with any NaN values
encoded_data = pd.get_dummies(data, columns=['sex', 'cp', 'fbs', 'restecg', 'dataset',"slope","thal"])
numbered_data = encoded_data.drop(["id"], axis=1)
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(numbered_data), columns=numbered_data.columns)

plt.figure(figsize=(10, 6))
plt.boxplot(df_standardized['age'], vert=False)

#reducing dimensionality
pca = PCA(n_components=None)  
pca_data = pca.fit_transform(df_standardized)
#visualizing pca
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c='blue', alpha=0.5)
plt.title('PCA of Heart Disease Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()
pca_data = PCA(n_components=5).fit_transform(df_standardized)
pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
#splitting data
y = data["cp"]
x = pca_df
x_train, x_test, y_train, y_test = train_test_split(pca_df,y,test_size=0.2, random_state=42)
#training model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x, y)
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': pca_df.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)
#applying RFE
estimator = LogisticRegression()
rfe_selector = RFE(estimator=estimator, n_features_to_select=10, step=1)
rfe_selector = rfe_selector.fit(x_train, y_train)
x_train_rfe = rfe_selector.transform(x_train)
x_test_rfe = rfe_selector.transform(x_test)
#Chi-square test
x_chi = numbered_data  # standardized and one-hot encoded data
y_chi = data["cp"]     # Target variable
chi_scores, p_values = chi2(x_chi, y_chi)
chi_scores, p_values = chi2(x_chi, y_chi)
chi2_df = pd.DataFrame({
    'Feature': x_chi.columns,
    'Chi2 Score': chi_scores,
    'P-value': p_values
})
print(chi2_df.sort_values(by='Chi2 Score', ascending=False))
#creating models
model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model4 = SVC(probability=True)
#training and evaluating models
model1.fit(x_train_rfe, y_train)
y_pred1 = model1.predict(x_test_rfe)
accuracy1 = accuracy_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1, average='weighted')
precision1 = precision_score(y_test, y_pred1, average='weighted')
recall1 = recall_score(y_test, y_pred1, average='weighted')
roc1_auc1 = roc_auc_score(y_test, model1.predict_proba(x_test_rfe), multi_class='ovr')
fpr1, tpr1, _ = roc_curve(y_test, model1.predict_proba(x_test_rfe)[:,1], pos_label=1)
plt.figure()
plt.plot(fpr1, tpr1, label='Logistic Regression (area = %0.2f)' % roc1_auc1)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print(f"Logistic Regression - Accuracy: {accuracy1}, F1 Score: {f1_1}, Precision: {precision1}, Recall: {recall1}")
model2.fit(x_train_rfe, y_train)
y_pred2 = model2.predict(x_test_rfe) 
accuracy2 = accuracy_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2, average='weighted')
precision2 = precision_score(y_test,y_pred2, average = 'weighted')
recall2 = recall_score(y_test, y_pred2, average='weighted')
roc2_auc2 = roc_auc_score(y_test, model2.predict_proba(x_test_rfe), multi_class='ovr')
fpr2, tpr2, _ = roc_curve(y_test, model2.predict_proba(x_test_rfe)[:,1], pos_label=1)
plt.figure()
plt.plot(fpr2, tpr2, label='Decision Tree (area = %0.2f)' % roc2_auc2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print(f"Decision Tree - Accuracy: {accuracy2}, F1 Score: {f1_2}, Precision: {precision2}, Recall: {recall2}")   
model3.fit(x_train_rfe, y_train)
y_pred3 = model3.predict(x_test_rfe)
accuracy3 = accuracy_score(y_test, y_pred3)
f1_3 = f1_score(y_test, y_pred3, average='weighted')
precision3 = precision_score(y_test, y_pred3, average='weighted')
recall3 = recall_score(y_test, y_pred3, average='weighted')
roc3_auc3 = roc_auc_score(y_test, model3.predict_proba(x_test_rfe), multi_class='ovr')
fpr3, tpr3, _ = roc_curve(y_test, model3.predict_proba(x_test_rfe)[:,1], pos_label=1)
plt.figure()
plt.plot(fpr3, tpr3, label='Random Forest (area = %0.2f)' % roc3_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print(f"Random Forest - Accuracy: {accuracy3}, F1 Score: {f1_3}, Precision: {precision3}, Recall: {recall3}")
model4.fit(x_train_rfe, y_train)
y_pred4 = model4.predict(x_test_rfe)
accuracy4 = accuracy_score(y_test, y_pred4)
f1_4 = f1_score(y_test, y_pred4, average='weighted')
precision4 = precision_score(y_test, y_pred4, average='weighted')
recall4 = recall_score(y_test, y_pred4, average='weighted')
roc4_auc4 = roc_auc_score(y_test, model4.predict_proba(x_test_rfe), multi_class='ovr')
fpr4, tpr4, _ = roc_curve(y_test, model4.predict_proba(x_test_rfe)[:,1], pos_label=1)
plt.figure()
plt.plot(fpr4, tpr4, label='SVM (area = %0.2f)' % roc4_auc4)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print(f"SVM - Accuracy: {accuracy4}, F1 Score: {f1_4}, Precision: {precision4}, Recall: {recall4}")
#unsupervised learning
#finding optimal K
wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, init = 'k-means++',max_iter = 300, n_init = 10, random_state=42)
    kmeans.fit(pca_df)
    
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1,16), wcss)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters') 
plt.ylabel('WCSS')
plt.show()
#applying k-means
kmeans = KMeans(n_clusters=5, init = 'k-means++',max_iter = 300, n_init = 10, random_state=42)
y_kmeans = kmeans.fit_predict(pca_df)
#visualizing clusters
plt.figure(figsize=(8, 6))
plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Means Clustering of Heart Disease Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
#hierarchical clustering
linked = linkage(pca_df, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
#comparing clusters with actual labels
ari = adjusted_rand_score(y, y_kmeans)
print(f'Adjusted Rand Index between K-Means clusters and actual labels: {ari}')
#hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(x_train_rfe, y_train)
best_rf_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)
y_pred_best = best_rf_model.predict(x_test_rfe)
accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best, average='weighted')
precision_best = precision_score(y_test, y_pred_best, average='weighted')
recall_best = recall_score(y_test, y_pred_best, average='weighted')
roc_best_auc = roc_auc_score(y_test, best_rf_model.predict_proba(x_test_rfe), multi_class='ovr')
fpr_best, tpr_best, _ = roc_curve(y_test, best_rf_model.predict_proba(x_test_rfe)[:,1], pos_label=1)
plt.figure()
plt.plot(fpr_best, tpr_best, label='Tuned Random Forest (area = %0.2f)' % roc_best_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print(f"Tuned Random Forest - Accuracy: {accuracy_best}, F1 Score: {f1_best}, Precision: {precision_best}, Recall: {recall_best}")
# Save the best model to a file
file_name = 'best_rf_model.pkl'
with open(file_name, 'wb') as f:
    pickle.dump(best_rf_model, f)
print(f"Best model saved to {file_name}")