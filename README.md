# elevatelabs-task6
# K-Nearest Neighbors (KNN) Classification

This project demonstrates how to implement the **K-Nearest Neighbors (KNN)** algorithm for classification using the **Iris dataset**. It includes data loading from CSV or SQLite, preprocessing, training, accuracy evaluation, confusion matrix plotting, and decision boundary visualization.

---

## ✅ Features

- Load data from CSV or SQLite.
- Clean and preprocess the data (handle missing values, remove duplicates, normalize).
- Encode target labels using `LabelEncoder`.
- Split data into training and testing sets.
- Train and evaluate `KNeighborsClassifier` with multiple values of K.
- Visualize:
  - Confusion matrices.
  - Decision boundaries (for first two features).

---

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- sqlite3

## 📊 Example Output

- Accuracy for K=3: 93.33%
- Accuracy for K=5: 93.33%
- Accuracy for K=7: 93.33%
- Accuracy for K=10: 93.33%
- Accuracy for K=20: 96.67%

