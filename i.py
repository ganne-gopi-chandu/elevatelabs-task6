import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

### LOAD DATA FROM CSV OR SQLITE ###
# Option 1: Load dataset from CSV file
try:
    df = pd.read_csv('Iris.csv')  # Ensure the file is in your working directory
    print("Data loaded from CSV.")
except FileNotFoundError:
    print("CSV file not found. Attempting to load from SQLite...")

# Option 2: Load dataset from SQLite database if CSV not found
try:
    conn = sqlite3.connect('database.sqlite')  # Ensure the SQLite file is available
    query = "SELECT * FROM iris_data;"  # Change table name if needed
    df = pd.read_sql_query(query, conn)
    conn.close()
    print("Data loaded from SQLite.")
except:
    print("Error loading data from SQLite database.")

# View dataset structure
print(df.head())

### DATA CLEANING ###
# Drop ID column if it's not needed for modeling
df = df.drop(columns=['Id'])

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Fill or drop missing values if necessary
df = df.dropna()  # Remove rows with missing values

# Remove duplicates
df = df.drop_duplicates()

# Check dataset after cleaning
print("Dataset after cleaning:\n", df.info())

### PREPROCESSING ###
# Encode categorical labels (Species)
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])  # Convert species names to numbers

# Normalize feature columns
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

### SPLIT DATA ###
X = df.iloc[:, :-1]
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### TRAIN KNN MODEL ###
k_values = [3, 5, 7,10,20]  # Test different values of K
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Ensure feature names match by converting X_test into a DataFrame
    X_test_df = pd.DataFrame(X_test, columns=X_train.columns)  # Ensure matching feature names
    y_pred = knn.predict(X_test_df)  

    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy for K={k}: {acc * 100:.2f}%')  # Show percentage accuracy

### CONFUSION MATRIX VISUALIZATION (MULTIPLE PLOTS) ###
fig, axes = plt.subplots(1, len(k_values), figsize=(18, 5))
for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test_df)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', ax=axes[i])
    axes[i].set_title(f'K={k}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

### DECISION BOUNDARIES FOR ALL K VALUES ###
fig, axes = plt.subplots(1, len(k_values), figsize=(18, 5))

for i, k in enumerate(k_values):
    X_vis = X_train.iloc[:, [0, 1]]
    y_vis = y_train

    knn_vis = KNeighborsClassifier(n_neighbors=k)
    knn_vis.fit(X_vis, y_vis)

    x_min, x_max = X_vis.iloc[:, 0].min() - 1, X_vis.iloc[:, 0].max() + 1
    y_min, y_max = X_vis.iloc[:, 1].min() - 1, X_vis.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # Match feature names from X_vis
    grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X_vis.columns)
    Z = knn_vis.predict(grid_points)
    Z = Z.reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, alpha=0.3)
    axes[i].scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_vis, edgecolors='k', marker='o')
    axes[i].set_title(f"K={k}")
    axes[i].set_xlabel("Sepal Length (Cm)")
    axes[i].set_ylabel("Sepal Width (Cm)")

plt.tight_layout()
plt.show()
