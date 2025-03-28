import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

# Load dataset
df = pd.read_csv("/Users/akshatyuvan/github-classroom/Finova-MIT/intro-to-ml-akshatyuvan/Logistic_Regression/Iris.csv")
df = df.drop(columns=["Id"])  # Remove Id column

# Encode target variable
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Select two features for visualization
X = df[["PetalLengthCm", "PetalWidthCm"]].values
y = df["Species"].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression model
model = LogisticRegression(multi_class="ovr", solver="lbfgs")
model.fit(X_train, y_train)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(["red", "blue", "green"]))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["red", "blue", "green"]), edgecolor="k")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Logistic Regression Decision Boundary")
    
    # Legend
    legend_labels = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
               for color in ["red", "blue", "green"]]
    plt.legend(handles, [legend_labels[i] for i in range(len(legend_labels))], title="Species")
    
    plt.show()

# Plot decision boundary
plot_decision_boundary(model, X, y)
