#!/usr/bin/env python
# coding: utf-8

# ###### ENGINEERING TECHNOLOGY FOR STRATEGY (AND SECURITY)
# ###  
# ### STRATEGIES FOR TELECOMMUNICATIONS 98228
# 
# ###  
# 
# ##### *Assignment for the machine learning part of the course*
# ##### Prof. MAURIZIO MONGELLI
#  ## 
#  
# 
# **| Student Name: Ammar Shaheen |**   
# **| ID: S5192825 |**

# In[9]:


get_ipython().system('pip install git+https://github.com/scikit-learn-contrib/skope-rules.git')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from skrules import SkopeRules
from sklearn.metrics import accuracy_score
# Load the Iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train a Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Train a Bagging of Decision Trees model
bagging = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)

# Train a Random Forests model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Train a Skope-Rules model
rules = SkopeRules(max_depth_duplication=2)
rules.fit(X_train, y_train)

# Evaluate the performance of the models on the testing set
y_pred_dt = dt.predict(X_test)
y_pred_bagging = bagging.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_rules = rules.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
acc_bagging = accuracy_score(y_test, y_pred_bagging)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_rules = accuracy_score(y_test, y_pred_rules)

print(f"Decision Tree accuracy: {acc_dt:.3f}")
print(f"Bagging of Decision Trees accuracy: {acc_bagging:.3f}")
print(f"Random Forests accuracy: {acc_rf:.3f}")
print(f"Skope-Rules accuracy: {acc_rules:.3f}")


# In[11]:


# Visualize the decision boundaries for the Random Forests model
import matplotlib.pyplot as plt
import numpy as np

try:
    xx, yy = np.meshgrid(np.arange(4, 8, 0.01), np.arange(1.5, 5, 0.01))
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test, alpha=0.8)
    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])
    plt.show()

except Exception as e:
    print(f"Error occurred during decision boundary visualization: {str(e)}")

# Plot the feature importance scores for the Random Forests model
from sklearn.inspection import permutation_importance

try:
    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()

    plt.barh(range(4), result.importances_mean[sorted_idx])
    plt.yticks(range(4), np.array(iris.feature_names)[sorted_idx])
    plt.xlabel("Importance score")
    plt.show()

except Exception as e:
    print(f"Error occurred during feature importance plot: {str(e)}")


# In[12]:


import matplotlib.pyplot as plt
import numpy as np

# Select the third and fourth features
X = iris.data[:, [2, 3]]
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# Create a meshgrid of points to visualize the decision boundaries
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.01),
                     np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.01))

# Predict the class for each point in the meshgrid
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])

# Reshape the predictions to match the meshgrid shape
Z = Z.reshape(xx.shape)

# Plot the decision boundaries and the test set
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

