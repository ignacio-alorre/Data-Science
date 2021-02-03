# importing required libraries 
# Importing required libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix 
from matplotlib.colors import ListedColormap 

# PART 1- APPLYING PCA (D2) TO A DATASET

# importing or loading the dataset 
dataset = pd.read_csv('wine.csv') 

# distributing the dataset into two components X and Y 
X = dataset.iloc[:, 0:13].values 
y = dataset.iloc[:, 13].values 
  
# Splitting the X and Y into the Training set and Testing set 
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 


# Performing preprocessing part 
sc = StandardScaler() 
  
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

# Applying PCA function on training and testing set of X component 
pca = PCA(n_components = 2) 
  
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 
  
explained_variance = pca.explained_variance_ratio_


# PART 2- USING THOSE DATASET TO BUILD A CLASSIFIER

# Fitting Logistic Regression To the training set 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 

# Predicting the test set result using predict function under LogisticRegression  
y_pred = classifier.predict(X_test) 

# Making confusion matrix between test set of Y and predicted value. 
cm = confusion_matrix(y_test, y_pred)

# Scatter plot with the observations classified
def scatter_plot_distribution(X_set, y_set, ptitle):
    
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                         stop = X_set[:, 0].max() + 1, step = 0.01), 
                         np.arange(start = X_set[:, 1].min() - 1, 
                         stop = X_set[:, 1].max() + 1, step = 0.01)) 

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
                 X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
                 cmap = ListedColormap(('yellow', 'white', 'aquamarine'))) 

    plt.xlim(X1.min(), X1.max()) 
    plt.ylim(X2.min(), X2.max()) 

    for i, j in enumerate(np.unique(y_set)): 
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                    c = ListedColormap(('red', 'green', 'blue'))(i), label = j) 
    
    plt.title(ptitle)
    plt.xlabel('PC1') # for Xlabel 
    plt.ylabel('PC2') # for Ylabel 
    plt.legend() # to show legend 

    # show scatter plot 
    plt.show()
 
# Scatter plot for Test Set
scatter_plot_distribution(X_train, y_train, 'Logistic Regression (Train set)')

# Scatter plot for Train Set
scatter_plot_distribution(X_test, y_test, 'Logistic Regression (Test set)')