import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
from sklearn import tree

iris = datasets.load_iris()
feature = pd.DataFrame(iris.data, columns= iris.feature_names);feature.head()
target = iris.target
iris.target_names # array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
iris_data = feature.copy()
iris_data['species'] = target
iris_data['species'] = iris_data['species'].map({0:'setosa', 1:'versicolor', 2:'virginica'})
" As the data has only three distinct species of flowers namely Setosa,\
 Versicolor, and Virginica"
iris_data.describe()
sns.pairplot(iris_data, hue = 'species')
# Splitting the dataset.
X_train, X_test, y_train, y_test= train_test_split(feature, target, test_size = 0.4, 
                                                   random_state=1)

dt = tree.DecisionTreeClassifier(max_depth=3, random_state = 1)
dt.fit(X_train, y_train)

predict = dt.predict(X_test)
print("The accuracy of the Decicion Tree is", '{:.2f}'.format(metrics.accuracy_score(predict, y_test)))

# PLOTTING THE TREE
    
plt.figure()
tree.plot_tree(dt, feature_names = iris.feature_names, class_names=iris.target_names, filled = True)



