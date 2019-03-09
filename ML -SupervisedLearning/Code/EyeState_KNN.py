#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:55:00 2019

@author: anshuta
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:35:57 2019

@author: anshuta
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import pydotplus as ptp
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from time import time
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline






def plot_learning_curve(estimator, title, X, y, ylim=None, xlim = None,cv=None, scoring=None, obj_line=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    scoring : string, callable or None, optional, default: None
              A string (see model evaluation documentation)
              or a scorer callable object / function with signature scorer(estimator, X, y)
              For Python 3.5 the documentation is here:
              http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
              For example, Log Loss is specified as 'neg_log_loss'
              
    obj_line : numeric or None (default: None)
               draw a horizontal line 
               

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
        
        
    Citation
    --------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        
    Usage
    -----
        plot_learning_curve(estimator = best_estimator, 
                            title     = best_estimator_title, 
                            X         = X_train, 
                            y         = y_train, 
                            ylim      = (-1.0, 1.1), # neg_log_loss is negative
                            cv        = StatifiedCV, # CV generator
                            scoring   = scoring,     # eg., 'neg_log_loss'
                            obj_line  = obj_line,    # horizontal line
                            n_jobs    = n_jobs)      # how many CPUs

         plt.show()
    """
    

    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlim =xlim
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    if obj_line:
        plt.axhline(y=obj_line, color='blue')

    plt.legend(loc="best")
    return plt








dataset = pd.read_csv("/Users/anshuta/Desktop/spring2019/ML/EEGEye/eyedataset.csv") 

print (dataset.shape)
#remove duplicates

#print (sorted(metrics.SCORERS.keys()))


dataset.drop_duplicates(keep=False, inplace=True)
print (dataset.shape)

X = dataset.drop('eyeDetection', axis=1)
y = dataset['eyeDetection'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state = 7)



knn  = KNeighborsClassifier(n_jobs= -1)
# 5. Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), knn)
print(pipeline.get_params())


weight_options = ['uniform', 'distance']
k_range = list(range(1, 31))


# 6. Declare hyperparameters to tune
param_grid = dict(kneighborsclassifier__n_neighbors=k_range, kneighborsclassifier__weights=weight_options)

# 7. Tune model using cross-validation pipeline
start = time()
grid = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy',refit = True)
grid.fit(X_train, y_train)

print(("time taken to train K-NN classifier is {:.2f}" "seconds"),format(time() - start))


print(grid.best_params_)

# Calling Method 
#plot_grid_search(grid.cv_results_, k_range, weight_options, 'N Neighbors', 'Weights')


start2 = time()
model = grid.best_estimator_
y_pred = model.predict(X_test)
print(("time taken to train K-NN classifier is {:.2f}" "seconds"),format(time() - start2))



scores =[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors =k,n_jobs= -1)
    score = cross_val_score(knn, X_train, y_train, cv=10, scoring='balanced_accuracy')
    scores.append(score.mean())
    
        
plt.plot(k_range,scores)
plt.xlabel('K value')
plt.ylabel('Cross validation Accuracy')
plt.title('Simple K-NN Classifier  CV Accuracy' )

y_pred_train = model.predict(X_train)



print ("Accuracy on training set with KNN is %5f",metrics.balanced_accuracy_score(y_train,y_pred_train))
print ("Accuracy with test set with KNN is %5f",metrics.balanced_accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

plot_learning_curve(model,'Eye state Learning Curve',X_train,y_train,ylim =(0.7,1.2,.05),cv =20,scoring = 'f1',n_jobs =-1)


    
