#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:55:10 2019

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
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import sys
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize
from IPython.display import display



def GridSearch_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):


  clf = grid_clf.best_estimator_
  clf_params = grid_clf.best_params_
  if negative:
    clf_score = -grid_clf.best_score_
  else:
    clf_score = grid_clf.best_score_
  clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
  cv_results = grid_clf.cv_results_

  print("best parameters: {}".format(clf_params))
  print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))

  if display_all_params:
    import pprint
    pprint.pprint(clf.get_params())

  # pick out the best results
  # =========================
  scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

  best_row = scores_df.iloc[0, :]
  if negative:
    best_mean = -best_row['mean_test_score']
  else:
    best_mean = best_row['mean_test_score']
  best_stdev = best_row['std_test_score']
  best_param = best_row['param_' + param_name]



  # display the top 'num_results' results
  display(pd.DataFrame(cv_results).sort_values(by='rank_test_score').head(num_results))

  # plot the results

  scores_df = scores_df.sort_values(by='param_' + param_name)

  if negative:
    means = -scores_df['mean_test_score']
  else:
    means = scores_df['mean_test_score']
  #stds = scores_df['std_test_score']
  params = scores_df['param_' + param_name]


  # plot
  if graph:
    plt.figure(figsize=(8, 8))
    plt.plot(params, means)

    #plt.axhline(y=best_mean + best_stdev, color='red')
    #plt.axhline(y=best_mean - best_stdev, color='red')
    plt.plot(best_param)

    plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.show()
  
        




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



if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
    
    
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

dataset = pd.read_csv("/Users/anshuta/Desktop/spring2019/ML/spamdata.csv") 

print (dataset.shape)
#remove duplicates

#print (sorted(metrics.SCORERS.keys()))


dataset.drop_duplicates(keep=False, inplace=True)
print (dataset.shape)

X = dataset.drop('Spam /Not Spam', axis=1)
y = dataset['Spam /Not Spam'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state = 7)

svc = SVC(kernel ='rbf')
# 5. Declare data preprocessing steps
pipe = Pipeline(steps = [("scaler",StandardScaler()),("rbf_svc",svc)])
print(pipe.get_params())

Cs = [0.1, 1, 10, 100, 1000]
gammas = [0.1, 1, 10, 100]


# 6. Declare hyperparameters to tune
param_grid = dict(rbf_svc__gamma=gammas, rbf_svc__C=Cs)

# 7. Tune model using cross-validation pipeline
start = time()
grid = GridSearchCV(pipe, param_grid, cv=10, scoring='balanced_accuracy',refit = True)
grid.fit(X_train, y_train)

print(("time taken to train RBF SVC classifier is {:.2f}" "seconds"),format(time() - start))


print(grid.best_params_)

# Calling Method 
#plot_grid_search(grid.cv_results_, k_range, weight_options, 'N Neighbors', 'Weights')



model = grid.best_estimator_
start2 = time()
y_pred = model.predict(X_test)
print(("time taken to test RBF SVC classifier is {:.2f}" "seconds"),format(time() - start2))


y_pred_train = model.predict(X_train)



print ("Accuracy on training set with RBF SVC is %5f",metrics.balanced_accuracy_score(y_train,y_pred_train))
print ("Accuracy with test set with RBF SVC is %5f",metrics.balanced_accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

plot_learning_curve(model,'Spambase Learning Curve',X_train,y_train,ylim =(0.7,1.2,.05),cv =10,n_jobs =-1)



GridSearch_table_plot(grid, "rbf_svc__gamma" ,negative=False)


GridSearch_table_plot(grid, "rbf_svc__C",negative=False)

'''# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [0.1, 1, 10, 100, 1000]
gamma_2d_range = [0.1, 1, 10, 100]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_train, y_train)
        classifiers.append((C, gamma, clf))

# #############################################################################
# Visualization
#
# draw visualization of parameter effects

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(Cs),
                                                     len(gammas))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gammas)), gammas, rotation=45)
plt.yticks(np.arange(len(Cs)), Cs)
plt.title('Validation accuracy')
plt.show()

'''



    

