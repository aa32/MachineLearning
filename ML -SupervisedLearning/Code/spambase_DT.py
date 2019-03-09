# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

'''import sklearn.model_selection as ms'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import pydotplus as ptp
import collections
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns
from time import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import itertools
from sklearn.metrics import mean_squared_error

'''
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
import numpy as np  
import matplotlib.pyplot as plt  '''


 


    
    

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)
    ax.ylim = (0.500, 0.999)

    

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
        

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    #ax.show()
    
    
    
     
    
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, scoring=None, obj_line=None,
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
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    if obj_line:
        plt.axhline(y=obj_line, color='blue')

    plt.legend(loc="best")
    return plt








setCount = 4


dataset = pd.read_csv("/Users/anshuta/Desktop/spring2019/ML/spamdata.csv") 

print (dataset.shape)
#remove duplicates

dataset.drop_duplicates(keep=False, inplace=True)

X = dataset.drop('Spam /Not Spam', axis=1)
y = dataset['Spam /Not Spam'] 

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state =0)

#decision Tree without Pruning:
start1= time()
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 0)
clf_gini.fit(X_train, y_train)

print (("time taken to train unpruned DT is {:.2f} "
           "seconds"),format(time() - start1))

start2 =time()

y_pred_gini_test = clf_gini.predict(X_test)
print (("time taken to test unpruned DT is {:.2f} "
           "seconds"),format(time() - start2))
y_pred_gini_train = clf_gini.predict(X_train)











'''
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)
y_pred_entropy_train = clf_entropy.predict(X_train)



plot_learning_curve(estimator = DecisionTreeClassifier(criterion = "gini", random_state = 7), 
                            title     = 'Learning Curve without Pruning', 
                            X         = X_train, 
                            y         = y_train, 
                            ylim      = (0.4, 1.2),
                            cv        = 10,# neg_log_loss is negative# horizontal line
                            n_jobs    = -1)      # how many CPUs

plt.show()


'''




 

data_feature_names =['word_freq_make',	'word_freq_address',	'word_freq_all',	'word_freq_3d'	,'word_freq_our',	'word_freq_over',	'word_freq_remove','	word_freq_internet','word_freq_order	', 'word_freq_mail','word_freq_receive','word_freq_will', 	'word_freq_people',	 'word_freq_report','word_freq_addresses' ,	'word_freq_free',	'word_freq_business','word_freq_email',	'word_freq_you', 	'word_freq_credit',	'word_freq_your',	'word_freq_font' ,	'word_freq_000' ,	'word_freq_money' ,	'word_freq_hp' ,'word_freq_hpl',	'word_freq_george', 	'word_freq_650', 'word_freq_lab',	'word_freq_labs',	'word_freq_telnet',	'word_freq_857','word_freq_data' ,	'word_freq_415','word_freq_85',	'word_freq_technology',	'word_freq_1999',	'word_freq_parts	','word_freq_pm' ,	'word_freq_direct',	'word_freq_cs',	'word_freq_meeting','word_freq_original',	'word_freq_project',	'word_freq_re',	 'word_freq_edu',	'word_freq_table',	'word_freq_conference',	'char_freq_;',	'char_freq_(',	'char_freq_[	','char_freq_!','char_freq_$',	'char_freq_#	','capital_run_length_average',	'capital_run_length_longest',	'capital_run_length_total']

scoring = {'Score': make_scorer(accuracy_score)}
tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
start = time()
clf = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), tree_para, cv=10 ,return_train_score=True)
clf.fit(X_train, y_train)
print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(clf.cv_results_)))
#print(clf.cv_results_)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_) 
print('Accuracy on the training subset: {:.3f}'.format(tree_model.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree_model.score(X_test, y_test)))

class_names = ['Spam','Not Spam']

start2 = time()


y_pred = tree_model.predict(X_test)




plot_learning_curve(tree_model,'Spambase Learning Curve',X_train,y_train,ylim =(0.7,1.2,.05),cv =10,n_jobs =-1)





y_pred_train = tree_model.predict(X_train)

plot_grid_search(clf.cv_results_ , [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],['gini','entropy'] ,'Max Depth','Info Gain')



results = clf.cv_results_


scoring = {'score': make_scorer(accuracy_score)}

#plot Train Vs Test Accuracy

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("Max Depth")
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(0, 250)
ax.set_ylim(0.73, 1)

results = clf.cv_results_

print (results['param_max_depth'].data)


# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_max_depth'].data, dtype=int)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()



dot_data = tree.export_graphviz(tree_model,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = ptp.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('/Users/anshuta/Desktop/spring2019/ML/tree.png')

print ("Accuracy on test data with Gini is %5f",accuracy_score(y_test,y_pred_gini_test))
print ("Accuracy with train  data with GIni is %5f",accuracy_score(y_train,y_pred_gini_train))




'''
print ("Accuracy on train data with Gini is %5f",accuracy_score(y_train,y_pred_gini_train))
print ("Accuracy with train data is %5f",accuracy_score(y_train,y_pred_entropy_train))
'''

print("Accuracy on the training subset after Pruning is %5f",accuracy_score(y_train,y_pred_train))
print("Accuracy on the test subset after Pruning is %5f",accuracy_score(y_test,y_pred))
print (clf.best_score_)



dot_data = tree.export_graphviz(clf_gini,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = ptp.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)


for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

print("Max Depth without Pruning:")    
print (clf_gini.tree_.max_depth)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))  

print("classification report")
print(classification_report(y_test, y_pred))  

#graph.write_png('/Users/anshuta/Desktop/spring2019/ML/tree2.png')


