#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from pandas import read_csv


from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.metrics import accuracy_score
from sklearn2pmml import make_pmml_pipeline
from sklearn2pmml import sklearn2pmml
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.ensemble._voting import VotingClassifier

from mlxtend.feature_selection import ColumnSelector
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE

print(__doc__)

fold1_train = 'D:/Mestrado/ARTIGO DRC/TREINAMENTO.CSV'
df_fold1_train = read_csv(fold1_train, header=None)
data_fold1_train = df_fold1_train.values

X_train_fold1 = data_fold1_train[:, :-1]
y_train_fold1 = data_fold1_train[:, -1]


preprocessor1 = ColumnSelector(cols=(2, 5, 7))
preprocessor2 = ColumnSelector(cols=(0, 1, 4, 5, 6))
preprocessor3 = ColumnSelector(cols=(1, 4))
preprocessor4 = ColumnSelector(cols=(0, 1, 4, 5, 7))


from sklearn.tree import DecisionTreeClassifier

pipeline = DecisionTreeClassifier()

pipeline = PMMLPipeline([
  ("classifier", VotingClassifier([
    ("pipe1", Pipeline(steps=[('preprocessor', preprocessor1),('rfc', DecisionTreeClassifier())])),
    ("pipe2", Pipeline(steps=[('preprocessor', preprocessor2),('rfc', DecisionTreeClassifier())])),
    ("pipe3", Pipeline(steps=[('preprocessor', preprocessor3),('rfc', DecisionTreeClassifier())])),
    ("pipe4", Pipeline(steps=[('preprocessor', preprocessor4),('rfc', DecisionTreeClassifier())])),
  ]))
])

#Create the parameter grid, entering the values to use for each parameter selected in the RandomForest estimator
parameters = [{"classifier__pipe1__rfc__criterion": ['entropy', 'gini'], "classifier__pipe1__rfc__ccp_alpha": [0.0, 'non-negative'],
                "classifier__pipe1__rfc__max_depth": [3, 5, 10, None], "classifier__pipe1__rfc__max_features": [None,'auto', 0.1, 0.2, 0.3],
                 "classifier__pipe1__rfc__min_samples_split": [2,3,4]
              },
              {"classifier__pipe2__rfc__criterion": ['entropy', 'gini'], "classifier__pipe2__rfc__ccp_alpha": [0.0, 'non-negative'],
               "classifier__pipe2__rfc__max_depth": [3, 5, 10, None], "classifier__pipe2__rfc__max_features": [None,'auto', 0.1, 0.2, 0.3],
                 "classifier__pipe2__rfc__min_samples_split": [2,3,4]
             },
              {"classifier__pipe3__rfc__criterion": ['entropy', 'gini'], "classifier__pipe3__rfc__ccp_alpha": [0.0, 'non-negative'],
                "classifier__pipe3__rfc__max_depth": [3, 5, 10, None], "classifier__pipe3__rfc__max_features": [None,'auto', 0.1, 0.2, 0.3],
                 "classifier__pipe3__rfc__min_samples_split": [2,3,4]
             },
              {"classifier__pipe4__rfc__criterion": ['entropy', 'gini'], "classifier__pipe4__rfc__ccp_alpha": [0.0, 'non-negative'],
                "classifier__pipe4__rfc__max_depth": [3, 5, 10, None], "classifier__pipe4__rfc__max_features": [None,'auto', 0.1, 0.2, 0.3],
                 "classifier__pipe4__rfc__min_samples_split": [2,3,4]
             }
             ]


# In[ ]:


# gridsearch aumento manual
scores = ['precision']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    
    cv = KFold(n_splits=5)   
    
    clf = GridSearchCV(
        pipeline, parameters,  n_jobs=-1, verbose=1, cv=cv, scoring='%s_macro' % score
    )
    clf.fit(X_train_fold1, y_train_fold1)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
 #   y_true, y_pred = y_test_fold1, clf.predict(X_test_fold1)
  #  print(classification_report(y_true, y_pred))
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




