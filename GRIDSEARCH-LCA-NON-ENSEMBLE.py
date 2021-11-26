#!/usr/bin/env python
# coding: utf-8

# In[247]:


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
from sklearn.tree import DecisionTreeClassifier
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE

print(__doc__)


fold1_train = 'D:/Mestrado/ARTIGO DRC/TREINAMENTO.CSV'
df_fold1_train = read_csv(fold1_train, header=None)
data_fold1_train = df_fold1_train.values

X_train_fold1 = data_fold1_train[:, :-1]
y_train_fold1 = data_fold1_train[:, -1]


pipeline = LCA()

#Create the parameter grid, entering the values to use for each parameter selected in the RandomForest estimator
parameters = [{"DFP": [False, True], "with_IH": [False, True], 
               "k": [2,3,5,7], "knn_classifier": ['knn', 'faiss', None],
               "safe_k": [None,4,5,6,7]
              }
             ]


# In[249]:


{'DFP': True, 'k': 7, 'knn_classifier': None, 'safe_k': None, 'with_IH': False}
#GRIDSARCH AUMENTO MANUAL
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
    y_true, y_pred = y_test_fold1, clf.predict(X_test_fold1)
   # print(classification_report(y_true, y_pred))
    print()


# In[ ]:





# In[ ]:





# In[ ]:




