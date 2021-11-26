#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import joblib
from sklearn.metrics import precision_score #precision
from sklearn.metrics import recall_score #recall
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier


import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA

from scipy import interp
from sklearn.metrics import roc_curve, auc
from itertools import cycle


# In[2]:


# FOLD1
baseUsada = 'HOLDOUT-KNORAE-SVMSMOTE-NON-ENSEMBLE-'
tipo = 'HOLDOUT'

fold1_train = 'D:/ARTIGO DRC/TREINAMENTO.CSV'
df_fold1_train = read_csv(fold1_train, header=None)
data_fold1_train = df_fold1_train.values

fold1_test = 'D:/ARTIGO DRC/TESTE.CSV'
df_fold1_test = read_csv(fold1_test, header=None)
data_fold1_test = df_fold1_test.values

X_train_fold1 = data_fold1_train[:, :-1]
X_test_fold1 = data_fold1_test[:, :-1]
y_train_fold1 = data_fold1_train[:, -1]
y_test_fold1 = data_fold1_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold1 = labelencoder_X.fit_transform(y_test_fold1)
y_train_fold1 = labelencoder_X.fit_transform(y_train_fold1)



pipeline = DecisionTreeClassifier(min_samples_split = 4,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features= None)

#pipeline = RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=3, max_features= 0.2, n_estimators=150, random_state=0)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion= 'gini',max_depth= None,max_features= 0.2)
#pipeline = AdaBoostClassifier(arvore, algorithm='SAMME',n_estimators=150)

#pipeline = OLA(DFP=False, k=5, knn_classifier= None, safe_k= 5,with_IH=True)

#pipeline = LCA(DFP=False, k=5, knn_classifier='knn', with_IH=False, safe_k=5)

#pipeline = METADES()#DFP=True, k=7, Kp=7, knn_classifier='knn', knne=False, mode='selection')#, safe_k= 6)

#pipeline = KNORAE(DFP=True, k=7, safe_k=7, knn_classifier='knn', knne=False, with_IH=True)

#pipeline = KNORAU(DFP=True, k=7, safe_k=None, knn_classifier=None, knne=False, with_IH=True)


pipeline.fit(X_train_fold1, y_train_fold1)
yhat_fold1 = pipeline.predict(X_test_fold1)
accuracy1  = accuracy_score(y_test_fold1, yhat_fold1)

precision1_macro = precision_score(y_test_fold1, yhat_fold1,average='macro')
recall1_macro     = recall_score(y_test_fold1, yhat_fold1, average='macro')

precision1_micro = precision_score(y_test_fold1, yhat_fold1,average='micro')
recall1_micro     = recall_score(y_test_fold1, yhat_fold1, average='micro')

precision1_weighted = precision_score(y_test_fold1, yhat_fold1,average='weighted')
recall1_weighted     = recall_score(y_test_fold1, yhat_fold1, average='weighted')

f1_score_macro1  = f1_score(y_test_fold1, yhat_fold1, average='macro')
f1_score_micro1  = f1_score(y_test_fold1, yhat_fold1, average='micro')
f1_score_weighted1  = f1_score(y_test_fold1, yhat_fold1, average='weighted')    
matthews_corrcoef1  = matthews_corrcoef(y_test_fold1, yhat_fold1)
fowlkes_mallows_score1  = fowlkes_mallows_score(y_test_fold1, yhat_fold1)

#X, y = load_iris(return_X_y=True)
#clf = LogisticRegression(solver="liblinear").fit(X, y)
#roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')
#roc_auc_score1 = roc_auc_score(y_test_fold1, yhat_fold1, multi_class='over')

print('Accuracy: %.3f' % (accuracy1 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef1)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score1)
print("---- MACRO ----")
print("Precision Score macro: ", precision1_macro)
print("Recall Score macro: ", recall1_macro)
print("f1_score_macro : ", f1_score_macro1)
print("---- MICRO ----")
print("Precision Score micro: ", precision1_micro)
print("Recall Score micro: ", recall1_micro)
print("f1_score_micro : ", f1_score_micro1)
print("---- WEIGHTED ----")
print("Precision Score : ", precision1_weighted)
print("Recall Score : ", recall1_weighted)
print("f1_score_weighted : ", f1_score_weighted1)


#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold1, yhat_fold1))

print(y_test_fold1)
#binarizo o treinamento e o teste
y_test_fold1 = label_binarize(y_test_fold1, classes=[0,1, 2, 3])

y = label_binarize(y_train_fold1, classes=[0,1, 2, 3])
n_classes = y.shape[1]

#treino usando um contra todos
clf1 = OneVsRestClassifier(pipeline)
clf1.fit(X_train_fold1, y)

y_score = clf1.predict_proba(X_test_fold1)



fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold1[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold1.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

roc1 = roc_auc["micro"]

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
    
    
    
colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='Classe {0} (AUC = {1:0.2f})' ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=1.0)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
# plt.title('Curva ROC de dados multiclasse',fontsize=12)
plt.xlabel('False Positive Rate',fontsize=12)
plt.ylabel('True Positive Rate',fontsize=12)
plt.xticks(np.arange(0.0, 1.1, step=0.2), fontsize=12)
plt.yticks(np.arange(0.0, 1.1, step=0.2), fontsize=12)
plt.legend(prop={'size':11}, loc='lower right')
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/'+tipo+'/ROC E PR/' + baseUsada + 'ROC'
plt.savefig(ende,dpi=300);


# PRECISION-RECALL


# For each class
precision1 = dict()
recall1 = dict()
average_precision1 = dict()
for i in range(n_classes):
    precision1[i], recall1[i], _ = precision_recall_curve(y_test_fold1[:, i],
                                                        y_score[:, i])
    average_precision1[i] = average_precision_score(y_test_fold1[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision1["micro"], recall1["micro"], _ = precision_recall_curve(y_test_fold1.ravel(),
    y_score.ravel())
average_precision1["micro"] = average_precision_score(y_test_fold1, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision1["micro"]))


CPR1 = average_precision1["micro"]

#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores1 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score1 in f_scores1:
    x = np.linspace(0.01, 1)
    y = f_score1 * x / (2 * x - f_score1)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score1), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall1["micro"], precision1["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision1["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall1[i], precision1[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision1[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
#plt.show()
ende2 = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/'+tipo+'/ROC E PR/' + baseUsada + 'PR'
plt.savefig(ende2,dpi=300);

print("PR:", CPR1)
print("roc:", roc1)


# In[ ]:




