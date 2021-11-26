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


# In[3]:


# FOLD1
baseUsada = 'SMOTE-DT-ITERAC1-NON-NESTED-ENSEMBLE'
tipo = 'arvore'

fold1_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/1 FOLD/1FOLD - TREINAMENTO.CSV'
df_fold1_train = read_csv(fold1_train, header=None)
data_fold1_train = df_fold1_train.values

fold1_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/1 FOLD/1FOLD - TESTE.CSV'
df_fold1_test = read_csv(fold1_test, header=None)
data_fold1_test = df_fold1_test.values

X_train_fold1 = data_fold1_train[:, :-1]
X_test_fold1 = data_fold1_test[:, :-1]
y_train_fold1 = data_fold1_train[:, -1]
y_test_fold1 = data_fold1_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold1 = labelencoder_X.fit_transform(y_test_fold1)
y_train_fold1 = labelencoder_X.fit_transform(y_train_fold1)


pipeline1 = DecisionTreeClassifier(min_samples_split = 2,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features= None)

#pipeline1 = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=10, max_features= 'auto', n_estimators=25)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion= 'entropy',max_depth= 3,max_features= None)
#pipeline1 = AdaBoostClassifier(arvore, algorithm='SAMME.R',n_estimators=50)

#pipeline1 = OLA(DFP=False, k=7, knn_classifier= 'knn', safe_k=None,with_IH=False)

#pipeline1 = LCA(DFP=True, k=7, knn_classifier='knn', with_IH=False, safe_k=4)

#pipeline1 = METADES(DFP=True, k=7, Kp=8, knn_classifier=None, knne=True, mode='weighting', safe_k= 5)

#pipeline1 = KNORAE(DFP=False, k=8, safe_k=7, knn_classifier='knn', knne=True, with_IH=True)

#pipeline1 = KNORAU(DFP=True, k=7, safe_k=None, knn_classifier='knn', knne=True, with_IH=False)



pipeline1.fit(X_train_fold1, y_train_fold1)
yhat_fold1 = pipeline1.predict(X_test_fold1)

print(yhat_fold1)

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
clf1 = OneVsRestClassifier(pipeline1)
clf1.fit(X_train_fold1, y)

y_score1 = clf1.predict_proba(X_test_fold1)



fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold1[:, i], y_score1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold1.ravel(), y_score1.ravel())
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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/'+tipo+'/' + baseUsada + '1FOLD-ROC'
plt.savefig(ende,dpi=300);


# PRECISION-RECALL


# For each class
precision1 = dict()
recall1 = dict()
average_precision1 = dict()
for i in range(n_classes):
    precision1[i], recall1[i], _ = precision_recall_curve(y_test_fold1[:, i],
                                                        y_score1[:, i])
    average_precision1[i] = average_precision_score(y_test_fold1[:, i], y_score1[:, i])

# A "micro-average": quantifying score on all classes jointly
precision1["micro"], recall1["micro"], _ = precision_recall_curve(y_test_fold1.ravel(),
    y_score1.ravel())
average_precision1["micro"] = average_precision_score(y_test_fold1, y_score1,
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
ende2 = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/'+tipo+'/' + baseUsada + '1FOLD-PR'
plt.savefig(ende2,dpi=300);



# In[5]:


# FOLD2

fold2_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/2 FOLD/2FOLD - TREINAMENTO.CSV'
df_fold2_train = read_csv(fold2_train, header=None)
data_fold2_train = df_fold2_train.values

fold2_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/2 FOLD/2FOLD - TESTE.CSV'
df_fold2_test = read_csv(fold2_test, header=None)
data_fold2_test = df_fold2_test.values

X_train_fold2 = data_fold2_train[:, :-1]
X_test_fold2 = data_fold2_test[:, :-1]
y_train_fold2 = data_fold2_train[:, -1]
y_test_fold2 = data_fold2_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold2 = labelencoder_X.fit_transform(y_test_fold2)
y_train_fold2 = labelencoder_X.fit_transform(y_train_fold2)


pipeline2 = DecisionTreeClassifier(min_samples_split = 4,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features= None)

#pipeline2 = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=10, max_features= 'auto', n_estimators=100)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion= 'entropy',max_depth= 3,max_features= None)
#pipeline2 = AdaBoostClassifier(arvore, algorithm='SAMME.R',n_estimators=50)

#pipeline2 = OLA(DFP=False, k=7, knn_classifier= 'knn', safe_k=None,with_IH=False)

#pipeline2 = LCA(DFP=False, k=7, knn_classifier=None, with_IH=False, safe_k=4)

#pipeline2 = METADES(DFP=False, k=7, Kp=8, knn_classifier=None, knne=False, mode='hybrid', safe_k= 4)

#pipeline2 = KNORAE(DFP=True, k=7, safe_k=6, knn_classifier=None, knne=True, with_IH=True)

#pipeline2 = KNORAU(DFP=False, k=5, safe_k=5, knn_classifier=None, knne=True, with_IH=True)



pipeline2.fit(X_train_fold2, y_train_fold2)
yhat_fold2 = pipeline2.predict(X_test_fold2)
accuracy2  = accuracy_score(y_test_fold2, yhat_fold2)

precision2_macro = precision_score(y_test_fold2, yhat_fold2,average='macro')
recall2_macro    = recall_score(y_test_fold2, yhat_fold2, average='macro')

precision2_micro = precision_score(y_test_fold2, yhat_fold2,average='micro')
recall2_micro    = recall_score(y_test_fold2, yhat_fold2, average='micro')

precision2_weighted = precision_score(y_test_fold2, yhat_fold2,average='weighted')
recall2_weighted    = recall_score(y_test_fold2, yhat_fold2, average='weighted')

f1_score_macro2  = f1_score(y_test_fold2, yhat_fold2, average='macro')
f1_score_micro2  = f1_score(y_test_fold2, yhat_fold2, average='micro')
f1_score_weighted2  = f1_score(y_test_fold2, yhat_fold2, average='weighted')    
matthews_corrcoef2  = matthews_corrcoef(y_test_fold2, yhat_fold2)
fowlkes_mallows_score2  = fowlkes_mallows_score(y_test_fold2, yhat_fold2)

#X, y = load_iris(return_X_y=True)
#clf = LogisticRegression(solver="liblinear").fit(X, y)
#roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')
#roc_auc_score1 = roc_auc_score(y_test_fold1, yhat_fold1, multi_class='over')

print('Accuracy: %.3f' % (accuracy2 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef2)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score2)
print("---- MACRO ----")
print("Precision Score macro: ", precision2_macro)
print("Recall Score macro: ", recall2_macro)
print("f1_score_macro : ", f1_score_macro2)
print("---- MICRO ----")
print("Precision Score micro: ", precision2_micro)
print("Recall Score micro: ", recall2_micro)
print("f1_score_micro : ", f1_score_micro2)
print("---- WEIGHTED ----")
print("Precision Score : ", precision2_weighted)
print("Recall Score : ", recall2_weighted)
print("f1_score_weighted : ", f1_score_weighted2)

#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold2, yhat_fold2))

#binarizo o treinamento e o teste
y_test_fold2 = label_binarize(y_test_fold2, classes=[0, 1, 2, 3])

y2 = label_binarize(y_train_fold2, classes=[0, 1, 2, 3])
n_classes2 = y2.shape[1]

#treino usando um contra todos
clf2 = OneVsRestClassifier(pipeline2)
clf2.fit(X_train_fold2, y2)

y_score2 = clf2.predict_proba(X_test_fold2)


fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold2[:, i], y_score2[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold2.ravel(), y_score2.ravel())
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


roc2 = roc_auc["micro"]

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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/'+tipo+'/' + baseUsada + '2FOLD-ROC'
plt.savefig(ende,dpi=300);

# PRECISION-RECALL


# For each class
precision2 = dict()
recall2 = dict()
average_precision2 = dict()
for i in range(n_classes2):
    precision2[i], recall2[i], _ = precision_recall_curve(y_test_fold2[:, i],
                                                        y_score2[:, i])
    average_precision2[i] = average_precision_score(y_test_fold2[:, i], y_score2[:, i])

# A "micro-average": quantifying score on all classes jointly
precision2["micro"], recall2["micro"], _ = precision_recall_curve(y_test_fold2.ravel(),
    y_score2.ravel())
average_precision2["micro"] = average_precision_score(y_test_fold2, y_score2,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision2["micro"]))


CPR2 = average_precision2["micro"]


#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores2 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score2 in f_scores2:
    x = np.linspace(0.01, 1)
    y = f_score2 * x / (2 * x - f_score2)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score2), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall2["micro"], precision2["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision2["micro"]))

for i, color in zip(range(n_classes2), colors):
    l, = plt.plot(recall2[i], precision2[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision2[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

ende2 = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/'+tipo+'/' + baseUsada+ '2FOLD-PR'
plt.savefig(ende2,dpi=300);


# In[6]:


# FOLD3

fold3_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/3 FOLD/3FOLD - TREINAMENTO.CSV'
df_fold3_train = read_csv(fold3_train, header=None)
data_fold3_train = df_fold3_train.values

fold3_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/3 FOLD/3FOLD - TESTE.CSV'
df_fold3_test = read_csv(fold3_test, header=None)
data_fold3_test = df_fold3_test.values

X_train_fold3 = data_fold3_train[:, :-1]
X_test_fold3 = data_fold3_test[:, :-1]
y_train_fold3 = data_fold3_train[:, -1]
y_test_fold3 = data_fold3_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold3 = labelencoder_X.fit_transform(y_test_fold3)
y_train_fold3 = labelencoder_X.fit_transform(y_train_fold3)


pipeline3 = DecisionTreeClassifier(min_samples_split= 2,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features= None)

#pipeline3 = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=10, max_features=0.1, n_estimators=75)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion= 'entropy',max_depth= 3,max_features=None)
#pipeline3 = AdaBoostClassifier(arvore, algorithm='SAMME.R',n_estimators=200)

#pipeline3 = OLA(DFP=True, k=3, knn_classifier= 'knn', safe_k=None,with_IH=False)

#pipeline3 = LCA(DFP=False, k=2, knn_classifier= None, with_IH=False, safe_k=None)

#pipeline3 = METADES(DFP=True, k=7, Kp=8, knn_classifier=None, knne=False, mode='selection', safe_k= 4)

#pipeline3 = KNORAE(DFP=True, k=8, safe_k=None, knn_classifier='knn', knne=False, with_IH=False)

#Pipeline3 = KNORAU(DFP=True, k=7, safe_k=7, knn_classifier=None, knne=False, with_IH=False)


pipeline3.fit(X_train_fold3, y_train_fold3)
yhat_fold3 = pipeline3.predict(X_test_fold3)
accuracy3  = accuracy_score(y_test_fold3, yhat_fold3)

print(yhat_fold3)

precision3_macro = precision_score(y_test_fold3, yhat_fold3,average='macro')
recall3_macro    = recall_score(y_test_fold3, yhat_fold3, average='macro')

precision3_micro = precision_score(y_test_fold3, yhat_fold3,average='micro')
recall3_micro    = recall_score(y_test_fold3, yhat_fold3, average='micro')

precision3_weighted = precision_score(y_test_fold3, yhat_fold3,average='weighted')
recall3_weighted    = recall_score(y_test_fold3, yhat_fold3, average='weighted')

f1_score_macro3  = f1_score(y_test_fold3, yhat_fold3, average='macro')
f1_score_micro3  = f1_score(y_test_fold3, yhat_fold3, average='micro')
f1_score_weighted3  = f1_score(y_test_fold3, yhat_fold3, average='weighted')    
matthews_corrcoef3  = matthews_corrcoef(y_test_fold3, yhat_fold3)
fowlkes_mallows_score3  = fowlkes_mallows_score(y_test_fold3, yhat_fold3)


print('Accuracy: %.3f' % (accuracy3 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef3)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score3)
print("---- MACRO ----")
print("Precision Score macro: ", precision3_macro)
print("Recall Score macro: ", recall3_macro)
print("f1_score_macro : ", f1_score_macro3)
print("---- MICRO ----")
print("Precision Score micro: ", precision3_micro)
print("Recall Score micro: ", recall3_micro)
print("f1_score_micro : ", f1_score_micro3)
print("---- WEIGHTED ----")
print("Precision Score : ", precision3_weighted)
print("Recall Score : ", recall3_weighted)
print("f1_score_weighted : ", f1_score_weighted3)
 

#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold3, yhat_fold3))

#binarizo o treinamento e o teste
y_test_fold3 = label_binarize(y_test_fold3, classes=[0, 1, 2, 3])

y3 = label_binarize(y_train_fold3, classes=[0, 1, 2, 3])
n_classes3 = y3.shape[1]

#treino usando um contra todos
clf3 = OneVsRestClassifier(pipeline3)
clf3.fit(X_train_fold3, y3)

y_score3 = clf3.predict_proba(X_test_fold3)


fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold3[:, i], y_score3[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold3.ravel(), y_score3.ravel())
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

roc3 = roc_auc["micro"]
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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/'+tipo+'/' + baseUsada + '3FOLD-ROC'
plt.savefig(ende,dpi=300);


# PRECISION-RECALL


# For each class
precision3 = dict()
recall3 = dict()
average_precision3 = dict()
for i in range(n_classes3):
    precision3[i], recall3[i], _ = precision_recall_curve(y_test_fold3[:, i],
                                                        y_score3[:, i])
    average_precision3[i] = average_precision_score(y_test_fold3[:, i], y_score3[:, i])

# A "micro-average": quantifying score on all classes jointly
precision3["micro"], recall3["micro"], _ = precision_recall_curve(y_test_fold3.ravel(),
    y_score3.ravel())
average_precision3["micro"] = average_precision_score(y_test_fold3, y_score3,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision3["micro"]))


CPR3 = average_precision3["micro"]


#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores3 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score3 in f_scores3:
    x = np.linspace(0.01, 1)
    y = f_score3 * x / (2 * x - f_score3)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score3), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall3["micro"], precision3["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision3["micro"]))

for i, color in zip(range(n_classes3), colors):
    l, = plt.plot(recall3[i], precision3[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision3[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/'+tipo+'/' + baseUsada + '3FOLD-PR'
plt.savefig(ende,dpi=300);


# In[7]:


# FOLD4


fold4_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/4 FOLD/4FOLD - TREINAMENTO.CSV'
df_fold4_train = read_csv(fold4_train, header=None)
data_fold4_train = df_fold4_train.values

fold4_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/4 FOLD/4FOLD - TESTE.CSV'
df_fold4_test = read_csv(fold4_test, header=None)
data_fold4_test = df_fold4_test.values

X_train_fold4 = data_fold4_train[:, :-1]
X_test_fold4 = data_fold4_test[:, :-1]
y_train_fold4 = data_fold4_train[:, -1]
y_test_fold4 = data_fold4_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold4 = labelencoder_X.fit_transform(y_test_fold4)
y_train_fold4 = labelencoder_X.fit_transform(y_train_fold4)

pipeline4 = DecisionTreeClassifier(min_samples_split = 4,ccp_alpha= 0.0,criterion= 'gini',max_depth= 5,max_features= None)

#pipeline4 = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=10, max_features= 'auto', n_estimators=75)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion= 'entropy',max_depth= 3,max_features=None)
#pipeline4 = AdaBoostClassifier(arvore, algorithm='SAMME',n_estimators=200)

#pipeline4 = OLA(DFP=True, k=2, knn_classifier= None, safe_k=None,with_IH=False)

#pipeline4 = LCA(DFP=True, k=7, knn_classifier=None, with_IH=False, safe_k=4)

#pipeline4 = METADES(DFP=True, k=2, Kp=8, knn_classifier='knn', knne=False, mode='selection', safe_k= None)

#pipeline4 = KNORAE(DFP=False, k=7, safe_k=5, knn_classifier='knn', knne=False, with_IH=False)

#pipeline4 = KNORAU(DFP=True, k=5, safe_k=4, knn_classifier='knn', knne=False, with_IH=False)


pipeline4.fit(X_train_fold4, y_train_fold4)
yhat_fold4 = pipeline4.predict(X_test_fold4)
accuracy4  = accuracy_score(y_test_fold4, yhat_fold4)

precision4_macro = precision_score(y_test_fold4, yhat_fold4,average='macro')
recall4_macro    = recall_score(y_test_fold4, yhat_fold4, average='macro')

precision4_micro = precision_score(y_test_fold4, yhat_fold4,average='micro')
recall4_micro    = recall_score(y_test_fold4, yhat_fold4, average='micro')

precision4_weighted = precision_score(y_test_fold4, yhat_fold4,average='weighted')
recall4_weighted    = recall_score(y_test_fold4, yhat_fold4, average='weighted')

f1_score_macro4  = f1_score(y_test_fold4, yhat_fold4, average='macro')
f1_score_micro4  = f1_score(y_test_fold4, yhat_fold4, average='micro')
f1_score_weighted4  = f1_score(y_test_fold4, yhat_fold4, average='weighted')    
matthews_corrcoef4  = matthews_corrcoef(y_test_fold4, yhat_fold4)
fowlkes_mallows_score4  = fowlkes_mallows_score(y_test_fold4, yhat_fold4)


print('Accuracy: %.3f' % (accuracy4 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef4)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score4)
print("---- MACRO ----")
print("Precision Score macro: ", precision4_macro)
print("Recall Score macro: ", recall4_macro)
print("f1_score_macro : ", f1_score_macro4)
print("---- MICRO ----")
print("Precision Score micro: ", precision4_micro)
print("Recall Score micro: ", recall4_micro)
print("f1_score_micro : ", f1_score_micro4)
print("---- WEIGHTED ----")
print("Precision Score : ", precision4_weighted)
print("Recall Score : ", recall4_weighted)
print("f1_score_weighted : ", f1_score_weighted4)


 

#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold4, yhat_fold4))

#binarizo o treinamento e o teste
y_test_fold4 = label_binarize(y_test_fold4, classes=[0, 1, 2, 3])

y4 = label_binarize(y_train_fold4, classes=[0, 1, 2, 3])
n_classes4 = y4.shape[1]

#treino usando um contra todos
clf4 = OneVsRestClassifier(pipeline4)
clf4.fit(X_train_fold4, y4)

y_score4 = clf4.predict_proba(X_test_fold4)



fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold3[:, i], y_score3[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold3.ravel(), y_score3.ravel())
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

roc4 = roc_auc["micro"]
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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '4FOLD-ROC'
plt.savefig(ende,dpi=300);


# PRECISION-RECALL


# For each class
precision4 = dict()
recall4 = dict()
average_precision4 = dict()
for i in range(n_classes4):
    precision4[i], recall4[i], _ = precision_recall_curve(y_test_fold4[:, i],
                                                        y_score4[:, i])
    average_precision4[i] = average_precision_score(y_test_fold4[:, i], y_score4[:, i])

# A "micro-average": quantifying score on all classes jointly
precision4["micro"], recall4["micro"], _ = precision_recall_curve(y_test_fold4.ravel(),
    y_score4.ravel())
average_precision4["micro"] = average_precision_score(y_test_fold4, y_score4,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision4["micro"]))


CPR4 = average_precision4["micro"]


#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores4 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score4 in f_scores4:
    x = np.linspace(0.01, 1)
    y = f_score4 * x / (2 * x - f_score4)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score4), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall4["micro"], precision4["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision4["micro"]))

for i, color in zip(range(n_classes4), colors):
    l, = plt.plot(recall4[i], precision4[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision4[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '4FOLD-PR'
plt.savefig(ende,dpi=300);



# In[8]:


# FOLD5

fold5_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/5 FOLD/5FOLD - TREINAMENTO.CSV'
df_fold5_train = read_csv(fold5_train, header=None)
data_fold5_train = df_fold5_train.values

fold5_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/5 FOLD/5FOLD - TESTE.CSV'
df_fold5_test = read_csv(fold5_test, header=None)
data_fold5_test = df_fold5_test.values

X_train_fold5 = data_fold5_train[:, :-1]
X_test_fold5 = data_fold5_test[:, :-1]
y_train_fold5 = data_fold5_train[:, -1]
y_test_fold5 = data_fold5_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold5 = labelencoder_X.fit_transform(y_test_fold5)
y_train_fold5 = labelencoder_X.fit_transform(y_train_fold5)

pipeline5 = DecisionTreeClassifier(min_samples_split = 2,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features= None)

#pipeline5 = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=10, max_features= 'auto', n_estimators=100)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion= 'entropy',max_depth= 3,max_features=None)
#pipeline5 = AdaBoostClassifier(arvore, algorithm='SAMME',n_estimators=50)

#pipeline5 = OLA(DFP=False, k=7, knn_classifier= 'knn', safe_k=7,with_IH=False)

#pipeline5 = LCA(DFP=True, k=7, knn_classifier='knn', with_IH=False, safe_k=4)

#pipeline5 = METADES(DFP=False, k=7, Kp=8, knn_classifier='knn', knne=False, mode='selection', safe_k= 5)

#pipeline5 = KNORAE(DFP=False, k=7, safe_k=4, knn_classifier='knn', knne=True, with_IH=False)

#pipeline5 = KNORAU(DFP=False, k=8, safe_k=None, knn_classifier='knn', knne=False, with_IH=False)



pipeline5.fit(X_train_fold5, y_train_fold5)
yhat_fold5 = pipeline5.predict(X_test_fold5)
accuracy5  = accuracy_score(y_test_fold5, yhat_fold5)

precision5_macro = precision_score(y_test_fold5, yhat_fold5,average='macro')
recall5_macro    = recall_score(y_test_fold5, yhat_fold5, average='macro')

precision5_micro = precision_score(y_test_fold5, yhat_fold5,average='micro')
recall5_micro    = recall_score(y_test_fold5, yhat_fold5, average='micro')

precision5_weighted = precision_score(y_test_fold5, yhat_fold5,average='weighted')
recall5_weighted    = recall_score(y_test_fold5, yhat_fold5, average='weighted')

f1_score_macro5  = f1_score(y_test_fold5, yhat_fold5, average='macro')
f1_score_micro5  = f1_score(y_test_fold5, yhat_fold5, average='micro')
f1_score_weighted5  = f1_score(y_test_fold5, yhat_fold5, average='weighted')    
matthews_corrcoef5  = matthews_corrcoef(y_test_fold5, yhat_fold5)
fowlkes_mallows_score5  = fowlkes_mallows_score(y_test_fold5, yhat_fold5)


print('Accuracy: %.3f' % (accuracy5 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef5)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score5)
print("---- MACRO ----")
print("Precision Score macro: ", precision5_macro)
print("Recall Score macro: ", recall5_macro)
print("f1_score_macro : ", f1_score_macro5)
print("---- MICRO ----")
print("Precision Score micro: ", precision5_micro)
print("Recall Score micro: ", recall5_micro)
print("f1_score_micro : ", f1_score_micro5)
print("---- WEIGHTED ----")
print("Precision Score : ", precision5_weighted)
print("Recall Score : ", recall5_weighted)
print("f1_score_weighted : ", f1_score_weighted5)
 

#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold5, yhat_fold5))

#binarizo o treinamento e o teste
y_test_fold5 = label_binarize(y_test_fold5, classes=[0, 1, 2, 3])

y5 = label_binarize(y_train_fold5, classes=[0, 1, 2, 3])
n_classes5 = y5.shape[1]

#treino usando um contra todos
clf5 = OneVsRestClassifier(pipeline5)
clf5.fit(X_train_fold5, y5)

y_score5 = clf5.predict_proba(X_test_fold5)



fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold5[:, i], y_score5[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold5.ravel(), y_score5.ravel())
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

roc5 = roc_auc["micro"]
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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '5FOLD-ROC'
plt.savefig(ende,dpi=300);




# PRECISION-RECALL


# For each class
precision5 = dict()
recall5 = dict()
average_precision5 = dict()
for i in range(n_classes5):
    precision5[i], recall5[i], _ = precision_recall_curve(y_test_fold5[:, i],
                                                        y_score5[:, i])
    average_precision5[i] = average_precision_score(y_test_fold5[:, i], y_score5[:, i])

# A "micro-average": quantifying score on all classes jointly
precision5["micro"], recall5["micro"], _ = precision_recall_curve(y_test_fold5.ravel(),
    y_score5.ravel())
average_precision5["micro"] = average_precision_score(y_test_fold5, y_score5,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision5["micro"]))


CPR5 = average_precision5["micro"]


#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores5 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score5 in f_scores5:
    x = np.linspace(0.01, 1)
    y = f_score5 * x / (2 * x - f_score5)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score5), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall5["micro"], precision5["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision5["micro"]))

for i, color in zip(range(n_classes5), colors):
    l, = plt.plot(recall5[i], precision5[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision5[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '5FOLD-PR'
plt.savefig(ende,dpi=300);


# In[11]:


# fold6

fold6_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/6 FOLD/6FOLD - TREINAMENTO.CSV'
df_fold6_train = read_csv(fold6_train, header=None)
data_fold6_train = df_fold6_train.values

fold6_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/6 FOLD/6FOLD - TESTE.CSV'
df_fold6_test = read_csv(fold6_test, header=None)
data_fold6_test = df_fold6_test.values

X_train_fold6 = data_fold6_train[:, :-1]
X_test_fold6 = data_fold6_test[:, :-1]
y_train_fold6 = data_fold6_train[:, -1]
y_test_fold6 = data_fold6_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold6 = labelencoder_X.fit_transform(y_test_fold6)
y_train_fold6 = labelencoder_X.fit_transform(y_train_fold6)


pipeline6= DecisionTreeClassifier(min_samples_split = 2,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features=None)

#pipeline6= RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=10, max_features= 0.3, n_estimators=25)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0, criterion= 'gini', max_depth= 3, max_features= 'auto')
#pipeline6= AdaBoostClassifier(arvore, algorithm='SAMME', n_estimators=150)

#pipeline6= OLA(DFP=False, k=3, knn_classifier= None, safe_k=None,with_IH=False)

#pipeline6= LCA(DFP=False, k=3, knn_classifier=None, with_IH=False, safe_k=None)

#pipeline6= METADES(DFP=True, k=7, Kp=7, knn_classifier='knn', knne=False, mode='weighting', safe_k= None)

#pipeline6= KNORAE(DFP=True, k=8, safe_k=4, knn_classifier='knn', knne=True, with_IH=True)

#pipeline6= KNORAU(DFP=True, k=8, safe_k=5, knn_classifier='knn', knne=False, with_IH=False)



pipeline6.fit(X_train_fold6, y_train_fold6)
yhat_fold6 = pipeline6.predict(X_test_fold6)
accuracy6  = accuracy_score(y_test_fold6, yhat_fold6)

precision6_macro = precision_score(y_test_fold6, yhat_fold6,average='macro')
recall6_macro    = recall_score(y_test_fold6, yhat_fold6, average='macro')

precision6_micro = precision_score(y_test_fold6, yhat_fold6,average='micro')
recall6_micro    = recall_score(y_test_fold6, yhat_fold6, average='micro')

precision6_weighted = precision_score(y_test_fold6, yhat_fold6,average='weighted')
recall6_weighted    = recall_score(y_test_fold6, yhat_fold6, average='weighted')

f1_score_macro6  = f1_score(y_test_fold6, yhat_fold6, average='macro')
f1_score_micro6  = f1_score(y_test_fold6, yhat_fold6, average='micro')
f1_score_weighted6  = f1_score(y_test_fold6, yhat_fold6, average='weighted')    
matthews_corrcoef6  = matthews_corrcoef(y_test_fold6, yhat_fold6)
fowlkes_mallows_score6  = fowlkes_mallows_score(y_test_fold6, yhat_fold6)


print('Accuracy: %.3f' % (accuracy6 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef6)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score6)
print("---- MACRO ----")
print("Precision Score macro: ", precision6_macro)
print("Recall Score macro: ", recall6_macro)
print("f1_score_macro : ", f1_score_macro6)
print("---- MICRO ----")
print("Precision Score micro: ", precision6_micro)
print("Recall Score micro: ", recall6_micro)
print("f1_score_micro : ", f1_score_micro6)
print("---- WEIGHTED ----")
print("Precision Score : ", precision6_weighted)
print("Recall Score : ", recall6_weighted)
print("f1_score_weighted : ", f1_score_weighted6)
 

#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold6, yhat_fold6))

#binarizo o treinamento e o teste
y_test_fold6 = label_binarize(y_test_fold6, classes=[0, 1, 2, 3])

y6 = label_binarize(y_train_fold6, classes=[0, 1, 2, 3])
n_classes6 = y6.shape[1]

#treino usando um contra todos
clf6 = OneVsRestClassifier(pipeline6)
clf6.fit(X_train_fold6, y6)

y_score6 = clf6.predict_proba(X_test_fold6)


# Calcular a curva ROC e a área ROC para cada classe
fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold6[:, i], y_score6[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold6.ravel(), y_score6.ravel())
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

roc6 = roc_auc["micro"]
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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '6FOLD-ROC'
plt.savefig(ende,dpi=300);



# PRECISION-RECALL


# For each class
precision6 = dict()
recall6 = dict()
average_precision6 = dict()
for i in range(n_classes6):
    precision6[i], recall6[i], _ = precision_recall_curve(y_test_fold6[:, i],
                                                        y_score6[:, i])
    average_precision6[i] = average_precision_score(y_test_fold6[:, i], y_score6[:, i])

# A "micro-average": quantifying score on all classes jointly
precision6["micro"], recall6["micro"], _ = precision_recall_curve(y_test_fold6.ravel(),
    y_score6.ravel())
average_precision6["micro"] = average_precision_score(y_test_fold6, y_score6,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision6["micro"]))


CPR6 = average_precision6["micro"]


#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores7 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score7 in f_scores7:
    x = np.linspace(0.01, 1)
    y = f_score7 * x / (2 * x - f_score7)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score7), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall6["micro"], precision6["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision6["micro"]))

for i, color in zip(range(n_classes6), colors):
    l, = plt.plot(recall6[i], precision6[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision6[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '6FOLD-PR'
plt.savefig(ende,dpi=300);


# In[12]:


# FOLD7

fold7_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/7 FOLD/7FOLD - TREINAMENTO.CSV'
df_fold7_train = read_csv(fold7_train, header=None)
data_fold7_train = df_fold7_train.values

fold7_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/7 FOLD/7FOLD - TESTE.CSV'
df_fold7_test = read_csv(fold7_test, header=None)
data_fold7_test = df_fold7_test.values

X_train_fold7 = data_fold7_train[:, :-1]
X_test_fold7 = data_fold7_test[:, :-1]
y_train_fold7 = data_fold7_train[:, -1]
y_test_fold7 = data_fold7_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold7 = labelencoder_X.fit_transform(y_test_fold7)
y_train_fold7 = labelencoder_X.fit_transform(y_train_fold7)


pipeline7 = DecisionTreeClassifier(min_samples_split = 2,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features=None)

#pipeline7 = RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=10, max_features= 0.3, n_estimators=25)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0, criterion= 'gini', max_depth= 3, max_features= 'auto')
#pipeline7 = AdaBoostClassifier(arvore, algorithm='SAMME', n_estimators=150)

#pipeline7 = OLA(DFP=False, k=3, knn_classifier= None, safe_k=None,with_IH=False)

#pipeline7 = LCA(DFP=False, k=3, knn_classifier=None, with_IH=False, safe_k=None)

#pipeline7 = METADES(DFP=True, k=7, Kp=7, knn_classifier='knn', knne=False, mode='weighting', safe_k= None)

#pipeline7 = KNORAE(DFP=True, k=8, safe_k=4, knn_classifier='knn', knne=True, with_IH=True)

#pipeline7 = KNORAU(DFP=True, k=8, safe_k=5, knn_classifier='knn', knne=False, with_IH=False)



pipeline7.fit(X_train_fold7, y_train_fold7)
yhat_fold7 = pipeline7.predict(X_test_fold7)
accuracy7  = accuracy_score(y_test_fold7, yhat_fold7)

precision7_macro = precision_score(y_test_fold7, yhat_fold7,average='macro')
recall7_macro    = recall_score(y_test_fold7, yhat_fold7, average='macro')

precision7_micro = precision_score(y_test_fold7, yhat_fold7,average='micro')
recall7_micro    = recall_score(y_test_fold7, yhat_fold7, average='micro')

precision7_weighted = precision_score(y_test_fold7, yhat_fold7,average='weighted')
recall7_weighted    = recall_score(y_test_fold7, yhat_fold7, average='weighted')

f1_score_macro7  = f1_score(y_test_fold7, yhat_fold7, average='macro')
f1_score_micro7  = f1_score(y_test_fold7, yhat_fold7, average='micro')
f1_score_weighted7  = f1_score(y_test_fold7, yhat_fold7, average='weighted')    
matthews_corrcoef7  = matthews_corrcoef(y_test_fold7, yhat_fold7)
fowlkes_mallows_score7  = fowlkes_mallows_score(y_test_fold7, yhat_fold7)


print('Accuracy: %.3f' % (accuracy7 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef7)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score7)
print("---- MACRO ----")
print("Precision Score macro: ", precision7_macro)
print("Recall Score macro: ", recall7_macro)
print("f1_score_macro : ", f1_score_macro7)
print("---- MICRO ----")
print("Precision Score micro: ", precision7_micro)
print("Recall Score micro: ", recall7_micro)
print("f1_score_micro : ", f1_score_micro7)
print("---- WEIGHTED ----")
print("Precision Score : ", precision7_weighted)
print("Recall Score : ", recall7_weighted)
print("f1_score_weighted : ", f1_score_weighted7)
 

#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold7, yhat_fold7))

#binarizo o treinamento e o teste
y_test_fold7 = label_binarize(y_test_fold7, classes=[0, 1, 2, 3])

y7 = label_binarize(y_train_fold7, classes=[0, 1, 2, 3])
n_classes7 = y7.shape[1]

#treino usando um contra todos
clf7 = OneVsRestClassifier(pipeline7)
clf7.fit(X_train_fold7, y7)

y_score7 = clf7.predict_proba(X_test_fold7)


# Calcular a curva ROC e a área ROC para cada classe
fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold7[:, i], y_score7[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold7.ravel(), y_score7.ravel())
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

roc7 = roc_auc["micro"]
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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '7FOLD-ROC'
plt.savefig(ende,dpi=300);



# PRECISION-RECALL


# For each class
precision7 = dict()
recall7 = dict()
average_precision7 = dict()
for i in range(n_classes7):
    precision7[i], recall7[i], _ = precision_recall_curve(y_test_fold7[:, i],
                                                        y_score7[:, i])
    average_precision7[i] = average_precision_score(y_test_fold7[:, i], y_score7[:, i])

# A "micro-average": quantifying score on all classes jointly
precision7["micro"], recall7["micro"], _ = precision_recall_curve(y_test_fold7.ravel(),
    y_score7.ravel())
average_precision7["micro"] = average_precision_score(y_test_fold7, y_score7,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision7["micro"]))


CPR7 = average_precision7["micro"]


#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores7 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score7 in f_scores7:
    x = np.linspace(0.01, 1)
    y = f_score7 * x / (2 * x - f_score7)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score7), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall7["micro"], precision7["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision7["micro"]))

for i, color in zip(range(n_classes7), colors):
    l, = plt.plot(recall7[i], precision7[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision7[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '7FOLD-PR'
plt.savefig(ende,dpi=300);


# In[13]:


# FOLD8

fold8_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/8 FOLD/8FOLD - TREINAMENTO.CSV'
df_fold8_train = read_csv(fold8_train, header=None)
data_fold8_train = df_fold8_train.values

fold8_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/8 FOLD/8FOLD - TESTE.CSV'
df_fold8_test = read_csv(fold8_test, header=None)
data_fold8_test = df_fold8_test.values

X_train_fold8 = data_fold8_train[:, :-1]
X_test_fold8 = data_fold8_test[:, :-1]
y_train_fold8 = data_fold8_train[:, -1]
y_test_fold8 = data_fold8_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold8 = labelencoder_X.fit_transform(y_test_fold8)
y_train_fold8 = labelencoder_X.fit_transform(y_train_fold8)


pipeline8 = DecisionTreeClassifier(min_samples_split = 2,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features= None)

#pipeline8 = RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=None, max_features= 'auto', n_estimators=150)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion= 'entropy',max_depth= 3,max_features=None)
#pipeline8 = AdaBoostClassifier(arvore, algorithm='SAMME.R',n_estimators=50)

#pipeline8 = OLA(DFP=False, k=5, knn_classifier= 'knn', safe_k=5,with_IH=False)

#pipeline8 = LCA(DFP=False, k=5, knn_classifier='knn', with_IH=False, safe_k=5)

#pipeline8 = METADES(DFP=True, k=7, Kp=7, knn_classifier=None, knne=False, mode='hybrid', safe_k=None)

#pipeline8 = KNORAE(DFP=False, k=7, safe_k=7, knn_classifier= None, knne=True, with_IH=True)

#pipeline8 = KNORAU(DFP=False, k=7, safe_k=None, knn_classifier=None, knne=True, with_IH=True)



pipeline8.fit(X_train_fold8, y_train_fold8)
yhat_fold8 = pipeline8.predict(X_test_fold8)
accuracy8  = accuracy_score(y_test_fold8, yhat_fold8)

precision8_macro = precision_score(y_test_fold8, yhat_fold8,average='macro')
recall8_macro    = recall_score(y_test_fold8, yhat_fold8, average='macro')

precision8_micro = precision_score(y_test_fold8, yhat_fold8,average='micro')
recall8_micro    = recall_score(y_test_fold8, yhat_fold8, average='micro')

precision8_weighted = precision_score(y_test_fold8, yhat_fold8,average='weighted')
recall8_weighted    = recall_score(y_test_fold8, yhat_fold8, average='weighted')

f1_score_macro8  = f1_score(y_test_fold8, yhat_fold8, average='macro')
f1_score_micro8  = f1_score(y_test_fold8, yhat_fold8, average='micro')
f1_score_weighted8  = f1_score(y_test_fold8, yhat_fold8, average='weighted')    
matthews_corrcoef8  = matthews_corrcoef(y_test_fold8, yhat_fold8)
fowlkes_mallows_score8  = fowlkes_mallows_score(y_test_fold8, yhat_fold8)


print('Accuracy: %.3f' % (accuracy8 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef8)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score8)
print("---- MACRO ----")
print("Precision Score macro: ", precision8_macro)
print("Recall Score macro: ", recall8_macro)
print("f1_score_macro : ", f1_score_macro8)
print("---- MICRO ----")
print("Precision Score micro: ", precision8_micro)
print("Recall Score micro: ", recall8_micro)
print("f1_score_micro : ", f1_score_micro8)
print("---- WEIGHTED ----")
print("Precision Score : ", precision8_weighted)
print("Recall Score : ", recall8_weighted)
print("f1_score_weighted : ", f1_score_weighted8)
 

#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold8, yhat_fold8))

#binarizo o treinamento e o teste
y_test_fold8 = label_binarize(y_test_fold8, classes=[0, 1, 2, 3])

y8 = label_binarize(y_train_fold8, classes=[0, 1, 2, 3])
n_classes8 = y8.shape[1]

#treino usando um contra todos
clf8 = OneVsRestClassifier(pipeline8)
clf8.fit(X_train_fold8, y8)

y_score8 = clf8.predict_proba(X_test_fold8)


# Calcular a curva ROC e a área ROC para cada classe

# Calcular a curva ROC e a área ROC para cada classe
fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold8[:, i], y_score8[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold8.ravel(), y_score8.ravel())
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

roc8 = roc_auc["micro"]
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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '8FOLD-ROC'
plt.savefig(ende,dpi=300);



# PRECISION-RECALL


# For each class
precision8 = dict()
recall8 = dict()
average_precision8 = dict()
for i in range(n_classes8):
    precision8[i], recall8[i], _ = precision_recall_curve(y_test_fold8[:, i],
                                                        y_score8[:, i])
    average_precision8[i] = average_precision_score(y_test_fold8[:, i], y_score8[:, i])

# A "micro-average": quantifying score on all classes jointly
precision8["micro"], recall8["micro"], _ = precision_recall_curve(y_test_fold8.ravel(),
    y_score8.ravel())
average_precision8["micro"] = average_precision_score(y_test_fold8, y_score8,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision8["micro"]))


CPR8 = average_precision8["micro"]


#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores8 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score8 in f_scores8:
    x = np.linspace(0.01, 1)
    y = f_score8 * x / (2 * x - f_score8)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score8), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall8["micro"], precision8["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision8["micro"]))

for i, color in zip(range(n_classes8), colors):
    l, = plt.plot(recall8[i], precision8[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision8[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '8FOLD-PR'
plt.savefig(ende,dpi=300);



# In[14]:


# FOLD9

fold9_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/9 FOLD/9FOLD - TREINAMENTO.CSV'
df_fold9_train = read_csv(fold9_train, header=None)
data_fold9_train = df_fold9_train.values

fold9_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/9 FOLD/9FOLD - TESTE.CSV'
df_fold9_test = read_csv(fold9_test, header=None)
data_fold9_test = df_fold9_test.values

X_train_fold9 = data_fold9_train[:, :-1]
X_test_fold9 = data_fold9_test[:, :-1]
y_train_fold9 = data_fold9_train[:, -1]
y_test_fold9 = data_fold9_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold9 = labelencoder_X.fit_transform(y_test_fold9)
y_train_fold9 = labelencoder_X.fit_transform(y_train_fold9)


pipeline9 = DecisionTreeClassifier(min_samples_split = 4,ccp_alpha= 0.0,criterion= 'gini',max_depth= 10,max_features= None)

#pipeline9 = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=None, max_features= 0.3, n_estimators=25)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion= 'entropy',max_depth= 3,max_features='auto')
#pipeline9 = AdaBoostClassifier(arvore, algorithm='SAMME.R',n_estimators=150)

#pipeline9 = OLA(DFP=True, k=5, knn_classifier= 'knn', safe_k=None,with_IH=True)

#pipeline9 = LCA(DFP=True, k=5, knn_classifier='knn', with_IH=True, safe_k=None)

#pipeline9 = METADES(DFP=False, k=5, Kp=, knn_classifier='knn', knne=False, mode='hybrid', safe_k=None)

#pipeline9 = KNORAE(DFP=True, k=7, safe_k=7, knn_classifier='knn', knne=False, with_IH=False)

#pipeline9 = KNORAU(DFP=False, k=5, safe_k=4, knn_classifier=None, knne=False, with_IH=False)


 

pipeline9.fit(X_train_fold9, y_train_fold9)
yhat_fold9 = pipeline9.predict(X_test_fold9)
accuracy9  = accuracy_score(y_test_fold9, yhat_fold9)

precision9_macro = precision_score(y_test_fold9, yhat_fold9,average='macro')
recall9_macro    = recall_score(y_test_fold9, yhat_fold9, average='macro')

precision9_micro = precision_score(y_test_fold9, yhat_fold9,average='micro')
recall9_micro    = recall_score(y_test_fold9, yhat_fold9, average='micro')

precision9_weighted = precision_score(y_test_fold9, yhat_fold9,average='weighted')
recall9_weighted    = recall_score(y_test_fold9, yhat_fold9, average='weighted')

f1_score_macro9  = f1_score(y_test_fold9, yhat_fold9, average='macro')
f1_score_micro9  = f1_score(y_test_fold9, yhat_fold9, average='micro')
f1_score_weighted9  = f1_score(y_test_fold9, yhat_fold9, average='weighted')    
matthews_corrcoef9  = matthews_corrcoef(y_test_fold9, yhat_fold9)
fowlkes_mallows_score9  = fowlkes_mallows_score(y_test_fold9, yhat_fold9)


print('Accuracy: %.3f' % (accuracy9 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef9)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score9)
print("---- MACRO ----")
print("Precision Score macro: ", precision9_macro)
print("Recall Score macro: ", recall9_macro)
print("f1_score_macro : ", f1_score_macro9)
print("---- MICRO ----")
print("Precision Score micro: ", precision9_micro)
print("Recall Score micro: ", recall9_micro)
print("f1_score_micro : ", f1_score_micro9)
print("---- WEIGHTED ----")
print("Precision Score : ", precision9_weighted)
print("Recall Score : ", recall9_weighted)
print("f1_score_weighted : ", f1_score_weighted9)

 

#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold9, yhat_fold9))

#binarizo o treinamento e o teste
y_test_fold9 = label_binarize(y_test_fold9, classes=[0, 1, 2, 3])

y9 = label_binarize(y_train_fold9, classes=[0, 1, 2, 3])
n_classes9 = y9.shape[1]

#treino usando um contra todos
clf9 = OneVsRestClassifier(pipeline9)
clf9.fit(X_train_fold9, y9)

y_score9 = clf9.predict_proba(X_test_fold9)


# Calcular a curva ROC e a área ROC para cada classe
fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold9[:, i], y_score9[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold9.ravel(), y_score9.ravel())
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

roc9 = roc_auc["micro"]
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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '9FOLD-ROC'
plt.savefig(ende,dpi=300);



# PRECISION-RECALL


# For each class
precision9 = dict()
recall9 = dict()
average_precision9 = dict()
for i in range(n_classes9):
    precision9[i], recall9[i], _ = precision_recall_curve(y_test_fold9[:, i],
                                                        y_score9[:, i])
    average_precision9[i] = average_precision_score(y_test_fold9[:, i], y_score9[:, i])

# A "micro-average": quantifying score on all classes jointly
precision9["micro"], recall9["micro"], _ = precision_recall_curve(y_test_fold9.ravel(),
    y_score9.ravel())
average_precision9["micro"] = average_precision_score(y_test_fold9, y_score9,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision9["micro"]))


CPR9 = average_precision9["micro"]


#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores9 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score9 in f_scores9:
    x = np.linspace(0.01, 1)
    y = f_score9 * x / (2 * x - f_score9)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score9), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall9["micro"], precision9["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision9["micro"]))

for i, color in zip(range(n_classes9), colors):
    l, = plt.plot(recall9[i], precision9[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision9[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '9FOLD-PR'
plt.savefig(ende,dpi=300);


# In[17]:


# FOLD10

fold10_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/10 FOLD/10FOLD - TREINAMENTO.CSV'
df_fold10_train = read_csv(fold10_train, header=None)
data_fold10_train = df_fold10_train.values

fold10_test = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/SMOTE/10 FOLD/10FOLD - TESTE.CSV'
df_fold10_test = read_csv(fold10_test, header=None)
data_fold10_test = df_fold10_test.values

X_train_fold10 = data_fold10_train[:, :-1]
X_test_fold10 = data_fold10_test[:, :-1]
y_train_fold10 = data_fold10_train[:, -1]
y_test_fold10 = data_fold10_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold10 = labelencoder_X.fit_transform(y_test_fold10)
y_train_fold10 = labelencoder_X.fit_transform(y_train_fold10)


pipeline10 = DecisionTreeClassifier(min_samples_split = 3,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features= None)

#pipeline10 = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=10, max_features= 'auto', n_estimators=150)

#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion='entropy',max_depth= 3,max_features= None)
#pipeline10 = AdaBoostClassifier(arvore, algorithm='SAMME.R',n_estimators=50)

#pipeline10 = OLA(DFP=False, k=7, knn_classifier= None, safe_k=5,with_IH=False)

#pipeline10 = LCA(DFP=False, k=7, knn_classifier=None, with_IH=False, safe_k=5)

#pipeline10 = METADES(DFP=True, k=3, Kp=6, knn_classifier='knn', knne=False, mode='hybrid', safe_k= None)

#pipeline10 = KNORAE(DFP=True, k=8, safe_k=7, knn_classifier='knn', knne=False, with_IH=False)

#pipeline10 = KNORAU(DFP=True, k=7, safe_k=None, knn_classifier='knn', knne=False, with_IH=False)


pipeline10.fit(X_train_fold10, y_train_fold10)
yhat_fold10 = pipeline10.predict(X_test_fold10)
accuracy10  = accuracy_score(y_test_fold10, yhat_fold10)

precision10_macro = precision_score(y_test_fold10, yhat_fold10,average='macro')
recall10_macro    = recall_score(y_test_fold10, yhat_fold10, average='macro')

precision10_micro = precision_score(y_test_fold10, yhat_fold10,average='micro')
recall10_micro    = recall_score(y_test_fold10, yhat_fold10, average='micro')

precision10_weighted = precision_score(y_test_fold10, yhat_fold10,average='weighted')
recall10_weighted    = recall_score(y_test_fold10, yhat_fold10, average='weighted')

f1_score_macro10  = f1_score(y_test_fold10, yhat_fold10, average='macro')
f1_score_micro10  = f1_score(y_test_fold10, yhat_fold10, average='micro')
f1_score_weighted10  = f1_score(y_test_fold10, yhat_fold10, average='weighted')    
matthews_corrcoef10  = matthews_corrcoef(y_test_fold10, yhat_fold10)
fowlkes_mallows_score10  = fowlkes_mallows_score(y_test_fold10, yhat_fold10)

print('Accuracy: %.3f' % (accuracy10 * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef10)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score10)
print("---- MACRO ----")
print("Precision Score macro: ", precision10_macro)
print("Recall Score macro: ", recall10_macro)
print("f1_score_macro : ", f1_score_macro10)
print("---- MICRO ----")
print("Precision Score micro: ", precision10_micro)
print("Recall Score micro: ", recall10_micro)
print("f1_score_micro : ", f1_score_micro10)
print("---- WEIGHTED ----")
print("Precision Score : ", precision10_weighted)
print("Recall Score : ", recall10_weighted)
print("f1_score_weighted : ", f1_score_weighted10)


#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold10, yhat_fold10))

#binarizo o treinamento e o teste
y_test_fold10 = label_binarize(y_test_fold10, classes=[0, 1, 2, 3])

y10 = label_binarize(y_train_fold10, classes=[0, 1, 2, 3])
n_classes10 = y10.shape[1]

#treino usando um contra todos
clf10 = OneVsRestClassifier(pipeline10)
clf10.fit(X_train_fold10, y10)

y_score10 = clf10.predict_proba(X_test_fold10)


# Calcular a curva ROC e a área ROC para cada classe
fpr = dict()
tpr = dict()
mean_fpr = np.linspace(0, 1, 100)
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold10[:, i], y_score10[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold10.ravel(), y_score10.ravel())
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

roc10 = roc_auc["micro"]
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
ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '10FOLD-ROC'
plt.savefig(ende,dpi=300);



# PRECISION-RECALL


# For each class
precision10 = dict()
recall10 = dict()
average_precision10 = dict()
for i in range(n_classes10):
    precision10[i], recall10[i], _ = precision_recall_curve(y_test_fold10[:, i],
                                                        y_score10[:, i])
    average_precision10[i] = average_precision_score(y_test_fold10[:, i], y_score10[:, i])

# A "micro-average": quantifying score on all classes jointly
precision10["micro"], recall10["micro"], _ = precision_recall_curve(y_test_fold10.ravel(),
    y_score10.ravel())
average_precision10["micro"] = average_precision_score(y_test_fold10, y_score10,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision10["micro"]))



#TRAÇA A CURVA MULTICLASSE

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores10 = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score10 in f_scores10:
    x = np.linspace(0.01, 1)
    y = f_score10 * x / (2 * x - f_score10)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score10), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall10["micro"], precision10["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision10["micro"]))

for i, color in zip(range(n_classes10), colors):
    l, = plt.plot(recall10[i], precision10[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision10[i]))

CPR10 = average_precision10["micro"]

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

ende = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/' + tipo +'/' + baseUsada + '10FOLD-PR'
plt.savefig(ende,dpi=300);


# In[18]:


print('TOTAL ')
Accuracy_total              = (accuracy1+accuracy2+accuracy3+accuracy4+accuracy5+accuracy6+accuracy7+accuracy8+accuracy9+accuracy10)/10
matthews_corrcoef_total     = (matthews_corrcoef1+matthews_corrcoef2+matthews_corrcoef3+matthews_corrcoef4+matthews_corrcoef5+matthews_corrcoef6+matthews_corrcoef7+matthews_corrcoef8+matthews_corrcoef9+matthews_corrcoef10)/10
fowlkes_mallows_score_total = (fowlkes_mallows_score1+fowlkes_mallows_score2+fowlkes_mallows_score3+fowlkes_mallows_score4+fowlkes_mallows_score5+fowlkes_mallows_score6+fowlkes_mallows_score7+fowlkes_mallows_score8+fowlkes_mallows_score9+fowlkes_mallows_score10)/10
precision_macro_total       = (precision1_macro+precision2_macro+precision3_macro+precision4_macro+precision5_macro+precision6_macro+precision7_macro+precision8_macro+precision9_macro+precision10_macro)/10
recall_macro_total          = (recall1_macro+recall2_macro+recall3_macro+recall4_macro+recall5_macro+recall6_macro+recall7_macro+recall8_macro+recall9_macro+recall10_macro)/10
f1_score_macro_total        = (f1_score_macro1+f1_score_macro2+f1_score_macro3+f1_score_macro4+f1_score_macro5+f1_score_macro6+f1_score_macro7+f1_score_macro8+f1_score_macro9+f1_score_macro10)/10
precision_micro_total       = (precision1_micro+precision2_micro+precision3_micro+precision4_micro+precision5_micro+precision6_micro+precision7_micro+precision8_micro+precision9_micro+precision10_micro)/10
recall_micro_total          = (recall1_micro+recall2_micro+recall3_micro+recall4_micro+recall5_micro+recall6_micro+recall7_micro+recall8_micro+recall9_micro+recall10_micro)/10
f1_score_micro_total        = (f1_score_micro1+f1_score_micro2+f1_score_micro3+f1_score_micro4+f1_score_micro5+f1_score_micro6+f1_score_micro7+f1_score_micro8+f1_score_micro9+f1_score_micro10)/10
precision_weighted_total    = (precision1_weighted+precision2_weighted+precision3_weighted+precision4_weighted+precision5_weighted+precision6_weighted+precision7_weighted+precision8_weighted+precision9_weighted+precision10_weighted)/10
recall_weighted_total       = (recall1_weighted+recall2_weighted+recall3_weighted+recall4_weighted+recall5_weighted+recall6_weighted+recall7_weighted+recall8_weighted+recall9_weighted+recall10_weighted)/10

roc_total                   = (roc1+roc2+roc3+roc4+roc5+roc6+roc7+roc8+roc9+roc10)/10
f1_score_weighted_total     = (f1_score_weighted1+f1_score_weighted2+f1_score_weighted3+f1_score_weighted4+f1_score_weighted5+f1_score_weighted6+f1_score_weighted7+f1_score_weighted8+f1_score_weighted9+f1_score_weighted10)/10
PR_total                    = (CPR1+CPR2+CPR3+CPR4+CPR5+CPR6+CPR7+CPR8+CPR9+CPR10)/10
 
print('Accuracy: %.3f' % (Accuracy_total * 100))
print("matthews_corrcoef1 : ", matthews_corrcoef_total)
print("fowlkes_mallows_score1 : ", fowlkes_mallows_score_total)
print("ROC: ", roc_total)
print("Curva Precision-Recall: ", PR_total)
print("---- MACRO ----")
print("Precision Score macro: ", precision_macro_total)
print("Recall Score macro: ", recall_macro_total)
print("f1_score_macro : ", f1_score_macro_total)
print("---- MICRO ----")
print("Precision Score micro: ", precision_micro_total)
print("Recall Score micro: ", recall_micro_total)
print("f1_score_micro : ", f1_score_micro_total)
print("---- WEIGHTED ----")
print("Precision Score : ", precision_weighted_total)
print("Recall Score : ", recall_weighted_total)
print("f1_score_weighted : ", f1_score_weighted_total)

print("ROC TOTAL : ", roc_total)


# In[ ]:





# In[ ]:




