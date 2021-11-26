#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


fold1_train = 'D:/ARTIGO DRC/TREINAMENTO.CSV'
df_fold1_train = read_csv(fold1_train, header=None)
data_fold1_train = df_fold1_train.values

fold1_test = 'D:/ARTIGO DRC/TESTE.CSV'
df_fold1_test = read_csv(fold1_test, header=None)
data_fold1_test = df_fold1_test.values

X_train_fold1 = data_fold1_train[:, :-1]
y_train_fold1 = data_fold1_train[:, -1]

X_test_fold1 = data_fold1_test[:, :-1]
y_test_fold1 = data_fold1_test[:, -1]


labelencoder_X = LabelEncoder()
y_test_fold1 = labelencoder_X.fit_transform(y_test_fold1)
print(y_test_fold1)
y_train_fold1 = labelencoder_X.fit_transform(y_train_fold1)
print(y_train_fold1)


# In[4]:


preprocessor1 = ColumnSelector(cols=(2, 5, 7))
preprocessor2 = ColumnSelector(cols=(0, 1, 4, 5, 6))
preprocessor3 = ColumnSelector(cols=(1, 4))
preprocessor4 = ColumnSelector(cols=(0, 1, 4, 5, 7))


from sklearn.tree import DecisionTreeClassifier


arvore = DecisionTreeClassifier(min_samples_split = 4,ccp_alpha= 0.0,criterion= 'entropy',max_depth= 5,max_features= None)
pipeline = PMMLPipeline([
  ("classifier", VotingClassifier([
    ("pipe1", Pipeline(steps=[('preprocessor', preprocessor1),('classifier', arvore)])),
    ("pipe2", Pipeline(steps=[('preprocessor', preprocessor2),('classifier', arvore)])),
    ("pipe3", Pipeline(steps=[('preprocessor', preprocessor3),('classifier', arvore)])),
    ("pipe4", Pipeline(steps=[('preprocessor', preprocessor4),('classifier', arvore)])),
  ],voting='soft'))
])

#random = RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=3, max_features= 0.2, n_estimators=150, random_state=0)
#pipeline = PMMLPipeline([
#  ("classifier", VotingClassifier([
#    ("pipe1", Pipeline(steps=[('preprocessor', preprocessor1),('classifier', random)])),
#    ("pipe2", Pipeline(steps=[('preprocessor', preprocessor2),('classifier', random)])),
#    ("pipe3", Pipeline(steps=[('preprocessor', preprocessor3),('classifier', random)])),
#    ("pipe4", Pipeline(steps=[('preprocessor', preprocessor4),('classifier', random)])),
#  ],voting='soft'))
#])


#arvore = DecisionTreeClassifier(ccp_alpha= 0.0,criterion= 'gini',max_depth= None,max_features= 0.2)
#adaboost = AdaBoostClassifier(arvore, algorithm='SAMME',n_estimators=150)
#pipeline = PMMLPipeline([
#  ("classifier", VotingClassifier([
#    ("pipe1", Pipeline(steps=[('preprocessor', preprocessor1),('rfc', adaboost)])),
#    ("pipe2", Pipeline(steps=[('preprocessor', preprocessor2),('rfc', adaboost)])),
#    ("pipe3", Pipeline(steps=[('preprocessor', preprocessor3),('rfc', adaboost)])),
#    ("pipe4", Pipeline(steps=[('preprocessor', preprocessor4),('rfc', adaboost)])),
#  ],voting='soft'))
#])



#ola = OLA(DFP=False, k=5, knn_classifier= None, safe_k= 5,with_IH=True)
#pipeline = PMMLPipeline([
#  ("classifier", VotingClassifier([
#    ("pipe1", Pipeline(steps=[('preprocessor', preprocessor1),('rfc', ola)])),
#    ("pipe2", Pipeline(steps=[('preprocessor', preprocessor2),('rfc', ola)])),
#    ("pipe3", Pipeline(steps=[('preprocessor', preprocessor3),('rfc', ola)])),
#    ("pipe4", Pipeline(steps=[('preprocessor', preprocessor4),('rfc', ola)])),
#  ],voting='soft'))
#])

#lca = LCA(DFP=False, k=5, knn_classifier='knn', with_IH=False, safe_k=5)
#pipeline= PMMLPipeline([
#  ("classifier", VotingClassifier([
#    ("pipe1", Pipeline(steps=[('preprocessor', preprocessor1),('rfc', lca)])),
#    ("pipe2", Pipeline(steps=[('preprocessor', preprocessor2),('rfc', lca)])),
#    ("pipe3", Pipeline(steps=[('preprocessor', preprocessor3),('rfc', lca)])),
#    ("pipe4", Pipeline(steps=[('preprocessor', preprocessor4),('rfc', lca)])),
#  ],voting='soft'))
#])


#metades = METADES()#DFP=True, k=7, Kp=7, knn_classifier='knn', knne=False, mode='selection')#, safe_k= 6)
#pipeline = PMMLPipeline([
#  ("classifier", VotingClassifier([
#    ("pipe1", Pipeline(steps=[('preprocessor', preprocessor1),('rfc', metades)])),
#    ("pipe2", Pipeline(steps=[('preprocessor', preprocessor2),('rfc', metades)])),
#    ("pipe3", Pipeline(steps=[('preprocessor', preprocessor3),('rfc', metades)])),
#    ("pipe4", Pipeline(steps=[('preprocessor', preprocessor4),('rfc', metades)])),
#  ],voting='soft'))
#])



#knorae = KNORAE(DFP=True, k=7, safe_k=7, knn_classifier='knn', knne=False, with_IH=True)
#pipeline = PMMLPipeline([
#  ("classifier", VotingClassifier([
#    ("pipe1", Pipeline(steps=[('preprocessor', preprocessor1),('rfc', knorae)])),
#    ("pipe2", Pipeline(steps=[('preprocessor', preprocessor2),('rfc', knorae)])),
#    ("pipe3", Pipeline(steps=[('preprocessor', preprocessor3),('rfc', knorae)])),
#    ("pipe4", Pipeline(steps=[('preprocessor', preprocessor4),('rfc', knorae)])),
#  ],voting='soft'))
#])


#knorau = KNORAU(DFP=True, k=7, safe_k=None, knn_classifier=None, knne=False, with_IH=True)
#pipeline1 = PMMLPipeline([
#  ("classifier", VotingClassifier([
#    ("pipe1", Pipeline(steps=[('preprocessor', preprocessor1),('rfc', knorau)])),
#    ("pipe2", Pipeline(steps=[('preprocessor', preprocessor2),('rfc', knorau)])),
#    ("pipe3", Pipeline(steps=[('preprocessor', preprocessor3),('rfc', knorau)])),
#    ("pipe4", Pipeline(steps=[('preprocessor', preprocessor4),('rfc', knorau)])),
#  ],voting='soft'))
#])



# In[5]:



pipeline.fit(X_train_fold1, y_train_fold1)
yhat_fold1 = pipeline.predict(X_test_fold1)
accuracy1  = accuracy_score(y_test_fold1, yhat_fold1)

precision1_macro = precision_score(y_test_fold1, yhat_fold1,average='macro')
recall1_macro     = recall_score(y_test_fold1, yhat_fold1, average='macro')

precision1_micro = precision_score(y_test_fold1, yhat_fold1,average='micro')
recall1_micro     = recall_score(y_test_fold1, yhat_fold1, average='micro')

precision1 = precision_score(y_test_fold1, yhat_fold1,average='weighted')
recall1     = recall_score(y_test_fold1, yhat_fold1, average='weighted')

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
print("Precision Score : ", precision1)
print("Recall Score : ", recall1)
print("f1_score_weighted : ", f1_score_weighted1)

#print("ROC AUC Score : ", roc_auc_score1)
print (classification_report(y_test_fold1, yhat_fold1))


# In[6]:


#binarizo o treinamento e o teste
y_test_fold1 = label_binarize(y_test_fold1, classes=[0, 1, 2, 3])

y = label_binarize(y_train_fold1, classes=[0, 1, 2, 3])
n_classes = y.shape[1]

#treino usando um contra todos
clf = OneVsRestClassifier(pipeline)
clf.fit(X_train_fold1, y)

y_score = clf.predict_proba(X_test_fold1)


# In[7]:


from sklearn.metrics import roc_curve, auc

# Calcular a curva ROC e a área ROC para cada classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_fold1[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_fold1.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print('roc_auc all classes: {0:0.2f}'
      .format(roc_auc["micro"]))

# Gráfico de uma curva ROC para uma classe específica
#plt.figure()
#plt.plot(fpr[3], tpr[3], label='ROC curve (area = %0.2f)' % roc_auc[3])
#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('curva ROC para uma classe específica')
#plt.legend(loc="lower right")
#plt.show()

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC multi-class')
plt.legend(loc="lower right")
plt.show()



# In[8]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_fold1[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test_fold1[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_fold1.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test_fold1, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

#CURVA PRA MEDIA 

plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
.format(average_precision["micro"]))


#TRAÇA A CURVA MULTICLASSE

from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


plt.show()


# In[ ]:





# In[ ]:




