#!/usr/bin/env python
# coding: utf-8

# In[8]:


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

#carrega o arquivo pra ser gerado os novos dados 
fold1_train = 'D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/CROSS-VALIDATION/AUMENTO MANUAL/10 FOLD/10FOLD - TREINAMENTO.CSV'
df_fold1_train = read_csv(fold1_train, header=None)
data_fold1_train = df_fold1_train.values

X_train_fold1 = data_fold1_train[:, :-1]
y_train_fold1 = data_fold1_train[:, -1]

labelencoder_X = LabelEncoder()
y_train_fold1 = labelencoder_X.fit_transform(y_train_fold1)


#técnicas de reamostragem
svmsmote = SVMSMOTE(random_state = 10,k_neighbors = 4)
oversample = SMOTE(k_neighbors = 3, random_state=10)
bsmote = BorderlineSMOTE(random_state = 10, kind = 'borderline-1', k_neighbors = 3)

#descomentar essa linha e comentar as outras pra usar svmsmote
#X_oversample_fold1, y_oversample_fold1 = svmsmote.fit_resample(X_train_fold1, y_train_fold1)

#bordelinesmote
X_oversample_fold1, y_oversample_fold1 = bsmote.fit_resample(X_train_fold1, y_train_fold1)

#descomentar essa linha e comentar as anteriores para usar smote
#X_oversample_fold1, y_oversample_fold1 = oversample.fit_resample(X_train_fold1, y_train_fold1)

#local pra salvar o arquivo
np.savetxt('D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/GITHUB/BASE DE DADOS/CV/X_oversample_fold1.csv', X_oversample_fold1, delimiter=',',fmt='%s')
np.savetxt('D:/Mestrado/ARTIGO DRC/dados_pos_revisao/cross validation - dados reavaliados/4 revisao/GITHUB/BASE DE DADOS/CV/y_oversample_fold1.csv', y_oversample_fold1, delimiter=',',fmt='%s')

print(pd.Series(y_oversample_fold1).value_counts())


# In[ ]:





# In[ ]:





# In[ ]:




