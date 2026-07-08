import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                              classification_report, roc_curve, auc)

pd.set_option('display.max_columns', None)

#passo 1
#lendo os dados

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

print(df.head().to_string())
print(df.info())
print("-"*50)

#passo 2
#a coluna 'time' não faz sentido como preditora (ela está ligada ao próprio
#desfecho, então é praticamente um vazamento de informação), então ela sai

df.drop(['time'], axis=1, inplace=True)

print(df.dtypes)
print(df.isna().sum())
print("-"*50)

#passo 3
#separando colunas categóricas de numéricas e ajustando o tipo das categóricas

colunas_categoricas = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
df[colunas_categoricas] = df[colunas_categoricas].astype('category')

colunas_numericas = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                      'platelets', 'serum_creatinine', 'serum_sodium']

print("Colunas categóricas:")
print(df[colunas_categoricas].describe().to_string())
print("-"*50)

print("Colunas numéricas:")
print(df[colunas_numericas].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_string())
print("-"*50)

#passo 4
#olhando a distribuição das variáveis que aparentam ter outliers

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
sns.histplot(data=df, x='creatinine_phosphokinase', bins=35, kde=True, ax=axes[0, 0])
sns.histplot(data=df, x='ejection_fraction', bins=35, kde=True, ax=axes[0, 1])
sns.histplot(data=df, x='platelets', bins=35, kde=True, ax=axes[1, 0])
sns.histplot(data=df, x='serum_creatinine', bins=35, kde=True, ax=axes[1, 1])
plt.tight_layout()
plt.show()

#passo 5
#matriz de correlação, pra entender como as variáveis numéricas se relacionam

plt.figure(figsize=(7, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', square=True,
            linewidth=.5, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de correlação', fontweight='bold')
plt.show()

#passo 6
#separando treino e teste antes de tratar outliers e normalizar, pra não
#deixar informação do teste vazar pro treino

X = df.drop(['DEATH_EVENT'], axis=1)
y = df['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("-"*50)

#passo 7
#tratando outliers pelo critério de Tukey:
#outlier inferior se < 1º quartil - 1.5 * IQR
#outlier superior se > 3º quartil + 1.5 * IQR

def Tukey_outliers(df, var, k=1.5):
    IQR = df[var].quantile(0.75) - df[var].quantile(0.25)
    lim_inf = df[var].quantile(0.25) - k * IQR
    lim_sup = df[var].quantile(0.75) + k * IQR
    return lim_inf, lim_sup

colunas_outliers = ['creatinine_phosphokinase', 'platelets', 'serum_creatinine']

for var in colunas_outliers:
    print(f'{var}: {Tukey_outliers(X_train, var)}')

for var in colunas_outliers:
    inf, sup = Tukey_outliers(X_train, var)
    X_train.loc[X_train[var] < inf, var] = inf
    X_train.loc[X_train[var] > sup, var] = sup
    X_test.loc[X_test[var] < inf, var] = inf
    X_test.loc[X_test[var] > sup, var] = sup

print("outliers tratados!!")
print("-"*50)

#passo 8
#conferindo o efeito do tratamento nas distribuições

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
sns.histplot(data=X_train, x='creatinine_phosphokinase', bins=35, kde=True, ax=axes[0, 0])
sns.histplot(data=X_train, x='ejection_fraction', bins=35, kde=True, ax=axes[0, 1])
sns.histplot(data=X_train, x='platelets', bins=35, kde=True, ax=axes[1, 0])
sns.histplot(data=X_train, x='serum_creatinine', bins=35, kde=True, ax=axes[1, 1])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
sns.boxplot(data=X_train, x='creatinine_phosphokinase', ax=axes[0, 0])
sns.boxplot(data=X_train, x='ejection_fraction', ax=axes[0, 1])
sns.boxplot(data=X_train, x='platelets', ax=axes[1, 0])
sns.boxplot(data=X_train, x='serum_creatinine', ax=axes[1, 1])
plt.tight_layout()
plt.show()

#passo 9
#normalizando os dados com base apenas no treino

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(pd.DataFrame(X_train).describe())
print("-"*50)

#passo 10
#função pra não ficar repetindo o mesmo bloco de avaliação (matriz de confusão,
#relatório de classificação e curva ROC) a cada modelo treinado

classes = ['Saudável', 'Exposto']

def avalia_modelo(model, X_test, y_test, nome='modelo'):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Acurácia ({nome}): {acc:.4f}")

    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title(f'Matriz de confusão - {nome}')
    plt.show()

    print(classification_report(y_test, predictions, target_names=classes))

    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkblue', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Curva ROC - {nome}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return acc

#passo 11 - KNN

model_knn = KNeighborsClassifier(n_neighbors=7)
model_knn.fit(X_train, y_train)
avalia_modelo(model_knn, X_test, y_test, 'KNN (k=7)')

cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=7), X_train, y_train,
                             scoring='accuracy', cv=5)
print(cv_scores)
print('cv_scores mean: {:.4f}'.format(np.mean(cv_scores)))
print("-"*50)

num_viz = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(KNeighborsClassifier(), num_viz, cv=5)
knn_gscv.fit(X_train, y_train)
print(f"O melhor score obtido foi {knn_gscv.best_score_} usando {knn_gscv.best_params_}")

model_knn = KNeighborsClassifier(**knn_gscv.best_params_)
model_knn.fit(X_train, y_train)
avalia_modelo(model_knn, X_test, y_test, f"KNN (melhor k={knn_gscv.best_params_['n_neighbors']})")

#passo 12 - árvore de decisão

dt_classifier = DecisionTreeClassifier(random_state=24)
dt_classifier.fit(X_train, y_train)
avalia_modelo(dt_classifier, X_test, y_test, 'Árvore de decisão')

#passo 13 - random forest

model_rf = RandomForestClassifier(random_state=24)
model_rf.fit(X_train, y_train)
print("Random forest (parâmetros padrão):", model_rf.score(X_test, y_test))

#atenção: essa busca pode demorar um pouco pra rodar
param_grid_rf = {
    'n_estimators': [200, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [7, 8, 9, 10],
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=24), param_grid_rf, cv=5, verbose=1)
grid_rf.fit(X_train, y_train)
print('\nMelhores parâmetros:', grid_rf.best_params_)
print('Score:', grid_rf.best_score_)

model_rf = RandomForestClassifier(random_state=24, **grid_rf.best_params_)
model_rf.fit(X_train, y_train)
avalia_modelo(model_rf, X_test, y_test, 'Random forest')

#passo 14 - svm

model_svm = SVC()
model_svm.fit(X_train, y_train)
print("SVM (parâmetros padrão):", model_svm.score(X_test, y_test))

param_grid_svm = {
    'C': [1, 10, 50],
    'gamma': [1, 0.1, 0.005, 0.00001, 0.0000001],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5, verbose=1)
grid_svm.fit(X_train, y_train)
print('\nMelhores parâmetros:', grid_svm.best_params_)
print('Score:', grid_svm.best_score_)

model_svm = SVC(**grid_svm.best_params_)
model_svm.fit(X_train, y_train)
avalia_modelo(model_svm, X_test, y_test, 'SVM')

#passo 15 - regressão logística

model_logit = LogisticRegression(random_state=24)
model_logit.fit(X_train, y_train)
print("Regressão logística (parâmetros padrão):", model_logit.score(X_test, y_test))

param_grid_logit = {
    'penalty': ['l1', 'l2'],
    'C': [100, 10, 1, 0.1, 0.01, 0.0001]
}
grid_logit = GridSearchCV(LogisticRegression(random_state=24), param_grid_logit,
                           scoring='accuracy', cv=5, verbose=1)
grid_logit.fit(X_train, y_train)
print('\nMelhores parâmetros:', grid_logit.best_params_)
print('Score:', grid_logit.best_score_)

#passo 16
#achando o threshold ideal numa fatia de validação separada do treino, pra não
#usar o teste nessa escolha

X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model_val = LogisticRegression(random_state=24)
model_val.fit(X_train_new, y_train_new)

y_scores_val = model_val.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC - validação')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Threshold ótimo para equilíbrio TPR-FPR: {optimal_threshold}")
print("-"*50)

#passo 17
#treinando o modelo final com os melhores parâmetros e avaliando dos dois jeitos:
#com o threshold padrão (0.5) e com o threshold ajustado

model_logit = LogisticRegression(random_state=24, **grid_logit.best_params_)
model_logit.fit(X_train, y_train)
avalia_modelo(model_logit, X_test, y_test, 'Regressão logística')

y_scores = model_logit.predict_proba(X_test)[:, 1]
y_pred_ajustado = (y_scores >= optimal_threshold).astype(int)
acc_ajustada = accuracy_score(y_test, y_pred_ajustado)
print(f"Acurácia com threshold ajustado ({optimal_threshold:.3f}): {acc_ajustada:.4f}")