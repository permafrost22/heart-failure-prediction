# Predição de óbito por insuficiência cardíaca

Análise exploratória e comparação de modelos de classificação para prever o
desfecho `DEATH_EVENT` a partir de registros clínicos de pacientes com
insuficiência cardíaca.

## Dataset

[Predict survival of patients with heart failure](https://www.kaggle.com/datasets/rabieelkharoua/predict-survival-of-patients-with-heart-failure) (Kaggle),
com 299 pacientes e 13 variáveis clínicas (idade, fração de ejeção, creatinina
sérica, sódio sérico, plaquetas, entre outras). A coluna `time` (tempo de
acompanhamento) é removida da análise por representar vazamento de informação
em relação ao desfecho.

## O que o script faz

1. **Exploração dos dados** — tipos, valores ausentes, estatísticas descritivas
   das variáveis categóricas e numéricas, distribuições e matriz de correlação.
2. **Separação treino/teste** feita antes de qualquer tratamento, pra evitar
   vazamento de informação do teste.
3. **Tratamento de outliers** pelo critério de Tukey (1.5×IQR), aplicado em
   `creatinine_phosphokinase`, `platelets` e `serum_creatinine`.
4. **Normalização** com `MinMaxScaler`, calibrado apenas no treino.
5. **Treinamento e comparação de modelos**:
   - KNN (com `GridSearchCV` para escolher o melhor *k*)
   - Árvore de decisão
   - Random Forest (com busca de hiperparâmetros)
   - SVM (com busca de hiperparâmetros)
   - Regressão Logística (com busca de hiperparâmetros e ajuste de threshold
     via curva ROC, calculado numa fatia de validação separada do teste)
6. Cada modelo é avaliado com a mesma função (`avalia_modelo`): acurácia,
   matriz de confusão, relatório de classificação e curva ROC.

## Como rodar

```bash
pip install -r requirements.txt
python heart_failure_limpo.py
```

O script lê o CSV a partir da própria pasta onde ele está salvo, então basta
colocar `heart_failure_clinical_records_dataset.csv` na mesma pasta do script

Se preferir rodar no Kaggle, troque a leitura do passo 1 pelo caminho do
dataset em `/kaggle/input/...`.

## Estrutura

```
.
├── heart_failure_limpo.py
├── requirements.txt
└── README.md
└──heart_failure_clinical_records_dataset.csv
```

## Observações

- As curvas ROC são calculadas em cima das classes previstas (`predict`), não
  das probabilidades, mantém a mesma abordagem usada em todos os modelos
  pra ficarem comparáveis entre si.
- A busca de hiperparâmetros do Random Forest e do SVM pode demorar alguns
  minutos, dependendo da máquina.
