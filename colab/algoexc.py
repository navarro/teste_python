import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import joblib
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#abrir arquivo / Validação de conteúdo
df = pd.read_csv('REF_S_RG2.csv',encoding='latin1')
df.head()

#Validação de shape
df.shape

# retorna um novo objeto com o número de valores ausentes (NA)
df.isna().sum()

# Guardar um eixo de aprendizagem baseado no score para os demais atributos definam as as projeções do Score
scores = df.columns[df.columns.str.contains('score')].tolist()
scores

#calcula o viés (skewness) da distribuição, ou seja, a assimetria da distribuição em relação à sua média. Calcula o achatamento da distribuição em relação à uma distribuição normal
for score in scores:
    sns.histplot(df[score], element='bars', kde=True)
    plt.text(x=20, y=90, s=f"Skew: {round(df[score].skew(),2)}\nKurtosis: {round(df[score].kurt(),2)}")
    plt.show()
	
#Retorna um resumo estatístico que inclui contagens, médias, desvios padrão, valores mínimos e máximos
df.describe()

# Gerar gráfico de dispersão que mostra todas as combinações possíveis de pares de variáveis numéricas de um conjunto de dados
sns.pairplot(data=df)

# Gerar gráfico para análise de datos para verificação de score em função da perspectiva de Instrução ( Use o mesmo para os demais atributos, como ocupação, idade...)
for score in scores: 
    sns.boxplot(y=df[score], x=df['INSTRUCAO'])
    plt.figure(figsize=(10,6))
    ax= plt.subplot()
    plt.setp(ax.get_xticklabels(), fontsize=7, rotation=45)
    plt.show()
	
for score in scores: 
    sns.boxplot(y=df[score], x=df['OCUPACAO'])
    plt.figure(figsize=(10,6))
    ax= plt.subplot()
    plt.setp(ax.get_xticklabels(), fontsize=7, rotation=45)
    plt.show()

for score in scores: 
    sns.boxplot(y=df[score], x=df['IDADE'])
    plt.figure(figsize=(10,6))
    ax= plt.subplot()
    plt.setp(ax.get_xticklabels(), fontsize=7, rotation=45)
    plt.show()

# Definir um dedo para obter uma projeção ( Prever ) um valor de Score. 

cols = ['dedo310 score']
X, y = df.drop(cols, axis=1), df['dedo310 score']


# Eixo de análise e aplicação de modelo de dados de machine learning para obtenção de projeção
cat_cols = X.dtypes[X.dtypes == 'O'].index.tolist()
cat_cols

ct = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_cols)
], remainder='passthrough')


ct.fit_transform(X).shape

# Modelo ML

pipe = Pipeline([
    ('trf', ct),
    ('model', LGBMRegressor(random_state=0))
])

#Parametros de Calibragem do modelo
params = {
    'model__n_estimators':[100,130,150,170,190],
    'model__boosting_type': ['dart', 'gbdt', 'goss']
}

gs = GridSearchCV(pipe, param_grid=params, scoring='neg_root_mean_squared_error', n_jobs=-1)
gs.fit(X, y)
pd.DataFrame(gs.cv_results_).sort_values(by='rank_test_score')
gs.best_params_
gs.best_score_
gs.best_estimator_
joblib.dump(gs.best_estimator_, 'model.joblib')
mdl = joblib.load('model.joblib')
q = pd.DataFrame([['RUDIMENTAR', 'COZINHEIRO(A)', 'M', 'SALVADOR', 'IDOSO',145,101,124,102,76,160,100,118,96]],
             columns=X.columns)
q
mdl.predict(q)