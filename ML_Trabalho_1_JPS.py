#!/usr/bin/env python
# coding: utf-8

# #**Trabalho Prático 1**

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
import time


# ##**Task 1**

# ###Task 1 - Setting the Baseline

# In[8]:


#Carregar os dados
X_train_df = pd.read_csv("X_train.csv")
X_test_df = pd.read_csv("X_test.csv")
X_train_df.head()
print(X_train_df.shape)


# In[4]:


# Trajetorias
traj_length = 258
num_trajectories = len(X_train_df) // traj_length
num_plots = 3
fig, axes = plt.subplots(num_plots, 1, figsize=(8, 6 * num_plots), squeeze=False)

for i in range(num_plots):
    start = i * traj_length
    end = start + traj_length
    traj = X_train_df.iloc[start:end]

    # Plot positions for each body
    axes[i, 0].plot(traj['x_1'], traj['y_1'], label='Body 1')
    axes[i, 0].plot(traj['x_2'], traj['y_2'], label='Body 2')
    axes[i, 0].plot(traj['x_3'], traj['y_3'], label='Body 3')
    axes[i, 0].set_title(f"Trajectory {i+1}")
    axes[i, 0].set_xlabel('X Position')
    axes[i, 0].set_ylabel('Y Position')
    axes[i, 0].legend()

plt.tight_layout()
plt.show()


# In[5]:


X_train_df.head()


# In[9]:


X_train_df.drop('Id', axis=1, inplace = True)
X_train_df['all_zero'] = (X_train_df == 0).all(axis=1)
X_train_df.head()


# In[10]:


# Definir trajetorias
X_train_df['traj_id'] = (X_train_df.index //traj_length) +1
X_train_df.set_index(['traj_id', X_train_df.index], inplace=True)
X_train_df


# In[12]:


# Posições iniciais
initial_positions = X_train_df[X_train_df['t'] == 0].copy()  
initial_positions = initial_positions[initial_positions['all_zero'] == False]  
initial_positions = initial_positions.groupby('traj_id').first().reset_index()
print(initial_positions.head())
print(initial_positions.shape)


# In[13]:


# Dataframe completo
initial_positions = initial_positions.rename(columns={
    'x_1': 'x0_1', 'y_1': 'y0_1',
    'x_2': 'x0_2', 'y_2': 'y0_2',
    'x_3': 'x0_3', 'y_3': 'y0_3'
})

colunas = ['traj_id', 'x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']
initial_positions = initial_positions[colunas]

X_train_joined = X_train_df.merge(initial_positions, on='traj_id', how='left')
X_train_joined.drop('all_zero', axis=1, inplace=True)
X_train_joined = X_train_joined.dropna()

print(X_train_joined.head())
print(X_train_joined.shape)


# In[14]:


# Features:  posições iniciais
feature_cols = ['t','x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']
x = X_train_joined[feature_cols]


# Targets: posições atuais
target_cols = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
y = X_train_joined[target_cols]

print(x.head())
print(y.head())
print(x.shape)


# In[15]:


# Divisao dos dados apartir das trajetorias
init_df = X_train_joined[X_train_joined['t'] == 0][['traj_id']]
init_df = init_df.drop_duplicates()

train_init, val_init = train_test_split(init_df, test_size=0.2, random_state=42)
X_train_split = X_train_joined[X_train_joined['traj_id'].isin(train_init['traj_id'])]
X_val_split = X_train_joined[X_train_joined['traj_id'].isin(val_init['traj_id'])]

X_train_final = X_train_split[feature_cols]
y_train_final = X_train_split[target_cols]
X_val_final = X_val_split[feature_cols]
y_val_final = X_val_split[target_cols]


# Alinhar indices
X_train_final = X_train_final.reset_index(drop=True)
y_train_final = y_train_final.reset_index(drop=True)
X_val_final = X_val_final.reset_index(drop=True)
y_val_final = y_val_final.reset_index(drop=True)


print(X_val_final.shape)
print(y_val_final.shape)
print(X_train_final.shape)
print(y_train_final.shape)


# In[17]:


#Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline.fit(X_train_final, y_train_final)


# In[ ]:


# Previsões com o modelo treinado no conjunto de validação
y_pred = pipeline.predict(X_val_final)


# In[19]:


#Função para realizar os gráficos
def plot(y_test, y_pred, plot_title="plot"):
    labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    MAX = 500
    idx = np.random.choice(len(y_test), min(MAX, len(y_test)), replace=False)

    plt.figure(figsize=(12,12))

    # Cria um gráfico para cada variável de saída 
    for i in range(6):
        x0, x1 = np.min(y_test.iloc[idx, i]), np.max(y_test.iloc[idx, i])
        plt.subplot(3, 2, i + 1)
        plt.scatter(y_test.iloc[idx, i], y_pred[idx, i], alpha=0.5)
        plt.xlabel(f"True {labels[i]}")
        plt.ylabel(f"Pred {labels[i]}")
        plt.plot([x0, x1], [x0, x1], color="red")
        plt.axis("square")
    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.show()
    plt.savefig('baseline.pdf')

 # Gerar o gráfico
plot(y_val_final, y_pred, plot_title="baseline_y_yhat")


# In[20]:


X_test_df


# In[21]:


# Avaliação do modelo com o RMSE
rmse_test = np.sqrt(mean_squared_error(y_pred, y_val_final))
print(f"RMSE: {rmse_test:.6f}")


# In[ ]:


# Previsões com o modelo treinado no conjunto de Test
X_test = X_test_df.drop('Id', axis=1)
predictions = pipeline.predict(X_test)
print(predictions.shape)


# In[ ]:


#Ficheiro para submeter
labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
predictions_df = pd.DataFrame(predictions, columns=labels)
predictions_df.insert(0, 'id', range(len(predictions)))
predictions_df.to_csv('baseline-model.csv', index=False)
predictions_df.shape


# ##**Task 2**

# ###Task 2 - Nonlinear models on the data - The Polynomial Regression model

# In[ ]:


#Separar os dados em samples, para reduzir o tempo de treino
X_train_sample, _, y_train_sample, _ = train_test_split(X_train_final, y_train_final, train_size=0.01, random_state=42)
X_val_sample, _, y_val_sample, _ = train_test_split(X_val_final,y_val_final, train_size=0.01, random_state=42)

#Função regressão polinomial
def validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None, degrees=range(1,9), max_features=None):
    if regressor is None:
        regressor = LinearRegression()

    best_rmse = float('inf')
    best_model = None
    best_degree = None
    rmses = []

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        pipeline = Pipeline([
            ('poly', poly),
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])

        # Fit nas samples
        pipeline.fit(X_train_sample, y_train_sample)

        # Número de features 
        n_features = pipeline.named_steps['poly'].n_output_features_
        print(f"Degree {degree}: = {n_features}")

        # Predict no conjunto de validação e calculo do RMSE
        y_pred_poly = pipeline.predict(X_val_sample)
        rmse_val = np.sqrt(mean_squared_error(y_val_sample, y_pred_poly))
        rmses.append(rmse_val)
        print(f"{degree}: RMSE = {rmse_val:.6f}")

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = pipeline
            best_degree = degree

    print(f"Best degree: {best_degree} with RMSE: {best_rmse:.6f}")
    return best_model, best_rmse


# In[37]:


#NAO CORRER
# Com LinearRegression (default)
best_model_lr, best_rmse_lr = validate_poly_regression(X_train_final, y_train_final, X_val_sample, y_val_sample, degrees=range(1,9))


# In[ ]:


#NAO CORRER
#Com RidgeCV para regularização
alphas = [0.1, 1.0, 10.0] 
ridge_cv = RidgeCV(alphas=alphas)
best_model_ridge, best_rmse_ridge = validate_poly_regression(X_train_final, y_train_final, X_val_sample, y_val_sample, regressor=ridge_cv, degrees=range(1,9))


# In[ ]:


#NAO CORRER
# Ciclo de 10 runs, para encontrar o melhor grau 
selected_degrees = []
for i in range(10):
    X_train_sample, _, y_train_sample, _ = train_test_split(X_train_final, y_train_final, train_size=0.01, random_state=i)
    X_val_sample, _, y_val_sample, _ = train_test_split(X_val_final,y_val_final, train_size=0.01, random_state=i)
    best_model, _ = validate_poly_regression(X_train_sample, y_train_sample, X_val_final, y_val_final, degrees=range(1,9))
    best_degree = best_model.named_steps['poly'].degree 
    selected_degrees.append(best_degree)


# Plot distribuição
plt.figure(figsize=(8,6))
sns.histplot(selected_degrees, bins=range(1,9), kde=False)
plt.title("Distribution of Selected Polynomial Degrees over 10 Runs")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()
plt.savefig('degree_distribution.pdf') 


# In[ ]:


#NAO CORRER
#Previsão com o conjunto de validação
poly = PolynomialFeatures(degree=3, include_bias=False)
pipeline_poly = Pipeline([
    ('poly', poly),
    ('scaler', StandardScaler()),
    ('regressor', RidgeCV(alphas=[0.1, 1.0, 10.0]))  
])
pipeline_poly.fit(X_train_final, y_train_final)

# Predict no test split interno
y_pred_poly = pipeline_poly.predict(X_val_final)
rmse_poly = np.sqrt(mean_squared_error(y_pred_poly, y_val_final))
print(f"Polynomial RMSE: {rmse_poly:.6f}")


# In[ ]:


#NAO CORRER
# Comparação RMSE
print(f"Baseline RMSE: {rmse_test:.6f}")  
print(f"Polynomial RMSE: {rmse_poly:.6f}")
print(abs(rmse_poly - rmse_test))

plot(y_val_final, y_pred_poly, plot_title="polynomial_y_yhat")


# In[ ]:


#NAO CORRER
#Ficheiro para submeter
predictions_poly = pipeline_poly.predict(X_test)
labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
predictions_df_poly = pd.DataFrame(predictions_poly, columns=labels)
predictions_df_poly.insert(0, 'id', range(len(predictions_poly)))
predictions_df_poly.to_csv('polynomial_submission.csv', index=False)


# ##**Task 3**

# ###Task 3.1 - Removing variables

# Objetivo: ver relações lineares entre features → eliminar redundantes.

# In[ ]:


# Pairplot (gráfico de dispersão)
sns.pairplot(X_train_joined[feature_cols + target_cols].sample(200), kind="hist")
plt.show()
plt.savefig('feature_pairplot.pdf')

# Matriz de correlação 
corr = X_train_joined[feature_cols + target_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
plt.savefig('correlation_heatmap.pdf')
corr_unstacked = corr.abs().unstack().sort_values(ascending=False)
corr_unstacked = corr_unstacked[corr_unstacked < 1.0]
print(corr_unstacked.head(20))  # top 20 correlações


# ###Task 3.2 - Evaluation of Variable Reduction

# In[28]:


# Função para testar remoção
def test_feature_removal(feature_to_remove, X_train, y_train, X_test, y_test):
    reduced_cols = [col for col in feature_cols if col != feature_to_remove]
    X_train_red = X_train[reduced_cols]
    X_test_red = X_test[reduced_cols]

    pipeline_red = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    pipeline_red.fit(X_train_red, y_train)

    y_pred_red = pipeline_red.predict(X_test_red)
    rmse_red = np.sqrt(mean_squared_error(y_pred_red, y_test))
    print(f"RMSE sem {feature_to_remove}: {rmse_red:.6f}")

    # Plot para comparação
    plot(y_test, y_pred_red, plot_title=f"baseline_after_remove_{feature_to_remove}")

    return rmse_red, pipeline_red

# Teste remoções sequenciais 
candidates_to_remove = ['x0_3', 'y0_2', 'y0_3']  
rmses_removal = {}
current_cols = feature_cols.copy()
for feat in candidates_to_remove:
    rmse, _ = test_feature_removal(feat, X_train_final, y_train_final, X_val_final, y_val_final)
    rmses_removal[feat] = rmse

print(f"Original Baseline RMSE: {rmse_test:.6f}")
print("RMSEs remoções:", rmses_removal)


# In[20]:


reduced_feature_cols = ['t','x0_1','y0_1','x0_2','y0_2','x0_3'] #sem o y_03
X_train_red = X_train_final[reduced_feature_cols]
X_val_red = X_val_final[reduced_feature_cols]


# In[ ]:


#NAO CORRER
# Melhor grau sem y0_3 (escolhido)
best_model_red, best_rmse_red = validate_poly_regression(X_train_red, y_train_final, X_val_red, y_val_final, regressor=RidgeCV(alphas=[0.1, 1.0, 10.0]), degrees=range(1,9))


# In[ ]:


#NAO CORRER
#Previsão com o conjunto de validação
poly = PolynomialFeatures(degree=6, include_bias=False)
regressor = RidgeCV(alphas=[0.1, 1.0, 10.0])  
pipeline_poly_red = Pipeline([
    ('poly', poly),
    ('scaler', StandardScaler()),
    ('regressor', regressor)
])

pipeline_poly_red.fit(X_train_red, y_train_final)
y_pred_poly_red = pipeline_poly_red.predict(X_val_red)
rmse_poly_red = np.sqrt(mean_squared_error(y_pred_poly_red, y_val_final))

# Comparação
rmse_poly =1.397675
print(f"Poly RMSE: {rmse_poly:.6f}")  
print(f"Reduced Poly RMSE: {rmse_poly_red:.6f}")

# Plot
plot(y_val_final, y_pred_poly_red, plot_title="reduced_poly_y_yhat")


# In[77]:


#NAO CORRER
# Submissão 
X_test_reduced = X_test_df[reduced_feature_cols]  
predictions_red = pipeline_poly_red.predict(X_test_reduced)
predictions_df_red = pd.DataFrame(predictions_red, columns=target_cols)
predictions_df_red.insert(0, 'id', range(len(predictions_red)))
predictions_df_red.to_csv('reduced_polynomial_submission.csv', index=False)


# ###Task 3.3 - Adding Variables

# In[32]:


# Função para novas features 
def add_custom_features(X):
    new_X = X.copy()
    #Calculo da distância entre os corpos
    new_X['dist_12'] = np.sqrt((X['x0_1'] - X['x0_2'])**2 + (X['y0_1'] - X['y0_2'])**2)
    new_X['dist_13'] = np.sqrt((X['x0_1'] - X['x0_3'])**2 + (X['y0_1'] - X['y0_3'])**2)
    new_X['dist_23'] = np.sqrt((X['x0_2'] - X['x0_3'])**2 + (X['y0_2'] - X['y0_3'])**2)

    # Calcula o inverso da distância entre corpos: no problema de três corpos, a força gravitacional entre dois corpos é inversamente proporcional à distância 
    new_X['inv_dist_12'] = 1 / (new_X['dist_12'] + 1e-6)  # Avoid div0
    new_X['inv_dist_13'] = 1 / (new_X['dist_13'] + 1e-6)
    new_X['inv_dist_23'] = 1 / (new_X['dist_23'] + 1e-6)
    # Energia potencial (assume G=1, massas=1 para simplicidade; soma inversos das distâncias)
    new_X['potential_energy'] = - (1 / (new_X['dist_12'] + 1e-6) + 1 / (new_X['dist_13'] + 1e-6) + 1 / (new_X['dist_23'] + 1e-6))
    # Vetores 
    vec12_x = X['x0_2'] - X['x0_1']
    vec12_y = X['y0_2'] - X['y0_1']
    vec13_x = X['x0_3'] - X['x0_1']
    vec13_y = X['y0_3'] - X['y0_1']
    vec23_x = X['x0_3'] - X['x0_2']
    vec23_y = X['y0_3'] - X['y0_2']
    # Ângulo entre vetores
    new_X['angle_1_23'] = np.arctan2(vec12_y, vec12_x) - np.arctan2(vec13_y, vec13_x)
    new_X['angle_2_13'] = np.arctan2(vec12_y, vec12_x) - np.arctan2(vec23_y, vec23_x)
    new_X['angle_3_12'] = np.arctan2(vec13_y, vec13_x) - np.arctan2(vec23_y, vec23_x)

    return new_X


# In[33]:


#Teste1 para descobrir as melhores features para adicionar
X_train_joined_aug = add_custom_features(X_train_joined)
feature_cols_aug =  feature_cols + ['dist_12', 'dist_13', 'dist_23']
X_train_aug = X_train_joined_aug.loc[X_train_split.index, feature_cols_aug]  
X_val_aug = X_train_joined_aug.loc[X_val_split.index, feature_cols_aug]

# Testar com baseline 
pipeline_aug_base = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline_aug_base.fit(X_train_aug, y_train_final)
y_pred_aug_base = pipeline_aug_base.predict(X_val_aug)
rmse_aug_base = np.sqrt(mean_squared_error(y_pred_aug_base, y_val_final))
print(f"Augmented Baseline RMSE: {rmse_aug_base:.6f}")

# Plot
plot(y_val_final, y_pred_aug_base, plot_title="augmented_baseline_y_yhat")


# In[35]:


#Teste2 para descobrir as melhores features para adicionar
X_train_joined_aug = add_custom_features(X_train_joined)
feature_cols_aug = feature_cols + [ 'dist_12', 'dist_13', 'dist_23','potential_energy']
X_train_aug = X_train_joined_aug.loc[X_train_split.index, feature_cols_aug]  
X_val_aug = X_train_joined_aug.loc[X_val_split.index, feature_cols_aug]

pipeline_aum_base = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline_aug_base.fit(X_train_aug, y_train_final)
y_pred_aug_base = pipeline_aug_base.predict(X_val_aug)
rmse_aug_base = np.sqrt(mean_squared_error(y_pred_aug_base, y_val_final))
print(f"Augmented Baseline RMSE: {rmse_aug_base:.6f}")

# Plot
plot(y_val_final, y_pred_aug_base, plot_title="augmented_baseline_y_yhat")


# In[36]:


#Teste3 para descobrir as melhores features para adicionar
X_train_joined_aug = add_custom_features(X_train_joined)
feature_cols_aug = feature_cols + ['dist_12', 'dist_13', 'dist_23','inv_dist_12','inv_dist_13','inv_dist_23']
X_train_aug = X_train_joined_aug.loc[X_train_split.index, feature_cols_aug]  
X_val_aug = X_train_joined_aug.loc[X_val_split.index, feature_cols_aug]

pipeline_aug_base = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline_aug_base.fit(X_train_aug, y_train_final)
y_pred_aug_base = pipeline_aug_base.predict(X_val_aug)
rmse_aug_base = np.sqrt(mean_squared_error(y_pred_aug_base, y_val_final))
print(f"Augmented Baseline RMSE: {rmse_aug_base:.6f}")

# Plot
plot(y_val_final, y_pred_aug_base, plot_title="augmented_baseline_y_yhat")


# In[38]:


#Teste4 para descobrir as melhores features para adicionar
X_train_joined_aug = add_custom_features(X_train_joined)
feature_cols_aug = feature_cols + ['angle_1_23','angle_2_13','angle_3_12']
X_train_aug = X_train_joined_aug.loc[X_train_split.index, feature_cols_aug]  
X_val_aug = X_train_joined_aug.loc[X_val_split.index, feature_cols_aug]

pipeline_aug_base = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline_aug_base.fit(X_train_aug, y_train_final)
y_pred_aug_base = pipeline_aug_base.predict(X_val_aug)
rmse_aug_base = np.sqrt(mean_squared_error(y_pred_aug_base, y_val_final))
print(f"Augmented Baseline RMSE: {rmse_aug_base:.6f}")

# Plot
plot(y_val_final, y_pred_aug_base, plot_title="augmented_baseline_y_yhat")


# In[76]:


# Função para novas features escolhids
def add_custom_features1(X):
    new_X = X.copy()
    #Calculo da distância entre os corpos
    new_X['dist_12'] = np.sqrt((X['x0_1'] - X['x0_2'])**2 + (X['y0_1'] - X['y0_2'])**2)
    new_X['dist_13'] = np.sqrt((X['x0_1'] - X['x0_3'])**2 + (X['y0_1'] - X['y0_3'])**2)
    new_X['dist_23'] = np.sqrt((X['x0_2'] - X['x0_3'])**2 + (X['y0_2'] - X['y0_3'])**2)

    return new_X


# ###Task 3.4 - Evaluation of Variable Augmentation

# In[74]:


#Função regressão linear com transformer
def validate_poly_regression_aug(X_train, y_train, X_val, y_val, regressor=None, degrees=range(1,6), max_features=None):

    X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=0.01, random_state=42)
    X_val_sample, _, y_val_sample, _ = train_test_split(X_val ,y_val, train_size=0.01, random_state=42)

    custom_transformer = FunctionTransformer(add_custom_features1) 
    best_rmse = float('inf')
    rmses = []
    best_model = None
    best_degree = None

    for degree in degrees:
        # ColumnTransformer: aplica custom + poly em todas
        preprocessor = ColumnTransformer(
            transformers=[
                ('custom', custom_transformer, X_train_sample.columns), 
            ], remainder='passthrough')

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', RidgeCV(alphas=[0.1, 1.0, 10.0]))
        ])

        pipeline.fit(X_train_sample, y_train_sample)
        n_features = pipeline.named_steps['poly'].n_output_features_
        print(f"Degree {degree}: = {n_features}")

        y_pred_poly = pipeline.predict(X_val_sample)
        rmse_val = np.sqrt(mean_squared_error(y_val_sample, y_pred_poly))
        rmses.append(rmse_val)
        print(f"{degree}: RMSE = {rmse_val:.6f}")

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = pipeline
            best_degree = degree

    print(f"Best degree: {best_degree} with RMSE: {best_rmse:.6f}")

    return best_model, best_rmse


# In[77]:


#NAO CORRER
#Melhor grau para novas features
best_model_aug, best_rmse_aug = validate_poly_regression_aug(X_train_final, y_train_final, X_val_final, y_val_final, regressor=RidgeCV(alphas=[0.1,1.0,10.0]), degrees=range(1,6))


# In[78]:


#NAO CORRER
#validar com o conjunto de validação
custom_transformer = FunctionTransformer(add_custom_features1) 

preprocessor = ColumnTransformer(
    transformers=[
    ('custom', custom_transformer, X_train_sample.columns), 
    ], remainder='passthrough')

pipeline_poly_aug = Pipeline([
('preprocessor', preprocessor),
('poly', PolynomialFeatures(degree=4, include_bias=False)),
('scaler', StandardScaler()),
('regressor', RidgeCV(alphas=[0.1, 1.0, 10.0]))
])

pipeline_poly_aug.fit(X_train_final, y_train_final)

# Previsão no teste interno
y_pred_aug = pipeline_poly_aug.predict(X_val_final)
rmse_aug = np.sqrt(mean_squared_error(y_pred_aug, y_val_final))
print(f"Augmented Poly RMSE: {rmse_aug:.6f}")


plot(y_val_final, y_pred_aug, plot_title="augmented_poly_y_yhat")

# Comparação
rmse_poly = 1.397675
rmse_poly_red=1.389091
print(f"Poly RMSE: {rmse_poly:.6f}")
print(f"Reduced Poly RMSE: {rmse_poly_red:.6f}")
print(f"Augmented Poly RMSE: {rmse_aug:.6f}")  


# In[80]:


X_test_aug = pd.DataFrame()
X_test_aug = add_custom_features1(X_test)
feature_cols_aug = feature_cols + ['dist_12', 'dist_13', 'dist_23']
X_test_aug = X_test_aug[feature_cols_aug]
predictions_aug = pipeline_poly_aug.predict(X_test_aug)
predictions_df_aug = pd.DataFrame(predictions_aug, columns=target_cols)
predictions_df_aug.insert(0, 'id', range(len(predictions_aug)))
predictions_df_aug.to_csv('augmented_polynomial_submission.csv', index=False)


# ##**Task 4**

# ##Task 4.1 - Development

# In[51]:


def validate_knn_regression(X_train, y_train, X_val, y_val, k=range(1, 30)):

    results = {
        'k_values': list(k),
        'rmse': [],
        'train_times': [],
        'infer_times': []
    }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    for current_k in k:
        knn = KNeighborsRegressor(n_neighbors=current_k, metric='euclidean') 

        # Medir training time
        start_train = time.time()
        knn.fit(X_train_scaled, y_train)
        end_train = time.time()
        train_time = end_train - start_train
        results['train_times'].append(train_time)

        # Medir inference time
        start_infer = time.time()
        y_pred = knn.predict(X_val_scaled)
        end_infer = time.time()
        infer_time = end_infer - start_infer
        results['infer_times'].append(infer_time)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        results['rmse'].append(rmse)

    return results


# In[52]:


results_base = validate_knn_regression(X_train_final, y_train_final, X_val_final, y_val_final)
print(results_base)


# In[53]:


def plot_times_and_rmse(results, title_suffix="", save=True):

    k_values = results['k_values']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, results['train_times'], label='Training Time', marker='o')
    plt.plot(k_values, results['infer_times'], label='Inference Time', marker='o')
    plt.xlabel('k')
    plt.ylabel('Time (seconds)')
    plt.title(f'Training and Inference Times vs k {title_suffix}')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, results['rmse'], label='RMSE', marker='o', color='red')
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.title(f'RMSE vs k {title_suffix}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig(f'knn_analysis{title_suffix}.pdf')
    plt.show()

plot_times_and_rmse(results_base, title_suffix="features")


# In[54]:


# Encontrar melhor k base
best_k_base = results_base['k_values'][np.argmin(results_base['rmse'])]
min_rmse_base = min(results_base['rmse'])
print(f"Melhor k base: {best_k_base} com RMSE: {min_rmse_base:.6f}")


# In[ ]:


remove_features = ['x0_3', 'y0_2', 'y0_3']
rmses_removal = {}
current_cols = feature_cols.copy()

for feat in remove_features:
    reduced_cols = [col for col in current_cols if col != feat]
    X_train_red = X_train_final[reduced_cols]
    X_val_red = X_val_final[reduced_cols]

    results_red = validate_knn_regression(X_train_red, y_train_final, X_val_red, y_val_final)
    plot_times_and_rmse(results_red, title_suffix=f"_reduced_{feat}")

    min_rmse_red = min(results_red['rmse'])
    rmses_removal[feat] = min_rmse_red
    print(f"RMSE red {feat}: {min_rmse_red:.6f}")


# In[66]:


X_train_joined_aug = add_custom_features(X_train_joined)
feature_cols_aug =  feature_cols + ['dist_12', 'dist_13', 'dist_23']
X_train_aug = X_train_joined_aug.loc[X_train_split.index, feature_cols_aug]  
X_val_aug = X_train_joined_aug.loc[X_val_split.index, feature_cols_aug]

results_custom = validate_knn_regression(X_train_aug, y_train_final, X_val_aug, y_val_final)
plot_times_and_rmse(results_custom, title_suffix="_custom_features")

best_k_custom = results_custom['k_values'][np.argmin(results_custom['rmse'])]
min_rmse_custom = min(results_custom['rmse'])
print(f"Melhor k custom: {best_k_custom} com RMSE: {min_rmse_custom:.6f}")


# In[67]:


# Comparação de RMSEs mínimos 
print("\nComparação de RMSEs:")
print("Base:", min_rmse_base)
print("Remoções:", rmses_removal)
print("Custom:", min_rmse_custom)


# In[70]:


reduced_cols=['t','x0_1','y0_1','x0_2','y0_2','y0_3']
X_train_best = X_train_final[reduced_cols]
X_val_best = X_val_final[reduced_cols]

results_red = validate_knn_regression(X_train_best, y_train_final, X_val_best, y_val_final)
best_k = results_red['k_values'][np.argmin(results_custom['rmse'])]

min_rmse_best = min(results_red['rmse'])


# In[72]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_best)
X_val_scaled = scaler.transform(X_val_best)

knn_best = KNeighborsRegressor(n_neighbors=best_k_custom, metric='euclidean')
knn_best.fit(X_train_scaled, y_train_final)

y_pred_knn = knn_best.predict(X_val_scaled)
rmse_knn = np.sqrt(mean_squared_error(y_val_final, y_pred_knn))
print(rmse_knn)
plot(y_val_final, y_pred_knn, plot_title="knn_y_yhat")


# In[81]:


X_test_k = X_test[reduced_cols]
predictions_k = knn_best.predict(X_test_k)
predictions_df_k = pd.DataFrame(predictions_k, columns=target_cols)
predictions_df_k.insert(0, 'id', range(len(predictions_k)))
predictions_df_k.to_csv('knn_submission.csv', index=False)


# é comum que o kNN (um modelo não-paramétrico e baseado em instâncias) dê resultados piores em problemas como o Three-Body Problem comparado a modelos paramétricos como Linear Regression (de Task 2/3), especialmente se o dataset for grande ou as relações forem lineares/aproximáveis. O kNN é sensível a ruído, dimensionalidade e escala, e no three-body (um sistema caótico e não-linear), ele pode capturar padrões locais bem, mas sofre com "curse of dimensionality" ou falta de features informativas
# 

# 
