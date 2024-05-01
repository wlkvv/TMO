import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import *
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    
    ds = load_diabetes()
    data = pd.DataFrame(data= np.c_[ds['data'], ds['target']],
                     columns= list(ds['feature_names']) + ['target'])
   
    return data


@st.cache_resource
def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    # Масштабирование признаков
    scale_cols = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    sc = MinMaxScaler()
    data_out[scale_cols] = sc.fit_transform(data_out[scale_cols])
    return data_out, data_out['target']

# Загрузка и предварительная обработка данных
data = load_data()

data_X, data_y = preprocess_data(data)

# Интерфейс пользователя
st.sidebar.header('Метод ближайших соседей')
cv_slider = st.sidebar.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1)
step_slider = st.sidebar.slider('Шаг для соседей:', min_value=1, max_value=50, value=10, step=1)

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)

# Подбор гиперпараметра
n_range_list = list(range(1, 50, step_slider)) # Установите максимальное значение в соответствии с вашими данными
n_range = np.array(n_range_list)
tuned_parameters = [{'n_neighbors': n_range}]



# clf_gs = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=cv_slider, scoring='accuracy')
clf_gs = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=cv_slider, scoring='neg_mean_squared_error')
clf_gs.fit(data_X, data_y)



st.subheader('Оценка качества модели')
st.write('Лучшее значение параметров - {}'.format(clf_gs.best_params_))

# Изменение качества на тестовой выборке в зависимости от К-соседей
fig1 = plt.figure(figsize=(7,5))
plt.plot(n_range, clf_gs.cv_results_['mean_test_score'])
st.pyplot(fig1)