import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf
from statsforecast import StatsForecast
from plotly.subplots import make_subplots

# Função para calcular WMAPE
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# Função para treinar o modelo ETS (treinamento otimizado)
@st.cache_data
def train_ets_model(train_data):
    season_length = 252  # Sazonalidade anual
    model_ets = sm.tsa.ExponentialSmoothing(train_data['realizado'], seasonal='mul', seasonal_periods=season_length).fit(optimized=True)
    return model_ets

# Configurações do Streamlit
st.set_page_config(page_title="Deploy | Tech Challenge 4 | FIAP", layout='wide')

# Carregar e preparar os dados (com cache)
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/ISQRS00/dashboard_tc4/main/barril.csv', sep=';')
    df.drop(columns=['Unnamed: 2'], inplace=True)
    df.rename(columns={'Data': 'data', 'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'realizado'}, inplace=True)
    df['data'] = pd.to_datetime(df['data'])
    df['realizado'] = df['realizado'].str.replace(',', '.').astype(float)
    df['realizado'] = df['realizado'].ffill()  # Preencher valores ausentes
    return df

# Carregar dados uma vez com cache
df_barril_petroleo = load_data()

# Explicação sobre o corte de dados
st.write("""
### Como Funciona o Corte dos Dados?

O modelo de previsão ETS é treinado com base em dados históricos do preço do petróleo. Para avaliar a precisão do modelo, os dados são divididos em duas partes: 
1. **Dados de Treinamento**: São usados para treinar o modelo e entender os padrões históricos.
2. **Dados de Validação**: São usados para testar a performance do modelo e avaliar suas previsões.

O **número de dias de corte** define a quantidade de dados mais recentes que serão usados para o teste do modelo. Ou seja, o modelo será treinado com os dados até um ponto específico e testado com os dados após esse ponto.

**Como isso funciona?**
- Se você selecionar um número menor de dias (ex: 7 dias), o modelo irá usar mais dados antigos para treinamento e preverá apenas os últimos dias mais recentes.
- Se selecionar um número maior de dias (ex: 30 dias), o modelo será testado em uma janela de previsão maior e, portanto, mais desafiadora.

Escolha o número de dias para o corte e veja como o modelo se comporta para diferentes períodos.
""")

# Input para o número de dias para corte
dias_corte = st.number_input('Selecione o número de dias para o corte entre 7 e 90:', min_value=7, max_value=90, value=7)

# Calcular a data de corte com base no número de dias
cut_date = df_barril_petroleo['data'].max() - timedelta(days=dias_corte)

# Dividir em treino e validação
train = df_barril_petroleo.loc[df_barril_petroleo['data'] < cut_date]
valid = df_barril_petroleo.loc[df_barril_petroleo['data'] >= cut_date]

# Treinando o modelo ETS com dados de treino
model_ets = train_ets_model(train)

# Função para previsão com o modelo ETS
def forecast_ets(train, valid):
    season_length = 252  # Sazonalidade anual
    model_ets = sm.tsa.ExponentialSmoothing(train['realizado'], seasonal='mul', seasonal_periods=season_length).fit()
    forecast_ets = model_ets.forecast(len(valid))
    forecast_dates = pd.date_range(start=train['data'].iloc[-1] + pd.Timedelta(days=1), periods=len(valid), freq='D')
    ets_df = pd.DataFrame({'data': forecast_dates, 'previsão': forecast_ets})
    ets_df = ets_df.merge(valid, on=['data'], how='inner')

    wmape_ets = wmape(ets_df['realizado'].values, ets_df['previsão'].values)
    MAE_ets = mean_absolute_error(ets_df['realizado'].values, ets_df['previsão'].values)
    MSE_ets = mean_squared_error(ets_df['realizado'].values, ets_df['previsão'].values)
    R2_ets = r2_score(ets_df['realizado'].values, ets_df['previsão'].values)
    
    return ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets

# Exibição das métricas de desempenho
ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets = forecast_ets(train, valid)

st.subheader('Métricas de Desempenho do Modelo ETS')
st.write(f'WMAPE: {wmape_ets:.2%}')
st.write(f'MAE: {MAE_ets:.3f}')
st.write(f'MSE: {MSE_ets:.4f}')
st.write(f'R²: {R2_ets:.2f}')

# Criar gráfico ETS
fig_ets = go.Figure()

fig_ets.add_trace(go.Scatter(x=valid['data'], y=valid['realizado'], mode='lines', name='Realizado'))
fig_ets.add_trace(go.Scatter(x=ets_df['data'], y=ets_df['previsão'], mode='lines', name='Forecast'))

fig_ets.update_layout(
    title="Previsão do Modelo ETS",
    xaxis_title="Data",
    yaxis_title="Valor do Petróleo (US$)",
    xaxis=dict(
        tickformat="%d-%m-%Y", 
        type="date", 
        dtick="D1", 
        nticks=30, 
        tickangle=45  # Inclinar as datas em 45 graus
    ),
    yaxis=dict(
        title="Valor (US$)", 
        tickformat=".3f"
    ),
    autosize=True
)

# Exibição do gráfico
st.plotly_chart(fig_ets)

# Adicionar descrição do gráfico
st.write("""
**Descrição do Gráfico:** Este gráfico compara os valores reais do preço do petróleo (Realizado) com as previsões geradas pelo modelo ETS (Forecast). 
O modelo ETS utiliza a sazonalidade anual para gerar previsões e busca capturar tendências e padrões nos dados históricos.
""")

# Explicação sobre o download do arquivo
st.write("""
### Baixar os Resultados da Previsão

Após gerar as previsões com o modelo ETS, você pode baixar um arquivo `.csv` contendo as **previsões** feitas pelo modelo para os próximos dias. O arquivo inclui a data e o valor previsto do preço do petróleo, que pode ser útil para análises futuras ou para comparar com os dados reais posteriormente.

Clique no botão abaixo para baixar as previsões em formato CSV.
""")

# Opção de Download dos Resultados
st.subheader('Baixar Resultados')
ets_df['previsão'] = ets_df['previsão'].round(2)
csv = ets_df.to_csv(index=False)

st.download_button(
    label="Baixar Previsões ETS",
    data=csv,
    file_name="previsoes_ets.csv",
    mime="text/csv"
)
