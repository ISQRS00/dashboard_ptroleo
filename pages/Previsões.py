import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

# Configurações do Streamlit
##st.set_page_config(page_title="Deploy | Tech Challenge 4 | FIAP", layout='wide')

# Função para calcular WMAPE
@st.cache_data
def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

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

# Função para treinar o modelo ETS
@st.cache_data
def train_ets_model(train_data, season_length=60):
    model_ets = sm.tsa.ExponentialSmoothing(train_data['realizado'], seasonal='mul', seasonal_periods=season_length).fit()
    return model_ets

# Função para previsão com o modelo ETS
@st.cache_data
def forecast_ets(train, valid, model_ets):
    forecast_ets = model_ets.forecast(len(valid))
    forecast_dates = pd.date_range(start=train['data'].iloc[-1] + pd.Timedelta(days=1), periods=len(valid), freq='D')
    ets_df = pd.DataFrame({'data': forecast_dates, 'previsão': forecast_ets})
    ets_df = ets_df.merge(valid, on=['data'], how='inner')

    wmape_ets = wmape(ets_df['realizado'].values, ets_df['previsão'].values)
    MAE_ets = mean_absolute_error(ets_df['realizado'].values, ets_df['previsão'].values)
    MSE_ets = mean_squared_error(ets_df['realizado'].values, ets_df['previsão'].values)
    R2_ets = r2_score(ets_df['realizado'].values, ets_df['previsão'].values)

    return ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets

# Carregar dados
df_barril_petroleo = load_data()

# Explicação sobre o corte de dados
st.write("""
### Como Funciona o Corte dos Dados?

O modelo de previsão ETS é treinado com base em dados históricos do preço do petróleo. 
Escolha o número de dias para o corte e veja como o modelo se comporta para diferentes períodos.
""")

# Input para o número de dias para corte
dias_corte = st.number_input('Selecione o número de dias para o corte entre 7 e 90:', min_value=7, max_value=90, value=7)

# Calcular a data de corte com base no número de dias
cut_date = df_barril_petroleo['data'].max() - timedelta(days=dias_corte)

# Dividir em treino e validação
train = df_barril_petroleo.loc[df_barril_petroleo['data'] < cut_date]
valid = df_barril_petroleo.loc[df_barril_petroleo['data'] >= cut_date]

# Treinar o modelo ETS
model_ets = train_ets_model(train)

# Forecast ETS
if st.button("Gerar Previsão"):
    ets_df, wmape_ets, MAE_ets, MSE_ets, R2_ets = forecast_ets(train, valid, model_ets)

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
        xaxis=dict(tickformat="%d-%m-%Y", tickangle=45),
        yaxis=dict(title="Valor (US$)", tickformat=".3f"),
        autosize=True
    )
    st.plotly_chart(fig_ets)

    # Opção de Download dos Resultados
    st.subheader('Baixar Resultados')
    csv = ets_df.to_csv(index=False)
    st.download_button(
        label="Baixar Previsões ETS",
        data=csv,
        file_name="previsoes_ets.csv",
        mime="text/csv"
    )
