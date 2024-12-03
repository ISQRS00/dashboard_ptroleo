import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
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


st.title('Tech Challenge: Fase 4')


st.markdown("## Objetivo do Projeto")


st.markdown("""
   
            
Este projeto tem como objetivo realizar previsões de preços do barril de petróleo utilizando modelos de séries temporais. Com base nos dados históricos de preços, foram aplicados modelos como Exponential Smoothing (ETS) para prever o comportamento futuro do mercado. O modelo ETS foi escolhido devido à sua capacidade de capturar padrões sazonais e tendências em séries temporais.

As previsões geradas são avaliadas por métricas como WMAPE (Weighted Mean Absolute Percentage Error) – (métrica que calcula o erro percentual absoluto ponderado para medir a precisão das previsões em relação aos valores reais), MAE (Mean Absolute Error) – (média dos erros absolutos entre as previsões e os valores reais, indicando a magnitude do erro médio), MSE (Mean Squared Error) – (média dos erros ao quadrado entre as previsões e os valores reais, que penaliza erros maiores), e R² (Coeficiente de Determinação) – (mede a proporção da variabilidade total dos dados explicada pelo modelo, indicando o quão bem os dados se ajustam à linha de tendência do modelo), com o intuito de medir a acurácia do modelo em relação aos dados reais. Além disso, a ferramenta permite que o usuário visualize gráficos interativos que ilustram as previsões e o desempenho do modelo, além de fazer o download das previsões em formato CSV.

            """)