# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, explained_variance_score, max_error,
    mean_squared_log_error
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Linear Regression App", layout="centered")
st.title("📈 Regressão Linear Interativa com Streamlit")

# Carrega dataset automaticamente
st.subheader("📂 Usando dataset padrão embutido")
df = pd.read_csv("meus_dados.csv")
st.success("Dataset carregado automaticamente com sucesso!")

st.subheader("📋 Pré-visualização do Dataset")
st.dataframe(df.head())

# Seleção de colunas
num_cols = df.select_dtypes(include='number').columns.tolist()
if len(num_cols) < 2:
    st.error("❌ O dataset precisa de pelo menos duas colunas numéricas.")
else:
    x_col = st.selectbox("📌 Selecione a variável independente (X):", num_cols)
    y_col = st.selectbox("🎯 Selecione a variável dependente (Y):", [c for c in num_cols if c != x_col])

    # Limpeza dos dados
    df = df[[x_col, y_col]].dropna()
    df = df[(df[x_col] != 0) & (df[y_col] != 0)]

    # Treinamento do modelo
    X = df[[x_col]]
    y = df[y_col]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Métricas
    st.subheader("📊 Avaliação do Modelo")
    st.markdown(f"**Equação da Reta:** y = `{model.coef_[0]:.4f}` * x + `{model.intercept_:.4f}`")
    st.markdown(f"- R²: `{r2_score(y, y_pred):.4f}`")
    st.markdown(f"- Adjusted R²: `{1 - (1 - r2_score(y, y_pred)) * (len(y) - 1) / (len(y) - X.shape[1] - 1):.4f}`")
    st.markdown(f"- MAE: `{mean_absolute_error(y, y_pred):.4f}`")
    st.markdown(f"- RMSE: `{np.sqrt(mean_squared_error(y, y_pred)):.4f}`")
    st.markdown(f"- MSE: `{mean_squared_error(y, y_pred):.4f}`")
    st.markdown(f"- Median AE: `{median_absolute_error(y, y_pred):.4f}`")
    st.markdown(f"- Max Error: `{max_error(y, y_pred):.4f}`")
    st.markdown(f"- Explained Variance Score: `{explained_variance_score(y, y_pred):.4f}`")
    if (y > 0).all() and (y_pred > 0).all():
        st.markdown(f"- MSLE: `{mean_squared_log_error(y, y_pred):.6f}`")

    # Plot Regressão
    st.subheader("📈 Gráfico da Regressão Linear")
    fig, ax = plt.subplots()
    ax.scatter(X, y, label='Observado', color='blue')
    ax.plot(X, y_pred, color='red', label='Predito')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Ajuste Linear")
    ax.legend()
    st.pyplot(fig)

    # Gráfico de Resíduos
    st.subheader("🔍 Resíduos vs Predito")
    residuals = y - y_pred
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_pred, residuals, color='purple')
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_xlabel("Predito")
    ax2.set_ylabel("Resíduo")
    ax2.set_title("Resíduos vs Predições")
    st.pyplot(fig2)

    # Previsão interativa
    st.subheader("🧮 Prever novo valor de Y")
    input_x = st.number_input(f"Insira um novo valor de {x_col}:", value=float(X[x_col].mean()))
    predicted_y = model.predict([[input_x]])[0]
    st.success(f"Para {x_col} = {input_x}, o valor previsto de {y_col} é aproximadamente `{predicted_y:.2f}`")
