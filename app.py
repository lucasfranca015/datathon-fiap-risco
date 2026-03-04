import streamlit as st
import pandas as pd
import joblib

# Configuração visual da página
st.set_page_config(page_title="Monitoramento de Risco Estudantil", layout="wide")

# Carregamento dos artefatos do modelo
modelo = joblib.load('modelo_risco_final.pkl')
features_finais = joblib.load('colunas_modelo.pkl')

st.title("📊 Preditor de Risco do Aluno")
st.write("Insira os indicadores abaixo para calcular a probabilidade de risco.")

# Organização da interface em colunas
col1, col2 = st.columns(2)

with col1:
    mat_media = st.number_input("Média Matemáticaaaaaaaaaaaaa", 0.0, 10.0, 5.0)
    psico_media = st.number_input("Média Psicossocial", 0.0, 10.0, 5.0)
    engaj_media = st.number_input("Média Engajamento", 0.0, 10.0, 5.0)
    delta_mat = st.number_input("Delta Matemática", -10.0, 10.0, 0.0)
    pedra_2022 = st.selectbox("Pedra 2022", [0, 1, 2, 3, 4])

with col2:
    port_media = st.number_input("Média Português", 0.0, 10.0, 5.0)
    auto_media = st.number_input("Média Autoavaliação", 0.0, 10.0, 5.0)
    anos_inst = st.number_input("Anos na Instituição", 0, 20, 1)
    delta_port = st.number_input("Delta Português", -10.0, 10.0, 0.0)
    pedra_2024 = st.selectbox("Pedra 2024", [0, 1, 2, 3, 4])

# Botão de ação
if st.button("Calcular Risco"):
    # Montagem do DataFrame para predição na ordem exata das colunas salvas
    dados_entrada = pd.DataFrame([[
        mat_media, port_media, psico_media, auto_media,
        engaj_media, anos_inst, pedra_2022, pedra_2024,
        delta_mat, delta_port
    ]], columns=features_finais)
    
    # Realização da predição
    prob = modelo.predict_proba(dados_entrada)[0][1]
    risco = "ALTO" if prob > 0.5 else "BAIXO"
    
    # Exibição dos resultados
    st.divider()
    if risco == "ALTO":
        st.error(f"Risco Detectado: {risco}")
    else:
        st.success(f"Risco Detectado: {risco}")
    st.metric("Probabilidade de Risco", f"{prob*100:.2f}%")