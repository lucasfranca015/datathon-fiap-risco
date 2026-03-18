import streamlit as st
import pandas as pd
import joblib
import os
import sys

# --- AJUSTE DE CAMINHOS PARA O GITHUB ---
# Descobre onde o app.py está (dentro de script_streamlit)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sobe um nível para a raiz do projeto
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Adiciona a raiz ao sistema para evitar erros de importação do modelo
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Caminhos exatos para os arquivos pkl
PATH_MODELO = os.path.join(ROOT_DIR, 'script_modelo', 'modelo_risco_final.pkl')
PATH_COLUNAS = os.path.join(ROOT_DIR, 'script_modelo', 'colunas_modelo.pkl')

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Monitoramento de Risco Estudantil", layout="wide")

@st.cache_resource
def carregar_artefatos():
    # Carrega o modelo e a lista de colunas
    mod = joblib.load(PATH_MODELO)
    cols = joblib.load(PATH_COLUNAS)
    return mod, cols

try:
    modelo, features_finais = carregar_artefatos()
except Exception as e:
    st.error(f"Erro ao carregar arquivos: {e}")
    st.stop()

# --- INTERFACE ---
st.title("📊 Preditor de Risco do Aluno")
st.write("Insira os indicadores abaixo para calcular a probabilidade de risco.")

col1, col2 = st.columns(2)

with col1:
    mat_media = st.number_input("Média Matemática", 0.0, 10.0, 5.0)
    port_media = st.number_input("Média Português", 0.0, 10.0, 5.0)
    psico_media = st.number_input("Média Psicossocial", 0.0, 10.0, 5.0)
    auto_media = st.number_input("Média Autoavaliação", 0.0, 10.0, 5.0)
    engaj_media = st.number_input("Média Engajamento", 0.0, 10.0, 5.0)

with col2:
    anos_inst = st.number_input("Anos na Instituição", 0, 20, 1)
    pedra_2022 = st.selectbox("Pedra 2022", [0, 1, 2, 3, 4])
    pedra_2024 = st.selectbox("Pedra 2024", [0, 1, 2, 3, 4])
    delta_mat = st.number_input("Delta Matemática", -10.0, 10.0, 0.0)
    delta_port = st.number_input("Delta Português", -10.0, 10.0, 0.0)

if st.button("Calcular Risco"):
    # Organiza os dados exatamente como o modelo espera
    dados_entrada = pd.DataFrame([[
        mat_media, port_media, psico_media, auto_media,
        engaj_media, anos_inst, pedra_2022, pedra_2024,
        delta_mat, delta_port
    ]], columns=features_finais)
    
    # Predição
    prob = modelo.predict_proba(dados_entrada)[0][1]
    risco = "ALTO" if prob > 0.5 else "BAIXO"
    
    st.divider()
    if risco == "ALTO":
        st.error(f"Risco Detectado: {risco}")
    else:
        st.success(f"Risco Detectado: {risco}")

    st.metric("Probabilidade de Risco", f"{prob*100:.2f}%")
