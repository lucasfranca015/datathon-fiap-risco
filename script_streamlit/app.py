import streamlit as st
import pandas as pd
import joblib
import os
import sys

# --- AJUSTE DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

PATH_MODELO = os.path.join(ROOT_DIR, 'script_modelo', 'modelo_risco_final.pkl')
PATH_COLUNAS = os.path.join(ROOT_DIR, 'script_modelo', 'colunas_modelo.pkl')

st.set_page_config(page_title="Monitoramento de Risco Estudantil", layout="wide")

@st.cache_resource
def carregar_objetos():
    mod = joblib.load(PATH_MODELO)
    cols = joblib.load(PATH_COLUNAS)
    return mod, cols

try:
    modelo, features_finais = carregar_objetos()
except Exception as e:
    st.error(f"Erro ao carregar arquivos: {e}")
    st.stop()

st.title("📊 Preditor de Risco do Aluno")
st.write("Insira os indicadores abaixo:")

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
    # IMPORTANTE: Usei exatamente os nomes que estão no seu arquivo colunas_modelo.pkl
    dados_entrada = pd.DataFrame([[
        mat_media, port_media, psico_media, auto_media,
        engaj_media, anos_inst, pedra_2022, pedra_2024,
        delta_mat, delta_port
    ]], columns=[
        'matematica_media', 'portugues_media', 'psicossocial_media', 
        'autoavaliacao_media', 'engajamento_media', 'anos_na_instituicao', 
        'pedra_2022_num', 'pedra_2024_num', 'delta_matematica', 'delta_portugues'
    ])
    
    prob = modelo.predict_proba(dados_entrada)[0][1]
    risco = "ALTO" if prob > 0.5 else "BAIXO"
    
    st.divider()
    if risco == "ALTO":
        st.error(f"Risco Detectado: {risco}")
    else:
        st.success(f"Risco Detectado: {risco}")
    st.metric("Probabilidade de Risco", f"{prob*100:.2f}%")
