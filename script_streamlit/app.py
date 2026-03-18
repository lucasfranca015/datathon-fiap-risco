import streamlit as st
import pandas as pd
import joblib
import os
import sys

# --- CONFIGURAÇÃO DE CAMINHOS DINÂMICOS ---
# 1. Descobre onde este app.py está (dentro de /script_streamlit/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define a raiz do projeto (uma pasta acima)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# 3. Adiciona a raiz ao PATH do sistema (ajuda o joblib a encontrar dependências do modelo)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 4. Caminhos para os arquivos dentro de 'script_modelo'
PATH_MODELO = os.path.join(ROOT_DIR, 'script_modelo', 'modelo_risco_final.pkl')
PATH_COLUNAS = os.path.join(ROOT_DIR, 'script_modelo', 'colunas_modelo.pkl')

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Monitoramento de Risco Estudantil", layout="wide")

# --- CARREGAMENTO SEGURO DOS ARTEFATOS ---
@st.cache_resource
def carregar_objetos():
    try:
        mod = joblib.load(PATH_MODELO)
        cols = joblib.load(PATH_COLUNAS)
        return mod, cols
    except FileNotFoundError as e:
        st.error(f"Erro: Arquivo não encontrado. Verifique se as pastas 'script_modelo' e 'script_streamlit' estão no mesmo nível na raiz do GitHub.")
        st.info(f"Tentando ler em: {PATH_MODELO}")
        return None, None
    except Exception as e:
        st.error(f"Erro inesperado ao carregar o modelo: {e}")
        return None, None

modelo, features_finais = carregar_objetos()

# --- INTERFACE ---
st.title("📊 Preditor de Risco do Aluno")
st.write("Insira os indicadores abaixo para calcular a probabilidade de risco.")

if modelo is not None and features_finais is not None:
    # Organização da interface em colunas
    col1, col2 = st.columns(2)

    with col1:
        mat_media = st.number_input("Média Matemática", 0.0, 10.0, 5.0)
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
else:
    st.warning("Aguardando carregamento do modelo...")
