# 📊 Monitoramento de Risco Estudantil - Datathon FIAP

Este projeto é uma aplicação web desenvolvida com **Streamlit** que utiliza Machine Learning para prever o risco de evasão ou baixo desempenho de alunos. O modelo foi treinado com base em indicadores de notas, comportamento psicossocial e engajamento.

## 📁 Estrutura do Repositório

Organizamos o repositório para facilitar a manutenção e o deploy automático:

* **`script_streamlit/`**: Contém o arquivo `app.py` (interface do usuário) e o `requirements.txt`.
* **`script_modelo/`**: Contém os arquivos binários do modelo (`modelo_risco_final.pkl`) e a lista de colunas (`colunas_modelo.pkl`).
* **Raiz**: Arquivos de configuração e documentação.

---

## 🛠️ Tecnologias e Versões

Para evitar erros de compatibilidade, o ambiente de produção utiliza:
* **Python 3.11**
* **Scikit-Learn 1.6.1**
* **Streamlit 1.55.0**
* **Joblib** (para carregamento do modelo)

---

## 🧠 Variáveis Analisadas

O modelo de predição utiliza os seguintes campos para gerar o diagnóstico:

| Variável | Descrição |
| :--- | :--- |
| **Média Matemática** | Média das notas de exatas. |
| **Média Português** | Média das notas de linguagens. |
| **Média Psicossocial** | Avaliação de comportamento e socialização. |
| **Média Autoavaliação** | Percepção do próprio aluno sobre seu desempenho. |
| **Média Engajamento** | Nível de participação em atividades. |
| **Anos na Instituição** | Tempo de permanência do aluno na escola. |
| **Pedras (2022/2024)** | Nível de classificação (0 a 4) nos anos correspondentes. |
| **Deltas** | Evolução ou queda de desempenho entre os períodos. |

---

## 🚀 Como Executar Localmente

Se desejar rodar o projeto no seu computador:

1. Clone o repositório:
   ```bash
   git clone [https://github.com/lucasfranca015/datathon-fiap-risco.git](https://github.com/lucasfranca015/datathon-fiap-risco.git)

2. Acesse a pasta do script:
   ```bash
   cd script_streamlit
   
3. Instale as dependências:
  ```bash
  pip install -r requirements.txt

4. Inicie o app:
   ```bash
   streamlit run app.py

🔗 Deploy Online
O app está configurado para deploy automático no Streamlit Cloud.
👉 Acesse o Dashboard de Risco [Aqui](https://datathon-fiap-risco-r9bswmxgnsz6mmkqbxqvee.streamlit.app/)
