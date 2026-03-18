📊 Monitoramento de Risco Estudantil - Datathon FIAP
Este projeto consiste em uma aplicação web interativa desenvolvida com Streamlit para prever a probabilidade de risco de alunos com base em indicadores educacionais, psicossociais e de engajamento.

O modelo de Machine Learning utiliza algoritmos de classificação (Random Forest) para identificar padrões e auxiliar na tomada de decisão pedagógica.

📁 Estrutura do Repositório
A organização do projeto segue a estrutura abaixo para garantir a modularidade e facilitar o deploy:

script_streamlit/: Contém o código da interface do usuário (app.py) e as dependências específicas do servidor.

script_modelo/: Armazena os artefatos do modelo treinado (.pkl) e a lista de colunas necessárias para a predição.

notebooks/ (Opcional): Espaço destinado aos arquivos de análise exploratória e treinamento do modelo.

data/ (Opcional): Base de dados utilizada no projeto (respeitando a LGPD).

🛠️ Tecnologias Utilizadas
Python 3.11

Streamlit: Interface Web.

Scikit-Learn 1.6.1: Inteligência Artificial e Machine Learning.

Pandas & Numpy: Manipulação de dados.

Joblib: Persistência do modelo.

🚀 Como Executar o Projeto
1. Requisitos
Certifique-se de ter o Python 3.11 instalado. É recomendável o uso de um ambiente virtual.

2. Instalação
Bash
# Clone o repositório
git clone https://github.com/lucasfranca015/datathon-fiap-risco.git

# Entre na pasta do script
cd script_streamlit

# Instale as dependências
pip install -r requirements.txt
3. Rodar a Aplicação
Bash
streamlit run app.py
🧠 Variáveis do Modelo
O modelo analisa os seguintes indicadores para calcular o risco:

Variável	Descrição
Média Matemática	Desempenho em avaliações de exatas.
Média Português	Desempenho em avaliações de linguagens.
Média Psicossocial	Indicadores de comportamento e bem-estar.
Deltas	Evolução das notas entre períodos.
Pedras (2022/2024)	Nível de classificação institucional do aluno.
