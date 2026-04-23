# Previsão de Doenças Cardíacas com Redes Neurais (Deep Learning)

Este projeto utiliza Redes Neurais Artificiais para prever a presença de doenças cardíacas em pacientes com base em indicadores clínicos. O modelo foi desenvolvido em Python, utilizando a biblioteca Keras/TensorFlow.

## 🚀 Objetivo do Projeto
O objetivo é classificar se um paciente possui ou não uma cardiopatia com base em 13 variáveis (como idade, colesterol, frequência cardíaca máxima, etc.). Este é um problema clássico de **classificação binária**.

## 🛠️ Tecnologias e Técnicas Utilizadas
* **Linguagem:** Python 3
* **Deep Learning:** Keras & TensorFlow (Sequential Model)
* **Processamento de Dados:** Pandas e NumPy
* **Pré-processamento:** Scikit-Learn (`StandardScaler` para normalização e `train_test_split`)
* **Arquitetura da Rede:**
    * Camada de Entrada / Oculta: 10 neurônios (ativação ReLU)
    * Camada de Saída: 1 neurônio (ativação Sigmoid para probabilidade binária)
    * Otimizador: **Adam**
    * Função de Perda: **Binary Crossentropy**

## 📊 Estrutura do Repositório
* `analise_doenças_cardiacas.py`: Implementação da Rede Neural.
* `regressão.py`: Script para análise comparativa ou testes estatísticos.
* `doencas_cardiacas.csv`: Dataset com as informações clínicas dos pacientes.

## 📈 Como Executar
1. Instale as dependências: `pip install pandas tensorflow scikit-learn`
2. Execute o arquivo principal: `python analise_doenças_cardiacas.py`

## ✒️ Autor
* *Pedro alyson ** - [www.linkedin.com/in/pedro-alyson-b79a32236]
