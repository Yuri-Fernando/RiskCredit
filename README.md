# Credit Risk Analysis — Análise de Risco de Crédito com Machine Learning

## Status

🟢 **Concluído — Projeto de portfólio / Machine Learning aplicado**

Pipeline completo de **Machine Learning aplicado à predição de inadimplência em cartões de crédito**, desenvolvido com foco em rigor metodológico, validação estatística, calibração de probabilidades e interpretabilidade dos modelos.

O projeto utiliza o dataset público **UCI Credit Card Default — Taiwan** e percorre todo o fluxo de análise, desde EDA e engenharia de atributos até modelagem, validação, explicabilidade e disponibilização dos resultados em dashboard interativo.

---

## Visão Geral

| Campo             | Detalhe                                                         |
| ----------------- | --------------------------------------------------------------- |
| **Dataset**       | UCI Credit Card Default — Taiwan                                |
| **Volume**        | 30.000 clientes, 23 features                                    |
| **Problema**      | Classificação binária: inadimplência no mês seguinte            |
| **Modelos**       | Baseline, Regressão Logística, Random Forest, XGBoost, LightGBM |
| **Melhor modelo** | LightGBM                                                        |
| **ROC AUC**       | 0.7645 no conjunto de avaliação                                 |
| **K-Fold AUC**    | 0.7770 ± 0.0099                                                 |

---

## Objetivo

O projeto busca avaliar diferentes abordagens de Machine Learning para **predição de default**, comparando desempenho, estabilidade, calibração e interpretabilidade.

Além da previsão, a solução investiga quais características mais contribuem para o risco estimado e disponibiliza uma interface para análise individual das predições.

---

## Pipeline

```text id="n4wy6r"
Dataset
   ↓
EDA
   ↓
Feature Engineering
   ↓
Train / Validation
   ↓
Model Comparison
   ↓
Hyperparameter Optimization
   ↓
Stratified K-Fold
   ↓
Statistical Tests
   ↓
Probability Calibration
   ↓
Explainability / SHAP
   ↓
Individual Risk Analysis
   ↓
Interactive Dashboard
```

---

## 1. Análise Exploratória

A etapa de EDA contempla:

* Distribuição do target;
* Identificação da taxa de default;
* Matriz de correlação;
* Distribuição das variáveis numéricas;
* Investigação de outliers;
* Análise das características demográficas;
* Taxa de inadimplência por faixa etária.

A distribuição observada para o target apresenta aproximadamente **22,1% de inadimplência**.

---

## 2. Engenharia de Atributos

Foram desenvolvidas e avaliadas **7 features derivadas**, com foco em histórico de pagamento, comportamento financeiro e tendências.

| Feature      | Descrição                                      |
| ------------ | ---------------------------------------------- |
| `mean_delay` | Média dos status de atraso (`PAY_0` a `PAY_6`) |
| `max_delay`  | Máximo atraso registrado                       |
| `total_bill` | Soma total das faturas                         |
| `total_pay`  | Soma total dos pagamentos                      |
| `pay_ratio`  | Relação entre pagamento e faturamento          |
| `bill_trend` | Tendência de crescimento da fatura             |
| `pay_trend`  | Tendência de crescimento do pagamento          |

---

## 3. Comparação de Modelos

Foram comparados cinco modelos e um baseline:

| Modelo              |   Accuracy |         F1 |    ROC AUC |     PR AUC |      Brier |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Baseline            |     0.7788 |     0.0000 |     0.5000 |     0.2212 |     0.2212 |
| Regressão Logística |     0.7015 |     0.4649 |     0.6992 |     0.4436 |     0.2004 |
| Random Forest       |     0.7978 |     0.5018 |     0.7545 |     0.5108 |     0.1510 |
| XGBoost             |     0.7907 |     0.4814 |     0.7445 |     0.5030 |     0.1536 |
| **LightGBM**        | **0.7972** | **0.4897** | **0.7645** | **0.5282** | **0.1456** |

O **LightGBM** apresentou o maior ROC AUC entre os modelos avaliados no experimento principal.

---

## 4. Otimização de Hiperparâmetros

Foi utilizado:

```text
RandomizedSearchCV
```

com validação cruzada para otimização dos modelos ensemble.

A etapa foi utilizada para investigar combinações de hiperparâmetros e avaliar o comportamento dos modelos sob diferentes configurações.

---

## 5. Validação Robusta — Stratified K-Fold

A avaliação foi complementada por **10 splits estratificados**, buscando medir a estabilidade do desempenho em diferentes particionamentos dos dados.

### LightGBM

```text
ROC AUC médio: 0.7770
Desvio padrão: 0.0099
```

O desvio padrão inferior a 0,01 indica baixa variação do ROC AUC entre os folds analisados.

---

## 6. Testes Estatísticos

O projeto inclui análises estatísticas para complementar a comparação dos modelos.

### Paired t-test

Aplicado para comparar o desempenho dos modelos em diferentes folds.

Resultado observado:

```text
Modelos ensemble vs. Regressão Logística
Significância estatística em 5/5 comparações
α = 0.05
```

### McNemar Test

Utilizado para analisar diferenças nas predições individuais entre pares de modelos.

---

## 7. Calibração de Probabilidades

Foi utilizada a técnica:

```text
CalibratedClassifierCV
```

com **regressão isotônica** para avaliar e melhorar a calibração das probabilidades previstas.

O Brier Score foi utilizado como uma das métricas de avaliação da qualidade das probabilidades produzidas.

---

## 8. Interpretabilidade Global

Diferentes métodos foram utilizados de acordo com o modelo:

### Regressão Logística

* Coeficientes padronizados.

### Random Forest

* Feature importance baseada em impureza.

### XGBoost

* SHAP;
* Beeswarm plot;
* Bar plot;
* Ranking das features mais impactantes.

### Top 5 Features — SHAP

1. `mean_delay`
2. `max_delay`
3. `MARRIAGE`
4. `PAY_0`
5. `total_bill`

---

## 9. Explicabilidade Individual

Foram analisados três perfis individuais utilizando **SHAP Waterfall**:

* **Cliente A** — baixo risco;
* **Cliente B** — risco intermediário;
* **Cliente C** — alto risco.

O objetivo é mostrar como diferentes características contribuem para a previsão individual de cada cliente.

---

## 10. Dashboard Interativo

O projeto inclui uma aplicação desenvolvida com **Dash** com três áreas principais:

### Visão Geral

Comparação de desempenho entre os modelos avaliados.

### Tabelas

Exibição dos resultados detalhados de:

* Modelos;
* K-Fold;
* Calibração;
* Interpretabilidade;
* Testes estatísticos;
* Otimização.

### Simulador Individual

Permite inserir características de um cliente e obter uma estimativa de risco utilizando o pipeline desenvolvido.

---

## Questões de Pesquisa

| # | Questão                                              | Resultado                                                    |
| - | ---------------------------------------------------- | ------------------------------------------------------------ |
| 1 | ML supera Regressão Logística?                       | LightGBM apresentou AUC 0.7645 vs. 0.6992                    |
| 2 | Qual apresentou melhor desempenho no K-Fold?         | LightGBM — 0.7770 ± 0.0099                                   |
| 3 | Houve diferença estatística nas comparações?         | Significância observada em 5/5 comparações                   |
| 4 | O desempenho apresentou estabilidade?                | Desvio padrão de 0.0099 no LightGBM                          |
| 5 | As probabilidades foram calibradas?                  | Calibração isotônica aplicada                                |
| 6 | Quais features mais influenciaram?                   | `mean_delay`, `max_delay`, `MARRIAGE`, `PAY_0`, `total_bill` |
| 7 | É possível combinar desempenho e interpretabilidade? | XGBoost + SHAP forneceu análise detalhada das contribuições  |

---

## Estrutura do Projeto

```text id="w7d3h8"
.
├── riskcredit_v2.ipynb
├── generate_notebook.py
├── UCI_Credit_Card.csv
│
├── outputs/
│   ├── relatorio_final.pdf
│   │
│   ├── images/
│   │   ├── 01_distribuicao_target.png
│   │   ├── 02_correlacao_heatmap.png
│   │   ├── ...
│   │   └── 18_resumo_final_performance.png
│   │
│   └── data/
│       ├── tabela_modelos.csv
│       ├── tabela_kfold.csv
│       ├── tabela_calibracao.csv
│       ├── tabela_interpretabilidade.csv
│       ├── tabela_ttest.csv
│       ├── tabela_mcnemar.csv
│       ├── tabela_otimizacao.csv
│       └── conclusoes_questoes_pesquisa.csv
│
└── README.md
```

---

## Como Executar

### Pré-requisitos

Python 3.9+

### Instalação

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap dash plotly reportlab
```

### Opção 1 — Executar o notebook

```bash
jupyter notebook riskcredit_v2.ipynb
```

Depois:

```text
Kernel → Restart & Run All
```

### Opção 2 — Regenerar o notebook

```bash
python generate_notebook.py
jupyter notebook riskcredit_v2.ipynb
```

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![SHAP](https://img.shields.io/badge/SHAP-explainability-purple)
![Dash](https://img.shields.io/badge/Dash-dashboard-blue)

| Categoria        | Tecnologias                         |
| ---------------- | ----------------------------------- |
| Linguagem        | Python                              |
| Dados            | Pandas · NumPy                      |
| Visualização     | Matplotlib · Seaborn · Plotly       |
| Machine Learning | scikit-learn · XGBoost · LightGBM   |
| Explicabilidade  | SHAP                                |
| Dashboard        | Dash                                |
| Exportação       | ReportLab                           |
| Avaliação        | ROC AUC · PR AUC · F1 · Brier Score |
| Estatística      | Paired t-test · McNemar             |

---

## Dataset

**UCI Machine Learning Repository — Default of Credit Card Clients**

Dataset referente a clientes de cartão de crédito de Taiwan, contendo:

* **30.000 registros**
* **23 features originais**
* Target binário: `default.payment.next.month`

As variáveis incluem informações relacionadas a:

* limite de crédito;
* características demográficas;
* histórico de pagamentos;
* valores de faturas;
* valores de pagamentos.

---

## O que este projeto demonstra

* Construção de pipeline completo de Machine Learning;
* Análise exploratória de dados;
* Feature Engineering;
* Comparação sistemática de modelos;
* Otimização de hiperparâmetros;
* Validação cruzada estratificada;
* Testes estatísticos;
* Calibração de probabilidades;
* Explainable AI;
* SHAP global e local;
* Avaliação de risco;
* Desenvolvimento de dashboards analíticos;
* Organização e reprodutibilidade de experimentos de Data Science.

---

## Limitações e Considerações

* O dataset utilizado é público e histórico;
* Os resultados são específicos ao conjunto de dados e metodologia utilizada;
* O modelo não deve ser interpretado como um sistema de decisão de crédito validado para uso regulado;
* A análise de viés apresentada é exploratória;
* Resultados de métricas podem variar conforme particionamento, versões das bibliotecas e configuração dos modelos.

---

## Melhorias Futuras

* Avaliação em outros datasets de risco de crédito;
* Validação temporal e *out-of-time testing*;
* Monitoramento de drift;
* Comparação com outros algoritmos;
* Experimentação com diferentes técnicas de calibração;
* API para disponibilização do modelo;
* Containerização da aplicação;
* Pipeline de inferência automatizado;
* Monitoramento de performance e qualidade do modelo;
* Expansão da análise de fairness e bias.

---

## Status do Projeto

🟢 **Concluído**

O pipeline principal, experimentos, avaliação estatística, análise de interpretabilidade, geração de resultados e dashboard foram implementados e documentados.

O repositório permanece disponível para reprodução dos experimentos e futuras extensões relacionadas a **Machine Learning, Explainable AI, Credit Risk e MLOps**.

---

## Autor

**Yuri Fernando Dubbern**

AI/ML Engineer · Data Science · Machine Learning · Risk Analytics

[LinkedIn](https://www.linkedin.com/in/yuridubbern) · [GitHub](https://github.com/Yuri-Fernando) · [Lattes](http://lattes.cnpq.br/7151392692642166) · [Linktree](https://linktr.ee/yuri.f.dubbern)
