# Credit Risk Analysis — Análise de Risco de Crédito com ML

Projeto de ciência de dados com pipeline completo de machine learning aplicado à predição de inadimplência de clientes de cartão de crédito, com rigor metodológico equivalente a artigo científico aplicado.

---

## Visão Geral

| Campo | Detalhe |
|---|---|
| **Dataset** | UCI Credit Card Default — Taiwan |
| **Volume** | 30.000 clientes, 23 features |
| **Problema** | Classificação binária: inadimplência no mês seguinte |
| **Modelos** | Baseline, Regressão Logística, Random Forest, XGBoost, LightGBM |
| **Melhor modelo** | LightGBM — AUC 0.7645, validado com Stratified K-Fold (10 splits) |

---

## Pipeline — 10 Fases

### 1. Análise Exploratória (EDA)
- Distribuição do target (default: 22,1%)
- Matriz de correlação entre features
- Distribuições e outliers das variáveis numéricas
- **Análise de viés por faixa etária**: taxa de inadimplência por grupo demográfico

### 2. Engenharia de Atributos
7 features derivadas criadas e validadas via comparação A/B:

| Feature | Descrição |
|---|---|
| `mean_delay` | Média dos status de atraso (PAY_0 a PAY_6) |
| `max_delay` | Máximo atraso registrado |
| `total_bill` | Soma total das faturas |
| `total_pay` | Soma total dos pagamentos |
| `pay_ratio` | Razão pagamento/fatura |
| `bill_trend` | Tendência de crescimento da fatura |
| `pay_trend` | Tendência de crescimento do pagamento |

### 3. Comparação de 5 Modelos

| Modelo | Accuracy | F1 | ROC AUC | PR AUC | Brier |
|---|---|---|---|---|---|
| Baseline | 0.7788 | 0.0000 | 0.5000 | 0.2212 | 0.2212 |
| Regressão Logística | 0.7015 | 0.4649 | 0.6992 | 0.4436 | 0.2004 |
| Random Forest | 0.7978 | 0.5018 | 0.7545 | 0.5108 | 0.1510 |
| XGBoost | 0.7907 | 0.4814 | 0.7445 | 0.5030 | 0.1536 |
| **LightGBM** | **0.7972** | **0.4897** | **0.7645** | **0.5282** | **0.1456** |

### 4. Otimização de Hiperparâmetros
`RandomizedSearchCV` com validação cruzada aplicada nos três modelos ensemble.

### 5. Validação Robusta — Stratified K-Fold
10 splits estratificados para avaliação estável:
- LightGBM: AUC médio **0.7770 ± 0.0099** (desvio < 0.01 = excelente estabilidade)

### 6. Testes Estatísticos
- **Paired t-test**: modelos ensemble são significativamente superiores à Regressão Logística em 5/5 comparações (α = 0.05)
- **Teste de McNemar**: diferenças nas predições individuais entre pares de modelos

### 7. Calibração de Probabilidades
`CalibratedClassifierCV` com isotônica regressão — Brier Score melhorado em todos os modelos testados.

### 8. Interpretabilidade Global
- **Regressão Logística**: coeficientes padronizados
- **Random Forest**: feature importance por impureza
- **XGBoost + SHAP**: beeswarm plot e bar plot com as 5 features mais impactantes

Top 5 features (SHAP — XGBoost):
1. `mean_delay` — atraso médio de pagamento
2. `max_delay` — pior atraso registrado
3. `MARRIAGE` — estado civil
4. `PAY_0` — status de pagamento mais recente
5. `total_bill` — valor total das faturas

### 9. Perfis Individuais — 3 Clientes
SHAP Waterfall para explicação individual de previsões:
- **Cliente A** — Baixo risco
- **Cliente B** — Médio risco
- **Cliente C** — Alto risco

### 10. Dashboard Interativo (Dash)
Aplicação web com 3 abas:
- **Visão Geral**: comparação de métricas entre modelos
- **Tabelas**: resultados detalhados de cada fase
- **Simulador Individual**: entrada de dados de um cliente e previsão de risco em tempo real

---

## Questões de Pesquisa Respondidas

| # | Questão | Resposta |
|---|---|---|
| 1 | ML supera Regressão Logística? | **SIM** — LightGBM: AUC 0.7645 vs 0.6992 (ganho +0.0653) |
| 2 | Melhor modelo preditivo? | **LightGBM** — AUC 0.7770 ± 0.0099 em K-Fold |
| 3 | Ganho é estatisticamente significativo? | **SIM** em 5/5 comparações (paired t-test, α=0.05) |
| 4 | Modelo é estável em diferentes amostras? | **SIM** — desvio padrão de 0.0099 (excelente) |
| 5 | Probabilidades previstas são confiáveis? | **SIM** — calibração isotônica melhorou todos os modelos |
| 6 | Variáveis mais influentes? | `mean_delay`, `max_delay`, `MARRIAGE`, `PAY_0`, `total_bill` |
| 7 | Performance e interpretabilidade conciliáveis? | **SIM** — XGBoost + SHAP: AUC 0.7645 com explicabilidade total |

---

## Estrutura do Projeto

```
.
├── riskcredit_v2.ipynb       # Notebook principal (28 células)
├── generate_notebook.py       # Script gerador do notebook
├── UCI_Credit_Card.csv        # Dataset (30.000 registros)
├── outputs/
│   ├── relatorio_final.pdf    # Relatório exportado
│   ├── images/                # 18+ gráficos gerados (PNG)
│   │   ├── 01_distribuicao_target.png
│   │   ├── 02_correlacao_heatmap.png
│   │   ├── ...
│   │   └── 18_resumo_final_performance.png
│   └── data/                  # Tabelas de resultados (CSV)
│       ├── tabela_modelos.csv
│       ├── tabela_kfold.csv
│       ├── tabela_calibracao.csv
│       ├── tabela_interpretabilidade.csv
│       ├── tabela_ttest.csv
│       ├── tabela_mcnemar.csv
│       ├── tabela_otimizacao.csv
│       └── conclusoes_questoes_pesquisa.csv
└── README.md
```

---

## Como Rodar

### Pré-requisitos
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap dash plotly reportlab
```

### Opção 1 — Notebook direto
```bash
jupyter notebook riskcredit_v2.ipynb
# Kernel → Restart & Run All
```

### Opção 2 — Regenerar o notebook do zero
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

| Categoria | Bibliotecas |
|---|---|
| Dados | pandas, numpy |
| Visualização | matplotlib, seaborn, plotly |
| ML | scikit-learn, xgboost, lightgbm |
| Explicabilidade | shap |
| Dashboard | dash |
| Exportação | reportlab |

---

## Dataset

**UCI Machine Learning Repository — Default of Credit Card Clients**  
Taiwan, 2005 — 30.000 clientes, 23 features originais + target binário (`default.payment.next.month`)

Features incluem: limite de crédito, sexo, escolaridade, estado civil, idade, histórico de pagamentos (6 meses), valores de fatura e pagamento.

---

*Projeto de portfólio — pipeline de ML aplicado a risco de crédito com ênfase em rigor estatístico, interpretabilidade e reprodutibilidade.*
