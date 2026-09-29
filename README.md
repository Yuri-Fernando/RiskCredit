# Credit Risk Analysis — Análise de Risco de Crédito com Machine Learning

## Status

🟢 **Concluído — Projeto de portfólio / Machine Learning aplicado** · 🆕 **v3 (set/2026): Credit Risk Analytics & Portfolio Strategy** — ver [o que há de novo](#-v3--credit-risk-analytics--portfolio-strategy-setembro2026)

Pipeline completo de **Machine Learning aplicado à predição de inadimplência em cartões de crédito**, desenvolvido com foco em rigor metodológico, validação estatística, calibração de probabilidades e interpretabilidade dos modelos.

O projeto utiliza o dataset público **UCI Credit Card Default — Taiwan** e percorre todo o fluxo de análise, desde EDA e engenharia de atributos até modelagem, validação, explicabilidade e disponibilização dos resultados em dashboard interativo.

---

## Versões

| Versão | Foco | Onde |
|---|---|---|
| **v3.0.2 (set/2026) — nova** | Credit Risk Analytics & Portfolio Strategy (modelo → decisão) | `src/`, `run_pipeline_v3.py`, `reports/v3/`, `artifacts/champion/` |
| v2 | Pipeline de ML em notebook (seções "Visão Geral" a "10. Dashboard") | `riskcredit_v2.ipynb` |
| v1 | Notebook exploratório original | `riskcredit.ipynb` |

## 🆕 V3 — Credit Risk Analytics & Portfolio Strategy (setembro/2026)

### O que há de novo na v3 (em relação à v2)

A v2 (notebook, seções abaixo) continua **intacta**. A v3 é **código novo**, em módulos Python testados:

| # | Novidade da v3 | Arquivo |
|---|---|---|
| 1 | **Scorecard tradicional**: binning monotônico, WoE, IV, logística, escala 600/50:1/PDO 20, reason codes | `src/features/woe.py`, `src/models/scorecard.py` |
| 2 | **Champion × challenger formal**, com regra definida antes dos resultados | `src/models/champion_challenger.py` |
| 3 | **Segmentação** K-Means, GMM e DBSCAN, com estabilidade por bootstrap | `src/segmentation/clustering.py` |
| 4 | **Ponto de corte por custo e lucro** (não 0,5) | `src/strategy/threshold.py` |
| 5 | **Estratégia de carteira**: EL, capital IRB, RAROC, pricing por faixa, limite, prioridade de cobrança | `src/strategy/portfolio.py` |
| 6 | **Stress testing** (utilização, atraso, limite, pagamento) | `src/stress/stress_test.py` |
| 7 | **Monitoramento**: PSI/CSI por feature, drift de score, alertas | `src/monitoring/drift.py` |
| 8 | **Equidade**: aprovação, TPR/FPR, calibração e AIR com IC por sexo, escolaridade e idade | `src/fairness/audit.py` |
| 9 | **Robustez de defasagem** (e documentação de por que não há OOT real neste dataset) | `src/validation/robustness.py` |
| 10 | **Export do champion** como contrato para o repositório AWS | `src/export/artifacts.py`, `artifacts/champion/` |
| 11 | **Dashboard executivo** (Dash, 6 páginas), pipeline único, configuração e **16 testes** | `dashboards/`, `run_pipeline_v3.py`, `configs/v3.yaml`, `tests/` |

**Mudança de desenho:** atributos demográficos (SEX, MARRIAGE, AGE, EDUCATION) saíram das features do
modelo e passaram a ser usados só na auditoria de equidade. Histórico completo em [CHANGELOG.md](CHANGELOG.md).

A v3 transforma o projeto de "classificação de default" em uma cadeia **modelo → decisão**:
o score vira política de aprovação, precificação, limite, prioridade de cobrança e monitoramento.
Todos os números abaixo vêm de `python run_pipeline_v3.py` (execução real, `reports/v3/summary.json`).

> **Escopo:** simulação de estratégia para portfólio. Premissas econômicas (LGD 75%, CCF 60%, receita 18%,
> funding 8%, opex NT$ 120/conta, hurdle 15%) são **ilustrativas**; dados de Taiwan, 2005 — sem validade
> para uma carteira brasileira atual. Não é sistema de concessão.

### Decisões de desenho da v3

- **Atributos demográficos fora do modelo** (SEX, MARRIAGE, AGE, EDUCATION) — usados só na auditoria de
  equidade. A v2 usava MARRIAGE como preditor; a v3 não.
- **Sem OOT fingido:** o UCI não tem data de originação nem safra. Em vez de "pseudo-OOT", a v3 mede a
  **robustez de defasagem** (modelo aplicado sem o mês mais recente) e deixa OOT real no roadmap.
- **Regra de champion definida antes dos resultados:** o challenger só substitui o scorecard com ganho de
  Gini ≥ 0,02, ECE ≤ 0,02 e PSI treino→teste < 0,10.

### Champion × Challenger (teste, 9.000 contas)

| Modelo | AUC | Gini | KS | Brier | ECE | PSI tr→te | Features | Explicabilidade |
|---|---|---|---|---|---|---|---|---|
| Scorecard WoE (logística) | 0,762 | 0,525 | 0,406 | 0,139 | 0,015 | 0,001 | 16 | alta |
| Logística (padronizada) | 0,753 | 0,506 | 0,396 | 0,142 | 0,013 | 0,001 | 28 | alta |
| LightGBM | 0,777 | 0,554 | 0,418 | 0,136 | 0,019 | 0,003 | 28 | média (SHAP) |
| **XGBoost — champion** | **0,780** | **0,560** | **0,426** | **0,135** | **0,014** | **0,000** | 28 | média (SHAP) |
| LightGBM calibrado | 0,777 | 0,555 | 0,421 | 0,136 | 0,009 | 0,003 | 28 | média (SHAP) |

Ganho do XGBoost sobre o scorecard: **+0,035 de Gini** (≥ 0,02) → promovido. O scorecard (600 pontos em
odds 50:1, PDO 20, reason codes) continua como modelo interpretável de referência.

**Robustez de defasagem (não é OOT):** Gini 0,560 com a janela completa → 0,497 sem o mês mais recente
(queda de 0,063) — mede sensibilidade a atraso na chegada dos dados.

### Threshold por custo e lucro (não 0,5)

| Política | Limiar de PD | Aprovação | Bad rate aprovados | Lucro realizado (teste) |
|---|---|---|---|---|
| Mínimo custo esperado | 0,10 | 32,6% | 6,8% | — |
| **Máximo lucro esperado** | **0,12** | **40,5%** | **7,8%** | **NT$ +22,7 mi** |
| Corte ingênuo 0,5 | 0,50 | 88,0% | 16,1% | NT$ −18,8 mi |

### Segmentação comportamental (K-Means k=5 por silhouette; GMM k=8 por BIC; DBSCAN)

| Segmento (rótulo descritivo) | Participação | Default | Jaccard bootstrap |
|---|---|---|---|
| Rotativo recorrente (utilização alta, pagamento mínimo) | 29,5% | 13,0% | 0,69 |
| Baixo risco, alto limite, bom pagamento | 43,1% | 13,1% | 0,88 |
| Uso moderado (sobrepagamento) | 1,3% | 17,5% | **0,36 — instável** |
| Deterioração recente / atraso | 17,7% | 39,6% | 0,93 |
| Atraso persistente | 8,3% | 65,0% | 0,97 |

ARI médio no bootstrap (30 réplicas) = 0,78; ARI K-Means × GMM = 0,51; DBSCAN: 4 grupos + 5,3% de ruído.
Clusters são descrições, não verdade econômica — o segmento instável é sinalizado, não interpretado.

### Estratégia de carteira

- **EL = PD × LGD × EAD**, EAD = saldo + CCF × limite não utilizado;
- **Capital** pela fórmula IRB de Basileia II para varejo rotativo (R = 0,04, 99,9%);
- **RAROC** = (receita − funding − EL − opex) / capital.

| Recorte | EAD | EL/EAD | RAROC |
|---|---|---|---|
| Carteira inteira de teste | NT$ 1.094 mi | 13,2% | −0,56 |
| Carteira aprovada no limiar 0,12 | NT$ 582 mi | 5,0% | **0,43** |

Com receita de 18%, só as faixas A (<5%) e B (5–10%) superam o hurdle; a taxa mínima para a faixa C seria
~20,7% (`reports/v3/pricing_bands.csv`). O pipeline também gera matriz segmento × faixa com ação sugerida,
sugestão de limite e ranking de cobrança (top 100 por PD×EAD: 69% de maus observados).

### Stress testing (PD e EL do champion)

| Cenário | Δ PD média | Δ EL |
|---|---|---|
| Utilização +20% | +0,9 p.p. | +14,0% |
| Atraso +1 mês | +9,4 p.p. | +48,9% |
| Corte de limite −20% | +0,8 p.p. | −10,0% (menor EAD) |
| Pagamento −30% (proxy de renda; o UCI não tem renda) | +1,3 p.p. | +8,3% |
| Combinado severo | +11,3 p.p. | +55,9% |

### Monitoramento (lotes de produção simulados a partir do teste)

Lotes 1–2 sem choque: PSI do score 0,003 (green). Lote 3 (utilização +25%): 0,017. Lote 4 (atraso +1 mês em
30% das contas): PSI do score 0,076 e **6 features em vermelho** (PAY_2–PAY_5, max_delay, n_months_delayed);
Gini cai de 0,573 para 0,510. PSI com bins por valor para variáveis discretas; CSI por bin de WoE.

### Equidade (limiar 0,12; atributos fora do modelo)

| Grupo | AIR (IC 95%) | Observação |
|---|---|---|
| Masculino vs feminino | 0,93 (0,88–0,98) | acima de 4/5 |
| Ensino médio vs universidade | 0,88 (0,81–0,97) | acima de 4/5 |
| Idade ≤25 vs 26–35 | **0,56 (0,51–0,63)** | **abaixo de 4/5 — investigar** |
| Idade 46–60 vs 26–35 | 0,82 (0,76–0,88) | limítrofe |

Mesmo sem AGE no modelo, clientes jovens são aprovados em proporção muito menor — efeito de proxy
(limite menor, histórico mais curto e mais atrasos). A v3 **não** ajusta limiar por grupo: o achado vai para
revisão de política (necessidade do negócio, variáveis alternativas, impacto regulatório).

### Integração com o portfólio

`artifacts/champion/` é o contrato consumido pelo repositório
[Credit-Score-Predictor---AWS-Streamlit](https://github.com/Yuri-Fernando/Credit-Score-Predictor---AWS-Streamlit):
`model.tar.gz` (joblib + `xgboost-model.json`), `model_card.json` (hash dos dados, hash do artefato,
`approval_status = PendingManualApproval`), `feature_schema.json` (features, faixas válidas, atributos
excluídos), `metrics.json`, `reference_profile.json` (distribuições de treino para drift) e
`golden_samples.json` (25 entradas brutas + PD esperada, usadas como teste de contrato no repositório AWS). A camada regulatória (PD/LGD/EAD, ECL IFRS 9, survival) está no
[IFRS17_Risk](https://github.com/Yuri-Fernando/IFRS17_Risk).

```text
RiskCredit (modelagem, estratégia) → Credit Score AWS (registry, deploy, monitoramento) → IFRS17_Risk (ECL, stress, capital)
```

---

## V2 — Visão Geral

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
├── run_pipeline_v3.py               # 🆕 v3 — orquestração
├── configs/v3.yaml                  # 🆕 v3 — premissas
├── src/                             # 🆕 v3 — data · features · models · segmentation · validation
│                                    #          monitoring · fairness · stress · strategy · export
├── dashboards/executive_dashboard.py  # 🆕 v3 — Dash (6 páginas)
├── reports/v3/                      # 🆕 v3 — resultados do run (CSV, figuras, summary.json)
├── artifacts/champion/              # 🆕 v3 — contrato para o repositório AWS
├── tests/test_v3.py                 # 🆕 v3 — 16 testes
├── CHANGELOG.md · ROADMAP.md        # 🆕 histórico e escopo
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

### 🆕 v3 — pipeline, testes e dashboard executivo

```bash
pip install -r requirements.txt
python run_pipeline_v3.py                  # ~45 s → reports/v3/ e artifacts/champion/
python -m pytest -q tests                  # 16 testes
python dashboards/executive_dashboard.py   # http://127.0.0.1:8050
```

### Opção 1 — Executar o notebook (v2)

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
* ~~Monitoramento de drift~~ — ✅ feito na v3;
* Comparação com outros algoritmos;
* Experimentação com diferentes técnicas de calibração;
* API para disponibilização do modelo — 🟡 contrato pronto; serviço no repositório [Credit-Score-Predictor---AWS-Streamlit](https://github.com/Yuri-Fernando/Credit-Score-Predictor---AWS-Streamlit);
* Containerização da aplicação;
* Pipeline de inferência automatizado;
* Monitoramento de performance e qualidade do modelo;
* ~~Expansão da análise de fairness e bias~~ — ✅ feito na v3;
* Validação *out-of-time* real (exige dataset com safras) — ver [ROADMAP.md](ROADMAP.md).

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
