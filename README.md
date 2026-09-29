# Credit Risk Analysis — Análise de Risco de Crédito com ML

Projeto de ciência de dados com pipeline completo de machine learning aplicado à predição de inadimplência de clientes de cartão de crédito, com rigor metodológico equivalente a artigo científico aplicado.

## Versões

| Versão | Foco | Onde |
|---|---|---|
| **v3.0.0 (set/2026)** | **Credit Risk Analytics & Portfolio Strategy** — scorecard WoE/IV, champion/challenger, segmentação, threshold por custo/lucro, EL/capital/RAROC, stress, monitoramento, equidade, export para AWS | `src/`, `run_pipeline_v3.py`, `reports/v3/`, `artifacts/champion/` |
| v2 | Pipeline de ML em notebook (10 fases, SHAP, Dash) | `riskcredit_v2.ipynb` (seção abaixo) |
| v1 | Notebook exploratório original | `riskcredit.ipynb` |

Histórico completo em [CHANGELOG.md](CHANGELOG.md) · escopo pendente em [ROADMAP.md](ROADMAP.md).

---

## V3 — Credit Risk Analytics & Portfolio Strategy

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
├── run_pipeline_v3.py         # V3 — orquestração completa
├── configs/v3.yaml            # premissas (scorecard, economia, stress, monitoramento)
├── src/                       # V3 — módulos
│   ├── data/                  # carga, limpeza, features, EAD
│   ├── features/woe.py        # binning monotônico, WoE, IV
│   ├── models/                # scorecard (PDO, reason codes) · champion/challenger
│   ├── segmentation/          # K-Means, GMM, DBSCAN, estabilidade bootstrap
│   ├── validation/            # métricas, PSI, robustez de defasagem
│   ├── monitoring/drift.py    # PSI/CSI, drift de score/alvo/calibração, alertas
│   ├── fairness/audit.py      # aprovação, TPR/FPR, calibração, AIR com IC
│   ├── stress/                # choques de utilização, atraso, limite, pagamento
│   ├── strategy/              # threshold custo/lucro · EL, capital IRB, RAROC, pricing, limites, cobrança
│   └── export/artifacts.py    # contrato para o repositório AWS
├── dashboards/executive_dashboard.py   # Dash — 6 páginas executivas
├── reports/v3/                # CSVs, figuras e summary.json do run
├── artifacts/champion/        # model.tar.gz, model_card.json, feature_schema.json, metrics.json
├── tests/test_v3.py           # 15 testes
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
pip install -r requirements.txt
```

### V3 — pipeline, testes e dashboard executivo
```bash
python run_pipeline_v3.py                  # ~50 s → reports/v3/ e artifacts/champion/
python -m pytest -q tests                  # 15 testes
python dashboards/executive_dashboard.py   # http://127.0.0.1:8050
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
