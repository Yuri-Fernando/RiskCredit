# CHANGELOG — RiskCredit

Documento mestre de histórico. Formato [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/) + SemVer.

## [3.0.2] — 2026-09-28

### Fixed
- `feature_schema.json`: faixa de `PAY_AMT*` era [−2, 9] porque o prefixo `PAY_` casava antes de `PAY_AMT`; agora usa o prefixo mais longo (teste de regressão adicionado). Detectado ao integrar o contrato no repositório AWS.

## [3.0.1] — 2026-09-28

### Added
- Export do champion inclui `reference_profile.json` (drift) e `golden_samples.json` (contrato/paridade com o repositório AWS).

## [3.0.0] — 2026-09-28 — Credit Risk Analytics & Portfolio Strategy

Origem: plano de evolução do portfólio financeiro (seções 4, 7 e 8), priorizando os itens P0 de RiskCredit.

### Added
- `src/` modular (data, features, models, segmentation, validation, monitoring, fairness, stress, strategy, export).
- **Scorecard tradicional**: binning monotônico, WoE, IV (com faixas de Siddiqi), filtro de correlação,
  remoção de sinal contraintuitivo, escala 600/50:1/PDO 20, tabela de pontos e reason codes.
- **Champion/challenger formal** com regra definida antes dos resultados (Gini, ECE, PSI, explicabilidade, latência).
- **Segmentação** K-Means / GMM / DBSCAN com silhouette, Davies-Bouldin, BIC, estabilidade por bootstrap
  (ARI e Jaccard por cluster) e perfis com rótulo descritivo.
- **Threshold por custo e lucro esperado** com curva de aprovação × bad rate e lucro realizado no teste.
- **Estratégia**: EL, capital IRB (varejo rotativo), RAROC, pricing bands, sugestão de limite, prioridade de cobrança,
  matriz segmento × faixa de PD.
- **Stress testing** (utilização, atraso, limite, pagamento, combinado) com EL e concentração.
- **Monitoramento**: PSI por feature (bins por valor em discretas), CSI por bin de WoE, drift de score/alvo/calibração,
  performance por lote, alertas green/amber/red.
- **Equidade**: aprovação, TPR/FPR, calibração e AIR com IC bootstrap por sexo, escolaridade e faixa etária.
- **Robustez de defasagem** (explicitamente não-OOT).
- **Export do champion** para o repositório AWS (`artifacts/champion/`).
- Dashboard executivo Dash (6 páginas), `configs/v3.yaml`, `requirements.txt`, 15 testes (16 a partir da 3.0.2).

### Changed
- Atributos demográficos saem das features do modelo e passam a ser só de auditoria.

### Não alterado
- `riskcredit_v2.ipynb`, `generate_notebook.py` e `outputs/` (v2) permanecem intactos.

## [2.0.0] — 2026-06-02
- Notebook de 28 células gerado por `generate_notebook.py`: EDA, 7 features, 5 modelos, otimização, K-Fold,
  t pareado, McNemar, calibração, SHAP, perfis individuais, Dash, PDF.

## [1.0.0] — 2025-07-17
- Notebook exploratório original (`riskcredit.ipynb`).
