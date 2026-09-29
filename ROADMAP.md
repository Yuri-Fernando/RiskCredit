# ROADMAP — RiskCredit

Legenda: ✅ feito na v3 · 🟡 parcial · ⏳ planejado

| Item do plano | Estado | Observação |
|---|---|---|
| Segmentação K-Means + GMM + DBSCAN | ✅ | HDBSCAN e UMAP não usados (dependências extras; PCA só para visualização) |
| WoE/IV + scorecard logístico, PDO, reason codes | ✅ | |
| Threshold por custo/retorno | ✅ | |
| PSI/CSI/drift, alert levels | ✅ | lotes de produção simulados (UCI sem série temporal) |
| Portfolio strategy + dashboard executivo | ✅ | Dash (Power BI não usado) |
| Champion/challenger formal | ✅ | |
| Fairness ampliado com IC | ✅ | achado de proxy etário aguarda revisão de política |
| Stress testing | ✅ | choque de renda substituído por pagamento (UCI sem renda) |
| **OOT verdadeiro** | ⏳ | exige dataset com originação/safra (ex.: base pública de empréstimos) |
| Validação externa com 2º dataset público | ⏳ | |
| Bayesian PD por segmento (Beta-Binomial) | ⏳ | implementado no motor ECL do IFRS17_Risk; trazer para segmentos daqui |
| Inferência causal (política de cobrança, IPW/DR) | ⏳ | só com desenho sintético apropriado |
| Séries temporais de delinquency/vintage | ⏳ | no IFRS17_Risk (vintage/migração); UCI não tem safra |
| Pipeline RiskCredit → Model Registry → AWS → Monitoring | 🟡 | contrato de artefatos pronto; registry local no repo AWS; deploy real pausado por custo |
| ECL consumindo os mesmos scores/model versions | ⏳ | P2 do plano |
