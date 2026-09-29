"""Dashboard executivo RiskCredit V3 (Dash) — 6 páginas, lê só reports/v3/.

    python run_pipeline_v3.py           # gera os artefatos
    python dashboards/executive_dashboard.py   → http://127.0.0.1:8050
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, dash_table, dcc, html

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports" / "v3"
S = json.loads((R / "summary.json").read_text(encoding="utf-8"))


def table(df: pd.DataFrame, fmt: int = 4):
    df = df.copy()
    for c in df.select_dtypes("number"):
        df[c] = df[c].round(fmt)
    return dash_table.DataTable(df.to_dict("records"), [{"name": c, "id": c} for c in df.columns],
                                style_table={"overflowX": "auto"}, style_cell={"fontSize": 12}, page_size=12)


def kpi(label, value):
    return html.Div([html.Div(label, style={"color": "#5b6474", "fontSize": 12}),
                     html.Div(value, style={"fontSize": 22, "fontWeight": 600, "color": "#2f5d8a"})],
                    style={"border": "1px solid #dfe3ea", "borderRadius": 8, "padding": 12, "minWidth": 170})


pf = S["portfolio"]
cc = pd.read_csv(R / "champion_challenger.csv")
seg = pd.read_csv(R / "segmentation_profile_kmeans.csv")
thr = pd.read_csv(R / "threshold_curve.csv")
mon = pd.read_csv(R / "monitoring_batches.csv")
monf = pd.read_csv(R / "monitoring_features.csv")
bands = pd.read_csv(R / "pricing_bands.csv")
points = pd.read_csv(R / "scorecard_points.csv")
iv = pd.read_csv(R / "iv_table.csv")

pages = {
    "1 · Visão da carteira": html.Div([
        html.Div([kpi("Exposição (EAD, teste)", f"{pf['total_ead'] / 1e6:,.0f} mi"),
                  kpi("Perda esperada", f"{pf['expected_loss'] / 1e6:,.1f} mi"),
                  kpi("EL / EAD", f"{pf['el_rate']:.1%}"),
                  kpi("RAROC carteira aprovada", f"{pf['approved_book_at_threshold']['raroc']:.2f}"),
                  kpi("Champion", S["champion"]["champion"])], style={"display": "flex", "gap": 12,
                                                                     "flexWrap": "wrap"}),
        html.P(pf["note"], style={"color": "#9a5b00"}),
        dcc.Graph(figure=px.bar(bands, x="pd_band", y="ead", title="Exposição por faixa de PD")),
        dcc.Graph(figure=px.bar(seg, x="label", y="share", color="default_rate", title="Segmentos de risco")),
    ]),
    "2 · Performance do modelo": html.Div([
        table(cc[["model", "auc", "gini", "ks", "brier", "ece", "psi_train_test", "passes_calibration",
                  "passes_stability", "explainability", "n_features", "latency_ms_per_1k"]]),
        html.P(f"Decisão: {S['champion']['reason']}"),
        html.P(f"Robustez de defasagem (não é OOT): Gini {S['robustness']['full_window']['gini']:.3f} → "
               f"{S['robustness']['lagged_1_month']['gini']:.3f} sem o mês mais recente."),
        html.H4("Information Value"), table(iv),
    ]),
    "3 · Segmentação": html.Div([
        table(seg), html.Img(src="/assets/segmentation_pca.png", style={"maxWidth": "100%"}),
        html.P(f"Estabilidade bootstrap: ARI médio {S['segmentation']['stability']['ari_mean']:.3f}; clusters "
               f"instáveis (Jaccard < 0,6): {S['segmentation']['stability']['unstable_clusters']}. "
               "Clusters são descritivos, não verdade econômica."),
    ]),
    "4 · Estratégia": html.Div([
        dcc.Graph(figure=px.line(thr, x="threshold", y=["expected_profit", "realized_profit_test"],
                                 title="Lucro esperado × realizado por limiar de PD")),
        dcc.Graph(figure=px.line(thr, x="approval_rate", y="bad_rate_approved", title="Aprovação × bad rate")),
        table(bands), html.P(json.dumps(S["threshold"]["max_expected_profit"], ensure_ascii=False)),
    ]),
    "5 · Monitoramento": html.Div([
        table(mon), dcc.Graph(figure=px.density_heatmap(monf, x="batch", y="feature", z="psi", histfunc="avg",
                                                        title="PSI por feature e lote (lotes 3–4 com choque simulado)",
                                                        color_continuous_scale="RdYlGn_r")),
    ]),
    "6 · Explicabilidade": html.Div([
        html.P("Scorecard: pontos por faixa (reason codes = características mais distantes da pontuação máxima)."),
        table(points),
        html.P(f"Exemplo de reason codes (5 contas): {S['reason_codes_example']}"),
    ]),
}

app = Dash(__name__, assets_folder=str(R / "figures"), title="RiskCredit V3")
app.layout = html.Div([
    html.H2("RiskCredit V3 — Credit Risk Analytics & Portfolio Strategy"),
    html.P("Dados UCI (Taiwan, 2005) · premissas econômicas ilustrativas · simulação de estratégia, "
           "não sistema de concessão.", style={"color": "#5b6474"}),
    dcc.Tabs([dcc.Tab(label=k, children=[v]) for k, v in pages.items()]),
], style={"maxWidth": 1200, "margin": "0 auto", "fontFamily": "system-ui, sans-serif", "padding": 16})

if __name__ == "__main__":
    app.run(debug=False)
