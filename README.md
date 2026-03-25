 Credit Default Prediction Dashboard

Este projeto aplica Machine Learning para prever inadimplência de clientes de cartão de crédito usando o dataset UCI Credit Card Default. O modelo é interpretado com SHAP e visualizado num dashboard interativo construído com Dash (Plotly).
 
Funcionalidades

    📊 Classificação binária com Random Forest e balanceamento de classes via SMOTE
    🧮 Métricas de performance: ROC AUC, matriz de confusão, classification report
    🔬 Análise interpretável com SHAP (importância global e local das features)
    🧠 Dashboard com múltiplas abas:
        Visão geral do modelo e resultados
        Curva ROC
        Distribuição dos scores
        SHAP summary (importância das variáveis)
        Análise por perfil de cliente (interativo)

 Stack usada
    Python (pandas, scikit-learn, imblearn, shap, matplotlib)
    Dash (Plotly) + Dash HTML & Core Components
    SMOTE para tratamento de desbalanceamento
    SHAP para interpretabilidade de modelos

 Como rodar
    Clone o repositório
    Instale as dependências:
pip install -r requirements.txt

    Rode a aplicação:
python app.py
    Acesse o dashboard em http://localhost:8050


Projeto em adamento ainda falta:
- Terminar o app
- Simular um caso real de score de crédito
- Usar técnicas aplicadas (engenharia de atributos + modelos interpretáveis)
- Amarrar com alertas automáticos via n8n
