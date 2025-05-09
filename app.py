import dash
from dash import dcc
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output
import joblib
import base64
import io
import pandas as pd

model = joblib.load('model.pkl') 

# Application
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Prédiction de Faillite d'Entreprise"),
    
    # Fichier CSV
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Button('Télécharger des données'),
            multiple=False
        ),
        html.Div(id='output-data-upload'),
    ]),
    
    # Prédiction
    html.Div([
        html.H3("Prédiction de Faillite"),
        html.Div(id='prediction-output'),
    ])
])

# Callback

@app.callback(
    [Output('prediction-output', 'children')],
    [Input('upload-data', 'contents')]
)
def update_output(content):
    if content is None:
        return "Aucune donnée téléchargée", {}

    # Supposons que l'utilisateur télécharge un fichier CSV
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=';', header=None)
    
    # Prédire avec le modèle sur les nouvelles données
    X_new = df 
    y_pred = model.predict(X_new)
    
    return [f'Prédiction : Faillite' if y_pred[0] == 1 else f'Prédiction : Non-Faillite']

if __name__ == '__main__':
    app.run(debug=True)