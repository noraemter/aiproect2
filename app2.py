import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained Random Forest model and scaler
model = joblib.load('traffic_congestion_model_with_rf.pkl')

# Load the dataset
data = pd.read_csv('dataset/urban_mobility_data_past_year.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Initialize the MinMaxScaler for features
scaler_X = MinMaxScaler(feature_range=(0, 1))
weather_conditions = ['Clear', 'Cloudy', 'Rainy']
scaler_X.fit(pd.DataFrame(np.zeros((1, len(weather_conditions) + 3)), columns=weather_conditions + ['temperature', 'humidity', 'population_density']))

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Traffic Congestion Prediction Dashboard", className="text-center")),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Date:"),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=data['timestamp'].min().date(),
                max_date_allowed=data['timestamp'].max().date(),
                date=data['timestamp'].min().date()  
            ),
        ], width=6),
        dbc.Col([
            html.Label("Select Time:"),
            dcc.Slider(
                id='time-slider',
                min=0,
                max=23,
                step=1,
                value=12,  # Default to noon
                marks={i: f'{i}:00' for i in range(0, 24)}
            ),
        ], width=6),
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Weather Condition:"),
            dcc.Dropdown(
                id='weather-dropdown',
                options=[
                    {'label': 'Clear', 'value': 'Clear'},
                    {'label': 'Cloudy', 'value': 'Cloudy'},
                    {'label': 'Rainy', 'value': 'Rainy'}
                ],
                value='Clear'
            ),
        ], width=6),
        dbc.Col([
            html.Label("Select Temperature (°C):"),
            dcc.Slider(
                id='temperature-slider',
                min=data['temperature'].min(),
                max=data['temperature'].max(),
                step=1,
                value=(data['temperature'].min() + data['temperature'].max()) / 2, 
                marks={i: f'{i}°C' for i in range(int(data['temperature'].min()), int(data['temperature'].max()) + 1, 5)}
            ),
        ], width=6),
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Humidity (%):"),
            dcc.Slider(
                id='humidity-slider',
                min=data['humidity'].min(),
                max=data['humidity'].max(),
                step=1,
                value=(data['humidity'].min() + data['humidity'].max()) / 2, 
                marks={i: f'{i}%' for i in range(int(data['humidity'].min()), int(data['humidity'].max()) + 1, 10)}
            ),
        ], width=6),
        dbc.Col([
            html.Label("Select Population Density (people/km²):"),
            dcc.Slider(
                id='population-density-slider',
                min=data['population_density'].min(),
                max=data['population_density'].max(),
                step=1,
                value=(data['population_density'].min() + data['population_density'].max()) / 2, 
                marks={i: f'{i}' for i in range(int(data['population_density'].min()), int(data['population_density'].max()) + 1, 1000)}
            ),
        ], width=6),
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            dbc.Button("Predict Congestion", id="predict-btn", className="mt-4"),
            html.Div(id='prediction-output')
        ], width=12),
    ], className="mt-4"),
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    [Input('date-picker', 'date'),
     Input('time-slider', 'value'),
     Input('weather-dropdown', 'value'),
     Input('temperature-slider', 'value'),
     Input('humidity-slider', 'value'),
     Input('population-density-slider', 'value'),
     Input('predict-btn', 'n_clicks')]
)
def predict_congestion(selected_date, selected_time, weather_condition, temperature, humidity, population_density, n_clicks):
    if n_clicks is not None:
        try:
            weather_map = {
                'Clear': [1, 0, 0],
                'Cloudy': [0, 1, 0],
                'Rainy': [0, 0, 1]
            }
            weather_encoded = weather_map.get(weather_condition, [0, 0, 0])

            features = np.array(weather_encoded + [temperature, humidity, population_density]).reshape(1, -1)
            features_scaled = scaler_X.transform(features)

            predicted_congestion = model.predict(features_scaled)[0] 

            return f'Predicted Congestion Level on {selected_date} at {selected_time}:00: {predicted_congestion:.2f}'
        except Exception as e:
            return f'Error: {str(e)}'
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
