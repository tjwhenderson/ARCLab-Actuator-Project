#dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
#import pandas
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#read csv
df = pd.read_csv('datas.csv')

#define available columns for graphing
available_indicators = df.columns[1:5]

app.layout = html.Div([
    html.Div([

        html.Div([
            #define dropdown for x values
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Strain (%)'
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            #define dropdown for y values
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Stress (MPa)'
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    #set up graph
    dcc.Graph(id='indicator-graphic'),

])

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value')])
#updates x and y values based on dropdown and indicates each dot by actuator type
def update_graph(xaxis_column_name, yaxis_column_name):
    traces = []
    for i in df["Actuator Type"].unique():
        df_by_actuator = df[df['Actuator Type'] == i]
        traces.append(dict(
            x=df_by_actuator[xaxis_column_name],
            y=df_by_actuator[yaxis_column_name],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i
        ))


    return {
        'data': traces,
        'layout': dict(
            xaxis={
                'title': xaxis_column_name
            },
            yaxis={
                'title': yaxis_column_name
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)