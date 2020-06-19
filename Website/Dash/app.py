import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('datas.csv')

fig = px.scatter(df, x="Strain (%)", y="Stress (MPa)", color="Actuator Type")
fig.update_layout(yaxis_range=[-20, 750],
                  title_text="Actuator Properties: Strain vs Stress")

fig2 = px.scatter_matrix(df[:4], dimensions=df.columns[1:5], color="Actuator Type")
fig2.update_layout(title_text="Actuator Properties")

app.layout = html.Div([
    html.H1(children='Muscle Actuators'),
    dcc.Graph(figure=fig),
    dcc.Graph(figure=fig2)
])

if __name__ == '__main__':
    app.run_server(debug=True)