# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from datetime import datetime as dt

external_stylesheets  =  ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app  =  dash.Dash(__name__, external_stylesheets = external_stylesheets)

df  =  pd.read_csv('/Users/jingfengma/Desktop/Workspace/Dash/Wastewater_Data.csv')
df  =  df[df['Date'].str.contains(r'2001')]
#df  =  df.dropna();

def generate_table(dataframe, max_rows = 10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], style = {'margin':'30px'})

app.title  =  'Influent Predictor'

app.layout  =  html.Div(children = [
    html.Div(children = [

        html.Div(children = [
            html.H6('Please choose parameters:', style = {'font-weight':'bold'}),
            dcc.Checklist(
                id = 'ColumesCheckList',
                options = [
                       {'label': u'TotalSolids', 'value': 'TotalSolids'},
                       {'label': u'SS', 'value': 'SS'},
                       {'label': u'BOD5', 'value': 'BOD5'},
                       {'label': 'Org-N', 'value': 'Org-N'},
                       {'label': 'P-TOT', 'value': 'P-TOT'},
                       {'label': 'NH3', 'value': 'NH3'},
                       {'label': 'TKN', 'value': 'TKN'},
                       {'label': 'SO4', 'value': 'SO4'},
                       {'label': 'PRCP_NOOA', 'value': 'PRCP_NOOA'}
                       ],
                value = ['TotalSolids', 'SS', 'BOD5'],
                style = {'font-size':'1.1em'}
                ),
        ], style = {'float': 'left', 'display': 'inline-block', 'width': '35%'}),



        html.Div(children = [
            html.H6('Please specify data range and parameters:', style = {'font-weight':'bold'}),

            html.Label('Start Date:'),
            dcc.DatePickerSingle(
                id = 'datePickerStart',
                min_date_allowed = dt.strptime(df['Date'][0],'%Y-%m-%d'),
                max_date_allowed = dt.strptime(df.iloc[len(df) - 1, 0],'%Y-%m-%d'),
                initial_visible_month = dt.strptime(df['Date'][0],'%Y-%m-%d'),
                date = df.iloc[0, 0]
            ),

            html.Label('End Date:'),
            dcc.DatePickerSingle(
                id = 'datePickerEnd',
                min_date_allowed = dt.strptime(df['Date'][0],'%Y-%m-%d'),
                max_date_allowed = dt.strptime(df.iloc[len(df) - 1, 0],'%Y-%m-%d'),
                initial_visible_month = dt.strptime(df.iloc[len(df) - 1, 0],'%Y-%m-%d'),
                date = df.iloc[len(df) - 1, 0]
            ),

            # dcc.Dropdown(
            #      options = [
            #               {'label': 'YY-MM-dd', 'value': 'NYC'},
            #               {'label': u'YY-MM-dd', 'value': 'MTL'},
            #               {'label': 'YY-MM-dd', 'value': 'SF'}
            #               ],
            #      value = 'MTL'
            #      ),


            html.Label('Show predicted value in __ days'),
            dcc.Input(value = '5', type = 'text'),

            html.Label('Show real value'),
            dcc.Input(value = 'a value', type = 'text'),

            html.Label('Form of report:'),
            dcc.Dropdown(
                id = 'reportTypeDropdown',
                options = [
                      {'label': u'Line Chart', 'value': 'LC'},
                      {'label': 'Bar Chart', 'value': 'BC'},
                      {'label': 'Table', 'value': 'TB'}
                      ],
                value = 'LC'
                )

            # html.Label('Slider'),
            # dcc.Slider(
            #            min = 0,
            #            max = 9,
            #            marks = {i: 'Label {}'.format(i) if i  =  =  1 else str(i) for i in range(1, 6)},
            #            value = 5,
            #            ),
        ], style = {'float': 'left', 'display': 'inline-block', 'width':'60%'}),

    ], style = {'float': 'left', 'display': 'inline-block', 'width': '45%'}),

    html.Div(children = [
        html.H6('Results:', style = {'font-weight':'bold'}),
        #generate_table(df),
        # dcc.Graph(
        #     id = 'graph',
        #     figure = fig
        # )
    ],
    id = 'reportDiv',
    style = {'float': 'left', 'display': 'inline-block', 'width': '55%'})
], style = {'margin':'20px'})

@app.callback(
    dash.dependencies.Output('reportDiv', 'children'),
    [dash.dependencies.Input('ColumesCheckList', 'value'),
    dash.dependencies.Input('datePickerStart', 'date'),
    dash.dependencies.Input('datePickerEnd', 'date'),
    dash.dependencies.Input('reportTypeDropdown', 'value')
    ])
def recalculate(checkListValues, startDate, endDate, reportType):
    global checkListBak, startDateBak, endDateBak, result


    if len(checkListValues) > 0:
        checkListBak  =  checkListValues

    if startDate < endDate:
        startDateBak = startDate
        endDateBak = endDate
        result = df[(df['Date'] >= startDateBak) & ((df['Date'] <= endDateBak))]

    if reportType == 'LC':
        result = result.melt(id_vars = 'Date', value_vars = checkListBak)
        print(result)
        return dcc.Graph(
            id = 'graph',
            figure = px.line(result, x = "Date", y = 'value', color = 'variable')
        )

    if reportType == 'BC':
        return dcc.Graph(
            id = 'graph',
            figure = px.bar(result, x = "Date", y = checkListBak)
        )

    if reportType == 'TB':
        return generate_table(result),

if __name__  ==  '__main__':
    app.run_server(debug = True)
