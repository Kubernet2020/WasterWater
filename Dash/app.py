# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from datetime import datetime as dt

#------------ML--------------
from pandas import read_csv
from pandas import datetime
from pandas import to_numeric
from pandas import concat
import numpy as np
import pandas as pd
#tensorflow_version 2.x
import tensorflow as tf

#tf.enable_v2_behavior()
from tensorflow.compat.v1.keras import backend as K
from keras.layers import SimpleRNN, Dense, LSTM, Bidirectional, GRU
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint

import os
import random as rn
#from keras import backend as K

print(tf.__version__)

#Setting a seed for the computer's pseudorandom number generator.
#This allows us to reproduce the results from our script:
n = 5
np.random.seed(100 * n)
rn.seed(10000 * n)

#Depending on the actual running environment, you may specify if using GPU, or only CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Tensorflow session configuration.
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

tf.compat.v1.set_random_seed(1000 * n)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

series = read_csv('/Users/jingfengma/Desktop/Workspace/Dash/Wastewater_Data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
series = series.replace('^\s*$', np.nan, regex=True)
series = series.fillna(method='ffill')
series = series.apply(to_numeric)

#lag defines how many historical data are used to predict a specific wasterwater characteristic (e.g., BOD5)
lag = 7

#num_features defines how many historical wasterwater characteristics are used. We have 9 in total (i.e., TS, BOD5, NH3, etc.)
num_features = 9
target_dict = {}
for i, j in enumerate(series.columns):
    target_dict[j] = i

#print(target_dict['Total Solids'])
from pandas import DataFrame
from sklearn import preprocessing
names = series.columns
x = series.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit(x)
Maxdata = list(x_scaled.data_max_)
Mindata = list(x_scaled.data_min_)
x_scaled = min_max_scaler.transform(x)
series_normalized = DataFrame(x_scaled, columns=names)



#The following line of code will do the same normalization as the code above.
#series_normalized = (series - np.min(series))/(np.max(series)-np.min(series))

# table2lags() Shifts a dataFrame along its time axis (i.e., index) n steps
# (moving down/up if step is a positive/negative number), determined by min_lag and max_lag,
# and merge all shiffted dataframes into a single one and return,
# without including the original DataFrame (the one shifted 0 step).
# "values" is a list, and each item in "values" is a shifted dataframe.
# Input: Table: a DataFrame; max_lag: the maximum shifting; min_lag: the minimum shifting;
# In this module, no need to include the original dataframe or shift the dataframe up, and thus min_lag = 1 by default
# separator: used to concatenate the step value (e.g., 1, 2, 3) to the original column label. E.g., SO4_1, SO4_2, etc.
# Output: a dataframe

def table2lags(table, max_lag, min_lag=1, separator='_'):
    values = []
    for i in range(min_lag, max_lag + 1):
        #append shiffted dataframe into the list (i.e., values)
        values.append(table.shift(i).copy())
        #replace the last item or dataframe's columns by column_n; n is the shift step
        values[-1].columns = [c + separator + str(i) for c in table.columns]
    #pandas.concat is used to merge all dataframes (as items in values) into a single dataframe
    return concat(values, axis=1)

#prepare all historical (e.g., one day ago, two day ago,...) data into one dataframe.
#For all missing data, replaced by its closest future values
X = table2lags(series_normalized, lag)
X = X.fillna(method='bfill')
print(X.columns)

#input start and end date(M/D/Y), target
#output prediction, real value, error.

def predictWithParams(output_targets = ["Total Solids", "SS"],
                      start_date = "12/28/2018",
                      end_date = "01/01/2019"):
    #output_targets = ["Total Solids", "SS"]
    #start_date = "12/28/2018"
    #end_date = "01/01/2019"

    timelist = pd.date_range('1/1/2001','12/31/2018')
    input_datasetdf = pd.DataFrame(X.values, index=timelist)[start_date: end_date]
    input_dataset = input_datasetdf.values.reshape(-1, lag, num_features).astype('float32')
    #Error mae mape
    #everyday error, mae, mape
    def mae_mape(actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        return np.abs(actual - pred), np.mean(np.abs(actual - pred)), np.mean(np.abs(actual - pred) / actual)

    #return a dictionary
    #list predict values,list actual values, array error, mae, mape no %
    def predictWithModel(start_date=start_date, end_date=end_date, output_targets=output_targets, input_dataset=input_dataset, realdata=series, Maxdata=Maxdata, Mindata=Mindata):
        ansdict = {}
        dataFrame = {'Date': pd.date_range(start_date, end_date, name = 'str')}
        # print(dataFrame['Date'])
        for target in output_targets:
            model_using = tf.keras.models.load_model('/Users/jingfengma/Desktop/Workspace/Dash/my_models/myIWS_RNNmodel' + str(target))
            y_predict = model_using.predict(input_dataset)
            origin_predict = []
            for ans in y_predict:
                origin_predict.append(ans[0] * (Maxdata[target_dict[str(target)]] - Mindata[target_dict[str(target)]]) +  Mindata[target_dict[str(target)]])
            # print(origin_predict)
            # print(realdata.loc[start_date: end_date][str(target)].values)

            dataFrame[str('Prediction Of '+target)] = origin_predict
            dataFrame[str('Real '+target)] = realdata.loc[start_date: end_date][str(target)].values
            error, mae, mape= mae_mape(origin_predict, list(realdata.loc[start_date: end_date][str(target)].values))

            # ansdict[str(target)] = (origin_predict, list(realdata.loc[start_date: end_date][str(target)].values), error, mae, mape)
            emmDic = {'error': error, 'mae': mae, 'mape': mape}
            ansdict[str('emm of ' + target)] = emmDic

        ansdict['PredictionFrame'] = dataFrame;

        return ansdict
    t = predictWithModel()
    # print(t)
    return t
    #print(t['Date'][0].strftime('%Y-%m-%d'))
#------------Dash--------------

external_stylesheets  =  ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app  =  dash.Dash(__name__, external_stylesheets = external_stylesheets)

df  =  pd.read_csv('/Users/jingfengma/Desktop/Workspace/Dash/Wastewater_Data.csv')
# df  =  df[df['Date'].str.contains(r'2001')]
#df  =  df.dropna();

def generate_table(dataframe, max_rows = 100):
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

    # html.Div(id = 'progress-background', style = {
    # 'filter':'alpha(opacity=50)', '-moz-opacity':'0.3', 'opacity': '0.3',
    # 'z-index':'1000',
    # 'width':'100%',
    # 'height':'100%',
    # 'background-color':'gray',
    # 'position':'absolute'
    # }),
    # html.Div(id = 'progress', children = [
    #     html.Label('Calculating...', style = {'font-weight':'bold', 'font-size':'28px'})
    # ],
    # style = {'margin':'auto', 'width':'300px', 'height':'80px', 'background-color':'white',
    #     'filter':'alpha(opacity=100)', '-moz-opacity':'1.0', 'opacity': '1.0', 'position':'fixed',
    #     'z-index':'1001', 'display':'flex', 'justify-content':'center', 'align-items':'center', 'border-radius':'10px'}),

    html.Div(children = [

        html.Div(children = [
            html.H6('Please choose parameters:', style = {'font-weight':'bold'}),
            dcc.Checklist(
                id = 'ColumesCheckList',
                options = [
                       {'label': u'Total Solids', 'value': 'Total Solids'},
                       {'label': u'SS', 'value': 'SS'},
                       {'label': u'BOD5', 'value': 'BOD5'},
                       {'label': 'Org-N', 'value': 'Org-N'},
                       {'label': 'P-TOT', 'value': 'P-TOT'},
                       {'label': 'NH3', 'value': 'NH3'},
                       {'label': 'TKN', 'value': 'TKN'},
                       {'label': 'SO4', 'value': 'SO4'},
                       {'label': 'PRCP_NOOA', 'value': 'PRCP_NOOA'}
                       ],
                value = ['Total Solids', 'SS', 'BOD5'],
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
                initial_visible_month = dt.strptime('2018-12-28','%Y-%m-%d'),
                date = df.iloc[365 * 3 + 100, 0]
            ),

            html.Label('End Date:'),
            dcc.DatePickerSingle(
                id = 'datePickerEnd',
                min_date_allowed = dt.strptime(df['Date'][0],'%Y-%m-%d'),
                max_date_allowed = dt.strptime(df.iloc[len(df) - 1, 0],'%Y-%m-%d'),
                #initial_visible_month = dt.strptime(df.iloc[len(df) - 1, 0],'%Y-%m-%d'),
                initial_visible_month = dt.strptime('2019-01-01','%Y-%m-%d'),
                date = df.iloc[365 * 3 + 130, 0]
            ),

            # dcc.Dropdown(
            #      options = [
            #               {'label': 'YY-MM-dd', 'value': 'NYC'},
            #               {'label': u'YY-MM-dd', 'value': 'MTL'},
            #               {'label': 'YY-MM-dd', 'value': 'SF'}
            #               ],
            #      value = 'MTL'
            #      ),

            html.Label('Show predicted value in __ days', style = {'margin-top':'15px'}),
            dcc.Input(id = 'daysOfPrediction', value = '5', type = 'text'),

            html.Label('Show real value'),
            dcc.Input(value = 'Yes', type = 'text'),

            html.Label('Form of report:', style = {'margin-top':'15px'}),
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

    dcc.Loading(
        id="loading",
        children = [
            html.Div(children = [
                # html.H6('Results:', style = {'font-weight':'bold'}),
                #generate_table(df),
                # dcc.Graph(
                #     id = 'graph',
                #     figure = fig
                # )
            ],
            id = 'reportDiv',
            style = {'float': 'left', 'display': 'inline-block', 'width': '55%'}),
        ],
        type = "default",
        style = {'filter':'alpha(opacity=50)', '-moz-opacity':'0.5', 'opacity': '0.5',},
        fullscreen = True,
    ),

    dcc.ConfirmDialog(
        id='confirm',
        message='Please check conditions and try again.',
    ),

], style = {'margin':'0px 20px 0px 20px'}) #'display':'flex', 'justify-content':'center', 'align-items':'center'

@app.callback(
    dash.dependencies.Output('confirm', 'displayed'),
    [dash.dependencies.Input('datePickerStart', 'date'),
    dash.dependencies.Input('datePickerEnd', 'date')])
def display_confirm(startDate, endDate):
    if startDate >= endDate:
        return True
    return False

@app.callback(
    dash.dependencies.Output('daysOfPrediction', 'value'),
    [dash.dependencies.Input('datePickerStart', 'date'),
    dash.dependencies.Input('datePickerEnd', 'date')])
def calculateDays(startDate, endDate):
    print(startDate);
    startTime= datetime.strptime(startDate,"%Y-%m-%d")
    endTime= datetime.strptime(endDate,"%Y-%m-%d")
    return (endTime- startTime).days

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

        result = pd.DataFrame.from_dict(predictWithParams(output_targets = checkListBak,
                              start_date = startDateBak,
                              end_date = endDateBak)['PredictionFrame'])
        print(result)
        #result = df[(df['Date'] >= startDateBak) & ((df['Date'] <= endDateBak))]

        if reportType == 'LC':
            result = result.melt(id_vars = 'Date', value_vars = result.iloc[:,1:])
            return dcc.Graph(
                id = 'graph',
                figure = px.line(result, x = "Date", y = result.columns.tolist(), color = 'variable')
            )

        if reportType == 'BC':
            result = result.melt(id_vars = 'Date', value_vars = result.iloc[:,1:])
            return dcc.Graph(
                id = 'graph',
                figure = px.bar(result, x = "Date", y = result.columns.tolist(), color = 'variable')
            )

        if reportType == 'TB':
            temp=list(result['Date'])
            for i in temp:
    	        result['Date'] = i.strftime("%Y-%m-%d")
            return generate_table(result)

    else:
        return

if __name__  ==  '__main__':
    app.run_server(debug = True)
