import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import pandas as pd
import plotly.graph_objs as go
import glob
import os
import base64
import flask
from datetime import datetime
import pickle as pickle
import xgboost as xgb
import gc
prop = pd.read_csv('properties_2016.csv')
train = pd.read_csv('train_2016_v2.csv')
sample = pd.read_csv('sample_submission.csv')[:100]
df_train = train.merge(prop, how='left', on='parcelid')
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
train_columns = x_train.columns
xgbmodel = pickle.load(open("xgboostmodel500.dat", "rb"))
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
d_test = xgb.DMatrix(x_test)
print('Predicting on test ...')
p_test = xgbmodel.predict(d_test)
# read sample_submission.csv in S3
sub = pd.read_csv('sample_submission.csv')[:100]
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test
# save predict result and score of model to S3
sub.to_csv('predict.csv', index=False, float_format='%.4f') # Thanks to @inversion
df = pd.read_csv('predict.csv')[:100]
timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
modelversion=timenow
server=flask.Flask(__name__)
app = dash.Dash(__name__,server=server)
image_filename = 'houseAds.jpg' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
validationmae = open("validation.rtf").read()


app.layout=html.Div(children=[

	html.H1('House Price Prediction with Model Release on '+ modelversion +'with validation scrore'+ validationmae),
	#dcc.Checklist(
	 #   options=[
	 #       {'label': 'Palo Alto', 'value': 'PA'},
	 #       {'label': 'San Jose', 'value': 'SJ'},
	 #   ],
	 #   values=['PA', 'SJ'],
	 #   labelStyle={'display': 'inline-block'}
	#)
	dcc.Graph(
        id='predicthouse',
        figure={
            'data': [
                go.Histogram(
                	x=df[i],
                	histnorm='probability'
                	)for i in ['201610']
            ],
            'layout': go.Layout(
                xaxis={'title': 'prediction'},
                yaxis={'title': 'target'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    ),

	#dcc.Graph(id='example',
			#figure={
				#'data':[
				#	{'x':preditdata_PA,'y':target_PA,'type':'markers','name':'PA'},
				#	{'x':preditdata_SJ,'y':target_SJ,'type':'markers','name':'SJ'}
				#	],
				#'layout':{
				#	'title':'prediction on different areas'

				#	}
			#}),

	html.Img(src='data:image/jpg;base64,{}'.format(encoded_image)),
	html.Div(id='output-container-button',
             children='Please enter your house properties'),
	dcc.Input(
	    placeholder='Enter house area',
	    type='number',
	    value=''
	),
	dcc.Input(
	    placeholder='Enter house age',
	    type='number',
	    value=''
	),
	dcc.Input(
	    placeholder='Enter house type',
	    type='text',
	    value=''
	),

	html.Button('Submit', id='button'),
    html.Div(id='prediction-button',
             children='If the prediction is good? (yes or no)'),


	html.Div(dcc.Input(id='input-box', type='text')),
    html.Button('Submit', id='button'),
    html.Div(id='evaluate-button',
             children='If the prediction is good? (yes or no)')


    #dcc.DatePickerRange(
    #id='date-picker-range',
    #start_date=dt(1997, 5, 3),
    #end_date_placeholder_text='Select a date!'
#)




	#dcc.Slider(
	#min=-5,
	#max=10,
	#step=0.5,
	#value=-3,
	#)


	])

@app.callback(
    dash.dependencies.Output('evaluate-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])

def update_output(n_clicks, value):
    return 'The input value was "{}" and the button has been clicked {} times'.format(
        value,
        n_clicks
    )

"""go.Scatter(
                    x=df[df['state'] == i]['prediction'],
                    y=df[df['state'] == i]['target'],
                    text= df[df['state'] == i]['area'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 10,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in ['201']"""

if __name__=='__main__':
	app.run_server(debug=True, host='0.0.0.0')
