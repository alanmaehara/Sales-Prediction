import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import telegram
import os
from io import BytesIO
from flask import Flask, request, Response

# constants
TOKEN = '1499227844:AAErjtmgpQnW5N5qU1c_CiVi_eEsgxBFzl0'
bot = telegram.Bot(token=TOKEN)

# # Info about the bot:
# https://api.telegram.org/bot1499227844:AAErjtmgpQnW5N5qU1c_CiVi_eEsgxBFzl0/getMe

# # Get updates from users:
# https://api.telegram.org/bot1499227844:AAErjtmgpQnW5N5qU1c_CiVi_eEsgxBFzl0/getUpdates

# # Webhook
# https://api.telegram.org/bot1499227844:AAErjtmgpQnW5N5qU1c_CiVi_eEsgxBFzl0/setWebhook?url=https://f690a0cf7282.ngrok.io

# # Webhook Heroku
# https://api.telegram.org/bot1499227844:AAErjtmgpQnW5N5qU1c_CiVi_eEsgxBFzl0/setWebhook?url=https://rossmann-telegram-bot-final.herokuapp.com


# # Send message to users:
# https://api.telegram.org/bot1499227844:AAErjtmgpQnW5N5qU1c_CiVi_eEsgxBFzl0/sendMessage?chat_id=1288672672&text=Hi! I will send your sales prediction soon.

def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot{}/'.format(TOKEN)
    url = url + 'sendMessage?chat_id={}'.format(chat_id)

    r = requests.post(url, json = {'text': text})
    print('Status Code {}'.format(r.status_code))
    
    return None

def load_dataset(store_Id):
    # loading test dataset
    df10 = pd.read_csv('test_with_customers.csv')
    df_store_raw = pd.read_csv('store.csv')

    # merge store dataset on test dataset
    df_test = pd.merge(df10, df_store_raw, how = 'left', on = 'Store')

    # choose one store for prediction
    df_test = df_test[df_test['Store'] == store_Id]

    if not df_test.empty:
        # remove closed days
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()]
        df_test = df_test.drop('Id', axis = 1)

        # convert dataframe to JSON
        data = json.dumps(df_test.to_dict(orient = 'records'))

    else:
        data = 'error'

    return data

def predict(data):
    # API call
    url = 'https://rossmann-model-teste.herokuapp.com/rossmann/predict' # cloud host
    header = {'Content-type': 'application/json'}
    data = data

    r = requests.post(url, data = data, headers = header)
    print( 'Status Code {}'.format(r.status_code)) # if returns 200, request worked

    d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())

    return d1


def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/','') # remove the / that telegram uses

    try:
        store_id = int(store_id)
    
    except ValueError:
        store_id = 'error'

    return chat_id, store_id



# API initialize
app = Flask(__name__)

@app.route( '/', methods = ['GET','POST']) # create the endpoint/route (from where the user message will come)
def index(): #runs every time the endpoint / is called;
    
    if request.method == 'POST':
        message = request.get_json() #get the json data
        chat_id, store_id = parse_message( message )

        if store_id != 'error':
            # loading data
            data = load_dataset( store_id)

            if data != 'error':
                # prediction
                d1 = predict(data)
                # calculation
                d2 = d1[['store','prediction']].groupby('store').sum(0).reset_index()
                # send intro message
                intro = 'Generating sales forecast for store {}...'.format(d2['store'].values[0])
                send_message(chat_id, intro)   
                # send lineplot
                fig = plt.figure()
                sns.lineplot(x = 'week_of_year', y = 'prediction', data = d1)
                plt.title('Weekly Sales Forecast for Store {}'.format(d2['store'].values[0]))
                plt.xlabel('Week Year (starting from week 31 - July 19th)')
                plt.ylabel('Sales Prediction (US$)')
                buffer = BytesIO()
                fig.savefig(buffer, format='png')
                buffer.seek(0)
                bot.send_photo(chat_id=chat_id, photo=buffer)
                # send intro message
                msg = 'Store {} will sell ${:,.2f} for the next six weeks. If you wish to get predictions for a different store, just text me another store number.'.format(d2['store'].values[0],d2['prediction'].values[0])
                send_message(chat_id, msg)

                return Response('Ok', status = 200)

            else:
                send_message(chat_id, 'Predictions cannot be made for store {}. Please contact maehara@salesforecast.com for more details.'.format(store_id))

                return Response('0k', status = 200) 

        else:
            send_message(chat_id, 'Sorry, something went wrong. Please use only integer numbers.')
            return Response('0k', status = 200)

    else:  # if GET, means that user didn't send any data
        return '<h1> Rossmann Telegram Bot </h1>'

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( host = '0.0.0.0', port = port) # run app at host 0.0.0.0 on port 5000.



