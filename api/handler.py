import pandas as pd
import pickle
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann #(from folder rossmann, import rossmann class)

# loading model
model = pickle.load(open('/home/alan/Sales-Prediction/model/model_rossmann.pkl','rb'))


# initialize API
app = Flask(__name__) # __name__ = construtor

@app.route('/rossmann/predict', methods = ['POST']) # creating endpoint with method POST
def rossmann_predict():  #function that is executed when an endpoint receives a POST request. This function works on the data received.
    test_json = request.get_json() # retrieve the json data 
    
    if test_json:  # test whether the data is there or not
        if isinstance(test_json, dict): # if data is a dict, then we have only one line of data
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys()) # if data is not a dict, then it has multiple data. We need to name the columns.
        
        # Instantiate Rossmann Class ("copy"/call Rossmann class)
        pipeline = Rossmann()
        # run Data Cleaning on raw data
        df1 = pipeline.data_cleaning( test_raw )
        # run feature engineering on df1
        df2 = pipeline.feature_engineering(df1)
        # run data preprocessing on df2
        df3 = pipeline.data_preparation(df2)
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3) #generate predictions with xgboost model, the data that the user sent, and the data to generate predictions on.
        
        return df_response
        
    else:
        return Response( '{}', status = 200, mimetype = 'application/json') # if there's no data, return a response answer 200 (request was correct but execution failed)
        #mimetype indicates the data type
        
if __name__ == '__main__':
    app.run('0.0.0.0') #running flask on local host