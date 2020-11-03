import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann(object):
    def __init__(self):
        self.promo2_time_week_rs = pickle.load(open('parameter/promo_time_week_rs.pkl', 'rb'))
        self.promo2_time_month_rs = pickle.load(open('parameter/promo_time_month_rs.pkl', 'rb'))
        self.competition_distance_yeojohn = pickle.load( open('parameter/competition_distance_yeojohn.pkl', 'rb'))
        self.customers_yeojohn = pickle.load(open('parameter/customers_yeojohn.pkl', 'rb'))
        self.competition_since_month_yeojohn = pickle.load( open('parameter/competition_since_month_yeojohn.pkl', 'rb'))
        self.year_mms = pickle.load( open('parameter/year_mms.pkl', 'rb'))
        self.store_type_le = pickle.load( open('parameter/store_type_le.pkl', 'rb'))
        self.competition_open_since_year_le = pickle.load(open('parameter/competition_open_since_year_le.pkl', 'rb'))

 
    def data_cleaning(self, df1):
        
        ## 1.1. Rename Columns
        # store columns of dataset on new variable
        old_cols = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday','SchoolHoliday', 'Customers', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval']

        # rename all columns to snakecase type
        new_cols = list(map(lambda x: inflection.underscore(x), old_cols))
        df1.columns = new_cols

        # transform variable 'date' to a date type variable
        df1['date'] = pd.to_datetime(df1['date'])

        ## 1.4. Checking & Filling out NA Values
        ### 1.4.1. competition_distance
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 100000.0 if math.isnan(x) else x)

        ### 1.4.2. competition_open_since_month competition_open_since_year
        df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month 
                                                        if math.isnan( x['competition_open_since_month'] ) 
                                                        else x['competition_open_since_month'], axis=1 )

        df1['competition_open_since_year'] = df1.apply( lambda x: x['date'].year 
                                                       if math.isnan( x['competition_open_since_year'] ) 
                                                       else x['competition_open_since_year'], axis=1 )

        ### 1.4.3. promo2_since_year & promo2_since_week
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year 
                                             if math.isnan(x['promo2_since_year']) 
                                             else x['promo2_since_year'], axis = 1)

        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week 
                                             if math.isnan(x['promo2_since_week']) 
                                             else x['promo2_since_week'], axis = 1)

        ### 1.4.4. promo_interval
        # perform imputation by mapping month names to numbers
        month_map = {1: 'Jan',  2: 'Fev',  3: 'Mar',  4: 'Apr',  5: 'May', 
                     6: 'Jun',  7: 'Jul',  8: 'Aug',  9: 'Sep',  10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['promo_interval'].fillna(0, inplace = True)

        # Create new variable 'is_promo'. If store is opened and running consecutive promo on that period (promo_interval), we assign 1 (and zero otherwise) 
        df1['is_promo2'] = df1.apply(lambda x: 0 if x['promo_interval'] == 0 
                                    else 1 if x['month_map'] in x['promo_interval'].split(',') 
                                    else 0, axis = 1)

        ## 1.5. Change Dtypes
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int) 
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int) 
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int) 
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int) 
        
        return df1

    def feature_engineering(self, df2):

        ## 2.4. Feature Engineering
        # date
        df2['date'] = pd.to_datetime(df2['date'])

        # competition_since
        df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1 ), axis=1 )

        # competition_since_month
        df2['competition_since_month'] = ((df2['date'] - df2['competition_since']) / 30).apply(lambda x: x.days).astype(int)

        # promo2_since
        df2['promo2_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo2_since'] = df2['promo2_since'].apply( lambda x: 
                                                      datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
        # promo2_time_week
        df2['promo2_time_week'] = ((df2['date'] - df2['promo2_since']) / 7).apply(lambda x: x.days).astype(int)

        # promo2_time_month
        df2['promo2_time_month'] = ((df2['date'] - df2['promo2_since']) / 30).apply(lambda x: x.days).astype(int)

        # day, month, year
        df2['day'] = df2['date'].dt.day

        df2['month'] = df2['date'].dt.month

        df2['year'] = df2['date'].dt.year


        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # year week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        # is_weekday
        df2['is_weekday'] = df2.apply(lambda x: 1 if x['date'].weekday() in [0,1,2,3,4] 
                                      else 1 if x['state_holiday'] == 0
                                      else 0, axis = 1)

        # assortment
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a'
                                                    else 'extra' if x == 'b'
                                                    else 'extended')

        # state_holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a'
                                                          else 'easter' if x == 'b'
                                                          else 'christmas' if x == 'c'
                                                          else 'regular_day')

        # 3.0. STEP 03 - FILTERING VARIABLES (according to business restritions)
        # We will exclude closed stores with zero sales
        # df2 = df2[df2['open'] != 0]

        # dropping irrelevant columns
        cols_drop = ['promo_interval', 'open', 'month_map']
        df2 = df2.drop(cols_drop, axis = 1)

        # reordering columns
        df2 = df2[['store', 'date', 'day', 'month', 'year','week_of_year','year_week','customers','day_of_week','is_weekday','state_holiday','school_holiday',
             'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year',
             'competition_since', 'competition_since_month', 'promo', 'is_promo2','promo2', 'promo2_since', 'promo2_since_week', 
             'promo2_since_year', 'promo2_time_week', 'promo2_time_month']]

        return df2

    def data_preparation(self, df3):

        ## RobustScaler
        # promo2_time_week (weeks of promo2 since it started)
        rs_promo2_time_week = self.promo2_time_week_rs.fit(df3[['promo2_time_week']].values)
        df3['promo2_time_week'] = rs_promo2_time_week.transform(df3[['promo2_time_week']].values)

        # promo2_time_month (months of promo2 since it started)
        rs_promo2_time_month = self.promo2_time_month_rs.fit(df3[['promo2_time_month']].values)
        df3['promo2_time_month'] = rs_promo2_time_month.transform(df3[['promo2_time_month']].values)

        ## yeo-john
        
        # customers
        yeojohn_customers = self.customers_yeojohn.fit(df3[['customers']].values)
        df3['customers'] = yeojohn_customers.transform(df3[['customers']].values)
        
        # competition_since_month (months of competition since it started)
        yeojohn_competition_since_month = self.competition_since_month_yeojohn.fit(df3[['competition_since_month']].values)
        df3['competition_since_month'] = yeojohn_competition_since_month.transform(df3[['competition_since_month']].values)
        # competition_distance
        yeojohn_competition_distance = self.competition_distance_yeojohn.fit(df3[['competition_distance']].values)
        df3['competition_distance'] = yeojohn_competition_distance.transform(df3[['competition_distance']].values)

        ## MinMaxScaler
        # year
        mms_year = self.year_mms.fit(df3[['year']].values)
        df3['year'] = mms_year.transform( df3[['year']].values)

        # # One Hot Encoding
        # state_holiday 
        df3 = pd.get_dummies(df3, prefix = ['state_holiday'], columns = ['state_holiday'])
        # promo2_since_year
        df3['promo2_since_year'] = df3['promo2_since_year'].astype(str)
        df3 = pd.get_dummies(df3, prefix = ['promo2_since_year'], columns = ['promo2_since_year'])

        # # Ordinal Encoding
        # assortment
        assortment_dict = {'basic': 1,  'extra': 2, 'extended': 3}
        df3['assortment'] = df3['assortment'].map( assortment_dict )

        # # Label Encoding
        # store_type
        le_store_type = self.store_type_le.fit(df3['store_type'])
        df3['store_type'] = le_store_type.transform(df3['store_type'])

        # competition_open_since_year
        df3['competition_open_since_year'] = df3['competition_open_since_year'].astype(str)
        le_competition_open_since_year = self.competition_open_since_year_le.fit(df3['competition_open_since_year'])
        df3['competition_open_since_year'] = le_competition_open_since_year.transform(df3['competition_open_since_year'])

        ### 5.3.3. Nature Transformation (cyclic)
        # month
        df3['month_sin'] = df3['month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
        df3['month_cos'] = df3['month'].apply(lambda x: np.cos(x*(2*np.pi/12)))
        # day
        df3['day_sin'] = df3['day'].apply(lambda x: np.sin(x*(2*np.pi/30)))
        df3['day_cos'] = df3['day'].apply(lambda x: np.cos(x*(2*np.pi/30)))
        # week_of_year (W)
        df3['week_of_year_sin'] = df3['week_of_year'].apply(lambda x: np.sin(x*(2*np.pi/52)))
        df3['week_of_year_cos'] = df3['week_of_year'].apply(lambda x: np.cos(x*(2*np.pi/52)))
        # day_of_week
        df3['day_of_week_sin'] = df3['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/7)))
        df3['day_of_week_cos'] = df3['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/7)))
        # promo2_since_week
        df3['promo2_since_week_sin'] = df3['promo2_since_week'].apply(lambda x: np.sin(x*(2*np.pi/52)))
        df3['promo2_since_week_cos'] = df3['promo2_since_week'].apply(lambda x: np.cos(x*(2*np.pi/52)))
        #competition_open_since_month
        df3['competition_open_since_month_sin'] = df3['competition_open_since_month'].apply(lambda x: np.sin(x*(2*np.pi/12)))
        df3['competition_open_since_month_cos'] = df3['competition_open_since_month'].apply(lambda x: np.cos(x*(2*np.pi/12)))
        
        cols_selected_boruta_adapted = ['store', 'customers', 'is_weekday', 'store_type', 'assortment','competition_distance',
                'competition_open_since_month_sin', 'competition_open_since_month_cos', 'competition_open_since_year', 'competition_since_month',
                'promo', 'promo2', 'promo2_time_week', 'promo2_time_month','month_sin','month_cos', 'day_sin', 'day_cos','week_of_year_sin', 'week_of_year_cos', 'day_of_week_sin', 'day_of_week_cos', 'promo2_since_week_sin', 'promo2_since_week_cos']

        df3['promo'] = df3['promo'].astype(int)

        return df3[cols_selected_boruta_adapted]
    
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict( test_data)
        
        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient = 'records',date_format = 'iso') #return dataset to API in json format