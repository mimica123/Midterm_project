def get_time(date):
    #################################
    # From a dataframe column extract month and year and return two list
    # Expected value:
    #      -A dataframe with dates
    # Return:
    #      -A dataframe with years
    #      -A dataframe with months
    #
    # Example:  get_time(1998-09-18)
    #
    # Return: 1998, 09
    #################################
    from datetime import datetime
    import pandas as pd
    import numpy as np
    date = pd.to_datetime(date,format='%Y-%m-%d')
    year = pd.DatetimeIndex(date).year
    month = pd.DatetimeIndex(date).month
    return pd.DataFrame(year), pd.DataFrame(month)


def get_hour(time):
    #################################
    # From a column that contain hour and minutes together in the
    # the same row in the format %H%M
    # Expected value:
    #      -A dataframe with time
    #     
    # Return:
    #      -A dataframe with hour
    #      -A dataframe with minutes
    #
    # Example:  get_hour(1708)
    #
    # Return: 17, 08
    #################################
    import math
    minutes=time%100
    trunc = lambda x: math.trunc(x / 100);
    hours=time.apply(trunc)
    return hours, minutes

def categorize(data):
    #################################
    # Changing a feature to a catagory
    # Expected value:
    #      -A dataframe 
    #     
    # Return:
    #      -A dataframe categorized
    #
    # Example:  categorize([10,12,5,1,7]) in 3 categories
    #
    # Return: [2,2,1,0,1]
    #################################
    data = data.astype('category')
    data = data.cat.codes
    return data

def weight(data,feature1, feature2, header):
    #################################
    # Creating weight for each value in feature 1 and then assign 
    # them acording to feature 2. We need to know the header of the data
    # Expected value:
    #      -A dataframe with the values to compare
    #      -A dataframe with a feature same as the one before
    #      -the same dataframe with another feature
    #      -The name of each of the features
    #     
    # Return:
    #      -A dataframe with weights
    #
    # Example:  weight(data[a,b,a,c,b],feature1[a,b,c],feature2[1,2,3,4,5],['code','num'])
    #                for a: 1+3=4   1/4, 3/4
    #                for b: 2+5=7   2/7, 5/7
    #                for c: 4       4/4=1
    # Return: [1/4,2/7,3/4,1,5/7]
    #################################
    import pandas as pd
    feature=pd.DataFrame(list(zip(feature1,feature2)), columns=header)
    data=pd.DataFrame(data)
    weight=[]
    for i in data[header[0]]:
        if not feature[feature[header[0]]==i][header[1]].values>1:
            if len(feature[feature[header[0]]==i][header[1]].values)>0:            
                weight.append(feature[feature[header[0]]==i][header[1]].values[-1])
            else:
                weight.append(0)
        else:
            weight.append(0)
    return weight

def ols(y,X):
    #################################
    # Running ordinary linear square with our features
    # Expected value:
    #      -A dataframe with the independent features
    #      -A dataframe with the target feature
    #
    #     
    # Return:
    #      -Print the result from OLS
    #
    # Example:  weight(X,y)
    #
    # Return: Sumary, parameters and R2
    #################################
    import statsmodels.api as sm
    model = sm.OLS(y,X)
    results = model.fit()
    print(results.summary())
    print('Parameters: ', results.params)
    print('R2: ', results.rsquared)
    
def linear_regression(X):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import pandas as pd
    #importing data for training
    df=pd.read_csv('C:/Users/bd/lighthouse-data-notes/Week_5/Day_5/mid-term-project-I/fligths_all_f.csv')
    features=df[['op_unique_carrier','mkt_carrier_fl_num','origin_airport_id','dest_airport_id','crs_dep_time','crs_arr_time','crs_elapsed_time','distance','year','month','minutes','hours','w_delay_origin','w_delay_destination','w_delay_distance','w_delay_month','w_delay_hour','w_delay_departure','arr_delay']]
    y_train=features['arr_delay']
    X_train=features.drop(columns=['arr_delay'])
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_test=X[['op_unique_carrier','mkt_carrier_fl_num','origin_airport_id','dest_airport_id','crs_dep_time','crs_arr_time','crs_elapsed_time','distance','year','month','minutes','hours','w_delay_origin','w_delay_destination','w_delay_distance','w_delay_month','w_delay_hour','w_delay_departure']]
    X['prediction'] = model.predict(X_test)

    return X[['fl_date','mkt_carrier','mkt_carrier_fl_num','origin','dest','prediction']]
    
    
def polynomial_regression(X):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    #importing data for training
    df=pd.read_csv('C:/Users/bd/lighthouse-data-notes/Week_5/Day_5/mid-term-project-I/fligths_all_f.csv')
    features=df[['op_unique_carrier','mkt_carrier_fl_num','origin_airport_id','dest_airport_id','crs_dep_time','crs_arr_time','crs_elapsed_time','distance','year','month','minutes','hours','w_delay_origin','w_delay_destination','w_delay_distance','w_delay_month','w_delay_hour','w_delay_departure','arr_delay']]
    y_train=features['arr_delay']
    X_train=features.drop(columns=['arr_delay'])
    polynomial = PolynomialFeatures(degree=2)

    Xpoly_train = polynomial.fit_transform(X_train)
    polynomial.fit(Xpoly_train, y_train)
  
    ypoly_test_pred = polynomial.predict(X)

    return ypoly_test_pred