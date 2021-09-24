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

def weight(feature1,feature2, header):
    #################################
    # Creating weight for each value in feature 1 and then assign 
    # them acording to feature 2. We need to know the header of the data
    # Expected value:
    #      -A dataframe with a feature
    #      -A dataframe with another feature
    #      -The name of each of the features
    #     
    # Return:
    #      -A dataframe with weights
    #
    # Example:  weight([1,2,3,4,5],[a,b,a,c,b],['num','code'])
    #                for a: 1+3=4   1/4, 3/4
    #                for b: 2+5=7   2/7, 5/7
    #                for c: 4       4/4=1
    # Return: [1/4,2/7,3/4,1,5/7]
    #################################
    import pandas as pd
    data=pd.DataFrame(list(zip(feature1,feature2)), columns=header)
    group=data[[header[0],header[1]]].groupby(header[0]).sum().reset_index()
    
    group['perc']=group[header[1]]/len(data)*100
    weight=[]
    for i in data[header[0]]:
        if not group[group[header[0]]==i]['perc'].values>1:
            weight.append(group[group[header[0]]==i]['perc'].values[-1])
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
    
def linear_regression(X,y):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,shuffle=True,random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f'Train R^2:\t{r2_train}\n\
    Test R^2:\t{r2_test}')
    
def polynomial_regression(X,y):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,shuffle=True,random_state=0)
    polynomial = PolynomialFeatures(degree=2)

    Xpoly_train = polynomial.fit_transform(X_train)
    Xpoly_test = polynomial.transform(X_test)

    print(f'Number of polynomial features: {Xpoly_train.shape[1]}')

    model.fit(Xpoly_train, y_train)
    ypoly_train_pred = polynomial.predict(Xpoly_train)
    ypoly_test_pred = polynomial.predict(Xpoly_test)

    r2poly_train = r2_score(y_train, ypoly_train_pred)
    r2poly_test = r2_score(y_test, ypoly_test_pred)
    print(f'Train R^2:\t{r2poly_train}\nTest R^2:\t{r2poly_test}')