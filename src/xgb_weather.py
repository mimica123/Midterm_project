import pandas as pd
from pickler import read_model

# Load CSVs
weather = pd.read_csv('../data/delta_weather.csv', parse_dates=['Date time'])
weather['Name'] = weather['Name'].str[:-len(", United States")]

carrierDelay = pd.read_csv('../data/monthly_carrier_delay.csv')
carrierDelay.set_index(['op_unique_carrier', 'month'], inplace=True)

planeSpeed = pd.read_csv('../data/plane_speed.csv')
planeSpeed.set_index(['tail_num'], inplace=True)

ratios = pd.read_csv('../data/airport_delays.csv', index_col=0)
dists = pd.read_csv('../data/distance_delays.csv')
months = pd.read_csv('../data/month_delays.csv')
hours = pd.read_csv('../data/hour_delays.csv')

def _prep_data(flightData):
    '''
    Converts data from the format found in the database to
    the columns this model focuses on.
    
    Also returns the supporting columns as found in sample_submission.csv,
    because columns aren't guaranteed to be in the same order as flightData
    '''
    # Set up the non-weather delay stats
    modelData = flightData.copy()
    modelData['month'] = modelData.fl_date.dt.month
    modelData['carrier_delay'] = pd.merge(
        modelData, carrierDelay,
        left_on = ['op_unique_carrier', 'month'],
        right_index = True
    )['mean_delay']
    modelData['plane_speed'] = pd.merge(
        modelData, planeSpeed,
        left_on = 'tail_num',
        right_index = True
    )['speed']

    modelData = pd.merge(
        pd.merge(modelData, ratios[['origin_ratio']], left_on='origin_airport_id', right_index=True),
        ratios[['dest_ratio']],
        left_on = 'dest_airport_id',
        right_index = True
    )
    modelData = pd.merge(
        pd.merge(modelData, dists),
        months
    )

    modelData['hour'] = modelData['crs_arr_time'] // 100
    modelData = pd.merge(modelData, hours)
    
    # Merge weather columns
    interestWeather = weather[['Name', 'Date time', 'Temperature', 'Precipitation', 'Snow', 'Wind Speed', 'Visibility']]
    originMerged = pd.merge(
        modelData,
        interestWeather,
        how = 'left',
        left_on = ['origin_city_name', 'fl_date'],
        right_on = ['Name', 'Date time']
    )

    weatherMerged = pd.merge(
        originMerged,
        interestWeather,
        how = 'left',
        left_on = ['dest_city_name', 'fl_date'],
        right_on = ['Name', 'Date time'],
        suffixes = ('_origin', '_dest')
    )
    
    # Limit only to the columns for the final model
    keepcols = [
        'crs_arr_time', 'distance', 'plane_speed', 'carrier_delay',
        'origin_ratio', 'dest_ratio', 'distance_ratio', 'month_ratio',
        'hour_ratio', 'Temperature_origin', 'Precipitation_origin',
        'Snow_origin', 'Wind Speed_origin', 'Visibility_origin',
        'Temperature_dest', 'Precipitation_dest', 'Snow_dest',
        'Wind Speed_dest', 'Visibility_dest',
    ]
    modelData = weatherMerged[keepcols].copy()
    supportData = weatherMerged[
        ['fl_date', 'mkt_carrier', 'mkt_carrier_fl_num', 'origin', 'dest']
    ].copy()
    
    
    # Fill null values
    temp_avg = modelData[['Temperature_origin', 'Temperature_dest']].median().mean()
    precip_avg = modelData[['Precipitation_origin', 'Precipitation_dest']].median().mean()
    snow_avg = modelData[['Snow_origin', 'Snow_dest']].median().mean()
    wind_avg = modelData[['Wind Speed_origin', 'Wind Speed_dest']].median().mean()
    vis_avg = modelData[['Visibility_origin', 'Visibility_dest']].median().mean()

    modelData[['Temperature_origin', 'Temperature_dest']] = \
        modelData[['Temperature_origin', 'Temperature_dest']].fillna(temp_avg)
    modelData[['Precipitation_origin', 'Precipitation_dest']] = \
        modelData[['Precipitation_origin', 'Precipitation_dest']].fillna(precip_avg)
    modelData[['Snow_origin', 'Snow_dest']] = \
        modelData[['Snow_origin', 'Snow_dest']].fillna(snow_avg)
    modelData[['Wind Speed_origin', 'Wind Speed_dest']] = \
        modelData[['Wind Speed_origin', 'Wind Speed_dest']].fillna(wind_avg)
    modelData[['Visibility_origin', 'Visibility_dest']] = \
        modelData[['Visibility_origin', 'Visibility_dest']].fillna(vis_avg)
    
    return modelData, supportData

def get_predictions(flightData):
    '''
    Given raw database data, gets the predicted delay.
    
        Parameters:
            flightData (pd.DataFrame):
                This dataframe should contain all the columns from the test database
                
        Returns: pd.DataFrame
            The predictions for each row, matching the format found in
            sample_submission.csv in the assignment repository
    '''
    model = read_model('../models/xgb-weather')
    modelData, supportData = _prep_data(flightData)
    pred = model.predict(modelData)
    supportData['predicted_delay'] = pred
    return supportData
    