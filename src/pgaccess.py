import json
import numpy as np
import pandas as pd
import psycopg2 as pg

# I did verify that python is smart enough to only
# run the file once, that being when the module name is
# encountered for the first time, then reuse the same
# variables etc that the module defines. So that answers my
# concern about the possibility of creating a new connection
# for every individual import statement

# Load database config
with open('../config.json') as f:
    cfg = json.loads(f.read())
pg_cfg = cfg['postgres']

# Initialize the connection
_conn = pg.connect(**pg_cfg)
# Don't initialize a cursor here, those should probably be
# function-specific. You know, avoid clashes and all that.

def execute_query(sql: str):
    '''
    Runs a single query against the database and returns the result
    as a pandas dataframe.
    
        Parameters:
            sql (str): The query to run
            
        Returns:
            result (pd.DataFrame): The query results
    '''
    # Without explicitly looking it up, I imagine the cursor
    # is meant more for individual queries, and that initializing
    # it each time draws from a pool of connections maintained by
    # the conn object. Then the cursor can be closed once the query
    # is finished and that connection will be returned to the pool
    # My memory is foggy but I believe that's the approximate
    # workflow for the pg.js library, I would hope that this is the same
    # It can always be changed if we discover that's incorrect
    with _conn.cursor() as cur:
        cur.execute(sql)
        data = cur.fetchall()
        columns = [c.name for c in cur.description]
        df = pd.DataFrame(data, columns=columns)
        return df

def get_test_data(data_length):
    '''
    Gets random rows from the flights database, only including those columns
    available in the test_flights table, along with the target column.
    
        Parameters:
            data_length (int): How many rows to return.
                The number of rows returned won't be exact, due to rounding
                errors and the random selection just isn't that accurate in
                order to remain fast. The amount should be close enough anyways.
            
        Returns:
            result (pd.DataFrame)
    '''
    ROWCOUNT = 15927485 # At latest count
    with _conn.cursor() as cur:
        selectionFrac = data_length / ROWCOUNT
        columns = [
            'fl_date', 'mkt_unique_carrier', 'branded_code_share', 'mkt_carrier',
            'mkt_carrier_fl_num', 'op_unique_carrier', 'tail_num', 'op_carrier_fl_num',
            'origin_airport_id', 'origin', 'origin_city_name', 'dest_airport_id',
            'dest', 'dest_city_name', 'crs_dep_time', 'crs_arr_time', 'dup',
            'crs_elapsed_time', 'flights', 'distance', 'arr_delay'
        ]
        sql = f'''
        SELECT {', '.join(columns)} AS target
        FROM flights
        TABLESAMPLE BERNOULLI ({selectionFrac * 100})
        WHERE arr_delay IS NOT NULL
        '''
        cur.execute(sql)
        data = cur.fetchall()
        columns = [c.name for c in cur.description]
        df = pd.DataFrame(data, columns=columns)
        # Convert flight date into an actual date column
        df['fl_date'] = pd.to_datetime(df['fl_date'])
        return df
    
# Commonly executed queries
def get_flight_delays():
    '''
    Get all flight delays as an array. Order is determined by
    the database.
    
        Returns:
            delays (np.array): Flight delays
    '''
    with _conn.cursor() as cur:
        cur.execute("""
        SELECT arr_delay FROM flights WHERE arr_delay IS NOT NULL
        """)
        data = cur.fetchall()
        delays = np.array(data).flatten()
        return delays