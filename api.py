# For Data Analysis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

# For Time Series Forecasting
# import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
# from statsmodels.stats.stattools import durbin_watson
# from statsmodels.tools.eval_measures import rmse, aic

# For Rest Services
import requests
from flask import Flask, make_response, jsonify, abort, request, Response, render_template, g
from flask_cors import CORS
import time

app = Flask(__name__, template_folder = "template")
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/')
def home():
    return render_template("index.html")

@app.before_request
def before_request():
    g.start = time.time()

@app.after_request
def after_request(response):
    diff = time.time() - g.start
    if ((response.response) and
        (200 <= response.status_code < 300) and
        (response.content_type.startswith('text/html'))):
        response.set_data(response.get_data().replace(
            b'__EXECUTION_TIME__', bytes(str(diff), 'utf-8')))
    return response

# =============================================================================
# @app.route('/country_line_graph', methods=["POST"])
# def line_plot_data_for_country():
#     if not request.json:
#         abort(400)
#     req = request.json
#     country_names = req["country_names"].strip().split(",")
#     ycol = req["ycol"]
#     
#     data = global_deaths_df if ycol == "Fatalities" else global_confirmed_cases_df
#     date_cols = [col for col in data.columns if col not in drop_cols]
#     
#     if len(country_names) == 0:
#         abort(Response("Please select a country"))
#     elif ycol == "":
#         abort(Response("Please select a type"))
#     
#     country_dict = {}
#     for country_name in country_names:
#         info_dict = {}
#         
#         print(f"country_name - {country_name}")
#         country_data = data[data['Country/Region'] == country_name]
#         country_lat = country_data['Lat'].values[0]
#         country_long = country_data['Long'].values[0]
#         #train_filt = train_filt.groupby(['Date'], as_index = False).agg({ycol:"sum"})
#         
#         # country_data_to_repeat = country_data[drop_cols].reset_index().drop(['index'],axis= 1)
#         # repeat_indices = np.repeat(np.arange(len(country_data_to_repeat)),len(date_cols))
#         # country_data_to_repeat = country_data_to_repeat[repeat_indices]
#         
#         country_data = country_data.drop(drop_cols, axis =1)
#         
#         na_fill_item = country_data['Country/Region'].unique()[0]
#         country_data["Province/State"] = country_data["Province/State"].fillna(na_fill_item)
#         country_data_tr = country_data.set_index("Province/State").T
#         country_data_tr.columns = ["Date", "count"]
#         country_data_tr["Date"] = pd.to_datetime(country_data_tr["Date"])
#         print(country_data_tr.head())
#         
#         info_dict['Date'] = country_data_tr.Date.astype(str).tolist()
#         info_dict[ycol] = country_data_tr['count'].tolist()
#         country_dict[country_name] = info_dict
#         
#     return jsonify({"result":country_dict})
# =============================================================================


#################################################
########## GLOBAL APIS ##########################
#################################################
    
@app.route('/global_overall_count', methods=["GET"])
def get_global_overall_count():
    
    # max_date = str(df_world.Date.values[-1])[:10]
    # df_world_ind = df_world.set_index('Date').agg(max)
    
    # max_date = df_world.Date.max()
    # df_world_overall = df_world[df_world.Date == max_date].drop('Date', axis = 1)
    # index = df_world_overall.index.values[0]
    # res_dict = df_world_overall.loc[index].to_dict()
    
    world_idx = len(df_world) - 1
    df_world_overall = df_world.loc[world_idx]
    world_dict = {
            'deaths': int(df_world_overall['deaths']),
            'confirmed': int(df_world_overall['confirmed']),
            'recovered': int(df_world_overall['recovered']),
            'deaths_incr': int(df_world_overall['deaths_incr']),
            'confirmed_incr': int(df_world_overall['confirmed_incr']),
            'recovered_incr': int(df_world_overall['recovered_incr']),
            'deaths_incr_rate': df_world_overall['deaths_incr_rate'],
            'confirmed_incr_rate': df_world_overall['confirmed_incr_rate'],
            'recovered_incr_rate': df_world_overall['recovered_incr_rate']
        }
    
    return jsonify({"count":1,"date":str(df_world_overall['Date']),"result":world_dict})


# =============================================================================
# @app.route('/global_datewise_count', methods=["GET"])
# def get_global_datewise_count():
#     
#     df_world['Date'] = df_world.Date.astype(str)
#     df_world_ind = df_world.set_index('Date')
#     df_world_ind_tr = df_world_ind.T
#     res_dict = df_world_ind_tr.to_dict()
#     
#     return jsonify({"count":len(df_world_ind),"result":res_dict})
# =============================================================================

@app.route('/global_datewise_count', methods=["POST"])
def get_global_datewise_count():
    
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    start_date = req["start_date"]
    end_date = req["end_date"]
    
    # DEFAULT ALL DATA BEING TAKEN
    start_date = pd.to_datetime("2020-01-22") if start_date == "" else pd.to_datetime(start_date)
    end_date = pd.to_datetime(pd.datetime.now().date()) if end_date == "" else pd.to_datetime(end_date)
    
    # SELECTING BETWEEN THE START AND END DATES
    df_world_range = df_world.set_index('Date').loc[start_date:end_date].reset_index()
    df_world_range['Date'] = df_world_range.Date.astype(str)
    
    # FORMING RESPONSE
    df_world_ind = df_world_range.set_index('Date')
    df_world_ind_tr = df_world_ind.T
    res_dict = df_world_ind_tr.to_dict()
    
    return jsonify({"count":len(df_world_ind),"result":res_dict})

@app.route('/global_countrywise_count', methods=["GET"])
def get_global_countrywise_count():
    
    country_dict_list = []
    for i in range(len(df_country)):
        country_dict = {
            'Country': df_country.loc[i,'Country'],
            'Lat': float(df_country.loc[i,'Lat']),
            'Long': float(df_country.loc[i,'Long']),
            'deaths': int(df_country.loc[i,'deaths']),
            'confirmed': int(df_country.loc[i,'confirmed']),
            'recovered': int(df_country.loc[i,'recovered']),
            'deaths_incr': int(df_country.loc[i,'deaths_incr']),
            'confirmed_incr': int(df_country.loc[i,'confirmed_incr']),
            'recovered_incr': int(df_country.loc[i,'recovered_incr']),
            'deaths_incr_rate': df_country.loc[i,'deaths_incr_rate'],
            'confirmed_incr_rate': df_country.loc[i,'confirmed_incr_rate'],
            'recovered_incr_rate': df_country.loc[i,'recovered_incr_rate']
        }
        country_dict_list.append(country_dict)
        
    res_list = country_dict_list
    country_count = len(dfd_grp)
    max_date = str(df_world.Date.values[-1])[:10]
    
    return jsonify({"count":country_count,"date":max_date,"result":res_list})

@app.route('/single_country_overall_data', methods=["POST"])
def get_single_country_overall_data():
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    country_name = req["country_name"]
    
    # CHECKING
    if country_name == "":
        abort(Response("Please select a country"))
    
    df_country_filt = df_country[df_country["Country"].str.lower() == country_name]
    idx = df_country_filt.index.values[0]
    df_country_filt_arr = df_country_filt.loc[idx]
    country_dict = {
            'Country': df_country_filt_arr['Country'],
            'deaths': int(df_country_filt_arr['deaths']),
            'confirmed': int(df_country_filt_arr['confirmed']),
            'recovered': int(df_country_filt_arr['recovered']),
            'deaths_incr': int(df_country_filt_arr['deaths_incr']),
            'confirmed_incr': int(df_country_filt_arr['confirmed_incr']),
            'recovered_incr': int(df_country_filt_arr['recovered_incr']),
            'deaths_incr_rate': df_country_filt_arr['deaths_incr_rate'],
            'confirmed_incr_rate': df_country_filt_arr['confirmed_incr_rate'],
            'recovered_incr_rate': df_country_filt_arr['recovered_incr_rate']
        }
    max_date = str(df_world.Date.values[-1])[:10]
    
    return jsonify({"count":1,"date":max_date,"result":country_dict})
    
# =============================================================================
# @app.route('/single_country_data', methods=["POST"])
# def get_single_country_data():
#     if not request.json:
#         abort(400)
#     req = request.json
#     
#     # EXTRACTING REQUEST JSON DATA
#     country_name = req["country_name"]
#     
#     # CHECKING
#     if country_name == "":
#         abort(Response("Please select a country"))
#  
#     dfd_grp.index = dfd_grp.index.str.lower()
#     dfc_grp.index = dfc_grp.index.str.lower()
#     dfr_grp.index = dfr_grp.index.str.lower()
#     
#     dfd_grp_filt = dfd_grp.loc[country_name]
#     dfc_grp_filt = dfc_grp.loc[country_name]
#     dfr_grp_filt = dfr_grp.loc[country_name]
#     
#     # DAILY INCREASE
#     dfd_grp_filt_diff = dfd_grp_filt.diff(periods = 1)
#     dfc_grp_filt_diff = dfc_grp_filt.diff(periods = 1) 
#     dfr_grp_filt_diff = dfr_grp_filt.diff(periods = 1) 
#     
#     # CONCATENATING
#     country_df = pd.concat([dfd_grp_filt, dfc_grp_filt, dfr_grp_filt, dfd_grp_filt_diff, dfc_grp_filt_diff, dfr_grp_filt_diff], axis = 1).reset_index()
#     country_df.columns = ["Date", "deaths", "confirmed", "recovered","deaths_incr","confirmed_incr","recovered_incr"]
#     
#     country_df["deaths_incr_rate"] = np.round(100 * country_df["deaths_incr"] / (country_df["deaths"] - country_df["deaths_incr"]))
#     country_df["confirmed_incr_rate"] = np.round(100 * country_df["confirmed_incr"] / (country_df["confirmed"] - country_df["confirmed_incr"]))
#     country_df["recovered_incr_rate"] = np.round(100 * country_df["recovered_incr"] / (country_df["recovered"] - country_df["recovered_incr"]))
#    
#     country_df.fillna(0, inplace = True)
#     
#     country_df["deaths_incr_rate"] = country_df["deaths_incr_rate"].astype(str) + '%'
#     country_df["confirmed_incr_rate"] = country_df["confirmed_incr_rate"].astype(str) + '%'
#     country_df["recovered_incr_rate"] = country_df["recovered_incr_rate"].astype(str) + '%'
#     
#     country_dict_list = []
#     for i in range(len(country_df)):
#         country_dict = {
#             'Date': country_df.loc[i,'Date'],
#             'deaths': int(country_df.loc[i,'deaths']),
#             'confirmed': int(country_df.loc[i,'confirmed']),
#             'recovered': int(country_df.loc[i,'recovered']),
#             'deaths_incr': int(country_df.loc[i,'deaths_incr']),
#             'confirmed_incr': int(country_df.loc[i,'confirmed_incr']),
#             'recovered_incr': int(country_df.loc[i,'recovered_incr']),
#             'deaths_incr_rate': country_df.loc[i,'deaths_incr_rate'],
#             'confirmed_incr_rate': country_df.loc[i,'confirmed_incr_rate'],
#             'recovered_incr_rate': country_df.loc[i,'recovered_incr_rate']
#         }
#         country_dict_list.append(country_dict)
#     
#     return jsonify({"count":len(country_df),"Country":country_name,"result":country_dict_list})
# =============================================================================
    
@app.route('/single_country_data', methods=["POST"])
def get_single_country_data():
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    country_name = req["country_name"].lower()
    start_date = req["start_date"]
    end_date = req["end_date"]
    # start_date = req["start_date"] if "start_date" in request.form else ""
    # end_date = req["end_date"] if "end_date" in request.form else ""
    
    # CHECKING
    if country_name == "":
        abort(Response("Please select a country"))
    
    # DEFAULT ALL DATA BEING TAKEN
    start_date = pd.to_datetime("2020-01-22") if start_date == "" else pd.to_datetime(start_date)
    end_date = pd.to_datetime(pd.datetime.now().date()) if end_date == "" else pd.to_datetime(end_date)
        
    dfd_grp.index = dfd_grp.index.str.lower()
    dfc_grp.index = dfc_grp.index.str.lower()
    dfr_grp.index = dfr_grp.index.str.lower()
    
    dfd_grp_filt = dfd_grp.loc[country_name]
    dfc_grp_filt = dfc_grp.loc[country_name]
    dfr_grp_filt = dfr_grp.loc[country_name]
    
    # DAILY INCREASE
    dfd_grp_filt_diff = dfd_grp_filt.diff(periods = 1)
    dfc_grp_filt_diff = dfc_grp_filt.diff(periods = 1) 
    dfr_grp_filt_diff = dfr_grp_filt.diff(periods = 1) 
    
    # CONCATENATING
    country_df = pd.concat([dfd_grp_filt, dfc_grp_filt, dfr_grp_filt, dfd_grp_filt_diff, dfc_grp_filt_diff, dfr_grp_filt_diff], axis = 1).reset_index()
    country_df.columns = ["Date", "deaths", "confirmed", "recovered","deaths_incr","confirmed_incr","recovered_incr"]
    
    country_df["deaths_incr_rate"] = np.round(100 * country_df["deaths_incr"] / (country_df["deaths"] - country_df["deaths_incr"]))
    country_df["confirmed_incr_rate"] = np.round(100 * country_df["confirmed_incr"] / (country_df["confirmed"] - country_df["confirmed_incr"]))
    country_df["recovered_incr_rate"] = np.round(100 * country_df["recovered_incr"] / (country_df["recovered"] - country_df["recovered_incr"]))
   
    country_df.fillna(0, inplace = True)
    
    country_df["deaths_incr_rate"] = country_df["deaths_incr_rate"].astype(str) + '%'
    country_df["confirmed_incr_rate"] = country_df["confirmed_incr_rate"].astype(str) + '%'
    country_df["recovered_incr_rate"] = country_df["recovered_incr_rate"].astype(str) + '%'
    
    country_df["Date"] = pd.to_datetime(country_df["Date"])
    
    # SELECTING BETWEEN THE START AND END DATES
    country_df = country_df.set_index('Date').loc[start_date:end_date].reset_index()
    country_df['Date'] = country_df.Date.astype(str)
    
    country_dict_list = []
    for i in range(len(country_df)):
        country_dict = {
            'Date': country_df.loc[i,'Date'],
            'deaths': int(country_df.loc[i,'deaths']),
            'confirmed': int(country_df.loc[i,'confirmed']),
            'recovered': int(country_df.loc[i,'recovered']),
            'deaths_incr': int(country_df.loc[i,'deaths_incr']),
            'confirmed_incr': int(country_df.loc[i,'confirmed_incr']),
            'recovered_incr': int(country_df.loc[i,'recovered_incr']),
            'deaths_incr_rate': country_df.loc[i,'deaths_incr_rate'],
            'confirmed_incr_rate': country_df.loc[i,'confirmed_incr_rate'],
            'recovered_incr_rate': country_df.loc[i,'recovered_incr_rate']
        }
        country_dict_list.append(country_dict)
    
    return jsonify({"count":len(country_df),"Country":country_name.upper(),"result":country_dict_list})

@app.route('/single_country_forecast_data', methods=["POST"])
def get_single_country_forecast_data():
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    country_name = req["country_name"].lower()      
    num_days = req["num_days"]               
    
    # DEFAULT ALL DATA BEING TAKEN
    country_name = "world" if country_name == "" else country_name # By Default taking World Data
    num_days = 10 if num_days == "" else int(num_days)             # By Default taking 10 days
    test_days = 0
    
    # FILTERING FOR THE GIVEN COUNTRY
    master_data_filt = master_data[master_data.country.str.lower() == country_name]
    
    # TRAIN TEST SPLIT
    train = master_data_filt.iloc[:,:3]
    # test = master_data_filt.iloc[-test_days:,:3] # Taking last 10 days as testing data
    
    # START AND END DATES OF FORECASTING
    start_date = train.index[-1] + datetime.timedelta(1)
    end_date = start_date + datetime.timedelta(test_days+num_days-1)      # forecasting for test data + num_days in future
    
    # LOG TRANSFORMATION
    train_log = np.log1p(train)
    
    # MAKING STATIONARY BY DIFFERENCING
    train_diff1 = train_log.diff(1).dropna()
    train_diff2 = train_diff1.diff(1).dropna()
    
    # STATIONARITY CHECK
    # adfuller_test(train_diff2.deaths)   
    # adfuller_test(train_diff2.confirmed)
    # adfuller_test(train_diff2.recovered)
    
    # MODELING
    model = VAR(train_diff2)  # VAR model
    
    # SELECTING ORDER p FOR VAR MODEL
    # x = model.select_order(maxlags=12)
    # x.summary()
    
    # FITTING THE MODEL
    res = model.fit(4)                                 # ORDER = 4 FOR MINIMUM AIC
        
    # DW TEST FOR SERIAL CORRELATION
    # durbin_watson(res.resid)                         # should be around 2 for zero correlation
    
    # INITIAL INPUT FOR FORECAST
    lag_order = res.k_ar                               # 4 as shosen above
    forecast_input = train_diff2.values[-lag_order:]   # Taking last 4 training data as initial input for forecasting
    
    # FORECASTING
    num_days_forecast = test_days+num_days                    # forecasting for test data + num_days in future
    pred = res.forecast(forecast_input, num_days_forecast)  # numpy array of forecast values
    idx = pd.date_range(start=start_date, end=end_date) # date range for forecast data
    pred_df = pd.DataFrame(pred, index=idx, columns=train_diff2.columns + '_2d') # converintg to dataframe
    
    # INVERSE TRANSFORMATION
    pred_df_inv1 = invert_transformation(pred_df,train_log, train_diff1, train_diff2, second_diff=True, third_diff=False)      # inverting 2nd diff to 1st diff
    pred_df_inv = invert_transformation(pred_df_inv1,train_log, train_diff1, train_diff2, second_diff=False, third_diff=False) # inverting 1st diff and taking inverse log tr
    
    # SELECTING REQUIRED COLUMNS
    df_forecast = pred_df_inv.iloc[:,-3:]
    df_forecast.columns = train_diff2.columns
    
    # EVALUATING PREDICTIONS
    # test_f = test.join(df_forecast, rsuffix='_f')
    # mape(test_f.deaths_f, test_f.deaths)
    # mape(test_f.confirmed_f, test_f.confirmed)
    # mape(test_f.recovered_f, test_f.recovered)
    
    # COMBINING TRAIN WITH FORECAST DATA
    new_cases_forecast = pd.concat([train, df_forecast], axis = 0)
    total_cases_forecast = new_cases_forecast.cumsum()
    
    # COMBINING 
    final_forecast = pd.concat([total_cases_forecast,new_cases_forecast], axis = 1).reset_index()
    final_forecast.columns = ['Date'] + new_cases_forecast.columns.tolist() + [col + '_incr' for col in new_cases_forecast.columns]
    
    
    # RESPONSE JSON
    # return jsonify({"count":len(final_forecast),"Country":country_name.upper(),"result":final_forecast.T.to_dict()})
    final_forecast["deaths_incr_rate"] = np.round(100 * final_forecast["deaths_incr"] / (final_forecast["deaths"] - final_forecast["deaths_incr"]))
    final_forecast["confirmed_incr_rate"] = np.round(100 * final_forecast["confirmed_incr"] / (final_forecast["confirmed"] - final_forecast["confirmed_incr"]))
    final_forecast["recovered_incr_rate"] = np.round(100 * final_forecast["recovered_incr"] / (final_forecast["recovered"] - final_forecast["recovered_incr"]))
   
    final_forecast.fillna(0, inplace = True)
    
    final_forecast["deaths_incr_rate"] = final_forecast["deaths_incr_rate"].astype(str) + '%'
    final_forecast["confirmed_incr_rate"] = final_forecast["confirmed_incr_rate"].astype(str) + '%'
    final_forecast["recovered_incr_rate"] = final_forecast["recovered_incr_rate"].astype(str) + '%'
    
    # SELECTING BETWEEN THE START AND END DATES
    final_forecast['Date'] = final_forecast.Date.astype(str)
    # final_forecast.set_index('Date',inplace = True)
    
    country_dict_list = []
    for i in range(len(final_forecast)):
        country_dict = {
            'Date': final_forecast.loc[i,'Date'],
            'deaths': int(final_forecast.loc[i,'deaths']),
            'confirmed': int(final_forecast.loc[i,'confirmed']),
            'recovered': int(final_forecast.loc[i,'recovered']),
            'deaths_incr': int(final_forecast.loc[i,'deaths_incr']),
            'confirmed_incr': int(final_forecast.loc[i,'confirmed_incr']),
            'recovered_incr': int(final_forecast.loc[i,'recovered_incr']),
            'deaths_incr_rate': final_forecast.loc[i,'deaths_incr_rate'],
            'confirmed_incr_rate': final_forecast.loc[i,'confirmed_incr_rate'],
            'recovered_incr_rate': final_forecast.loc[i,'recovered_incr_rate']
        }
        country_dict_list.append(country_dict)
    
    return jsonify({
        "count":len(final_forecast),
        "Country":country_name.upper(),
        "forecast_start_date":str(start_date)[:10],
        "forecast_end_date":str(end_date)[:10],
        "result":country_dict_list})
    
    
        
#################################################
########## INDIA APIS ###########################
#################################################

@app.route('/india_all_states_overall_data', methods=["GET"])
def get_india_all_states_overall_data():
    
    state_dict_list = []
    for i in range(len(df_states_overall)):
        state_dict = {
            'State': df_states_overall.loc[i,'state'].upper(),
            'State Code': df_states_overall.loc[i,'state_code'].upper(),
            'Lat': float(df_states_overall.loc[i,'latitude']),
            'Long': float(df_states_overall.loc[i,'longitude']),
            'deaths': int(df_states_overall.loc[i,'deaths']),
            'confirmed': int(df_states_overall.loc[i,'confirmed']),
            'recovered': int(df_states_overall.loc[i,'recovered']),
            'deaths_incr': int(df_states_overall.loc[i,'deaths_incr']),
            'confirmed_incr': int(df_states_overall.loc[i,'confirmed_incr']),
            'recovered_incr': int(df_states_overall.loc[i,'recovered_incr']),
            'deaths_incr_rate': df_states_overall.loc[i,'deaths_incr_rate'],
            'confirmed_incr_rate': df_states_overall.loc[i,'confirmed_incr_rate'],
            'recovered_incr_rate': df_states_overall.loc[i,'recovered_incr_rate']
        }
        state_dict_list.append(state_dict)
        
    res_list = state_dict_list
    state_count = len(df_states_overall)
    max_date = str(df_states.date.values[-1])[:10]
    
    return jsonify({"count":state_count,"date":max_date,"result":res_list})
    
@app.route('/india_single_state_overall_data', methods=["POST"])
def get_india_single_state_overall_data():
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    state_name = req["state_name"].lower()
    state_name = "total" if state_name == "" else state_name
    
    # FILTERING FOR THE GIVEN STATE - BY DEFAULT GIVING TOTAL
    state_code_df_mod = state_code_df.copy()
    state_code_df_mod['state'] = state_code_df_mod['state'].str.lower()
    state_code_df_filt = state_code_df_mod[state_code_df_mod.state == state_name]
    
    state_lat = float(state_code_df_filt.latitude.values[0])
    state_long = float(state_code_df_filt.longitude.values[0])
    state_code = "tt" if state_name == "total" else state_code_df_filt.index[0]
    
    # GETTING STATE DATA
    state_df = get_state_data(state_code).reset_index()
    idx = len(state_df) - 1
    
    # LATEST STATE DATA
    state_df = state_df.iloc[idx,:]
    # state_df['date'] = state_df.date.astype(str)
    end_date = str(state_df.date)[:10]
    
    state_dict = {
            'deaths': int(state_df['deaths']),
            'confirmed': int(state_df['confirmed']),
            'recovered': int(state_df['recovered']),
            'deaths_incr': int(state_df['deaths_incr']),
            'confirmed_incr': int(state_df['confirmed_incr']),
            'recovered_incr': int(state_df['recovered_incr']),
            'deaths_incr_rate': state_df['deaths_incr_rate'],
            'confirmed_incr_rate': state_df['confirmed_incr_rate'],
            'recovered_incr_rate': state_df['recovered_incr_rate']
        }
    
    return jsonify({
        "count":1,
        "date":end_date,
        "State": state_name.upper(),
        "State Code": state_code.upper(),
        "Lat": state_lat,
        "Long": state_long,
        "result":state_dict
        })
    

@app.route('/india_single_state_datewise_data', methods=["POST"])
def get_india_single_state_datewise_data():
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    state_name = req["state_name"].lower()
    start_date = req["start_date"]
    end_date = req["end_date"]
    
    # BY DEFAULT ALL DATA BEING TAKEN
    state_name = "total" if state_name == "" else state_name
    start_date = pd.to_datetime("2020-01-22") if start_date == "" else pd.to_datetime(start_date)
    end_date = pd.to_datetime(pd.datetime.now().date()) if end_date == "" else pd.to_datetime(end_date)
    
    # FILTERING FOR THE GIVEN STATE - BY DEFAULT GIVING TOTAL
    state_code_df_mod = state_code_df.copy()
    state_code_df_mod['state'] = state_code_df_mod['state'].str.lower()
    state_code_df_filt = state_code_df_mod[state_code_df_mod.state == state_name]
    
    state_lat = float(state_code_df_filt.latitude.values[0])
    state_long = float(state_code_df_filt.longitude.values[0])
    state_code = "tt" if state_name == "total" else state_code_df_filt.index[0]
    
    # GETTING STATE DATA
    state_df = get_state_data(state_code)
    
    # SELECTING BETWEEN THE START AND END DATES
    state_df = state_df.loc[start_date:end_date].reset_index()
    state_df['date'] = state_df.date.astype(str)
    
    state_dict_list = []
    for i in range(len(state_df)):
        state_dict = {
            'Date': state_df.loc[i,'date'],
            'deaths': int(state_df.loc[i,'deaths']),
            'confirmed': int(state_df.loc[i,'confirmed']),
            'recovered': int(state_df.loc[i,'recovered']),
            'deaths_incr': int(state_df.loc[i,'deaths_incr']),
            'confirmed_incr': int(state_df.loc[i,'confirmed_incr']),
            'recovered_incr': int(state_df.loc[i,'recovered_incr']),
            'deaths_incr_rate': state_df.loc[i,'deaths_incr_rate'],
            'confirmed_incr_rate': state_df.loc[i,'confirmed_incr_rate'],
            'recovered_incr_rate': state_df.loc[i,'recovered_incr_rate']
        }
        state_dict_list.append(state_dict)
    
    return jsonify({
        "count":len(state_df),
        "State":state_name.upper(), 
        "State Code":state_code.upper(),
        "Lat": state_lat,
        "Long": state_long,
        "result":state_dict_list
        })

@app.route('/india_single_state_district_data', methods=["POST"])
def get_india_single_state_district_data():
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    state_name = req["state_name"].lower()
    # state_name = "total" if state_name == "" else state_name
    
    # FILTERING FOR THE GIVEN STATE - BY DEFAULT GIVING TOTAL
    state_code_df_mod = state_code_df.copy()
    state_code_df_mod['state'] = state_code_df_mod['state'].str.lower()
    state_code_df_filt = state_code_df_mod[state_code_df_mod.state == state_name]
    
    state_lat = float(state_code_df_filt.latitude.values[0])
    state_long = float(state_code_df_filt.longitude.values[0])
    state_code = state_code_df_filt.index[0]
    
    district_data_list = df_dist.districtData[df_dist.statecode == state_code].values[0]
    district_data_list_new = []
    for dct in district_data_list:
        dist_dict = {}
        dist_dict['district'] = dct['district']
        dist_dict['confirmed'] = dct['confirmed']
        dist_dict['deaths'] = dct['deceased']
        dist_dict['recovered'] = dct['recovered']
        dist_dict['confirmed_incr'] = dct['delta']['confirmed']
        dist_dict['deaths_incr'] = dct['delta']['deceased']
        dist_dict['recovered_incr'] = dct['delta']['recovered']
        dist_dict['confirmed_incr_rate'] = 0.0 if dist_dict['confirmed'] - dist_dict['confirmed_incr'] == 0 else 100 * dist_dict['confirmed_incr'] / (dist_dict['confirmed'] - dist_dict['confirmed_incr'])
        dist_dict['deaths_incr_rate'] = 0.0 if dist_dict['deaths'] - dist_dict['deaths_incr'] == 0 else 100 * dist_dict['deaths_incr'] / (dist_dict['deaths'] - dist_dict['deaths_incr'])
        dist_dict['recovered_incr_rate'] = 0.0 if dist_dict['recovered'] - dist_dict['recovered_incr'] == 0 else 100 * dist_dict['recovered_incr'] / (dist_dict['recovered'] - dist_dict['recovered_incr'])
        dist_dict['confirmed_incr_rate'] = str(dist_dict['confirmed_incr_rate']) + '%'
        dist_dict['deaths_incr_rate'] = str(dist_dict['deaths_incr_rate']) + '%'
        dist_dict['recovered_incr_rate'] = str(dist_dict['recovered_incr_rate']) + '%'
        
        district_data_list_new.append(dist_dict)
        
    
    return jsonify({
        "count":len(district_data_list_new),
        "State":state_name.upper(), 
        "State Code":state_code.upper(),
        "Lat": state_lat,
        "Long": state_long,
        "result":district_data_list_new
        })
    
@app.route('/india_single_state_forecast_data', methods=["POST"])
def get_india_single_state_forecast_data():
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    state_name = req["state_name"].lower()
    test_days = 0
    num_days = int(req["num_days"])
    
    # BY DEFAULT ALL DATA BEING TAKEN
    state_name = "total" if state_name == "" else state_name
    
    # FILTERING FOR THE GIVEN STATE - BY DEFAULT GIVING TOTAL
    state_code_df_mod = state_code_df.copy()
    state_code_df_mod['state'] = state_code_df_mod['state'].str.lower()
    state_code_df_filt = state_code_df_mod[state_code_df_mod.state == state_name]
    
    state_lat = float(state_code_df_filt.latitude.values[0])
    state_long = float(state_code_df_filt.longitude.values[0])
    state_code = "tt" if state_name == "total" else state_code_df_filt.index[0]
    
    # GETTING STATE DATA
    dfsc_filt = dfsc.loc[state_code]
    dfsd_filt = dfsd.loc[state_code]
    dfsr_filt = dfsr.loc[state_code]
    
    # FINDING IMPORTANT DATA
    state_df = pd.concat([dfsc_filt, dfsd_filt, dfsr_filt], axis = 1)
    state_df = state_df.astype(int)
    state_df.columns = ['confirmed', 'deaths', 'recovered']
    
    # TRAIN TEST SPLIT
    train = state_df.iloc[:,:3]
    # test = master_data_filt.iloc[-test_days:,:3] # Taking last 10 days as testing data
    
    # START AND END DATES OF FORECASTING
    start_date = train.index[-1] + datetime.timedelta(1)
    end_date = start_date + datetime.timedelta(test_days+num_days-1)      # forecasting for test data + num_days in future
    
    # LOG TRANSFORMATION
    train_log = np.log1p(train)
    
    # MAKING STATIONARY BY DIFFERENCING
    train_diff1 = train_log.diff(1).dropna()
    train_diff2 = train_diff1.diff(1).dropna()
    
    # STATIONARITY CHECK
    # adfuller_test(train_diff2.deaths)   
    # adfuller_test(train_diff2.confirmed)
    # adfuller_test(train_diff2.recovered)
    
    # MODELING
    model = VAR(train_diff2)  # VAR model
    
    # SELECTING ORDER p FOR VAR MODEL
    # x = model.select_order(maxlags=12)
    # x.summary()
    
    # FITTING THE MODEL
    res = model.fit(4)                                 # ORDER = 4 FOR MINIMUM AIC
        
    # DW TEST FOR SERIAL CORRELATION
    # durbin_watson(res.resid)                         # should be around 2 for zero correlation
    
    # INITIAL INPUT FOR FORECAST
    lag_order = res.k_ar                               # 4 as shosen above
    forecast_input = train_diff2.values[-lag_order:]   # Taking last 4 training data as initial input for forecasting
    
    # FORECASTING
    num_days_forecast = test_days+num_days                    # forecasting for test data + num_days in future
    pred = res.forecast(forecast_input, num_days_forecast)  # numpy array of forecast values
    idx = pd.date_range(start=start_date, end=end_date) # date range for forecast data
    pred_df = pd.DataFrame(pred, index=idx, columns=train_diff2.columns + '_2d') # converintg to dataframe
    
    # INVERSE TRANSFORMATION
    pred_df_inv1 = invert_transformation(pred_df,train_log, train_diff1, train_diff2, second_diff=True, third_diff=False)      # inverting 2nd diff to 1st diff
    pred_df_inv = invert_transformation(pred_df_inv1,train_log, train_diff1, train_diff2, second_diff=False, third_diff=False) # inverting 1st diff and taking inverse log tr
    
    # SELECTING REQUIRED COLUMNS
    df_forecast = pred_df_inv.iloc[:,-3:]
    df_forecast.columns = train_diff2.columns
    
    # EVALUATING PREDICTIONS
    # test_f = test.join(df_forecast, rsuffix='_f')
    # mape(test_f.deaths_f, test_f.deaths)
    # mape(test_f.confirmed_f, test_f.confirmed)
    # mape(test_f.recovered_f, test_f.recovered)
    
    # COMBINING TRAIN WITH FORECAST DATA
    new_cases_forecast = pd.concat([train, df_forecast], axis = 0)
    total_cases_forecast = new_cases_forecast.cumsum()
    
    # COMBINING 
    final_forecast = pd.concat([total_cases_forecast,new_cases_forecast], axis = 1).reset_index()
    final_forecast.columns = ['Date'] + new_cases_forecast.columns.tolist() + [col + '_incr' for col in new_cases_forecast.columns]
    
    
    # RESPONSE JSON
    # return jsonify({"count":len(final_forecast),"Country":country_name.upper(),"result":final_forecast.T.to_dict()})
    final_forecast["deaths_incr_rate"] = np.round(100 * final_forecast["deaths_incr"] / (final_forecast["deaths"] - final_forecast["deaths_incr"]))
    final_forecast["confirmed_incr_rate"] = np.round(100 * final_forecast["confirmed_incr"] / (final_forecast["confirmed"] - final_forecast["confirmed_incr"]))
    final_forecast["recovered_incr_rate"] = np.round(100 * final_forecast["recovered_incr"] / (final_forecast["recovered"] - final_forecast["recovered_incr"]))
   
    final_forecast.fillna(0, inplace = True)
    
    final_forecast["deaths_incr_rate"] = final_forecast["deaths_incr_rate"].astype(str) + '%'
    final_forecast["confirmed_incr_rate"] = final_forecast["confirmed_incr_rate"].astype(str) + '%'
    final_forecast["recovered_incr_rate"] = final_forecast["recovered_incr_rate"].astype(str) + '%'
    
    # SELECTING BETWEEN THE START AND END DATES
    final_forecast['Date'] = final_forecast.Date.astype(str)
    # final_forecast.set_index('Date',inplace = True)
    
    country_dict_list = []
    for i in range(len(final_forecast)):
        country_dict = {
            'Date': final_forecast.loc[i,'Date'],
            'deaths': int(final_forecast.loc[i,'deaths']),
            'confirmed': int(final_forecast.loc[i,'confirmed']),
            'recovered': int(final_forecast.loc[i,'recovered']),
            'deaths_incr': int(final_forecast.loc[i,'deaths_incr']),
            'confirmed_incr': int(final_forecast.loc[i,'confirmed_incr']),
            'recovered_incr': int(final_forecast.loc[i,'recovered_incr']),
            'deaths_incr_rate': final_forecast.loc[i,'deaths_incr_rate'],
            'confirmed_incr_rate': final_forecast.loc[i,'confirmed_incr_rate'],
            'recovered_incr_rate': final_forecast.loc[i,'recovered_incr_rate']
        }
        country_dict_list.append(country_dict)
    
    return jsonify({
        "count":len(final_forecast),
        "state":state_name.upper(),
        "state_code":state_code.upper(),
        "Lat": state_lat,
        "Long": state_long,
        "forecast_start_date":str(start_date)[:10],
        "forecast_end_date":str(end_date)[:10],
        "result":country_dict_list})
    
    
    
# =============================================================================
# @app.route('/country_line_graph', methods=["POST"])
# def line_plot_data_for_country():
#     if not request.json:
#         abort(400)
#     req = request.json
#     
#     # EXTRACTING REQUEST JSON DATA
#     country_names = req["country_names"].strip().split(",")
#     ycol = req["ycol"]
#     
#     # CHECKING
#     if len(country_names) == 0:
#         abort(Response("Please select a country"))
#     elif ycol == "":
#         abort(Response("Please select a type"))
#         
#     # SELECTING REQUIRED DATAFRAME
#     data = dfd if ycol == "Fatalities" else dfc
#     req_cols = [col for col in data.columns if col not in ["Lat","Long","Province/State"]]
#     
#     # AGGREGATING PER COUNTRY
#     data = data[req_cols].groupby(["Country/Region"]).agg(sum)
#     
#     # 
#     country_dict = {}
#     for country_name in country_names:
#         info_dict = {}
#         
#         print(f"country_name - {country_name}")
#         country_data = data[data['Country/Region'] == country_name]
#         
#         country_data = country_data.drop(drop_cols, axis =1)
#         
#         na_fill_item = country_data['Country/Region'].unique()[0]
#         country_data["Province/State"] = country_data["Province/State"].fillna(na_fill_item)
#         country_data_tr = country_data.set_index("Province/State").T
#         country_data_tr.columns = ["Date", "count"]
#         country_data_tr["Date"] = pd.to_datetime(country_data_tr["Date"])
#         print(country_data_tr.head())
#         
#         info_dict['Date'] = country_data_tr.Date.astype(str).tolist()
#         info_dict[ycol] = country_data_tr['count'].tolist()
#         country_dict[country_name] = info_dict
#         
#     return jsonify({"result":country_dict})
# =============================================================================

####################################
## HELPER FUNCTIONS
####################################

def get_state_data(state_code):
    
    dfsc_filt = dfsc.loc[state_code]
    dfsd_filt = dfsd.loc[state_code]
    dfsr_filt = dfsr.loc[state_code]
    
    # FINDING IMPORTANT DATA
    state_df = pd.concat([dfsc_filt, dfsd_filt, dfsr_filt], axis = 1)
    state_df = state_df.astype(int)
    state_df.columns = ['confirmed_incr', 'deaths_incr', 'recovered_incr']
    state_df['confirmed'] = state_df['confirmed_incr'].cumsum()
    state_df['deaths'] = state_df['deaths_incr'].cumsum()
    state_df['recovered'] = state_df['recovered_incr'].cumsum()
    
    state_df["deaths_incr_rate"] = np.round(100 * state_df["deaths_incr"] / (state_df["deaths"] - state_df["deaths_incr"]))
    state_df["confirmed_incr_rate"] = np.round(100 * state_df["confirmed_incr"] / (state_df["confirmed"] - state_df["confirmed_incr"]))
    state_df["recovered_incr_rate"] = np.round(100 * state_df["recovered_incr"] / (state_df["recovered"] - state_df["recovered_incr"]))
   
    # CLEANING NA DATA
    state_df = state_df.replace([np.inf, -np.inf], np.nan)
    state_df.fillna(0, inplace = True)
    
    state_df["deaths_incr_rate"] = state_df["deaths_incr_rate"].astype(str) + '%'
    state_df["confirmed_incr_rate"] = state_df["confirmed_incr_rate"].astype(str) + '%'
    state_df["recovered_incr_rate"] = state_df["recovered_incr_rate"].astype(str) + '%'
    
    return state_df

def is_cumulative_increasing(series):
    s = series.copy()
    bad_idx_1 = [i for i in range(len(s)-1) if (s[i] > s[i+1]) & (s[i] > s[i-1])]
    bad_idx_2 = [i+1 for i in range(len(s)-2) if (s[i+1] < s[i]) & (s[i+1] < s[i+2])]

    # print(bad_idx_1, bad_idx_2)
    bad_idx_arr = bad_idx_1 + bad_idx_2
    
    if len(bad_idx_arr):    
        return False, bad_idx_arr
    else:
        return True, bad_idx_arr

def check_source_data(series):
    s = series.copy()
    is_inc, bad_idx = is_cumulative_increasing(s)
    if not is_inc:
        for idx in bad_idx:
            update_val = s[idx-1]
#             print(f"idx:{idx}, current val:{s[idx]}, update_val:{update_val}")
            s[idx] = update_val
#             print(f"idx:{idx}, current val:{s[idx]}")
    return s

def get_country_data(d,c,r, country):
    
    d = check_source_data(d)
    c = check_source_data(c)
    r = check_source_data(r)
    df = pd.concat([d,c,r], axis = 1).reset_index()
    df.columns = ["Date","deaths", "confirmed", "recovered"]
    df["Date"] = pd.to_datetime(df["Date"])
    df_pos = df[df.confirmed > 0.0].set_index('Date')
    df_diff = df_pos.diff(1)
    df_diff['deaths'].iloc[0] =  df_pos['deaths'].iloc[0]
    df_diff['confirmed'].iloc[0] =  df_pos['confirmed'].iloc[0]
    df_diff['recovered'].iloc[0] =  df_pos['recovered'].iloc[0]
    df_diff['country'] = country
    
    return df_diff
            
def get_master_data(countries):
    master_data = pd.DataFrame()
    for country in countries:
        if country == "World":
            df_world_diff = get_country_data(dfd_world,dfc_world,dfr_world, country)
            master_data = pd.concat([master_data,df_world_diff], axis = 0)
        else:
            country_dfd = dfd_grp.loc[country].T
            country_dfc = dfc_grp.loc[country].T
            country_dfr = dfr_grp.loc[country].T
            
            country_data_diff = get_country_data(country_dfd,country_dfc,country_dfr, country)
            master_data = pd.concat([master_data,country_data_diff], axis = 0)
        
    return master_data

def adfuller_test(series):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    return p_value,p_value <= 0.05

def invert_transformation(df_forecast, train_log, train_diff1, train_diff2, second_diff=False, third_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = train_log.columns
    
    for col in columns:      
        
        # Roll back 3rd Diff
        if third_diff:
            df_fc[str(col)+'_2d'] = train_diff2[col].iloc[-1] + df_fc[str(col)+'_3d'].cumsum()
        
        # Roll back 2nd Diff
        elif second_diff:
            df_fc[str(col)+'_1d'] = train_diff1[col].iloc[-1] + df_fc[str(col)+'_2d'].cumsum()
        
        # Roll back 1st Diff and take inverse log transformation
        else:
            df_fc[str(col)+'_forecast'] = round(np.expm1(train_log[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()))
    
    return df_fc

if __name__ == '__main__':
    
    #####################################################
    ####### COVID-19 DATA ANALYSIS - WORLD ##############
    #####################################################
    
    # DATA SOURCES
    global_deaths_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    global_confirmed_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    global_recovered_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
    
    
    # READING DATA
    dfd = pd.read_csv(global_deaths_url)
    dfc = pd.read_csv(global_confirmed_url)
    dfr = pd.read_csv(global_recovered_url)
    
    # COLUMNS TO SELECT
    drop_cols = ["Country/Region","Lat","Long"]
    cols = [col for col in dfd.columns if col not in ["Lat","Long","Province/State"]]
    date_cols = [col for col in dfd.columns if col not in ["Lat","Long","Province/State", "Country/Region"]]
    
    # LAT / LONG DATA
    lat_long_df = dfd[["Country/Region","Lat","Long"]]
    lat_long_df = lat_long_df.groupby(["Country/Region"]).agg({"Lat":"first","Long":"first"})
    
    
    # WORLDWIDE COUNT
    dfd_world = dfd[date_cols].agg(sum)
    dfc_world = dfc[date_cols].agg(sum)
    dfr_world = dfr[date_cols].agg(sum)
    
    # WORLDWIDE COUNT - DAILY INCREASE
    dfd_world_diff = dfd_world.diff(periods = 1)
    dfc_world_diff = dfc_world.diff(periods = 1) 
    dfr_world_diff = dfr_world.diff(periods = 1) 

    # WORLDWIDE COUNT - CONCATENATING
    df_world = pd.concat([dfd_world,dfc_world,dfr_world,dfd_world_diff,dfc_world_diff,dfr_world_diff], axis = 1).reset_index()
    df_world.columns = ["Date","deaths", "confirmed", "recovered","deaths_incr","confirmed_incr","recovered_incr"]
    df_world["deaths_incr_rate"] = np.round(100 * df_world["deaths_incr"] / (df_world["deaths"] - df_world["deaths_incr"]))
    df_world["confirmed_incr_rate"] = np.round(100 * df_world["confirmed_incr"] / (df_world["confirmed"] - df_world["confirmed_incr"]))
    df_world["recovered_incr_rate"] = np.round(100 * df_world["recovered_incr"] / (df_world["recovered"] - df_world["recovered_incr"]))
   
    df_world["Date"] = pd.to_datetime(df_world["Date"])
    df_world.fillna(0, inplace = True)
    
    df_world["deaths_incr_rate"] = df_world["deaths_incr_rate"].astype(str) + '%'
    df_world["confirmed_incr_rate"] = df_world["confirmed_incr_rate"].astype(str) + '%'
    df_world["recovered_incr_rate"] = df_world["recovered_incr_rate"].astype(str) + '%'
    
    
    # COUNTRY-WISE COUNT - SUMMING FOR ALL STATES
    dfd_grp = dfd[cols].groupby(["Country/Region"], as_index=False).agg(sum).set_index('Country/Region')
    dfc_grp = dfc[cols].groupby(["Country/Region"], as_index=False).agg(sum).set_index('Country/Region')
    dfr_grp = dfr[cols].groupby(["Country/Region"], as_index=False).agg(sum).set_index('Country/Region')
    
    # COUNTRY-WISE COUNT - SUMMING FOR ALL DAYS 
    dfd_grp_agg = dfd_grp.T.agg(max)
    dfc_grp_agg = dfc_grp.T.agg(max)
    dfr_grp_agg = dfr_grp.T.agg(max)
    
    # COUNTRY-WISE COUNT - DAILY INCREASE
    dfd_grp_diff_tr = dfd_grp.T.diff(periods = 1).reset_index().drop('index', axis = 1)
    dfc_grp_diff_tr = dfc_grp.T.diff(periods = 1).reset_index().drop('index', axis = 1)
    dfr_grp_diff_tr = dfr_grp.T.diff(periods = 1).reset_index().drop('index', axis = 1)
    
    idx = len(dfd_grp_diff_tr)-1
    dfd_grp_diff_tr_ind = dfd_grp_diff_tr.loc[idx]
    dfc_grp_diff_tr_ind = dfc_grp_diff_tr.loc[idx]
    dfr_grp_diff_tr_ind = dfr_grp_diff_tr.loc[idx]
    # max_date = df_world.Date.max()
    # df_world_overall = df_world[df_world.Date == max_date].drop('Date', axis = 1)
    # index = df_world_overall.index.values[0]
    # res_dict = df_world_overall.loc[index].to_dict()
    
    df_country = pd.concat([lat_long_df,dfd_grp_agg, dfc_grp_agg, dfr_grp_agg, dfd_grp_diff_tr_ind, dfc_grp_diff_tr_ind, dfr_grp_diff_tr_ind], axis = 1).reset_index()
    df_country.columns = ["Country","Lat","Long","deaths", "confirmed", "recovered","deaths_incr","confirmed_incr","recovered_incr"]
    df_country["deaths_incr_rate"] = np.round(100 * df_country["deaths_incr"] / (df_country["deaths"] - df_country["deaths_incr"]))
    df_country["confirmed_incr_rate"] = np.round(100 * df_country["confirmed_incr"] / (df_country["confirmed"] - df_country["confirmed_incr"]))
    df_country["recovered_incr_rate"] = np.round(100 * df_country["recovered_incr"] / (df_country["recovered"] - df_country["recovered_incr"]))
    
    df_country.fillna(0, inplace = True)
    df_country["deaths_incr_rate"] = df_country["deaths_incr_rate"].astype(str) + '%'
    df_country["confirmed_incr_rate"] = df_country["confirmed_incr_rate"].astype(str) + '%'
    df_country["recovered_incr_rate"] = df_country["recovered_incr_rate"].astype(str) + '%'
    
    # print(country_dict_list[0:3])
    #train = pd.read_csv("train.csv", parse_dates = ['Date'])
    
    # MASTER DATA FOR MODELING
    countries = dfd_grp.index.tolist()       # list of all countries
    countries.append('World')
    master_data = get_master_data(countries)   # contains time series data for all the countries
    
    
    #####################################################
    ####### COVID-19 DATA ANALYSIS - INDIA ##############
    #####################################################
    
    # INDIA DATA
    india_district_url = "https://api.covid19india.org/v2/state_district_wise.json"
    india_states_daily_url = "https://api.covid19india.org/states_daily.json"
    
    dict_dist = requests.get(india_district_url).json()
    dict_states = requests.get(india_states_daily_url).json()
    
    df_dist = pd.DataFrame(dict_dist)
    df_dist['statecode'] = df_dist['statecode'].str.lower()
    df_dist['state'] = df_dist['state'].str.lower()
    
    df_states = pd.DataFrame(dict_states['states_daily'])
    df_states['date'] = pd.to_datetime(df_states['date'])
    
    dfsd = df_states[df_states.status == 'Deceased'].set_index('date').T
    dfsc = df_states[df_states.status == 'Confirmed'].set_index('date').T
    dfsr = df_states[df_states.status == 'Recovered'].set_index('date').T
    
    # STATES DATA
    # state_code_dict = {
    #     "state_code":["TT","MH","GJ","DL","RJ","MP","TN","UP","AP","TG","WB","JK","KA","KL","PB","HR","BR","OR","JH","UT","HP","CT","AS","CH","AN","LA","ML","PY","GA","MN","TR","MZ","AR","NL","DN","DD","LD","SK"],
    #     "state":["Total","Maharashtra","Gujarat","Delhi","Rajasthan","Madhya Pradesh","Tamil Nadu","Uttar Pradesh","Andhra Pradesh","Telangana","West Bengal","Jammu and Kashmir","Karnataka","Kerala","Punjab","Haryana","Bihar","Odisha","Jharkhand","Uttarakhand","Himachal Pradesh","Chhattisgarh","Assam","Chandigarh","Andaman and Nicobar Islands","Ladakh","Meghalaya","Puducherry","Goa","Manipur","Tripura","Mizoram","Arunachal Pradesh","Nagaland","Dadra and Nagar Haveli","Daman and Diu","Lakshadweep","Sikkim"]
    #     }
    
    # state_code_df = pd.DataFrame(state_code_dict).sort_values('state_code')
    state_code_df = pd.read_csv("state_coords.csv")
    state_code_df['state_code'] = state_code_df['state_code'].str.lower()
    state_code_df.set_index('state_code', inplace = True)
    
    # STATES DATA - LATEST
    df_states_last = df_states[-3:].drop('date', axis = 1)
    states_cols = df_states_last.columns
    df_states_last = df_states_last.set_index('status').replace('',0).astype(int).T
    
    # STATES DATA - OVERALL
    df_states_drop = df_states.drop(['date','status'],axis = 1)
    df_states_drop = df_states_drop.replace('',0).astype(int)
    
    df_states_status = df_states.status
    df_states_comb = pd.concat([df_states_drop,df_states_status], axis = 1)  # Appending status col
    df_states_grp = df_states_comb.groupby('status', as_index = False).agg(sum)
    df_states_grp = df_states_grp[states_cols].set_index('status').T
    states_grp_cols = df_states_grp.columns
    df_states_last = df_states_last[states_grp_cols]
    
    df_dist_sel = df_dist[['state','statecode']].set_index('statecode')
    # df_dist_sel.columns = ['state_name','state_code']
    
    # STATES DATA - COMBINED
    df_states_overall = pd.concat([state_code_df,df_states_grp,df_states_last], axis = 1).reset_index()
    df_states_overall.columns = ["state_code","state","latitude","longitude","confirmed", "deaths", "recovered","confirmed_incr","deaths_incr","recovered_incr"]
    # df_states_overall.state[df_states_overall.state == np.nan] = ['']
    
    df_states_overall["deaths_incr_rate"] = np.round(100 * df_states_overall["deaths_incr"] / (df_states_overall["deaths"] - df_states_overall["deaths_incr"]))
    df_states_overall["confirmed_incr_rate"] = np.round(100 * df_states_overall["confirmed_incr"] / (df_states_overall["confirmed"] - df_states_overall["confirmed_incr"]))
    df_states_overall["recovered_incr_rate"] = np.round(100 * df_states_overall["recovered_incr"] / (df_states_overall["recovered"] - df_states_overall["recovered_incr"]))
    
    df_states_overall.fillna(0, inplace = True)
    df_states_overall["deaths_incr_rate"] = df_states_overall["deaths_incr_rate"].astype(str) + '%'
    df_states_overall["confirmed_incr_rate"] = df_states_overall["confirmed_incr_rate"].astype(str) + '%'
    df_states_overall["recovered_incr_rate"] = df_states_overall["recovered_incr_rate"].astype(str) + '%'
    
    
    app.run(debug=True)
    