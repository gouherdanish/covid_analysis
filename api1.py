# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:56:25 2020

@author: gdanish
"""

# Our Modules
import constants, master

# For Data Analysis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For Rest Services
from flask import Flask, make_response, jsonify, abort, request, Response, render_template, g, redirect
from flask_cors import CORS
import time

app = Flask(__name__, template_folder = "template")
cors = CORS(app, resources={r"/*": {"origins": "*"}})

def add_response(response, params = False):
    if params:
        response["Cache-Control"] = "no-cache, no-store, must-revalidate" # HTTP 1.1.
        response["Pragma"] = "no-cache" # HTTP 1.0.
        response["Expires"] = "0" # Proxies.
    return response

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/')
def home():
    return redirect("/static")

@app.route('/static/')
@app.route('/static/prediction')
@app.route('/static/about')
@app.route('/static/technologies')
@app.route('/static/country/<name>')
@app.route('/static/state/<name>')
def homeStatic(name=''):
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


#################################################
########## GLOBAL APIS ##########################
#################################################
    
@app.route('/global_overall_count', methods=["GET"])
def get_global_overall_count():
    
    df_world = master_data[master_data.region.str.lower() == "world"].reset_index()
    world_idx = len(df_world) - 1
    max_date = str(df_world.Date[world_idx])[:10]
    # df_world_overall = df_world.loc[world_idx]
    world_dict = transformer.get_region_ts_dict(df_world, world_idx)
    res_dict = add_response({"count":1,"date":max_date,"result":world_dict})
    
    return jsonify(res_dict)

@app.route('/global_datewise_count', methods=["POST"])
def get_global_datewise_count():
    
    if not request.json:
        abort(400)
    req = request.json
    
    
    # EXTRACTING REQUEST JSON DATA
    # start_date = request.args.get('start_date','')
    # end_date = request.args.get('end_date','')
    start_date = req["start_date"]
    end_date = req["end_date"]
    
    # DEFAULT ALL DATA BEING TAKEN
    start_date = pd.to_datetime("2020-01-22") if start_date == "" else pd.to_datetime(start_date)
    end_date = pd.to_datetime(pd.datetime.now().date()) if end_date == "" else pd.to_datetime(end_date)
    
    # SELECTING BETWEEN THE START AND END DATES
    df_world = master_data[master_data.region.str.lower() == "world"].drop('region', axis = 1)
    df_world_range = df_world.loc[start_date:end_date].reset_index()
    df_world_range['Date'] = df_world_range.Date.astype(str)
    
    # FORMING RESPONSE
    df_world_ind = df_world_range.set_index('Date')
    df_world_ind_tr = df_world_ind.T
    # res_dict = df_world_ind_tr.to_dict()
    res_dict = add_response(df_world_ind_tr.to_dict())
    
    return jsonify({"count":len(df_world_ind),"result":res_dict})

@app.route('/global_countrywise_count', methods=["GET"])
def get_global_countrywise_count():
    
    df_country = master_data.groupby('region').agg('last')
    # cols = df_country.columns
    df_country = lat_long_df.join(df_country).reset_index()
    cols = ['region']
    cols.extend(df_country.columns[1:])
    df_country.columns =  cols
    # TRANSFORMING TIME SERIES DATA INTO DICT
    country_dict_list = [transformer.get_countrywise_stats_dict(df_country, i) for i in range(len(df_country))]
            
    country_count = len(df_country)
    max_date = str(master_data.index.values[-1])[:10]
    res_dict = add_response({"count":country_count,"date":max_date,"result":country_dict_list})
    
    return jsonify(res_dict)

@app.route('/single_country_overall_data', methods=["GET"])
def get_single_country_overall_data():
    # if not request.json:
    #     abort(400)
    # req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    country_name = request.args.get('country','').lower()
    
    # CHECKING
    if country_name == "":
        abort(Response("Please select a country"))
    
    df_country_filt = master_data[master_data.region.str.lower() == country_name].reset_index()
    idx = len(df_country_filt) - 1
    country_dict = transformer.get_region_ts_dict(df_country_filt, idx)
    max_date = str(df_country_filt.Date.values[-1])[:10]
    res_dict = add_response({"count":1,"date":max_date,"Country":country_name.upper(),"result":country_dict})
    
    return jsonify(res_dict)
  
@app.route('/single_country_data', methods=["POST"])
def get_single_country_data():
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    country_name = req["country"].lower()
    start_date = req["start_date"]
    end_date = req["end_date"]
    # country_name = request.args.get('country').lower()
    # start_date = request.args.get('start_date')
    # end_date = request.args.get('end_date')
    
    # CHECKING
    if country_name == "":
        abort(Response("Please select a country"))
    
    # DEFAULT ALL DATA BEING TAKEN
    start_date = pd.to_datetime("2020-01-22") if start_date == "" else pd.to_datetime(start_date)
    end_date = pd.to_datetime(pd.datetime.now().date()) if end_date == "" else pd.to_datetime(end_date)
        
    country_df = master_data[master_data.region.str.lower() == country_name]
    
    # SELECTING BETWEEN THE START AND END DATES
    country_df = country_df.loc[start_date:end_date].reset_index()
    country_df['Date'] = country_df.Date.astype(str)
    
    # TRANSFORMING TIME SERIES DATA INTO DICT
    country_dict_list = [transformer.get_region_ts_dict(country_df, i) for i in range(len(country_df))]
    res_dict = add_response({"count":len(country_df),"Country":country_name.upper(),"result":country_dict_list})
    return jsonify(res_dict)

@app.route('/single_country_forecast_data', methods=["GET"])
def get_single_country_forecast_data():
    # if not request.json:
    #     abort(400)
    # req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    # country_name = req["country_name"].lower()      
    # num_days = req["num_days"]   
    country_name = request.args.get('country','')
    num_days = request.args.get('days','')
    # print(num_days)
    
    # DEFAULT ALL DATA BEING TAKEN
    country_name = "world" if country_name == "" else country_name # By Default taking World Data
    num_days = 10 if num_days == "" else int(num_days)             # By Default taking 10 days
        
    # FILTERING FOR THE GIVEN COUNTRY
    master_data_filt = master_data_var[master_data_var.region.str.lower() == country_name]
        
    # TRAIN TEST SPLIT
    train = master_data_filt.iloc[:,:3]
    # test = master_data_filt.iloc[-test_days:,:3] # Taking last 10 days as testing data
    
    # FITTING VAR MODEL
    model = master.VectorAutoRegression(train, num_days)
    res = model.fit()
    
    # DW TEST FOR SERIAL CORRELATION
    # durbin_watson(res.resid)                         # should be around 2 for zero correlation
    
    # PREDICTING
    final_forecast = model.predict(res).reset_index()
    # start_date = final_forecast.Date.values[0]
    # end_date = final_forecast.Date.values[-1]
    start_date = model.start_date
    end_date = model.end_date
    
    # TRANSFORMING TIME SERIES DATA INTO DICT
    country_dict_list = [transformer.get_region_ts_dict(final_forecast, i) for i in range(len(final_forecast))]
    res_dict = add_response({
        "count":len(final_forecast),
        "Country":country_name.upper(),
        "forecast_start_date":str(start_date)[:10],
        "forecast_end_date":str(end_date)[:10],
        "result":country_dict_list})
    return jsonify(res_dict)
    
    
        
#################################################
########## INDIA APIS ###########################
#################################################

@app.route('/india_all_states_overall_data', methods=["GET"])
def get_india_all_states_overall_data():
    
    df_states_overall = state_master_data.groupby('region').agg('last')
    df_states_overall = state_code_df.join(df_states_overall).reset_index()
    state_dict_list = [statesman.get_statewise_stats_dict(df_states_overall, i) for i in range(len(df_states_overall))]
            
    state_count = len(df_states_overall)
    max_date = str(df_states.date.values[-1])[:10]
    res_dict = add_response({"count":state_count,"date":max_date,
                    "result":state_dict_list})
    return jsonify(res_dict)
    
@app.route('/india_single_state_overall_data', methods=["GET"])
def get_india_single_state_overall_data():
    # if not request.json:
    #     abort(400)
    # req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    state_name = request.args.get('state','').lower()
    state_name = "total" if state_name == "" else state_name
    
    # STATE DATA
    state_code, state_lat, state_long = statesman.get_statecode_latlong(state_code_df,state_name)
    
    # FILTERING FOR THE GIVEN STATE - BY DEFAULT GIVING TOTAL
    state_df = state_master_data[state_master_data.region.str.lower() == state_code].reset_index()
    idx = len(state_df) - 1
    max_date = str(state_df.Date[idx])[:10]
    
    
    # TRANSFORMING DATA TO DICT
    state_dict = statesman.get_region_ts_dict(state_df, idx)
    res_dict = add_response({
        "count":1,
        "date":max_date,
        "State": state_name.upper(),
        "State Code": state_code.upper(),
        "Lat": state_lat,
        "Long": state_long,
        "result":state_dict
        })
    return jsonify(res_dict)

@app.route('/india_single_state_datewise_data', methods=["POST"])
def get_india_single_state_datewise_data():
    if not request.json:
        abort(400)
    req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    state_name = req["state"].lower()
    start_date = req["start_date"]
    end_date = req["end_date"]
    # state_name = request.args.get('state_name').lower()
    # start_date = request.args.get('start_date')
    # end_date = request.args.get('end_date')
    
    # BY DEFAULT ALL DATA BEING TAKEN
    state_name = "total" if state_name == "" else state_name
    start_date = pd.to_datetime("2020-01-22") if start_date == "" else pd.to_datetime(start_date)
    end_date = pd.to_datetime(pd.datetime.now().date()) if end_date == "" else pd.to_datetime(end_date)
    
    # STATE DATA
    state_code, state_lat, state_long = statesman.get_statecode_latlong(state_code_df,state_name)
    
    # TIME SERIES DATA FOR THE GIVEN STATE - BY DEFAULT FOR TOTAL
    state_df = state_master_data[state_master_data.region.str.lower() == state_code]
      
    # SELECTING BETWEEN THE START AND END DATES
    state_df = state_df.loc[start_date:end_date].reset_index()
    state_df['Date'] = state_df.Date.astype(str)
    
    # TRANSFORMING TIME SERIES DATA INTO DICT
    state_dict_list = [statesman.get_region_ts_dict(state_df, i) for i in range(len(state_df))]
    res_dict = add_response({
        "count":len(state_df),
        "State":state_name.upper(), 
        "State Code":state_code.upper(),
        "Lat": state_lat,
        "Long": state_long,
        "result":state_dict_list
        })
    return jsonify(res_dict)

@app.route('/india_single_state_district_data', methods=["GET"])
def get_india_single_state_district_data():
    # if not request.json:
    #     abort(400)
    # req = request.json
    
    # EXTRACTING REQUEST JSON DATA
    state_name = request.args.get('state','').lower()
    state_name = "total" if state_name == "" else state_name
   
    # STATE DATA
    state_code, state_lat, state_long = statesman.get_statecode_latlong(state_code_df,state_name)
        
    district_data_list = df_dist.districtData[df_dist.statecode == state_code].values[0]
    district_data_list_new = [statesman.get_districtwise_stats_dict(dct) for dct in district_data_list]
    res_dict = add_response({
        "count":len(district_data_list_new),
        "State":state_name.upper(), 
        "State Code":state_code.upper(),
        "Lat": state_lat,
        "Long": state_long,
        "result":district_data_list_new
        })
    return jsonify(res_dict)
    
@app.route('/india_single_state_forecast_data', methods=["GET"])
def get_india_single_state_forecast_data():
    # if not request.json:
    #     abort(400)
    # req = request.json
    
    # # EXTRACTING REQUEST JSON DATA
    # state_name = req["state_name"].lower()
    # num_days = int(req["num_days"])
    state_name = request.args.get('state','')
    num_days = request.args.get("days",'')
    
    # BY DEFAULT ALL DATA BEING TAKEN
    state_name = "total" if state_name == "" else state_name
    num_days = 10 if num_days == "" else int(num_days)             # By Default taking 10 days
    
    # STATE DATA
    state_code, state_lat, state_long = statesman.get_statecode_latlong(state_code_df,state_name)
    
    # FILTERING FOR THE GIVEN STATE - BY DEFAULT GIVING TOTAL
    state_df = state_master_data[state_master_data.region.str.lower() == state_code]
          
    # TRAIN TEST SPLIT
    train = state_df.iloc[:,:3]
    # test = master_data_filt.iloc[-test_days:,:3] # Taking last 10 days as testing data
    
    # FITTING VAR MODEL
    model = master.VectorAutoRegression(train, num_days)
    res = model.fit()
    
    # DW TEST FOR SERIAL CORRELATION
    # durbin_watson(res.resid)                         # should be around 2 for zero correlation
    
    # PREDICTING
    final_forecast = model.predict(res).reset_index()
    
    # start_date = final_forecast['Date'].values[0]
    # end_date = final_forecast['Date'].values[-1]
    start_date = model.start_date
    end_date = model.end_date
    
    # TRANSFORMING TIME SERIES DATA TO DICT
    country_dict_list = [statesman.get_region_ts_dict(final_forecast, i) for i in range(len(final_forecast))]
    # for i in range(len(final_forecast)):
    #     country_dict = statesman.get_country_dict(final_forecast, i)
    #     country_dict_list.append(country_dict)
    res_dict = add_response({
        "count":len(final_forecast),
        "state":state_name.upper(),
        "state_code":state_code.upper(),
        "Lat": state_lat,
        "Long": state_long,
        "forecast_start_date":str(start_date)[:10],
        "forecast_end_date":str(end_date)[:10],
        "result":country_dict_list})
    return jsonify(res_dict)

if __name__ == '__main__':
    
    #####################################################
    ####### COVID-19 DATA ANALYSIS - WORLD ##############
    #####################################################
    
    
    # DATA SOURCES
    global_deaths_url = constants.GLOBAL['global_deaths_url']
    global_confirmed_url = constants.GLOBAL['global_confirmed_url']
    global_recovered_url = constants.GLOBAL['global_recovered_url']
    
    # print(global_deaths_url)
    
    # CREATING TRANSFORMER INSTANCE
    transformer = master.DataTransformer()
    
    # CREATING INSTANCES
    # inst_d = master.DataTransformer(global_deaths_url)
    # inst_c = master.DataTransformer(global_confirmed_url)
    # inst_r = master.DataTransformer(global_recovered_url)
    
    # LOADING DATA
    dfd = transformer.load_data(global_deaths_url)
    dfc = transformer.load_data(global_confirmed_url)
    dfr = transformer.load_data(global_recovered_url)
    
    # print(dfd.head())
    
    
    # LAT / LONG DATA
    lat_long_df = transformer.prepare_latlong_data(dfd)
    # print(lat_long_df.loc['France'])
    # print(lat_long_df.head())
    
    # COLUMNS TO SELECT
    cols = [col for col in dfd.columns if col not in ["Lat","Long","Province/State"]]
    date_cols = [col for col in dfd.columns if col not in ["Lat","Long","Province/State", "Country/Region"]]
    
    ###################### MODELING DATA ########################
    
    # list_df = [dfd_world, dfc_world, dfr_world, dfd_grp, dfc_grp, dfr_grp]
    master_data = transformer.prepare_master_data([dfd, dfc, dfr], cols, date_cols, region_col = "Country/Region", for_world = True, for_model = False, is_sir_model = False)   # contains time series data for all the countries
    master_data_var = transformer.prepare_master_data([dfd, dfc, dfr], cols, date_cols, region_col = "Country/Region", for_world = True, for_model = True, is_sir_model = False)   # contains time series data for all the countries
    master_data_sir = transformer.prepare_master_data([dfd, dfc, dfr], cols, date_cols, region_col = "Country/Region", for_world = True, for_model = True, is_sir_model = True)   # contains time series data for all the countries
    print(master_data_sir[master_data_sir.region.str.lower() == "india"].head())
    # mydf_country = master_data.groupby('region').agg('last')
    # print(lat_long_df.head())
    # print(mydf_country.head())
    # mydf_country = lat_long_df.join(mydf_country).reset_index()
    # print(['region'].extend(mydf_country.columns[1:]))
    # print(mydf_country.head())
    # print(mydf_country.loc[0,'region'])
    # print("FULL DATA - ")
    # print(master_data.tail(20))
    # print("VAR MODEL DATA - ")
    # print(master_data_var.head())
    print("SIR MODEL DATA - ")
    print(master_data_sir.head())
    sir = master.CompartmentalModel(master_data_sir, "india", 5)
    res = sir.fit() 
    print(res.x)
    
    # print(master_data.groupby('country').agg('last').join(lat_long_df).head())
       
    #####################################################
    ####### COVID-19 DATA ANALYSIS - INDIA ##############
    #####################################################
    
    # INDIA DATA
    india_district_url = constants.INDIA['india_district_url']
    india_states_daily_url = constants.INDIA['india_states_daily_url']
    
    # CREATING INSTANCE
    statesman = master.DataTransformerIndia(india_district_url, india_states_daily_url)
    
    # DISTRICTS AND STATES DATA
    df_dist = statesman.load_district_data()
    df_states = statesman.load_states_data()
    dfsd, dfsc, dfsr = [statesman.load_data(df_states, status) for status in ['Deceased', 'Confirmed', 'Recovered']]
    # print("STATE DATA:")
    # print(df_states.head())
    # print(df_dist.head())
    # print(dfsd.head())
    
    # STATES COORDS DATA
    state_code_df = statesman.load_state_coord_data()
    # print(state_code_df.head())
    
    # REQUIRED COLUMN NAMES
    cols_state = dfsd.columns.tolist()
    date_cols_state = [col for col in dfsd.columns if col not in ["state_code"]]
    # print(cols_state)
    # print(date_cols_state)
    
    # FINDING STATES MASTER DATA
    state_master_data = statesman.prepare_master_data([dfsd, dfsc, dfsr], cols_state, date_cols_state, region_col = "state_code", for_world = False, for_model = False, is_sir_model = False)   # contains time series data for all the countries
    state_master_data_var = statesman.prepare_master_data([dfsd, dfsc, dfsr], cols_state, date_cols_state, region_col = "state_code", for_world = False, for_model = True, is_sir_model = False)   # contains time series data for all the countries
    state_master_data_sir = statesman.prepare_master_data([dfsd, dfsc, dfsr], cols_state, date_cols_state, region_col = "state_code", for_world = False, for_model = True, is_sir_model = True)   # contains time series data for all the countries
    # print("FULL DATA - ")
    # print(state_master_data.head())
    # print("VAR MODEL DATA - ")
    # print(state_master_data_var.head())
    # print("SIR MODEL DATA - ")
    # print(state_master_data_sir.head())
    
    
    
    # app.run(debug=True)
    # app.run(host = 0.0.0.0, port = 80)
