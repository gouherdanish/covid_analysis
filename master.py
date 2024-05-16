# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:39:25 2020

@author: gdanish
"""

import numpy as np
import pandas as pd
import datetime
import constants
import requests

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from scipy import integrate, optimize
from sklearn.metrics import mean_squared_log_error
    
class DataLoader():
    
    # def __init__(self, url):
    #     self.url = url
        
    def load_data(self, url):
        return pd.read_csv(url)
    
    def prepare_latlong_data(self, df):
        
        # LAT / LONG DATA
        lat_long_df = df[["Country/Region","Lat","Long"]]
        lat_long_df.columns = ['country', 'Lat', 'Long']
        lat_long_df = lat_long_df.groupby(["country"]).agg({"Lat":"first","Long":"first"})
        # lat_long_df = lat_long_df.groupby(["Country/Region"]).agg({"Lat":"first","Long":"first"})
        
        # COLONIAL/PROVINCIAL COUNTRIES
        coord_list = constants.COORDS["coord_list"]
        
        # SETTING ONE SINGLE LAT/LONG FOR THESE COUNTRIES
        coord_df = pd.DataFrame(coord_list).set_index('country')
        coord_df_join =  lat_long_df.join(coord_df, rsuffix = 'c')
        # print(coord_df_join.head())
        
        coord_df_join['Latm'] = np.where(coord_df_join['Latc'].isnull(),coord_df_join['Lat'],coord_df_join['Latc'])
        coord_df_join['Longm'] = np.where(coord_df_join['Longc'].isnull(),coord_df_join['Long'],coord_df_join['Longc'])
        # print(coord_df_join.head())
        
        coord_df_join = coord_df_join[['Latm','Longm']]
        coord_df_join.columns = ['Lat','Long']
        
        return coord_df_join

    def is_cumulative_increasing(self, series):
        s = series.copy()
        # print(s)
        bad_idx_1 = [i for i in range(len(s)-1) if (s[i] > s[i+1]) & (s[i] > s[i-1])]
        bad_idx_2 = [i+1 for i in range(len(s)-2) if (s[i+1] < s[i]) & (s[i+1] < s[i+2])]
    
        # print(bad_idx_1, bad_idx_2)
        bad_idx_arr = bad_idx_1 + bad_idx_2
        
        if len(bad_idx_arr):    
            return False, bad_idx_arr
        else:
            return True, bad_idx_arr

    def check_source_data(self, series):
        s = series.copy()
        is_inc, bad_idx = self.is_cumulative_increasing(s)
        if not is_inc:
            for idx in bad_idx:
                update_val = s[idx-1]
    #             print(f"idx:{idx}, current val:{s[idx]}, update_val:{update_val}")
                s[idx] = update_val
    #             print(f"idx:{idx}, current val:{s[idx]}")
        return s
    
    def prepare_timeseries(self, dfx, cols, region, region_col):
        # TS DATA FOR WORLD
        if region.lower() == "world":
            ser = dfx[cols].agg(sum) # aggregating for the world
            ser = self.check_source_data(ser) # checking the source data is cumulative increasing or not
            
        # TS DATA FOR COUNTRY
        else:
            ser = dfx[cols].groupby([region_col]).agg(sum) # some countries have state-wise data, so aggreagting
            ser = ser.loc[region]
            ser = self.check_source_data(ser) # checking the source data is cumulative increasing or not
            # print(ser)
        # print(ser[:2])
        ser.index = pd.to_datetime(ser.index)
        if len(ser) == 0:
            print(f"ERROR: CHECK REGION - {region}")
        # print(ser[:2])
        return ser
        
    def find_infection_start_date(self, s, region):
        # if region.lower() == 'an':
        #     print(s[:2])
        #     print(s[s > 0][:2])
        s_pos = s[s > 0]
        isd = pd.to_datetime(s.index[-1]) if len(s_pos) == 0 else pd.to_datetime(s_pos.index[0])
        
        # print(f"region : {region} len:{len(s_pos)} isd: {isd}")
        return isd
        
    def concatenate_series(self, list_series, colnames):
        df = pd.concat(list_series, axis = 1).reset_index()
        df.columns = colnames
        return df
    
    def get_daily_data(self, series, infection_start_date, for_model):
        
        # print(infection_start_date)
        
        # model is trained on data starting from first infection
        if for_model:
            series = series.loc[series.index > infection_start_date]
            if len(series) == 0: # no cases in this region so far,
                return series
        
        series_diff = series.diff(periods = 1)
        # print(f"series 0 : {len(series)} , diff 0 : {len(series_diff)}")
        series_diff[0] = series[0] # substituting for NA
        # print(lseries)
        # print(series_diff)
        # print(series[:3])
        # print(series_diff[:5])
        
        return series_diff
        
    def calculate_growth_rate(self, df):
        
        df_world = df.copy()
        df_world = df_world.set_index('Date').astype(float)
        # if region.lower() == "mh":
        #     print(df_world.head())
        #     print(df_world.info())
        df_world["active_incr_rate"] = np.round(100 * df_world["active_incr"] / (df_world["active"] - df_world["active_incr"]), 2)
        df_world["deaths_incr_rate"] = np.round(100 * df_world["deaths_incr"] / (df_world["deaths"] - df_world["deaths_incr"]), 2)
        df_world["confirmed_incr_rate"] = np.round(100 * df_world["confirmed_incr"] / (df_world["confirmed"] - df_world["confirmed_incr"]), 2)
        df_world["recovered_incr_rate"] = np.round(100 * df_world["recovered_incr"] / (df_world["recovered"] - df_world["recovered_incr"]), 2)
        
        # df_world["active_incr_frac"] = np.round(100 * df_world["active_incr"] / (df_world["confirmed_incr"]), 2)
        # df_world["deaths_incr_frac"] = np.round(100 * df_world["deaths_incr"] / (df_world["confirmed_incr"]), 2)
        # df_world["recovery_incr_frac"] = np.round(100 * df_world["recovered_incr"] / (df_world["confirmed_incr"]), 2)
        
        df_world["active_frac"] = np.round(100 * df_world["active"] / (df_world["confirmed"]), 2)
        df_world["deaths_frac"] = np.round(100 * df_world["deaths"] / (df_world["confirmed"]), 2)
        df_world["recovery_frac"] = np.round(100 * df_world["recovered"] / (df_world["confirmed"]), 2)
        
        # df_world["transmission_rate"] = np.round(df_world["active_incr"] / df_world["active"], 2)
        # df_world["case_fatality_rate"] = np.round(df_world["deaths_incr"] / df_world["active"], 2)
        # df_world["removal_rate"] = np.round((df_world["recovered_incr"] + df_world["deaths_incr"])/ df_world["active"], 2)
        
        # df_world["incubation_period"] = np.round(1.0 / df_world["transmission_rate"], 2)
        # df_world["infection_period"] = np.round(1.0 / df_world["removal_rate"], 2)
        df_world["reproduction_number"] = np.round((1 + df_world["active_incr"] / df_world["recovered_incr"]), 2)
        
        # df_world['incubation_period'] = df_world['incubation_period'].replace([np.inf, -np.inf], 1)
        # df_world['infection_period'] = df_world['infection_period'].replace([np.inf, -np.inf], 1)
        df_world['reproduction_number'] = df_world['reproduction_number'].replace([np.inf, -np.inf], 1)
        
        df_world = df_world.replace([np.inf, -np.inf], 100)
        df_world.fillna(0, inplace = True)
        
        df_world["active_incr_rate"] = df_world["active_incr_rate"].astype(str) + '%'
        df_world["deaths_incr_rate"] = df_world["deaths_incr_rate"].astype(str) + '%'
        df_world["confirmed_incr_rate"] = df_world["confirmed_incr_rate"].astype(str) + '%'
        df_world["recovered_incr_rate"] = df_world["recovered_incr_rate"].astype(str) + '%'
        
        # df_world["active_incr_frac"] = df_world["active_incr_frac"].astype(str) + '%'
        # df_world["deaths_incr_frac"] = df_world["deaths_incr_frac"].astype(str) + '%'
        # df_world["recovery_incr_frac"] = df_world["recovery_incr_frac"].astype(str) + '%'
        
        df_world["active_frac"] = df_world["active_frac"].astype(str) + '%'
        df_world["deaths_frac"] = df_world["deaths_frac"].astype(str) + '%'
        df_world["recovery_frac"] = df_world["recovery_frac"].astype(str) + '%'
        
        # df_world.reset_index(inplace = True)
        return df_world
    
    def get_region_ts_dict(self, final_forecast, i):
        dct = {
            'Date': final_forecast['Date'].astype(str).loc[i][:10],
            'active': int(final_forecast.loc[i,'active']),
            'deaths': int(final_forecast.loc[i,'deaths']),
            'confirmed': int(final_forecast.loc[i,'confirmed']),
            'recovered': int(final_forecast.loc[i,'recovered']),
            'active_incr': int(final_forecast.loc[i,'active_incr']),
            'deaths_incr': int(final_forecast.loc[i,'deaths_incr']),
            'confirmed_incr': int(final_forecast.loc[i,'confirmed_incr']),
            'recovered_incr': int(final_forecast.loc[i,'recovered_incr']),
            'active_incr_rate': final_forecast.loc[i,'active_incr_rate'],
            'deaths_incr_rate': final_forecast.loc[i,'deaths_incr_rate'],
            'confirmed_incr_rate': final_forecast.loc[i,'confirmed_incr_rate'],
            'recovered_incr_rate': final_forecast.loc[i,'recovered_incr_rate'],
            # 'active_incr_frac': final_forecast.loc[i,'active_incr_frac'],
            # 'deaths_incr_frac': final_forecast.loc[i,'deaths_incr_frac'],
            # 'recovery_incr_frac': final_forecast.loc[i,'recovery_incr_frac'],
            'active_frac': final_forecast.loc[i,'active_frac'],
            'deaths_frac': final_forecast.loc[i,'deaths_frac'],
            'recovery_frac': final_forecast.loc[i,'recovery_frac'],
            # 'transmission_rate': final_forecast.loc[i,'transmission_rate'],
            # 'case_fatality_rate': final_forecast.loc[i,'case_fatality_rate'],
            # 'removal_rate': final_forecast.loc[i,'removal_rate'],
            # 'incubation_period': final_forecast.loc[i,'incubation_period'],
            # 'infection_period': final_forecast.loc[i,'infection_period'],
            'reproduction_number': final_forecast.loc[i,'reproduction_number']
        }
        return dct
    
    def get_countrywise_stats_dict(self, df_country, i):
        country_dict = {
            'Country': df_country.loc[i,'region'],
            'Lat': float(df_country.loc[i,'Lat']),
            'Long': float(df_country.loc[i,'Long']),
            'deaths': int(df_country.loc[i,'deaths']),
            'confirmed': int(df_country.loc[i,'confirmed']),
            'recovered': int(df_country.loc[i,'recovered']),
            'active': int(df_country.loc[i,'active']),
            'deaths_incr': int(df_country.loc[i,'deaths_incr']),
            'confirmed_incr': int(df_country.loc[i,'confirmed_incr']),
            'recovered_incr': int(df_country.loc[i,'recovered_incr']),
            'active_incr': int(df_country.loc[i,'active_incr']),
            'deaths_incr_rate': df_country.loc[i,'deaths_incr_rate'],
            'confirmed_incr_rate': df_country.loc[i,'confirmed_incr_rate'],
            'recovered_incr_rate': df_country.loc[i,'recovered_incr_rate'],
            # 'active_incr_frac': df_country.loc[i,'active_incr_frac'],
            # 'deaths_incr_frac': df_country.loc[i,'deaths_incr_frac'],
            # 'recovery_incr_frac': df_country.loc[i,'recovery_incr_frac'],
            'active_frac': df_country.loc[i,'active_frac'],
            'deaths_frac': df_country.loc[i,'deaths_frac'],
            'recovery_frac': df_country.loc[i,'recovery_frac'],
            # 'transmission_rate': df_country.loc[i,'transmission_rate'],
            # 'case_fatality_rate': df_country.loc[i,'case_fatality_rate'],
            # 'removal_rate': df_country.loc[i,'removal_rate'],
            # 'incubation_period': df_country.loc[i,'incubation_period'],
            # 'infection_period': df_country.loc[i,'infection_period'],
            'reproduction_number': df_country.loc[i,'reproduction_number']
        }
        return country_dict
    
    def get_statewise_stats_dict(self, df_states_overall, i):
        state_dict = {
            # 
            'State': df_states_overall.loc[i,'state'],
            'State Code': df_states_overall.loc[i,'region'].upper(),
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
            'recovered_incr_rate': df_states_overall.loc[i,'recovered_incr_rate'],
            # 'active_incr_frac': df_states_overall.loc[i,'active_incr_frac'],
            # 'deaths_incr_frac': df_states_overall.loc[i,'deaths_incr_frac'],
            # 'recovery_incr_frac': df_states_overall.loc[i,'recovery_incr_frac'],
            'active_frac': df_states_overall.loc[i,'active_frac'],
            'deaths_frac': df_states_overall.loc[i,'deaths_frac'],
            'recovery_frac': df_states_overall.loc[i,'recovery_frac'],
            # 'transmission_rate': df_states_overall.loc[i,'transmission_rate'],
            # 'case_fatality_rate': df_states_overall.loc[i,'case_fatality_rate'],
            # 'removal_rate': df_states_overall.loc[i,'removal_rate'],
            # 'incubation_period': df_states_overall.loc[i,'incubation_period'],
            # 'infection_period': df_states_overall.loc[i,'infection_period'],
            'reproduction_number': df_states_overall.loc[i,'reproduction_number']
        }
        return state_dict
    
    def get_districtwise_stats_dict(self, dct):
        
        dist_dict = {}
        dist_dict['district'] = dct['district']
        dist_dict['confirmed'] = dct['confirmed']
        dist_dict['deaths'] = dct['deceased']
        dist_dict['recovered'] = dct['recovered']
        dist_dict['active'] = dist_dict['confirmed'] - dist_dict['deaths'] - dist_dict['recovered']
        
        dist_dict['deaths_frac'] = np.round(100 * dist_dict['deaths'] / dist_dict['confirmed'])
        dist_dict['recovered_frac'] = np.round(100 * dist_dict['recovered'] / dist_dict['confirmed'])
        dist_dict['active_frac'] = np.round(100 * dist_dict['active'] / dist_dict['confirmed'])
        
        dist_dict['confirmed_incr'] = dct['delta']['confirmed']
        dist_dict['recovered_incr'] = dct['delta']['recovered']
        dist_dict['deaths_incr'] = dct['delta']['deceased']
        dist_dict['active_incr'] = dist_dict['confirmed_incr'] - dist_dict['deaths_incr'] - dist_dict['recovered_incr']
        
        dist_dict['confirmed_incr_rate'] = 0.0 if dist_dict['confirmed'] - dist_dict['confirmed_incr'] == 0 else np.round(100 * dist_dict['confirmed_incr'] / (dist_dict['confirmed'] - dist_dict['confirmed_incr']), 2)
        dist_dict['recovered_incr_rate'] = 0.0 if dist_dict['recovered'] - dist_dict['recovered_incr'] == 0 else np.round(100 * dist_dict['recovered_incr'] / (dist_dict['recovered'] - dist_dict['recovered_incr']), 2)
        dist_dict['deaths_incr_rate'] = 0.0 if dist_dict['deaths'] - dist_dict['deaths_incr'] == 0 else np.round(100 * dist_dict['deaths_incr'] / (dist_dict['deaths'] - dist_dict['deaths_incr']), 2)
        dist_dict['active_incr_rate'] = 0.0 if dist_dict['active'] - dist_dict['active_incr'] == 0 else np.round(100 * dist_dict['active_incr'] / (dist_dict['active'] - dist_dict['active_incr']), 2)
        
        # dist_dict["transmission_rate_daily"] = 0.0 if dist_dict['active'] - dist_dict['active_incr'] == 0 else np.round(100 * dist_dict["active_incr"] / (dist_dict["active"] - dist_dict["active_incr"]), 2)
        # dist_dict["case_fatality_rate_daily"] = 0.0 if dist_dict['confirmed_incr'] == 0 else np.round(100 * dist_dict["deaths_incr"] / (dist_dict["confirmed_incr"]), 2)
        # dist_dict["recovery_rate_daily"] = 0.0 if dist_dict['confirmed_incr'] == 0 else np.round(100 * dist_dict["recovered_incr"] / (dist_dict["confirmed_incr"]), 2)
        
        # dist_dict["reproduction_number"] = 0.0 if dist_dict['recovered_incr'] == 0 else np.round((1 + dist_dict["active_incr"] / dist_dict["recovered_incr"]), 2)
        # dist_dict["case_fatality_rate_cum"] = 0.0 if dist_dict['confirmed'] == 0 else np.round(100 * dist_dict["deaths"] / (dist_dict["confirmed"]), 2)
        # dist_dict["recovery_rate_cum"] = 0.0 if dist_dict['confirmed'] == 0 else np.round(100 * dist_dict["recovered"] / (dist_dict["confirmed"]), 2)
        
        dist_dict['confirmed_incr_rate'] = str(dist_dict['confirmed_incr_rate']) + '%'
        dist_dict['recovered_incr_rate'] = str(dist_dict['recovered_incr_rate']) + '%'
        dist_dict['deaths_incr_rate'] = str(dist_dict['deaths_incr_rate']) + '%'
        dist_dict['active_incr_rate'] = str(dist_dict['active_incr_rate']) + '%'
        
        dist_dict['recovered_frac'] = str(dist_dict['recovered_frac']) + '%'
        dist_dict['deaths_frac'] = str(dist_dict['deaths_frac']) + '%'
        dist_dict['active_frac'] = str(dist_dict['active_frac']) + '%'
        # dist_dict['transmission_rate_daily'] = str(dist_dict['transmission_rate_daily']) + '%'
        # dist_dict['case_fatality_rate_daily'] = str(dist_dict['case_fatality_rate_daily']) + '%'
        # dist_dict['recovery_rate_daily'] = str(dist_dict['recovery_rate_daily']) + '%'
        # dist_dict['case_fatality_rate_cum'] = str(dist_dict['case_fatality_rate_cum']) + '%'
        # dist_dict['recovery_rate_cum'] = str(dist_dict['recovery_rate_cum']) + '%'
        
        return dist_dict
    
class DataTransformer(DataLoader):
            
    def prepare_region_ts_dataframe(self, list_df, cols, region, region_col, for_model, is_sir_model):
        
        region_case_series_list = [self.prepare_timeseries(dfx, cols, region, region_col) for dfx in list_df]
        # print(len(region_case_series_list))
        # print(region_case_series_list[0])
        region_active_case_series = region_case_series_list[1] - region_case_series_list[0] - region_case_series_list[2] # active = confirmed - recovered - deaths
        
        region_case_series_list.append(region_active_case_series)
        # print(len(region_case_series_list))
        
        infection_start_date = self.find_infection_start_date(region_case_series_list[1], region)
        # infection_start_date = pd.to_datetime(infection_start_str)
        
        region_case_series_diff_list = [self.get_daily_data(ser, infection_start_date, for_model) for ser in region_case_series_list ]
        
        # print(len(region_case_series_diff_list))
        # print(infection_start_date)
        
        if for_model: 
            # print("INSIDE FOR MODEL")
            model_df_cols =  ["Date","deaths", "confirmed", "recovered", "active"]
            
            # DATA FOR SIR MODEL / VAR MODEL
            df = self.concatenate_series(region_case_series_list, model_df_cols) if is_sir_model else self.concatenate_series(region_case_series_diff_list, model_df_cols)
            df["Date"] = pd.to_datetime(df["Date"])
            
            # TAKING DATA FROM 1st CONFIRMED CASE
            df = df[df.Date > infection_start_date]
            df.set_index('Date', inplace = True)
        else:
            # print(f"INSIDE FOR DATA - {region}")
            comb_lists = region_case_series_list + region_case_series_diff_list
            comb_cols = ["Date","deaths", "confirmed", "recovered","active","deaths_incr","confirmed_incr","recovered_incr","active_incr"]
            df = self.concatenate_series(comb_lists, comb_cols)
            # print(df.head())
        
            # DAILY GROWTH RATE
            df = self.calculate_growth_rate(df)
    
        df['region'] = region
        return df
    
    def prepare_master_data(self, list_df, cols, date_cols, region_col, for_world, for_model, is_sir_model):
        
        # list of all countries
        df = list_df[0]
        regions = df[region_col].unique().tolist()
        if for_world:
            regions.append('World')
        # print(countries)
        
        print(f"for_model :{for_model}")
        # LIST OF DATAFRAMES FOR EACH REGION
        master_list_regions_df = [self.prepare_region_ts_dataframe(list_df, date_cols, region, region_col, for_model, is_sir_model) if region.lower() == "world" 
                           else self.prepare_region_ts_dataframe(list_df, cols, region, region_col, for_model, is_sir_model)  for region in regions]
        
        # CONCATENATING TO ONE SINGLE DATAFRAME
        master_df = pd.concat(master_list_regions_df, axis = 0)
        
        return master_df
            
class DataTransformerIndia(DataTransformer):
    
    def __init__(self, url1, url2):
        self.india_district_url = url1
        self.india_states_daily_url = url2
        
    def load_district_data(self):
        dict_dist = requests.get(self.india_district_url).json()
        df_dist = pd.DataFrame(dict_dist)
        df_dist['statecode'] = df_dist['statecode'].str.lower()
        df_dist['state'] = df_dist['state'].str.lower()
        return df_dist
    
    def load_states_data(self):
        dict_states = requests.get(self.india_states_daily_url).json()
        df_states = pd.DataFrame(dict_states['states_daily'])
        # df_states['date'] = pd.to_datetime(df_states['date'])
        
        # STATES DATA - OVERALL
        int_cols = [col for col in df_states.columns if col not in ['date','status']]
        # df_states_drop = df_states.drop(['date','status'],axis = 1)
        # df_states_drop = df_states_drop.replace('',0).astype(int)
        
        df_states[int_cols] = df_states[int_cols].replace('',0).astype(int)
        
        # df_states_status = df_states.status
        # df_states_comb = pd.concat([df_states_drop,df_states_status], axis = 1)  # Appending status col
        
        return df_states
    
    def load_data(self,df_states, status):
        dfsx = df_states[df_states.status == status]
        dfsx_w0 = dfsx.drop(['status','date'], axis = 1).cumsum()
        dfsx_dt = dfsx['date']
        
        dfsx_comb = pd.concat([dfsx_dt, dfsx_w0], axis = 1)
        dfsx_comb = dfsx_comb.set_index('date').T
        cols = dfsx_comb.columns.tolist()
        dfsx_comb.reset_index(inplace =True)
        dfsx_comb.columns = ['state_code'] + cols
        return dfsx_comb
    
    def load_state_coord_data(self):
        
        # state_code_df = pd.DataFrame(state_code_dict).sort_values('state_code')
        state_code_df = pd.read_csv("state_coords.csv")
        state_code_df['region'] = state_code_df['region'].str.lower()
        state_code_df.set_index('region', inplace = True)
        return state_code_df
    
    def get_statecode_latlong(self, state_code_df, state_name):
        
        # FILTERING FOR THE GIVEN STATE - BY DEFAULT GIVING TOTAL
        state_code_df_mod = state_code_df.copy()
        state_code_df_filt = state_code_df_mod[state_code_df_mod.state.str.lower() == state_name]
        
        state_lat = float(state_code_df_filt.latitude.values[0])
        state_long = float(state_code_df_filt.longitude.values[0])
        state_code = "tt" if state_name == "total" else state_code_df_filt.index[0]
        return state_code, state_lat, state_long

class VectorAutoRegression(DataLoader):
    
    def __init__(self, df, forecast_days):
        
        self.train = df
        self.forecast_days = forecast_days
        self.test_days = 0
        
        # START AND END DATES OF FORECASTING
        self.start_date = self.train.index[-1] + datetime.timedelta(1)
        self.end_date = self.start_date + datetime.timedelta(self.test_days+self.forecast_days-1)      # forecasting for test data + num_days in future
        
        # LOG TRANSFORMATION
        self.train_log = np.log1p(self.train)
        
        # MAKING STATIONARY BY DIFFERENCING
        self.train_diff1 = self.train_log.diff(1).dropna()
        self.train_diff2 = self.train_diff1.diff(1).dropna()
        
    def fit(self):
        
        # STATIONARITY CHECK
        # self.adfuller_test(self.train_diff2.deaths)   
        # self.adfuller_test(self.train_diff2.confirmed)
        # self.adfuller_test(self.train_diff2.recovered)
        
        # MODELING
        model = VAR(self.train_diff2)  # VAR model
        
        # SELECTING ORDER p FOR VAR MODEL
        # x = model.select_order(maxlags=12)
        # x.summary()
        
        # FITTING THE MODEL
        res = model.fit(4)                                 # ORDER = 4 FOR MINIMUM AIC
        return res
    
    def predict(self, res):
        
        # INITIAL INPUT FOR FORECAST
        lag_order = res.k_ar                               # 4 as shosen above
        forecast_input = self.train_diff2.values[-lag_order:]   # Taking last 4 training data as initial input for forecasting
        
        # FORECASTING
        num_days_forecast = self.test_days+self.forecast_days                    # forecasting for test data + num_days in future
        pred = res.forecast(forecast_input, num_days_forecast)  # numpy array of forecast values
        idx = pd.date_range(start=self.start_date, end=self.end_date) # date range for forecast data
        pred_df = pd.DataFrame(pred, index=idx, columns=self.train_diff2.columns + '_2d') # converintg to dataframe
        
        # INVERSE TRANSFORMATION
        pred_df_inv1 = self.invert_transformation(pred_df,self.train_log, self.train_diff1, self.train_diff2, second_diff=True, third_diff=False)      # inverting 2nd diff to 1st diff
        pred_df_inv = self.invert_transformation(pred_df_inv1,self.train_log, self.train_diff1, self.train_diff2, second_diff=False, third_diff=False) # inverting 1st diff and taking inverse log tr
        
        # SELECTING REQUIRED COLUMNS
        df_forecast = pred_df_inv.iloc[:,-3:]
        df_forecast.columns = self.train_diff2.columns
        
        # EVALUATING PREDICTIONS
        # test_f = test.join(df_forecast, rsuffix='_f')
        # mape(test_f.deaths_f, test_f.deaths)
        # mape(test_f.confirmed_f, test_f.confirmed)
        # mape(test_f.recovered_f, test_f.recovered)
        
        # COMBINING TRAIN WITH FORECAST DATA
        new_cases_forecast = pd.concat([self.train, df_forecast], axis = 0)
        new_cases_forecast['active'] = new_cases_forecast['confirmed'] - new_cases_forecast['deaths']- new_cases_forecast['recovered']
        total_cases_forecast = new_cases_forecast.cumsum()
        
        # COMBINING 
        final_forecast = pd.concat([total_cases_forecast,new_cases_forecast], axis = 1).reset_index()
        final_forecast.columns = ['Date'] + new_cases_forecast.columns.tolist() + [col + '_incr' for col in new_cases_forecast.columns]
    
        final_forecast['Date'] = final_forecast.Date.astype(str)
        final_forecast = self.calculate_growth_rate(final_forecast)
        return final_forecast
    
    def adfuller_test(self, series):
        """
        Parameters
        ----------
        series : a time series to check stationarity for
        Null Hypothesis H0 : Series is not stationary

        Returns
        -------
        p_value : Probability that null hypothesis is true

        """
        r = adfuller(series, autolag='AIC')
        output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
        p_value = output['pvalue']
        return p_value,p_value <= 0.05
    
    def invert_transformation(self, df_forecast, train_log, train_diff1, train_diff2, second_diff, third_diff):
        """
        Revert back the differencing to get the forecast to original scale.
        """
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
    
    
class CompartmentalModel(DataLoader):
    
    def __init__(self, df, country, forecast_days):
        
        self.train = df
        self.country = country
        self.forecast_days = forecast_days
        self.test_days = 0
        
        # START AND END DATES OF FORECASTING
        self.start_date = self.train.index[-1] + datetime.timedelta(1)
        self.end_date = self.start_date + datetime.timedelta(self.test_days+self.forecast_days-1)      # forecasting for test data + num_days in future
        
        # LOG TRANSFORMATION
        # self.train_log = np.log1p(self.train)
        
    def get_population_data(self):
        """
        Population Size for the Selected Country
        """
        dfp = pd.read_csv("population.csv")
        N = dfp.loc[dfp['Country (or dependency)'].str.lower() == self.country, 'Population (2020)'].values[0]
        
        return N
    
    def get_initial_guess_for_hyperparams(self):
        # TUNING PARAMETERS
        beta = 0.001 # transmission_rate
        gamma = 0.001 # removal_rate
        cfr = 0.2 # case_fatality_rate
        initial_guess = (beta, gamma, cfr)
        return initial_guess
    
    def get_sir_model_data(self):
        
        # Population data for the country
        N = self.get_population_data()
        df = self.train
        
        # Filtering TS data for the country
        df = df[df.region.str.lower() == self.country].iloc[:,:3]
        
        # Creating S, I, R series data
        df['removed'] = df['deaths'] + df['recovered']
        df['infected'] = df['confirmed'] - df['removed']
        df['susceptible'] = N - df['infected'] - df['removed']
        
        # Selecting required cols and normalizing
        dfm = df.iloc[:,3:] / N
        return dfm
    
    def get_initial_condition(self, df):
        # Initial Conditions
        y_0 = df.values[0]
        
        return y_0.tolist()
    
    def fit(self):
        
        sir_df = self.get_sir_model_data()
        print(sir_df.head())
        y_0 = self.get_initial_condition(sir_df)
        print(y_0)
        params_0 = self.get_initial_guess_for_hyperparams()
        print(params_0)
        optimal = optimize.minimize(
            self.loss,
            params_0[:-1],
            args=(sir_df, y_0),
            method='L-BFGS-B',
            bounds=[(0.00000001, 0.4), (0.00000001, 0.4)]
        )
        beta, gamma = optimal.x
        return optimal
        
    def loss(self, params, df, y_0):
        
        size = len(df)
        
        def sir_model(t, y):
            """
            # Model Definition
            """
            # Unpacking variables and parameters
            r, i, s = y
            beta, gamma = params
            
            # Differential Equations of SIR Model
            dsdt = -beta * s * i
            didt = beta * s * i - gamma * i
            drdt = gamma * i
            return [drdt, didt, dsdt]
        
        # Solving Initial Value Problem - Uses Runge-Kutta RK45 Algorithm
        solution = integrate.solve_ivp(sir_model, [0, size], y_0, t_eval=np.arange(0, size, 1), vectorized=True)
        
        # Calculate MSLE Loss
        l1 = mean_squared_log_error(df.infected, solution.y[1])
        l2 = mean_squared_log_error(df.removed, solution.y[0])
        
        # Put more emphasis on recovered people
        alpha = 0.1
        msle_weighted = alpha * l1 + (1 - alpha) * l2
        print(f"MSLE Infected :{l1}, MSLE Removed :{l2}, MSLE Average :{msle_weighted}")
        return msle_weighted
        
        