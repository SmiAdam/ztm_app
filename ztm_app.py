import json
from io import StringIO
import pandas as pd
from datetime import datetime
from time import sleep
import requests
import numpy as np
import os, psutil
from math import radians, cos, sin, asin, sqrt
from datetime import datetime

# process = psutil.Process(os.getpid())
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


my_station = [52.263922301047515, 21.038307838309706]
stations = {1: [52.300199185889674, 21.042236494667645],
            2: [52.25299421102346, 21.051558414657702]}
my_list = [0, 0]
line_list = ['3', '4', '25', '169']
stations_dict = {   '3': {1: [52.298580161956025, 21.02190249118404],
                    2: [52.23793866107101, 21.119464411156407]},
                    '4': {1: [52.311240967076756, 21.012447978650222],
                    2: [52.16795982115995, 21.016122775302918]},
                    '25': {1: [52.298580161956025, 21.02190249118404],
                    2: [52.20983089337748, 20.98208747692909]},
                    '169': {1: [52.300199185889674, 21.042236494667645],
                    2: [52.25299421102346, 21.051558414657702]},
                    }
apikey = '8fa64a24-ae3c-4d9c-ab79-7eb5eaa5ae37'
refresh_rate = 30
url = 'https://api.um.warszawa.pl/api/action/busestrams_get/?resource_id=%20f2e5503e-927d-4ad3-9500-4ab9e55deb59&apikey={}&type={}&line={}'


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    meters = round(km * 1000, 1)
    return meters


def most_common(lst):
    if len(lst)!=0:
        return max(set(lst), key=lst.count)
    else:
        return '???'


def rounduptomultiple(number, multiple):
    num = number + (multiple - 1)
    return num - (num % multiple)


def calculate_direction(c_lat, c_lon, v_m, previous_dirs, line):
    dist_m_1 = haversine(my_station[1], my_station[0], stations_dict[line][1][1], stations_dict[line][1][0])
    dist_m_2 = haversine(my_station[1], my_station[0], stations_dict[line][2][1], stations_dict[line][2][0])
    dist_t_1 = haversine(c_lon, c_lat, stations_dict[line][1][1], stations_dict[line][1][0])
    dist_t_2 = haversine(c_lon, c_lat, stations_dict[line][2][1], stations_dict[line][2][0])

    dir_m = v_m[0] - v_m[1]

    # if v_1[1] < 100 or v_2[1] < 100:
    if v_m[0] != 0:
        if dist_m_1 > dist_t_1:
            if dir_m < 0:
                previous_dirs.append('going to station 1')
            elif dir_m > 0:
                previous_dirs.append('going to station 2')
            else:
                pass
        elif dist_m_2 > dist_t_2:
            if dir_m < 0:
                previous_dirs.append('going to station 2')
            elif dir_m > 0:
                previous_dirs.append('going to station 1')
            else:
                pass
        else:
            pass
    try:
        if dist_t_1 < 50 and previous_dirs[-1] == 'stationary':
            previous_dirs = ['going to station 2', 'going to station 2', 'going to station 2', 'going to station 2', 'stationary']
        elif dist_t_2 < 50 and previous_dirs[-1] == 'stationary':
            previous_dirs = ['going to station 2', 'going to station 1', 'going to station 1', 'going to station 1', 'stationary']
    except IndexError:
        pass

    return previous_dirs


def main_app(df, df_new):
    # df_temp = pd.DataFrame.from_dict(ret['result'])
    df = pd.merge(df, df_new, on=["Lines", "VehicleNumber", 'Brigade'], how='outer')
    df = df[df['Lat'].notna()]
    df['line'] = df['Lines'].astype(str)

    for i, row in df.iterrows():
        dist_my = haversine(my_station[1], my_station[0], row['Lon'], row['Lat'])

        orig = row['time_pos']
        extra = {row['Time']: [row['Lat'], row['Lon'], 0, rounduptomultiple(dist_my, 250)]}

        df.at[i, 'lat'] = round(df.loc[i, 'Lat'], 3).astype(str)
        df.at[i, 'lon'] = round(df.loc[i, 'Lon'], 3).astype(str)

        # set after_me at 0 unless it actualy crossed by station
        if df.loc[i, 'after_me'] == 1:
            pass
        else:
            df.at[i, 'after_me'] = 0

        try:
            orig.update(extra)
            dist_s1 = haversine(stations_dict[df.loc[i, 'Lines']][1][1], stations_dict[df.loc[i, 'Lines']][1][0], list(orig.values())[-1][1], list(orig.values())[-1][0])
            dist_s2 = haversine(stations_dict[df.loc[i, 'Lines']][2][1], stations_dict[df.loc[i, 'Lines']][2][0], list(orig.values())[-1][1], list(orig.values())[-1][0])

            df.at[i, 'v_me'] = [row['v_me'][1], dist_my]
            df.at[i, 'v_s1'] = [row['v_s1'][1], dist_s1]
            df.at[i, 'v_s2'] = [row['v_s2'][1], dist_s2]

            try:
                dir = calculate_direction(row['Lat'], row['Lon'],df.loc[i, 'v_me'], df.loc[i, 'direction_list'], df.loc[i, 'Lines'])
            except IndexError:
                dir = calculate_direction(row['Lat'], row['Lon'],df.loc[i, 'v_me'], df.loc[i, 'direction_list'], df.loc[i, 'Lines'])

            df.at[i, 'direction_list'] = dir
            df.loc[i, 'direction'] = most_common(dir)

        # except for new buses:
        except AttributeError:
            df.at[i, 'time_pos'] = extra
            df.loc[i, 'direction'] = 'initiated'
            df.at[i, 'direction_list'] = []
            df.at[i, 'v_me'] = my_list
            df.at[i, 'v_s1'] = my_list
            df.at[i, 'v_s2'] = my_list

        # here the events:
        # 1: check if bus is actualy moving (by significant amount)
        if abs(df.loc[i, 'v_me'][0]-df.loc[i, 'v_me'][1]) < 5 and (df.loc[i, 'v_s1'][1] < 100 or df.loc[i, 'v_s2'][1] < 100):
            df.loc[i, 'direction'] = 'stationary'

        # 2: check if the bus just finished his route
        if (df.loc[i, 'v_s1'][0] > 100 > df.loc[i, 'v_s1'][1]) or (df.loc[i, 'v_s2'][0] > 100 > df.loc[i, 'v_s2'][1]):
            df.loc[i, 'direction'] = 'just finished'

        # 3: check if the bus just crossed my station
        if df.loc[i, 'v_me'][0] > 50 > df.loc[i, 'v_me'][1]:
            extra2 = {row['Time']: [row['Lat'], row['Lon'], 1, rounduptomultiple(dist_my, 250)]}
            orig.update(extra2)
            # set market when the tram is already past my station
            df.at[i, 'after_me'] = 1

            # upload path:
            if len(dict(df.loc[i, 'time_pos']))>5:
                for d_k, d_i in dict(df.loc[i, 'time_pos']).items():
                    if d_i[2] == 1:
                        my_time = d_k

                # calculate time differences between my stop and points along the line
                try:
                    tt = pd.read_csv('timetable.csv')
                    print(dict(df.loc[i, 'time_pos']))
                    for d_k, d_i in dict(df.loc[i, 'time_pos']).items():
                        time_diff = (datetime.strptime(my_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(d_k, "%Y-%m-%d %H:%M:%S")).total_seconds()
                        lat = round(d_i[0], 3)
                        lon = round(d_i[1], 3)
                        if time_diff > 0 and d_i[3] != 0:
                            if len(tt.query("lat == {} and lon == {} and line == {} and direction == '{}'".format(lat, lon, df.loc[i, 'Lines'], df.loc[i, 'direction']))) != 0:
                                time_diff_old = max(tt.loc[tt.query("lat == {} and lon == {} and line == {} and direction == '{}'".format(lat, lon, df.loc[i, 'Lines'], df.loc[i, 'direction'])).index,'time_diff'])
                                n_old = max(tt.loc[tt.query("lat == {} and lon == {} and line == {} and direction == '{}'".format(lat, lon, df.loc[i, 'Lines'], df.loc[i, 'direction'])).index,'n'])

                                if n_old > 5 and time_diff > 2 * time_diff_old:
                                    pass
                                else:
                                    tt.loc[tt.query("lat == {} and lon == {} and line == {} and direction == '{}'".format(lat, lon, df.loc[i, 'Lines'], df.loc[i, 'direction'])).index,'time_diff'] = (tt['time_diff'] * tt['n'] + time_diff)/(tt['n'] + 1)
                                    tt.loc[tt.query("lat == {} and lon == {} and line == {} and direction == '{}'".format(lat, lon, df.loc[i, 'Lines'], df.loc[i, 'direction'])).index,'n'] = tt['n'] + 1

                            else:
                                tt = pd.concat([tt, pd.DataFrame.from_records([{'time_diff': time_diff, 'lat': lat, 'lon': lon, 'line': df.loc[i, 'line'], 'direction': df.loc[i, 'direction'], 'n': 1}])], ignore_index=True)

                    tt = tt.groupby(['lat', 'lon', 'direction', 'line']).agg(time_diff=('time_diff', np.mean), n=('n', np.sum))
                    tt = tt.reset_index()
                    print('uploading data')
                    tt.to_csv('timetable.csv', index=False)
                except NameError:
                    pass

    df.drop(['Lon', 'Lat', 'Time'], axis=1, inplace=True)
    df = df[~df['direction'].isin(['just finished'])]

    return df


def get_responses(line_list, main_df, init_flag):
    df_temp = pd.DataFrame()
    for line in line_list:
        if len(line) < 3:
            bus_tram = 2
        else:
            bus_tram = 1
        response = requests.get(url.format(apikey,bus_tram,line))

        js = json.load(StringIO(response.text))
        if len(str(js)) > 150:
            df_temp = pd.concat([df_temp, pd.DataFrame.from_dict(js['result'])], ignore_index=True)
        sleep(1)

    if init_flag == 1:
        main_df = df_temp
    else:
        main_df = main_app(main_df, df_temp)

    return main_df


def initialize(line_list):
    init_df = get_responses(line_list, '', 1)
    init_df['time_pos'] = ''
    init_df['lat'] = 0
    init_df['lon'] = 0
    init_df['after_me'] = 0
    init_df['time_diff'] = np.NaN
    init_df['direction'] = '???'

    init_df['v_me'] = init_df.apply(lambda _: my_list, axis=1)
    init_df['v_s1'] = init_df.apply(lambda _: my_list, axis=1)
    init_df['v_s2'] = init_df.apply(lambda _: my_list, axis=1)
    init_df['direction_list'] = init_df.apply(lambda _: [], axis=1)

    for i, row in init_df.iterrows():
        init_df.at[i, 'time_pos'] = {row['Time']: [row['Lat'], row['Lon'], 0, 0]}
    init_df.drop(['Lon', 'Lat', 'Time'], axis=1, inplace=True)

    return init_df


def prepare_display_df(input_df):
    tt = pd.read_csv('timetable.csv')
    tt['lon'] = round(tt['lon'], 3).astype(str)
    tt['lat'] = round(tt['lat'], 3).astype(str)
    tt['line'] = tt['line'].astype(str)

    try:
        input_df.drop(['last_time_diff'], axis=1, inplace=True)
    except KeyError:
        pass

    input_df.rename(columns={'time_diff': 'last_time_diff'}, inplace=True)
    input_df = pd.merge(input_df, tt[['line', 'lat', 'lon', 'direction', 'time_diff']],
                       left_on=['line', 'lat', 'lon', 'direction'], right_on=['line', 'lat', 'lon', 'direction'],
                       how='left')

    input_df.time_diff.fillna(input_df.last_time_diff - 35, inplace=True)
    input_df['display_time'] = round(input_df['time_diff'] / 60, 1).astype(str) + " minutes"

    return input_df


def display_controller(df_display, direction_flag):
    if direction_flag == 1:
        title = 'BRÓDNOWSKA 01 - kier. CENTRUM'
    else:
        title = 'BRÓDNOWSKA 02 - kier. BRÓDNO'

    for i, row in df_display.iterrows():
        print(row)


if __name__ == '__main__':
    init_df = initialize(line_list)
    row_controller = 0
    while True:
        down_date = datetime.now()
        init_df = get_responses(line_list, init_df, 0)
        init_df = prepare_display_df(init_df)
        # display_controller(init_df, 1)
        print(init_df.query("direction == 'going to station 2' and time_diff>=0 and after_me==0")[['line', 'VehicleNumber', 'Brigade', 'time_diff', 'display_time']].sort_values(by=['time_diff']))
        up_date = datetime.now()
        sleep(refresh_rate-(up_date - down_date).total_seconds())
