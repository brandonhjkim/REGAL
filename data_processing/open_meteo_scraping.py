# Open Meteo API Scraping

import pandas as pd 
import requests
import time
from datetime import datetime 
import requests_cache
from retry_requests import retry
import openmeteo_requests

# read in data
df = pd.read_csv('cleaned_mra.csv')
# Important data for API scraping 
only_useful = df[['Longitude', 'Latitude', 'Date']]

# setting up API 
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
url = "https://customer-archive-api.open-meteo.com/v1/archive"

# 1 massive loop 
for row in only_useful.itertuples(index=False):
	params = {
		"latitude": row.Latitude,
		"longitude": row.Longitude,
		"start_date": str(row.Date.date()),
		"end_date": str(row.Date.date()), 
		"daily": ["weather_code", "apparent_temperature_max", "sunset", "rain_sum", "wind_gusts_10m_max", "temperature_2m_max", "apparent_temperature_min", "snowfall_sum", "wind_direction_10m_dominant", "daylight_duration", "temperature_2m_min", "apparent_temperature_mean", "sunshine_duration", "precipitation_hours", "shortwave_radiation_sum", "temperature_2m_mean", "sunrise", "precipitation_sum", "wind_speed_10m_max", "et0_fao_evapotranspiration"],
		"hourly": ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "boundary_layer_height", "wet_bulb_temperature_2m", "total_column_integrated_water_vapour", "is_day", "albedo", "sunshine_duration", "snow_depth_water_equivalent", "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_100_to_255cm", "soil_moisture_28_to_100cm"],
		"timezone": "GMT",
		"apikey": "" # replace with your own 
        }
	response = openmeteo.weather_api(url, params=params)[0]

	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
	hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
	hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
	hourly_boundary_layer_height = hourly.Variables(3).ValuesAsNumpy()
	hourly_wet_bulb_temperature_2m = hourly.Variables(4).ValuesAsNumpy()
	hourly_total_column_integrated_water_vapour = hourly.Variables(5).ValuesAsNumpy()
	hourly_is_day = hourly.Variables(6).ValuesAsNumpy()
	hourly_albedo = hourly.Variables(7).ValuesAsNumpy()
	hourly_sunshine_duration = hourly.Variables(8).ValuesAsNumpy()
	hourly_snow_depth_water_equivalent = hourly.Variables(9).ValuesAsNumpy()
	hourly_apparent_temperature = hourly.Variables(10).ValuesAsNumpy()
	hourly_precipitation = hourly.Variables(11).ValuesAsNumpy()
	hourly_rain = hourly.Variables(12).ValuesAsNumpy()
	hourly_snowfall = hourly.Variables(13).ValuesAsNumpy()
	hourly_snow_depth = hourly.Variables(14).ValuesAsNumpy()
	hourly_weather_code = hourly.Variables(15).ValuesAsNumpy()
	hourly_pressure_msl = hourly.Variables(16).ValuesAsNumpy()
	hourly_surface_pressure = hourly.Variables(17).ValuesAsNumpy()
	hourly_cloud_cover = hourly.Variables(18).ValuesAsNumpy()
	hourly_cloud_cover_low = hourly.Variables(19).ValuesAsNumpy()
	hourly_cloud_cover_mid = hourly.Variables(20).ValuesAsNumpy()
	hourly_cloud_cover_high = hourly.Variables(21).ValuesAsNumpy()
	hourly_et0_fao_evapotranspiration = hourly.Variables(22).ValuesAsNumpy()
	hourly_vapour_pressure_deficit = hourly.Variables(23).ValuesAsNumpy()
	hourly_wind_speed_10m = hourly.Variables(24).ValuesAsNumpy()
	hourly_wind_speed_100m = hourly.Variables(25).ValuesAsNumpy()
	hourly_wind_direction_10m = hourly.Variables(26).ValuesAsNumpy()
	hourly_wind_direction_100m = hourly.Variables(27).ValuesAsNumpy()
	hourly_wind_gusts_10m = hourly.Variables(28).ValuesAsNumpy()
	hourly_soil_temperature_0_to_7cm = hourly.Variables(29).ValuesAsNumpy()
	hourly_soil_temperature_7_to_28cm = hourly.Variables(30).ValuesAsNumpy()
	hourly_soil_temperature_28_to_100cm = hourly.Variables(31).ValuesAsNumpy()
	hourly_soil_temperature_100_to_255cm = hourly.Variables(32).ValuesAsNumpy()
	hourly_soil_moisture_0_to_7cm = hourly.Variables(33).ValuesAsNumpy()
	hourly_soil_moisture_7_to_28cm = hourly.Variables(34).ValuesAsNumpy()
	hourly_soil_moisture_100_to_255cm = hourly.Variables(35).ValuesAsNumpy()
	hourly_soil_moisture_28_to_100cm = hourly.Variables(36).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature_2m"] = hourly_temperature_2m
	hourly_data["dew_point_2m"] = hourly_dew_point_2m
	hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
	hourly_data["boundary_layer_height"] = hourly_boundary_layer_height
	hourly_data["wet_bulb_temperature_2m"] = hourly_wet_bulb_temperature_2m
	hourly_data["total_column_integrated_water_vapour"] = hourly_total_column_integrated_water_vapour
	hourly_data["is_day"] = hourly_is_day
	hourly_data["albedo"] = hourly_albedo
	hourly_data["sunshine_duration"] = hourly_sunshine_duration
	hourly_data["snow_depth_water_equivalent"] = hourly_snow_depth_water_equivalent
	hourly_data["apparent_temperature"] = hourly_apparent_temperature
	hourly_data["precipitation"] = hourly_precipitation
	hourly_data["rain"] = hourly_rain
	hourly_data["snowfall"] = hourly_snowfall
	hourly_data["snow_depth"] = hourly_snow_depth
	hourly_data["weather_code"] = hourly_weather_code
	hourly_data["pressure_msl"] = hourly_pressure_msl
	hourly_data["surface_pressure"] = hourly_surface_pressure
	hourly_data["cloud_cover"] = hourly_cloud_cover
	hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
	hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
	hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
	hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
	hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
	hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
	hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
	hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
	hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
	hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
	hourly_data["soil_temperature_0_to_7cm"] = hourly_soil_temperature_0_to_7cm
	hourly_data["soil_temperature_7_to_28cm"] = hourly_soil_temperature_7_to_28cm
	hourly_data["soil_temperature_28_to_100cm"] = hourly_soil_temperature_28_to_100cm
	hourly_data["soil_temperature_100_to_255cm"] = hourly_soil_temperature_100_to_255cm
	hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm
	hourly_data["soil_moisture_7_to_28cm"] = hourly_soil_moisture_7_to_28cm
	hourly_data["soil_moisture_100_to_255cm"] = hourly_soil_moisture_100_to_255cm
	hourly_data["soil_moisture_28_to_100cm"] = hourly_soil_moisture_28_to_100cm

	hourly_dataframe = pd.DataFrame(data = hourly_data).iloc[[row.Date.hour]]
	hourly_dataframe.drop(labels='date', axis='columns', inplace=True)

	daily = response.Daily()
	daily_weather_code = daily.Variables(0).ValuesAsNumpy()
	daily_apparent_temperature_max = daily.Variables(1).ValuesAsNumpy()
	daily_sunset = daily.Variables(2).ValuesAsNumpy()
	daily_rain_sum = daily.Variables(3).ValuesAsNumpy()
	daily_wind_gusts_10m_max = daily.Variables(4).ValuesAsNumpy()
	daily_temperature_2m_max = daily.Variables(5).ValuesAsNumpy()
	daily_apparent_temperature_min = daily.Variables(6).ValuesAsNumpy()
	daily_snowfall_sum = daily.Variables(7).ValuesAsNumpy()
	daily_wind_direction_10m_dominant = daily.Variables(8).ValuesAsNumpy()
	daily_daylight_duration = daily.Variables(9).ValuesAsNumpy()
	daily_temperature_2m_min = daily.Variables(10).ValuesAsNumpy()
	daily_apparent_temperature_mean = daily.Variables(11).ValuesAsNumpy()
	daily_sunshine_duration = daily.Variables(12).ValuesAsNumpy()
	daily_precipitation_hours = daily.Variables(13).ValuesAsNumpy()
	daily_shortwave_radiation_sum = daily.Variables(14).ValuesAsNumpy()
	daily_temperature_2m_mean = daily.Variables(15).ValuesAsNumpy()
	daily_sunrise = daily.Variables(16).ValuesAsNumpy()
	daily_precipitation_sum = daily.Variables(17).ValuesAsNumpy()
	daily_wind_speed_10m_max = daily.Variables(18).ValuesAsNumpy()
	daily_et0_fao_evapotranspiration = daily.Variables(19).ValuesAsNumpy()

	daily_data = {"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)}

	daily_data["weather_code_day"] = daily_weather_code
	daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
	daily_data["sunset"] = daily_sunset
	daily_data["rain_sum"] = daily_rain_sum
	daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
	daily_data["temperature_2m_max"] = daily_temperature_2m_max
	daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
	daily_data["snowfall_sum"] = daily_snowfall_sum
	daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
	daily_data["daylight_duration"] = daily_daylight_duration
	daily_data["temperature_2m_min"] = daily_temperature_2m_min
	daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
	daily_data["sunshine_duration"] = daily_sunshine_duration
	daily_data["precipitation_hours"] = daily_precipitation_hours
	daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
	daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
	daily_data["sunrise"] = daily_sunrise
	daily_data["precipitation_sum"] = daily_precipitation_sum
	daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
	daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration

	daily_dataframe = pd.DataFrame(data = daily_data)
	daily_dataframe.drop(labels='date', axis='columns', inplace=True)
		
	weather = pd.concat([hourly_dataframe.reset_index(drop=True).add_prefix('hour_'), daily_dataframe.reset_index(drop=True).add_prefix('day_')], axis=1)
	weather_vals = pd.concat([weather_vals, weather], ignore_index=True)
	
	i += 1
	print(i)

	time.sleep(1) 

# combining
final = pd.concat([df.reset_index(drop=True), weather_vals.reset_index(drop=True)], axis=1)

# saving 
final.to_csv('cleaned_mra_with_weather.csv', index=False)

