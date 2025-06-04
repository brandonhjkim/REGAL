import pandas as pd 
import requests
from shapely.geometry import shape, Point
import numpy as np
import geopandas as gpd

# read in 
df = pd.read_csv('mra_wi_img.csv')

# dropping unimportant columns 
df.drop(columns=['globalid', 'Status'], axis=1, inplace=True)

# getting states via coords 
us_states_geojson = requests.get('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json').json()

states = [] 

for feature in us_states_geojson['features']:
    geom = shape(feature['geometry'])
    state_name = feature['properties']['name']

    if not geom.is_valid:
        geom = geom.buffer(0)

    states.append({
        'name': state_name,
        'geometry': geom
    })

def get_state(lat, lon):
    point = Point(lon, lat) 

    state_found = 'unknown'
    
    for state in states:
        if point.within(state['geometry']):
            state_found = state['name']
            break
    
    return state_found

df['state'] = df.apply(lambda row: get_state(row['Latitude'], row['Longitude']), axis = 1)

# cleaning up 
df['state'] = df['state'].replace('unknown', np.nan)
df['state'] = df['state'].str.upper()
df['state'] = df['state'].fillna(df['State'])

# cleaning up region
df['Region'] = df['Region'].fillna(df['state'] + ' REGION')

# getting counties
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])] 
gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

counties = gpd.read_file('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json')
counties = counties.to_crs(gdf_points.crs)

joined = gpd.sjoin(gdf_points, counties, how="left", predicate="intersects")
df['county'] = joined['NAME']

# cleaning up
df['county']=df['county'].str.upper() + ' COUNTY'
df['County'] = df['County'].apply(
    lambda x: f"{str(x)} COUNTY" if pd.notnull(x) and 'COUNTY' not in str(x).upper() else x
)
df['county'] = df['county'].fillna(df['County'])

# edge cases 
df['County'] = df['County'].str.replace(
    'LAS ANGELOS COUNTY', 'LOS ANGELES COUNTY', case=False
)

condition = df['state'].isna() & df['county'].notna()
df.loc[condition, 'state'] = 'CALIFORNIA'
df.loc[condition, 'Region'] = 'CALIFORNIA REGION'

# imputation
df['Number_Volunteers'].fillna(0, inplace=True)
df['Total_Hours'].fillna(df['Total_Hours'].mean(), inplace=True)
df['Age_Sub_1'].fillna(df['Age_Sub_1'].mean(), inplace=True)
df['Aircraft_Hours'].fillna(0, inplace=True)
df['Hoist_Used'].fillna('NO', inplace=True)
df['Mutual_Aid'].fillna('NO', inplace=True)
df['hour_snow_depth'].fillna(0, inplace=True)

columns_to_fill = [
    'Team', 
    'Region', 
    'Area_Type',
    'Land_Ownership',
    'Category',
    'Age_Sub_1',
    'Gender_Sub_1',
    'Fitness_Sub_1',
    'Experience_Sub_1',
    'Mental_Factor_Sub_1',
    'Mental_Rating_Sub_1'
]

df[columns_to_fill] = df[columns_to_fill].fillna('UNKNOWN')

df['state'] = df['state'].fillna('NOT APPLICABLE')
df['county'] = df['county'].fillna('NOT APPLICABLE')

# saving 
df.to_csv('final_mra.csv', index=False)