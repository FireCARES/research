import psycopg2
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from psycopg2.extras import RealDictCursor
from shapely.geometry import Point
from shapely import wkb
from shapely import geometry
import json
import pygeohash as pgh

state = 'VA'
fdid = '76000'
firecares_id = '93345'

#We take an arbitrary point in the AVL data frame (from Tyler G.) to create the example input
current_idx = 6303

#Specifying the number of incidents to sample for estimating the distribution of locations
n_samples = 1000

#Specifying the "covered" travel time in minutes. Model will try to maximize the incident_weighted area that is within covered_time
#of an available unit

covered_time = 4

#Specifiying the unit status tags from the AVL data that indicate available (per Tyler G.)
available_tags = ['AQ', 'AV', 'AM']

#############################################################################
"""
Building the unit_status component of the input

Note, we're only pulling in engines by using the unit id. If a unit id begins with an E, we use it. 
Again, this is just to build an example of an input to the model. The model does not make this assumption.

"""  
unit_status = [] #List of dictionaries

avl_df_raw = pd.read_csv('avl_data.csv')
avl_df = avl_df_raw.rename(columns={'Column1':'timestamp', 'Column5': 'unit_id', 'Column17':'unit_status',
                                   'Column7':'longitude', 'Column6':'latitude'})
avl_df = avl_df[['timestamp','unit_id','unit_status','longitude','latitude']]

#Isolating only units that are engines
isengine = [(True if i[0] == 'E' else False) for i in avl_df['unit_id'].unique()]
engine_list = avl_df['unit_id'].unique()[isengine]
avl_df = avl_df[avl_df['unit_id'].isin(engine_list)].reset_index(drop=True)

#Getting a list of all units by retrieving the distinct unit ids from the AVL data
unit_list = avl_df['unit_id'].unique()

#Going to the "current_idx" location in the AVL data, and then getting the most recent status of each unit
for i,unit in enumerate(unit_list):
    
    #Add a dictionary to the unit_status list
    unit_status.append({})
    unit_status[i]['unit_id'] = unit
    
    #Gives the index in the dataframe of the most recent update for that unit (before current_idx)
    most_recent_idx = np.where(avl_df['unit_id'].iloc[:current_idx] == unit)[0][-1]
    most_recent_info = avl_df.iloc[most_recent_idx]
    
    #Adding the unit's most recent location as a geohash
    current_loc = pgh.encode(most_recent_info['latitude'], most_recent_info['longitude'])
    unit_status[i]['current_location'] = current_loc
    
    #Adding the most recent availablity status
    status = most_recent_info['unit_status']
    if status in available_tags:
        unit_status[i]['status'] = 'AVAILABLE'
    else:
        unit_status[i]['status'] = 'UNAVAILABLE'

#############################################################################
"""
Building the station_status component of the input
"""

station_status = [] #List of dictionaries

#Query station locations from FireCARES table
q = """
select fs.station_number, addr.geom
from firestation_firestation fs
inner join firecares_core_address addr
on addr.id = fs.station_address_id
inner join firestation_firedepartment fd
on fd.id = fs.department_id
where fd.fdid = '""" + fdid + """'
and fd.state = '"""+state+"""'
"""

nfirs = psycopg2.connect(service='nfirs')
with nfirs.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(q)
    items=cur.fetchall()
    station_df = pd.DataFrame(items)

station_df['geom'] = station_df['geom'].apply(lambda x: wkb.loads(x,hex=True))

#Iterating through query results and saving them to station_status
for i, row in station_df.iterrows():
    station_status.append({})
    station_status[i]['station_id'] = row['station_number']
    
    #Converting the shapely point to a geohash
    station_status[i]['location'] = pgh.encode(row['geom'].coords.xy[1][0],
                                               row['geom'].coords.xy[0][0])

#############################################################################
"""
Building the incident distribution component of the input
"""

incident_distribution = [] #List of strings (geohashes)

#Performing Elasticsearch query to get all incidents from NFORS
es = Elasticsearch()
s = Search(using=es,index='*-fire-incident-*')
response = s.source(['description.incident_number',
                    'fire_department.firecares_id',
                    'address.latitude',
                    'address.longitude']).query('match',fire_department__firecares_id=firecares_id)

results_df = pd.DataFrame((d.to_dict() for d in response.scan()))
json_struct = json.loads(results_df.to_json(orient="records"))
df_flat = pd.io.json.json_normalize(json_struct)

#Taking the first entry from each incident number (we queried the apparatus data)
incident_df = df_flat.groupby('description.incident_number').first().reset_index()

#Then taking a random sample of incidents
incident_sample = incident_df.sample(n_samples).reset_index(drop=True)

#Iterating through the dataframe to get a list of incident geohashes
for i, row in incident_sample.iterrows():
    incident_distribution.append(pgh.encode(row['address.latitude'], row['address.longitude']))


#Finally converting everything to a json and saving it
example_input = {}
example_input['covered_time'] = covered_time
example_input['unit_status'] = unit_status
example_input['station_status'] = station_status
example_input['incident_distribution'] = incident_distribution

with open('example_input.json', 'w') as fp:
    json.dump(example_input, fp, indent=4)