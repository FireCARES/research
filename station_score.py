from backends import PostgresBackend
import pandas as pd
import psycopg2
import geopandas as gpd
from shapely import wkb
from shapely.geometry import Point
import simplejson
import urllib
import requests, json 
import numpy as np
import googlemaps
import os
 
api_key = os.environ['GOOGLE_MAPS_API_KEY']
gmaps = googlemaps.Client(key=api_key) 


class station_score():
	def __init__(self, fdid, state, sample_size=25):
		self.fdid = fdid
		self.state = state
		self.sample_size = sample_size
		self.df = self.incident_query()
		self.stations = self.station_query()
		self.station_calc()
		self.drive_times()
		self.scorer()

	def incident_query(self):
		#Querying the incidents and addresses from the department
		keys=['fdid', 'state', 'geom']
		query = """
		SELECT """ + ','.join(keys)+ """
		FROM incidentaddress
		where fdid = '""" + self.fdid + """'
		and state = '""" + self.state + """'
		limit """ + str(self.sample_size)

		with PostgresBackend(dict(service='nfirs')) as backend:
		    results = backend.query(query=query).fetchall()

		#storing all the values into a dictionary and then into a pandas dataframe
		d = {}
		for key, value in enumerate(keys):
			d[value] = [results[i][key] for i in range(len(results))]

		df = pd.DataFrame(d)
		df.dropna(inplace=True)

		df['geom'] = df['geom'].apply(lambda x: wkb.loads(x, hex=True))
		return df

	
	def station_query(self):
		#Querying the stations in the community
		keys = ['fs.station_number', 'addr.geom']
		query = """
		SELECT """ + ','.join(keys)+ """
		from firestation_firestation fs inner join firecares_core_address addr on addr.id = fs.station_address_id inner join firestation_firedepartment 
		fd on fd.id = fs.department_id 
		where fd.fdid = '""" + self.fdid + """' and fd.state = '""" + self.state + """'"""

		with PostgresBackend(dict(service='nfirs')) as backend:
			results = backend.query(query=query).fetchall()

		#storing all the values into a dictionary and then into a pandas dataframe
		d = {}
		for key, value in enumerate(keys):
			d[value] = [results[i][key] for i in range(len(results))]

		df = pd.DataFrame(d)
		df.dropna(inplace=True)
		df['geom'] = df['addr.geom'].apply(lambda x: wkb.loads(x, hex=True))
		return df


	def station_calc(self):
		#determines the nearest station for each incident in self.df
		nearest = [None]*len(self.df['geom'])

		#Iterate through list of incidents and append the nearest station geom (as the crow flies)
		for i,geom in enumerate(self.df['geom']):
			station_dist = np.ones(len(self.stations['geom'])) #holds the distances from the incident to all stations
			for j,station in enumerate(self.stations['geom']):
				station_dist[j] = geom.distance(station)

			#recording nearest station location
			nearest[i] = self.stations['geom'].iloc[np.argmin(station_dist)]

		#save results into the main dataframe
		self.df['nearest'] = nearest


	def drive_times(self):
		#Now that the dataframe is populated with geometries for incidents and the nearest station,
		#query Google for drive times 
		times = [None]*len(self.df)
		for i, incident in enumerate(self.df['geom']):
			origins = self.df['nearest'].iloc[i].coords[0][1], self.df['nearest'].iloc[i].coords[0][0]
			destinations = self.df['geom'].iloc[i].coords[0][1], self.df['geom'].iloc[i].coords[0][0]
			matrix = gmaps.distance_matrix(origins, destinations)
			times[i] = matrix['rows'][0]['elements'][0]['duration']['value']
			

		self.df['times'] = times

	def scorer(self):
		#Take the 90th percentile of the drive time distribution
		self.score = np.percentile(self.df['times'], 90)
		print self.score



station_score('WP801', 'TX')



