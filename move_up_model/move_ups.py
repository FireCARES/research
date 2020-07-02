import pygeohash as pgh
import json
from tqdm import tqdm
from shapely import geometry
from shapely import wkb
from shapely.geometry import Point
from shapely.geometry import Polygon
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from matplotlib.path import Path
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncio
import aiohttp
import os
from shapely.ops import cascaded_union


class move_up_model:
    """
    Class that holds methods for issuing movement recommendations of available units
    """

    def __init__(self, input_data, fdid, state):
        # Assigning the parameters in the input data to the class
        for key in input_data.keys():
            setattr(self, key, input_data[key])
        self.fdid = fdid
        self.state = state
        
        #Getting long/lat coordinates out of incident distribution
        self.incident_coords = np.zeros((len(self.incident_distribution), 2))
        for i, location in enumerate(self.incident_distribution):
            self.incident_coords[i,:] = np.flip(pgh.decode(location))

        # Getting jurisdictional boundary
        self.get_boundary()

        # Counting number of units currently available
        self.count_available()

        # Getting coverage polygons of currently available units
        self.get_unit_coverage_paths()

        # Getting station coverage polygons
        self.get_station_coverage_paths()

        # Assigning each incident to coverage polygons
        self.build_sets()

        # Finding ideal set of stations
        self.max_coverage()

        # Calculating optimial movement strategy
        self.balanced_assignment()

        # Generating json output
        self.output_recommendations()

    def count_available(self):
        """
        Counts how many units are available in the unit status section of input json
        
        Attributes
        ----------
        self.num_available: int
            Number of units currently available
        """

        self.num_available = 0
        for status in self.unit_status:
            if status['status'] == 'AVAILABLE':
                self.num_available += 1

    def get_station_coverage_paths(self):
        """
        Using the station location geohashes, creates a dictionary of "coverage polygons" for the stations.
        For each station, the coverage polygon is the set of locations that is within a self.coverage_time 
        travel time from the station. Also compute the centroid of stations as a reference point.
        
        Attributes
        ----------
        self.station_coverage_paths: dict
            Dictionary of shapely polygons describing the coverage polygons for each station
        self.station_locs: list
            List of long/lat coordinates for each station
        self.station_list: list
            List of station ids
        """

        self.station_coverage_paths = {}
        self.station_coverage_polys = {}
        self.station_list = []  # A list of all station ids
        self.station_locs = []  # A list of all station locations

        for i, status in enumerate(tqdm(self.station_status)):
            lat, long = pgh.decode(status['location'])
            self.station_locs.append([long, lat])
            self.station_list.append(status['station_id'])
            
            
        polygons = self.drivetime_poly(self.station_locs, self.covered_time)
        for i,station in enumerate(self.station_list):
            self.station_coverage_paths[station] = polygons[i]
            self.station_coverage_polys[station] = Polygon(polygons[i].to_polygons()[0]).buffer(0)

        #Computing centroid of all stations so we can convert distances to miles 
        self.station_centroid = [np.mean(np.array(self.station_locs)[:,0]), 
                              np.mean(np.array(self.station_locs)[:,1])]
            
    def get_unit_coverage_paths(self):
        """
        Using the locations of currently available units, get corresponding coverage polygons and compute
        fraction of incidents that are within the coverage polygon of a unit 
        
        Attributes
        ----------
        self.unit_coverage_paths: dict
            Dictionary of shapely polygons describing the coverage polygons for each currently available unit   
        self.unit_locs: list
            List of long/lat coordinates for each available unit
        self.unit_list: list
            List of available unit ids
        """

        self.unit_coverage_paths = {}
        self.unit_coverage_polys = {}
        self.unit_list = []  # A list of all available unit ids
        self.unit_locs = []  # A list of all current locations of available units
        self.current_unionized_poly = Polygon() #Unionized coverage polygon
        
        for i, status in enumerate(tqdm(self.unit_status)):
            if status['status'] == 'AVAILABLE':
                lat, long = pgh.decode(status['current_location'])
                self.unit_locs.append([long, lat])
                self.unit_list.append(status['unit_id'])
                
                
                
        
        polygons = self.drivetime_poly(self.unit_locs, self.covered_time)
        for i,unit in enumerate(self.unit_list):
            self.unit_coverage_paths[unit] = polygons[i]
            self.unit_coverage_polys[unit] = Polygon(polygons[i].to_polygons()[0]).buffer(0)
                
        self.current_unionized_poly = cascaded_union(self.unit_coverage_polys.values())

    def drivetime_poly(self, loc_list, drivetime=4):
        """Generates a travel time polygon surrounding a location
        
        Params
        ------
        long: float
            The longitude of the point
        lat: float
            The latitiude of the point
        drivetime: float
            The travel time window (default 4 minutes)
            
        Returns
        -------
        poly_list: list
            List of matplotlib paths to describe drivetime geometry
        """
        
        token = os.environ['mapbox_key']

        #Make a list of isochrone request URLs
        urls = []
        poly_list = []
        for long, lat in loc_list:
            base_url = 'https://api.mapbox.com/isochrone/v1/mapbox/driving/'
            urls.append(base_url+"""{longitude},{latitude}?contours_minutes={contours_minutes}
                                        &polygons=true&access_token={token}""".format(longitude=long,
                                                                                    latitude = lat,
                                                                                    contours_minutes = drivetime,
                                                                                    token = token))
        async def fetch(session, url):
            async with session.get(url) as response:
                return await response.json()

        async def fetch_all(urls, loop):
            async with aiohttp.ClientSession(loop=loop) as session:
                results = await asyncio.gather(*[fetch(session, url) for url in urls], return_exceptions=True)
                return results

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(fetch_all(urls, loop))
        for result in results:
            poly_list.append(Path(result['features'][0]['geometry']['coordinates'][0]))
            
        return poly_list

    def get_boundary(self):
        """
        Determines the jurisidictional boundary of the department
        
        Attributes
        ----------
        self.boundary: shapely.Polygon
            Polygon of the department's juridictional boundary
        """

        q = """
        select * from firestation_firedepartment
        where fdid='""" + self.fdid + """'
        and state='""" + self.state + """'"""

        nfirs = psycopg2.connect(service='nfirs')
        with nfirs.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q)
            items = cur.fetchall()
            boundary_df = pd.DataFrame(items)

        self.boundary = wkb.loads(boundary_df['geom'].iloc[0], hex=True)

    def build_sets(self):
        """
        Makes a set of incidents for each station and available unit in the department. 
        These sets contain the incidents from the sample that are within a drivetime polygon from the station.
        Note, incidents can be within multiple polygons.
        
        Attributes
        ----------
        self.station_subsets: dict
            Dictionary of sets. The keys are the station ids. The sets contain the incidents within
            the specified drivetime of that station. Incidents can be in multiple sets because of overlaps.
        self.unit_subsets: dict
            Dictionary of sets. The keys are the unit ids. The sets contain the incidents within
            the specified drivetime of that AVAILABLE unit. Incidents can be in multiple sets because of overlaps.   
        self.current_frac_covered: float
            Fraction of incidents within a coverage polygon of a currently available unit 
        """

        # Generate a set for each station that belongs to the department
        self.station_subsets = {}
        for status in tqdm(self.station_status):
            self.station_subsets[status['station_id']] = set(np.where(
                self.station_coverage_paths[status['station_id']].contains_points(self.incident_coords))[0])

        # Generate a set for each currently available unit
        self.unit_subsets = {}
        self.currently_covered = set()
        for available_unit in tqdm(self.unit_coverage_paths.keys()):
            incident_set = set(np.where(
                self.unit_coverage_paths[available_unit].contains_points(self.incident_coords))[0])
            # Make a list of sets at the aggregate level too
            self.unit_subsets[available_unit] = incident_set
            self.currently_covered |= self.unit_subsets[available_unit]
        self.current_frac_covered = len(self.currently_covered) / len(self.incident_distribution)

    def movement_improvement(self, unit, station):
        """
	    Calculates the improvement of individual moves. The improvement refers to the net change in the fraction of incidents
	    that are covered. Note that the overall improvement is not the sum of the individual improvements.
	    """
        covered = set()
        for key in self.unit_subsets.keys():
            if key == unit:
                covered |= self.station_subsets[station]
            else:
                covered |= self.unit_subsets[key]
        improvement = len(covered) - len(self.currently_covered)
        improvement = improvement / len(self.incident_distribution) * 100
        return improvement

    def max_coverage(self):
        """
        Uses the greedy algorithm to determine the set of stations that should have a unit
        given the current number of units available

        Attributes
        ----------
        self.ideal_stations: list
            List of ideal stations in order of fraction of ADDITIONAL incidents they cover
        self.moveup_frac_covered: float
            Fraction of incidents from the sample that are within the coverage polygon of at least one station in 
            the set of ideal stations. This is the estimated fraction covered if the recommended strategy is implemented.

        """
        self.covered = set()
        self.ideal_stations = []
        
        #Compute unionized polygon of all "ideal" stations
        poly_list = []
        used_all_stations = False
        for i in range(self.num_available):
            if not used_all_stations:
                # First make a list of stations that have not been added to self.ideal_stations
                remaining_stations = [station for station in list(self.station_subsets.keys()) if
                                      station not in self.ideal_stations]
                # Then add the station that has the most uncovered incidents
                append = max(remaining_stations, key=lambda idx: len(self.station_subsets[idx] - self.covered))
                self.ideal_stations.append(append)
                poly_list.append(self.station_coverage_polys[append])

                self.covered |= self.station_subsets[append]
                
                if len(remaining_stations) == 1:
                    #If we fill all the stations, then we're done
                    used_all_stations = True
                
        self.moveup_frac_covered = len(self.covered) / len(self.incident_distribution)
        self.optimized_unionized_poly = cascaded_union(poly_list)
    def miles_convert(self, locations):
        """
        Converts a list of lat/long pairs to x-y coordinates based on miles. The x coordinate is 
        the distance east of the centroid of stations and the y coordinate is the distance north of 
        the centroid of stations.

        Parameters
        ----------
        locations: array_like
            A list of lat/long pairs

        Returns
        -------
        mile_locs: numpy.array
            An array with the first column representing x coordinates in miles and the second column
            representing y coordinates in miles.
        """

        long_locs = np.array(locations)[:,0]-self.station_centroid[0]
        lat_locs = np.array(locations)[:,1]-self.station_centroid[1]
        mile_locs = np.zeros((len(locations),2))
        mile_locs[:,0] = (long_locs)*69.0*np.cos(self.station_centroid[1]*np.pi/180)
        mile_locs[:,1] = (lat_locs)*69.0 #converting to miles
        return mile_locs

    def balanced_assignment(self, exponent=1):
        """
        Determines which available unit should go to which available station
        At a later point- use miles instead of decimal degrees for distance 
        (shouldn't really matter, unless you're really far North or South)
        
        Even better- use travel time

        Attributes
        ----------
        self.distance_matrix: numpy.array
            matrix of distances (in decimal degrees) between current available unit locations and ideal stations
        self.movement_rec: list
            list of movement recommendations
        """

        ideal_station_locs = [self.station_locs[self.station_list.index(i)] for i in self.ideal_stations]
        ideal_stations_miles = self.miles_convert(ideal_station_locs)
        unit_locs_miles = self.miles_convert(self.unit_locs)

        self.distance_matrix = distance_matrix(np.array(unit_locs_miles), np.array(ideal_stations_miles))** exponent
        movements = linear_sum_assignment(self.distance_matrix)
        self.movement_rec = []
        for i in range(len(self.ideal_stations)):
            self.movement_rec.append({'unit': self.unit_list[movements[0][i]],
                                      'station': self.ideal_stations[movements[1][i]],
                                      'distance': self.distance_matrix[movements[0][i], movements[1][i]]})

    def output_recommendations(self):
        """
        Outputs recommendations into desired json format and saves the json

        Attributes
        ----------
        self.output: dict
            Nested dictionary following Joe Chop's example template
        """

        self.output = {}
        self.output['current'] = {}
        self.output['current']['metrics'] = {}
        self.output['current']['metrics']['percentage_under_4_minute_travel'] = self.current_frac_covered * 100
        self.output['current']['unoptimized_iso'] = geometry.mapping(self.current_unionized_poly)
        self.output['move_up'] = {}
        self.output['move_up']['strategy'] = 'maximize fraction of incidents within 4 minute travel time'
        self.output['move_up']['metrics'] = {}
        self.output['move_up']['metrics']['percentage_under_4_minute_travel'] = self.moveup_frac_covered * 100
        self.output['move_up']['optimized_iso'] = geometry.mapping(self.optimized_unionized_poly)
        self.output['move_up']['moves'] = []

        for rec in self.movement_rec:
            # Assumption is that if a unit id ends with the station id, then it's not a move
            if rec['distance'] > 0.01:
                append = {}
                append['unit_id'] = rec['unit']
                append['station'] = rec['station']
                append['distance'] = rec['distance']
                append['improvement'] = self.movement_improvement(rec['unit'], rec['station'])
                self.output['move_up']['moves'].append(append)

        with open('example_output.json', 'w') as fp:
            json.dump(self.output, fp, indent=4)
