from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import pandas as pd
import json
import numpy as np
import pdb
from tqdm import tqdm
from datetime import datetime
from matplotlib import rcParams, cycler
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from itertools import cycle
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from shapely import wkb
from shapely.geometry import Point
import scipy as sp






class heatmap:
    def __init__(self, df,y_pred, boundary, b=.2,n=100, cutoff_num=None, cutoff_rad=None):
        """

        Initializes a heatmap class

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe containing a 'latitude' column, a 'longitude' column, and a column with the name of y_pred
        y_pred: str
            The name of the column of values that will be smoothed over (ex. 'travel_time')
        boundary: shapely polygon
            A polygon of the department's jurisdictional boundary
        b: float
            The bandwidth parameter
        n: int
            The grid resolution. Method will make an nxn grid for interpolation
        cutoff_num: int
            The number of incidents that must be within cutoff_rad of a point for the method to interpolate
        cutoff_rad: float
            The radius within which cutoff_num incidents must be located or else the method will not interpolate there


        Returns
        -------
        self.y_pred: array_like
            An array of predicted values
        self.density: array_like
            An array of the data density
        self.latlocs: array_like
            An array of the latitude coordinates
        self.longlocs: array_like
            An array of the longitude coordinates

        """
        min_long, min_lat, max_long, max_lat = boundary.bounds

        latlocs = np.linspace(min_lat,max_lat,n)
        longlocs = np.linspace(min_long,max_long,n)

        ylocs = (latlocs - min_lat)*69.0 #converting to miles
        xlocs = (longlocs-min_long)*69.0*np.cos(min_lat*np.pi/180)

        X = np.zeros((len(df),2))
        X[:,0] = (df['longitude']-min_long)*69.0*np.cos(min_lat*np.pi/180)
        X[:,1] = (df['latitude']- min_lat)*69.0 #converting to miles


        y = np.array(df[y_pred])


        q = np.zeros((n*n,2))
        coords = np.zeros((n*n,2))
        points = np.zeros(n*n)
        y_pred = np.ones(n*n)*np.nan
        density = np.ones(n*n)*np.nan
        coverage_index = np.ones(n*n)*np.nan



        xx,yy = np.meshgrid(xlocs,ylocs)
        q[:,0] = np.reshape(xx, n*n,order='f')
        q[:,1] = np.reshape(yy, n*n,order='f')
        long,lat = np.meshgrid(longlocs,latlocs)
        coords[:,0] = np.reshape(long, n*n,order='f')
        coords[:,1] = np.reshape(lat, n*n,order='f')
        points = np.array([Point(i[0],i[1]).within(boundary) for i in coords])

        distances = sp.spatial.distance_matrix(q,X)

        distance_slice = distances[points==1]

        q_slice = q[points==1,:]
        coords_slice = coords[points==1,:]

        #Currently assumes a 2D Gaussian kernel
        weights = 1/(2*np.pi*b**2)*np.exp(-.5*(distance_slice/b)**2)

        row_sums = weights.sum(axis=1)
        norm_weights = weights/row_sums[:,np.newaxis]
        values = y
        y_pred[points==1] = norm_weights@values
        density[points==1] = row_sums
        if cutoff_rad:
            cutoff_mat = distances < cutoff_rad
            cutoff_mat_sums = cutoff_mat.sum(axis=1)
            y_pred[cutoff_mat_sums < cutoff_num] = np.nan


        self.y_pred = y_pred
        self.density = density
        self.latlocs = latlocs
        self.longlocs = longlocs

class station:
    """

    A class to hold station level attributes and methods


    """
    def __init__(self,df,station_num):
        #Find the probability of that station being first due
        self.station_df = df[df['first_due_station']==station_num].copy()
        self.first_due_prob = len(self.station_df)/len(df)

        def my_agg(x):
            d = {
                'prob': len(x)/len(self.station_df), #probability of that many incidents being already in progress
                'dispatch_prob': np.sum(x['any_from_first_due'])/len(x),
                'count': len(x),
                'fd_units_active': x['fd_units_active'].iloc[0]
            }

            return pd.Series(d, index=d.keys())


        #Grouping incidents by how many incidents they have
        self.reliability = self.station_df.groupby('fd_units_active').apply(my_agg).reset_index(drop=True)


        #Creating the distributions for travel times
        self.sent_unit_times =  np.array(self.station_df[self.station_df['any_from_first_due'] == True]['travel_time'])

        self.no_unit_not_busy = np.array(self.station_df[(self.station_df['any_from_first_due'] == False)
                                              & ((self.station_df['first_due_busy'] == False))]['travel_time'])

        self.no_unit_and_busy = np.array(self.station_df[(self.station_df['any_from_first_due'] == False)
                                              & ((self.station_df['first_due_busy'] == True))]['travel_time'])

        if (len(self.sent_unit_times)==0) | (len(self.no_unit_not_busy) == 0) | (len(self.no_unit_and_busy)==0):
            self.first_due_prob = 0.0



    def timedraw(self):
        #Draw how many units are already out
        probs  = np.array(self.reliability['prob'])
        probs = probs/np.sum(probs)
        num_already_active =  np.random.choice(range(len(self.reliability)),p=probs)
        #Then determine probability of dispatching based on how many units are out
        dispatch_prob = self.reliability['dispatch_prob'].iloc[num_already_active]

        #Then draw whether that station will send a unit based on dispatch_prob
        if np.random.uniform() <= dispatch_prob:
            dispatched = True
        else:
            dispatched = False
        #If they send a unit
        if dispatched == True:
            return np.percentile(self.sent_unit_times, np.random.uniform()*100)
        #If they are not busy, but don't send unit
        elif dispatched == False and num_already_active == 0:
            return np.percentile(self.no_unit_not_busy, np.random.uniform()*100)
        #If they are busy and don't send a unit
        else:
            return np.percentile(self.no_unit_and_busy, np.random.uniform()*100)


class unit_analysis:

    """

    A class for department-level analysis of apparatus information.


    """

    def __init__(self,firecares_id, fdid, state, load_all=False,unit_type=None):

        """

        Initializes the unit_analysis class

        Parameters
        ----------
        firecares_id : str
            The firecares id string
        fdid: str
            The fire department id
        state: str
            The state abbreviation (ex. 'TX')
        load_all: (optional) bool
            If true, will load data from default folders


        """
        self.firecares_id = firecares_id
        self.fdid = fdid
        self.state = state
        self.boundary_query()
        self.unit_type = unit_type

        if load_all==True:
            self.apparatus_query(load_dir='./apparatus_df')
            self.first_due_analysis(load_dir='./first_due_df')
            self.station_maker()

    def boundary_query(self):

        """

        Queries the NFIRS table for the jurisdictional boundary polygon

        Returns
        -------
        self.boundary: shapely.polygon
            The polygon for the jurisdictional boundary
        self.verified: bool
            Whether or not the boundary has been verified

        """

        print("Acquiring deprartment's jurisdictional boundary...")
        q = """
        select * from firestation_firedepartment
        where fdid='"""+self.fdid+"""'
        and state='"""+self.state+"""'"""

        nfirs=psycopg2.connect('service=nfirs')
        with nfirs.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q)
            items=cur.fetchall()
            boundary_df = pd.DataFrame(items)

        self.boundary = wkb.loads(boundary_df['geom'].iloc[0], hex=True)
        self.verified =  boundary_df['boundary_verified']
        print("Done! \n")





    def apparatus_query(self, savedir='./apparatus_df',load_dir=None):
        """

        This method gives the unit_analysis object a dataframe that holds all the apparatus information.


        If no load directory is specified, it will generate an apparatus dataframe by first querying the incident
        data, and then unpacking the apparatus json for each incident. If a load directory is specified, 
        it will simply load the dataframe from a pickle whose name corresponds to the firecares_id + "apparatus_df."

        Parameters
        ----------
        save_dir: (optional) str
            The location to save the dataframe if it is generated from an incident query.
        load_dir: (optional)  str
            The directory from which to load an existing apparatus dataframe.

        Returns
        -------
        self.apparatus_df: (pandas.DataFrame)
            A dataframe containing all of the apparatus data merged with selected incident data

        """

        if not load_dir:
            #If not load directory is specified, generate the apparatus_df from incident data and save it
            print("Querying incident data from NFORS...")

            es = Elasticsearch()
            s = Search(using=es,index='*-fire-incident-*')
            response = s.source(['apparatus','address.latitude','address.longitude',
                                 'durations.travel.seconds','address.first_due',
                                'description.event_opened','description.event_closed','description.event_opened',
                     'weather.daily.precipIntensity',
                     'weather.daily.precipType',
                     'description.day_of_week',
                     'weather.daily.temperatureHigh',
                    'NFPA.type',
                  'weather.daily.windGust',
                  'weather.daily.visibility',
                  'weather.daily.uvindex',
                  'weather.daily.pressure',
                  'weather.daily.summary',
                  'weather.daily.windSpeed',
                  'weather.daily.cloudCover',
                  'weather.currently.humidity',
                  'weather.currently.dewPoint',
                  'weather.daily.precipAccumulation',
                  'description.comments',
                'fire_department.firecares_id']).query('match',fire_department__firecares_id=self.firecares_id)


            results_df = pd.DataFrame((d.to_dict() for d in response.scan()))
            json_struct = json.loads(results_df.to_json(orient="records"))
            df_flat = pd.io.json.json_normalize(json_struct)
            df_flat['incident'] = df_flat.index.values

            #Expand apparatus entry of df_flat so now we get a pd series of dataframes
            df_list = df_flat.apparatus.apply(pd.io.json.json_normalize)

            print("Done!")


            print("Unpacking the apparatus data...")
            #Because we haven't exploded the apparatus json yet, assigning the index will tell us which incident each unit corresponds to
            for i in range(len(df_list)):
                df_list[i]['incident'] = [i]*len(df_list[i])

            apparatus_df = pd.concat(df_list.to_list()).merge(df_flat, on='incident', sort=True)

            #Saving the dataframe to a pickle for speed
            apparatus_df.to_pickle(savedir + "/" + self.firecares_id+'apparatus')
            self.apparatus_df = apparatus_df
            print("Done!")

        else:
            #If a load_dir is specified, simply load the dataframe.
            self.apparatus_df = pd.read_pickle(load_dir+"/"+self.firecares_id+'apparatus')
        if self.unit_type:
            self.apparatus_df = self.apparatus_df[self.apparatus_df['unit_type'] == self.unit_type].reset_index(drop=True)

    def active_stations(self, start_var = 'unit_status.dispatched.timestamp', duration_var='extended_data.event_duration'):
        """

        This method creates a copy of the apparatus dataframe, removes necessary NaN rows, and then creates a new column
        for which each row is a list of all stations that had units active when the first unit of the incident was dispatched.

        Parameters
        ----------

        Returns
        -------
        self.apparatus_station: pandas.DataFrame
            A dataframe containing the apparatus data along with a list of all stations with units active when the first unit is dispatched. 

        """

        #Make a copy of the apparatus_df because we will manipulate it
        #Drop Nan entries for the two relevant timestamps
        self.apparatus_station = self.apparatus_df.dropna(subset=[start_var,duration_var]).copy()
        self.apparatus_station['station'] = self.apparatus_station['station'].apply(lambda x: str(x).zfill(3))

        #Converting the relevant times into timstamps rather than strings
        self.apparatus_station['start'] = self.apparatus_station.apply(lambda x: datetime.strptime(x[start_var][:-6], "%Y-%m-%dT%H:%M:%S"),axis=1)
        self.apparatus_station['end'] = self.apparatus_station['start'] + self.apparatus_station[duration_var].apply(lambda x: timedelta(seconds=x))

        #Sort by the start times
        self.apparatus_station = self.apparatus_station.sort_values('start').reset_index(drop=True)

        #Defining the start and end arrays. Numpy arrays are faster than pandas dataframes here
        end = np.array(self.apparatus_station['end'])
        start = np.array(self.apparatus_station['start'])

        #Call the method that makes the list of all active stations
        self.__inprogressatstart__(start,end)

    def __inprogressatstart__(self,start,end,list_size = 1000):

        """

        Runs an algorithm that determines all stations that had units active when the first unit of the incident was dispatched.

        Parameters
        ----------
        start: array_like
            An array of timestamps corresponding to the relevant start times (i.e. when a unit was dispatched)
        end: array_like
            An array of timestamps corresponding to the end of a unit's active period (i.e. when it is available)
        list_size: (optional) int
            The maximum number of units that could be active at the same time. DO NOT CHANGE THIS UNLESS YOU GET AN ERROR THAT "active_station_list" doesn't
            have enough columns.

        Returns
        -------
        self.apparatus_station['station_list']: array_like
            A list of lists of the stations that had units active when the unit was dispatched
        """

        print("Running the algorithm to find units active at the start of each incident")
        active_station_list = np.full( (len(self.apparatus_station),list_size), 'empty'    )
        for i in tqdm(range(len(end))):
            for j in range(i+1, len(end)):
                if end[i] >= start[j]:
                    active_station_list[j,np.where(active_station_list[j,:] == 'empty')[0][0]] = str(self.apparatus_station.iloc[i]['station'])
                else:
                    break

        self.apparatus_station['station_list'] = list(active_station_list)
        print('Done!')

    def first_due_analysis(self,save_dir='./first_due_df',load_dir=None):

        """

        Creates a dataframe called first_due_df where each row is an INCIDENT. It gives information about the location of the incident,
        whether or not the first due station responded to it, and how many units the first due station already had active when the
        first unit for the incident was dispatched. Also creates a dict containing relevant probabilities for the department.

        Note: The active_stations method must be run first if no load directory is specified.

        Parameters:
        ----------
        save_dir: (optional) str
            The directory to save the dataframe if it is to be generated
        load_dir:
            The directory from which to load the dataframe if it already exists 

        Returns
        -------
        self.first_due_df: pandas.DataFrame
            The dataframe giving information about the status of the first due station at the start of the incident (when the first unit was dispatched.)
        self.first_due_stats: dict
            Dictionary containing probabilities of certain types of incidents

        """

        if not load_dir:
            try: self.apparatus_station
            except AttributeError: self.active_stations()

            def my_agg(x_raw):
                x = x_raw.reset_index()
                first = x['start'].idxmin()
                first_due_station = str(x.iloc[first]['address.first_due']).zfill(3)
                d = {
                    'first_due_station': first_due_station,
                    'latitude': x.iloc[first]['address.latitude'],
                    'longitude': x.iloc[first]['address.longitude'],
                    'longitude': x.iloc[first]['address.longitude'],
                    'travel_time': x.iloc[first]['durations.travel.seconds'],
                    'any_from_first_due': np.any(x['station'] == first_due_station),
                    'fd_units_active':  np.sum(x.iloc[first]['station_list'] == first_due_station)
                }
                return(pd.Series(d, index=d.keys()))
            print("Acquiring information about status of first due station for each incident...")
            self.apparatus_station = self.apparatus_station.dropna(subset=['durations.travel.seconds'])
            self.first_due_df = self.apparatus_station.groupby('incident').apply(my_agg).reset_index()
            self.first_due_df.to_pickle(save_dir+"/"+self.firecares_id)
            print("Done!")
        else:
            self.first_due_df = pd.read_pickle(load_dir+"/"+self.firecares_id)

        #We say that a station is "busy" when it already has a unit active
        self.first_due_df['first_due_busy'] = self.first_due_df['fd_units_active'] > 0
        self.first_due_stats = {
                'first_due_dispatch': np.sum(self.first_due_df['any_from_first_due'])/len(self.first_due_df),
                'no_fd_and_busy': np.sum((self.first_due_df['any_from_first_due']==False)
                                            & self.first_due_df['fd_units_active']>0 )/len(self.first_due_df)
                }

    def first_due_plot(self,save_dir='./figures'):

        """

        Generates a cdf of the travel times for all incidents and for incidents for which the first due station sent at least one unit.
        This gives an idea of the maximum possible improvement from improving station reliability. Said differently, the idealized cdf
        represents the case for which the first due station is never so busy that it cannot respond (infinite units available).

        Parameters
        ----------
        savedir: string
            The directory to save the figure to


        """
        df = self.first_due_df
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['figure.figsize'] = [10,6]

        percentiles = np.linspace(0,100)


        plt.xlabel('Travel time (sec)')
        plt.ylabel('cdf')
        plt.xlim([0,800])


        plt.plot(np.percentile(df[df['any_from_first_due']==True]['travel_time'],percentiles),.01*percentiles, color='b')
        plt.plot(np.percentile(df[(df['any_from_first_due']==False) & (df['first_due_busy']==True)]['travel_time'],percentiles),.01*percentiles, color='r' )

        plt.ylim([0,1])
        plt.legend(['At least one unit dispatched from first-due station',
                   'First due busy and does not dispatch'])

        plt.savefig(save_dir+"/"+self.firecares_id+"_first_due_comparison")
        plt.show()
        change = np.percentile(df['travel_time'],90) - np.percentile(df[df['any_from_first_due']==True]['travel_time'],90)
        print("The expected improvement to the 90th percentile travel time is at most " + str(int(change)) + " seconds")

    def station_maker(self,save_dir='./station_reliability',load_dir=None):
        """

        Generates a list of station objects for the department and calculates the probability of each being first due.

        Note: self.first_due_analysis must be run first

        Parameters
        ----------
        savedir: str
            The location to save the station reliabilities

        Returns
        -------
        self.station_objects: array_like
            a list of station objects
        self.first_due_probs: array_like
            a list of the probability of each station being first due

        """

        stations = """
        SELECT FS.STATION_NUMBER, ADDR.GEOM
        FROM FIRESTATION_FIRESTATION FS
        INNER JOIN FIRECARES_CORE_ADDRESS ADDR ON ADDR.ID = FS.STATION_ADDRESS_ID
        INNER JOIN FIRESTATION_FIREDEPARTMENT FD ON FD.ID = FS.DEPARTMENT_ID
        WHERE FD.FDID = %(fdid)s AND FD.STATE = %(state)s"""

        nfirs = psycopg2.connect('service=nfirs')
        with nfirs.cursor(cursor_factory=RealDictCursor) as c:
                c.execute(stations, dict(fdid=self.fdid, state=self.state))
                items = c.fetchall()
                station_df = pd.DataFrame(items)



        station_df['geom'] = station_df['geom'].apply(lambda x: wkb.loads(x, hex=True))



        station_list = np.unique(self.first_due_df['first_due_station'])

        try:
            station_list = station_list[~np.isnan(station_list)]
        except TypeError:
            station_list = station_list[station_list != 'nan']

        #Make a list of station_objects
        self.station_objects = [None]*len(station_list)
        self.first_due_probs = np.zeros(len(station_list))

        for i,station_num in enumerate(station_list):
            self.station_objects[i] = station(self.first_due_df,station_num)
            try:
                self.station_objects[i].geom = station_df.iloc[np.where(station_df['station_number'] == station_num)[0][0]]['geom']
            except(IndexError):
                self.station_objects[i].geom = np.nan
            self.first_due_probs[i] = self.station_objects[i].first_due_prob
            if load_dir:
                self.station_objects[i].reliability = pd.read_csv(load_dir+'/'+self.firecares_id+'/station_'+str(int(station_num)))
            else:
                try:
                    self.station_objects[i].reliability.to_csv(save_dir+'/'+self.firecares_id+'/station_'+str(int(station_num)))
                except(FileNotFoundError):
                    os.mkdir(save_dir+ '/' + self.firecares_id)
                    self.station_objects[i].reliability.to_csv(save_dir+'/'+self.firecares_id+'/station_'+str(int(station_num)))
                except(ValueError):
                    self.station_objects[i].reliability.to_csv(save_dir+'/'+self.firecares_id+'/station_'+str(station_num))


        self.station_list = station_list
        #scaling the first_due probabilities so they sum to one
        self.first_due_probs = self.first_due_probs/np.sum(self.first_due_probs)

    def station_plot(self,cutoff=20,save_dir='./figures'):
        """

        Makes a plot of the reliability of each station. This means the probablity that it will dispatch to an incident in its first
        due area as a function of how many units it already has active.

        Parameters
        ----------
        cutoff: int
            The minimum number of incidents for the a data point to show up on the plot

        save_dir: str
            The directory to save the figure


        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.rcParams['figure.figsize'] = [16,10]
        plt.rcParams.update({'font.size': 13})
        sns.set_palette('tab20')
        sns.set_style('whitegrid')

        markers = ["X","*","o", "+","^","s","1"]
        lines = ["-","--","-."]
        markercycler = cycle(markers)
        linecycler = cycle(lines)
        legend_list = []


        #Plotting the relibility of each station
        for i in self.station_objects:
            plot_df = i.reliability[i.reliability['count'] > cutoff].copy()
            plt.plot(plot_df['fd_units_active'],plot_df['dispatch_prob'],linestyle=next(linecycler),
                    linewidth=2.0, marker=next(markercycler), markersize=12.0)

        plt.xlabel('Number of units already active')
        plt.ylabel('Probability of dispatching unit')
        legend_list = [None]*len(self.station_list)
        for i,station in enumerate(self.station_list):
            try:
                legend_list[i] = "Station "+str(int(station))
            except(ValueError):
                legend_list[i] = "Station "+str(station)

        plt.legend(legend_list,ncol=3)
        plt.xticks([0,1,2,3,4,5])
        plt.show()
        plt.savefig(save_dir+"/"+self.firecares_id+'_station_reliability')


    def time_simulator(self,num_iter=50000):
        """

        Simulates the travel time distribution for a hypothetical set of station reliabilities

        Parameters
        ----------
        num_iter: int
            Number of iterations

        """


        #Running the simulator
        times = np.zeros(num_iter)

        for i in range(num_iter):
            times[i] = np.random.choice(self.station_objects, p=self.first_due_probs).timedraw()

        #Response time cdf
        plt.rcParams.update({'font.size': 12})
        plt.rcParams['figure.figsize'] = [10,6]

        percentiles = np.linspace(0,100)


        plt.xlabel('Travel time (sec)')
        plt.ylabel('cdf')
        plt.xlim([0,800])


        plt.plot(np.percentile(times,percentiles),.01*percentiles, color='b')
        plt.plot(np.percentile(self.first_due_df['travel_time'],percentiles),.01*percentiles, color='r' )

        plt.ylim([0,1])
        plt.legend(['Simulated distribution',
                   'Actual distribution'])


        change = np.percentile(self.first_due_df['travel_time'],90) - np.percentile(times,90)
        print("The expected improvement to the 90th percentile travel time is " + str(int(change)) + " seconds")





