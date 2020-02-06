import googlemaps 
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.style as style
import seaborn as sns
import pickle
import pandas as pd
import pdb
import psycopg2
from shapely import wkb
from psycopg2.extras import RealDictCursor
from copy import deepcopy
from sklearn.neighbors import KDTree
from datetime import datetime
from scipy.stats import lognorm
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

#Make a class for the department, which will hold aggregate level information
class department:

    """

    A class for department-level analysis of apparatus information.


    """

    def __init__(self,firecares_id, fdid, state, load_all=False, unit_types=None, bad_units=None,p = (10,50,90),
                weights = (0.3,0.4,0.3)):

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
        
        #Assigning inputs to the object
        self.firecares_id = firecares_id
        self.fdid = fdid
        self.state = state
        self.unit_types = unit_types
        self.bad_units = bad_units
        self.p = p
        self.weights = weights
        
        #Running standard methods at start
        self.boundary_query()
        self.station_query()

        if load_all==True:
            self.apparatus_query(load_dir='./apparatus_df')
        self.unit_reset(recalculate=False)

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
        it will simply load the dataframe from a pickle whose name corresponds to the firecares_id + "df."

        Parameters
        ----------
        save_dir: (optional) str
            The location to save the dataframe if it is generated from an incident query.
        load_dir: (optional)  str
            The directory from which to load an existing apparatus dataframe.

        Returns
        -------
        self.df: (pandas.DataFrame)
            A dataframe containing all of the apparatus data merged with selected incident data

        """

        if not load_dir:
            #If not load directory is specified, generate the df from incident data and save it
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

            print("Done! \n")


            print("Unpacking the apparatus data...")
            #Because we haven't exploded the apparatus json yet, assigning the index will tell us which incident each unit corresponds to
            for i in range(len(df_list)):
                df_list[i]['incident'] = [i]*len(df_list[i])

            df = pd.concat(df_list.to_list()).merge(df_flat, on='incident', sort=True)

            #Saving the dataframe to a pickle for speed
            df.to_pickle(savedir + "/" + self.firecares_id+'apparatus')
            self.df = df
            print("Done! \n")

        else:
            #If a load_dir is specified, simply load the dataframe.
            self.df = pd.read_pickle(load_dir+"/"+self.firecares_id+'apparatus')
        if self.unit_types:
            #If unit types are specified, only consider them in the dataframe
            self.df = self.df[self.df['unit_type'].isin(self.unit_types)].reset_index(drop=True)
        #Ignore specific unit_ids if specified
        if self.bad_units:
            self.df = self.df[~self.df['unit_id'].isin(self.bad_units)].reset_index(drop=True)
            
        #Removing first due areas that don't correspond to a station that sends units
        self.df = self.df[self.df['address.first_due'].isin(self.df['station'].unique())].reset_index(drop=True)
        self.df['from_first_due'] = self.df['address.first_due']  == self.df['station']
        
        #Making a list of all the stations
        station_list = np.array(self.df['address.first_due'].unique())

        #Then dropping incidents where the unit came from a different station than one that shows up in first due
        self.df = self.df[self.df['address.first_due'].isin(station_list)].reset_index(drop=True)

        #Gives a dataframe of all units, where they are housed, and 
        self.stat_units = self.df.groupby('station').apply(lambda x: x['unit_id'].unique()).reset_index()
        self.stat_units = self.stat_units.explode(0).rename(columns = {0:'unit_id'})
        self.stat_units = self.stat_units.merge(self.df[['unit_id','unit_type']].drop_duplicates())   
        
        
    def unit_reset(self, recalculate=True, revert_df=None):
        """
        Calculates the number of units belonging to each station. 
        Also resets the model to the real station counts.
        """
        def unit_agg(x):
            d = {}
            for unit in self.unit_types:
                d[unit] = np.sum(x['unit_type'] == unit)
            return pd.Series(d, index=d.keys())
        
        
        #This allows the program to work recursively because it can revert to a different state than the actual
        #unit counts
        if revert_df is not None:
            self.unit_counts = revert_df.copy()
        else:
            self.unit_counts = self.stat_units.groupby('station').apply(unit_agg).reset_index()
        
        if recalculate:
            self.build_distribution(self.p, self.weights)
            self.build_lognorm()
        
    def station_query(self):
        """

        Queries information about the stations and puts it in self.station_df

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
                self.station_df = pd.DataFrame(items)

        self.station_df['geom'] = self.station_df['geom'].apply(lambda x: wkb.loads(x, hex=True))

    def cleaner(self, subset=None, min_dist=0.0, max_dist=20, min_time=10.0, max_time=1200.0, min_long=-81.0):
        """
        This method cleans the dataframe by removing outlier incidents and dropping NaNs along specified fields.
        """
        
        print('Cleaning the apparatus dataframe...')
        
        #Dropping rows with NaNs in specified fields
        if subset:
            self.df = self.df.dropna(subset=subset).reset_index(drop=True)
        
        #Dropping rows with travel times or distances that are outside specified intervals
        self.df = self.df[self.df['address.first_due'] != '']
        self.df = self.df[self.df['extended_data.response_duration'] > min_time]
        self.df = self.df[self.df['extended_data.response_duration'] < max_time]
        self.df = self.df[self.df['distance'] > min_dist]
        self.df = self.df[self.df['distance'] < max_dist].reset_index(drop=True)
        self.df = self.df[self.df['address.longitude']>min_long].reset_index(drop=True)
        print('Done! \n')

    def from_station(self,prefix='11',tol = 0.25):
        """
        This method determines if a unit was dispatched from the station based on 
        comparing the distance field, which gives the reported distance from the unit 
        to the incident at the moment the unit was dispatched, to the distance between the 
        incident and the unit's station. If the distances line up, the unit was likely at the station
        when the unit was dispatched. This is not a perfect method, however. 
        
        *Note this method should only be used for Delray Beach because other departments don't have 
        a distance field. 
        """
        
        print('Determining if units were sent from their station...')
        
        #Need to drop if this column already exists, otherwise merge gives strange results
        if 'dist' in self.df.columns:
            self.df = self.df.drop('dist',axis=1)
        if 'geom' in self.df.columns:
            self.df = self.df.drop('geom',axis=1)

        #Getting the geometry for the station associated with the unit
        self.station_df['station'] = self.station_df['station_number'].apply(lambda x: prefix+str(x))
        if 'station_loc' not in self.df.columns:
            self.df = self.df.merge(self.station_df[['station','geom']], on='station')
            self.df = self.df.rename(columns={'geom': 'station_loc'})

        #Getting the geometry of the first due station, which is not necessarily the unit's station
        self.station_df['station'] = self.station_df['station_number'].apply(lambda x: prefix+str(x))
        if 'first_due_loc' not in self.df.columns:
            self.df = self.df.merge(self.station_df.rename(columns={'station':'address.first_due'})[['address.first_due','geom']], on='address.first_due')
            self.df = self.df.rename(columns={'geom': 'first_due_loc'})

        def dist(x,key):
            """
            Approximate distance calculation between the station and the incident address
            """
            x_diff  = (x[key].coords[0][0] -  x['address.longitude'])* \
                        np.cos(x['address.latitude']/180*np.pi)*69
            y_diff =  (x[key].coords[0][1] -  x['address.latitude'])*69
            d = np.sqrt(x_diff**2 + y_diff**2)
            return d

        self.df['dist_from_station'] = self.df.apply( lambda x: dist(x,key='station_loc')  ,axis=1)
        self.df['dist_from_due'] = self.df.apply( lambda x: dist(x,key='first_due_loc')  ,axis=1)
        
        #Need to drop if this column already exists, otherwise merge gives strange results
        if 'dist' in self.df.columns:
            self.df = self.df.drop('dist',axis=1)
        if 'geom' in self.df.columns:
            self.df = self.df.drop('geom',axis=1)

        #Getting the geometry for the station associated with the unit
        self.station_df['station'] = self.station_df['station_number'].apply(lambda x: prefix+str(x))
        if 'station_loc' not in self.df.columns:
            self.df = self.df.merge(self.station_df[['station','geom']], on='station')
            self.df = self.df.rename(columns={'geom': 'station_loc'})

        #Getting the geometry of the first due station, which is not necessarily the unit's station
        self.station_df['station'] = self.station_df['station_number'].apply(lambda x: prefix+str(x))
        if 'first_due_loc' not in self.df.columns:
            self.df = self.df.merge(self.station_df.rename(columns={'station':'address.first_due'})[['address.first_due','geom']], on='address.first_due')
            self.df = self.df.rename(columns={'geom': 'first_due_loc'})

    def __inprogressatstart__(self,start,end, stations, first_due,list_size = 15):

        """

        Runs an algorithm that determines all stations that had units active when the first unit of the incident was dispatched.

        Parameters
        ----------
        start: array_like
            An array of timestamps corresponding to the relevant start times (i.e. when a unit was dispatched)
        end: array_like
            An array of timestamps corresponding to the end of a unit's active period (i.e. when it is available)
        stations: array_like
            An array of stations that correspond to where the units are housed
        first_due: array_like
            An array of address designations corresponding to the first due station
        list_size: (optional) int
            The maximum number of units that could be active at the same time. DO NOT CHANGE THIS UNLESS YOU GET AN ERROR THAT "active_station_list" doesn't
            have enough columns.

        Returns
        -------
        active_stations: array_like
            A list of lists of the stations that had units active when the unit was dispatched
        active_regions: array_like
            A list of lists of the stations that had units active when the unit was dispatched
        """

        print("Running the algorithm to find units active at the start of each incident")
        active_stations = np.full( (len(start),list_size), 'empty'   )
        active_regions = np.full( (len(start),list_size), 'empty'   )
        for i in tqdm(range(len(end))):
            for j in range(i+1, len(end)):
                if end[i] >= start[j]:
                    active_stations[j,np.where(active_stations[j,:] == 'empty')[0][0]] = stations[i]
                    active_regions[j,np.where(active_regions[j,:] == 'empty')[0][0]] = first_due[i]
                else:
                    break

        return active_stations, active_regions


    def first_due_analysis(self):
        print('Getting information about units that are busy when units are dispatched...')
        #Need to make sure dataframe is sorted!!!
        self.df = self.df.sort_values(by='unit_status.dispatched.timestamp').reset_index(drop=True)
        df = pd.DataFrame()

        for unit in self.unit_types:
            append = self.df[self.df['unit_type']==unit].reset_index(drop=True)
            start = append['unit_status.dispatched.timestamp']
            end = append['unit_status.available.timestamp']
            start = np.array(start.apply(lambda x: datetime.strptime(x[:-6], "%Y-%m-%dT%H:%M:%S")))
            end = np.array(end.apply(lambda x: datetime.strptime(x[:-6], "%Y-%m-%dT%H:%M:%S")))
            active_stations, active_regions = self.__inprogressatstart__(start,end, append['station'], append['address.first_due'])

            #Initializing lists
            due_busy = np.zeros(len(append))
            active_from_due = np.zeros(len(append))
            active_in_area = np.zeros(len(append))
            due_in_area = np.zeros(len(append))

            for i,row in tqdm(append.iterrows()):
                #the number of units currently active from the first due station
                active_from_due[i] = np.sum(active_stations[i,:]  == row['address.first_due'])
                #The number of units responding to incidents in the first due area
                active_in_area[i] = np.sum(active_regions[i,:]  == row['address.first_due'])

                #The number of units from first due station responding to incidents in the first due area
                due_in_area[i] = np.sum( (active_regions[i,:]  == row['address.first_due']) & 
                                        (active_stations[i,:]  == row['address.first_due']))
                #If all from the first due are active, then due busy is True
                due_busy[i] = active_from_due[i] >= self.unit_counts[self.unit_counts['station'] == row['address.first_due']][unit]
            append['due_busy'] = due_busy
            append['num_active'] = active_from_due
            append['active_in_area'] = active_in_area
            append['due_in_area'] = due_in_area
            df = df.append(append)

        #df is now just the dataframe with more columns appended
        self.df = df.copy()

        #Adding a column that indicates how many units a station would need to respond to that incident
        self.df['num_required'] = self.df['active_in_area'] + self.df['num_active'] - self.df['due_in_area'] + 1
        print('Done! \n')
        
    def station_builder(self):
        """
        Creates a dictionary of station objects called self.station_dict and then determines the
        probablity of each station being first due. 
        """
        self.station_dict = {}
        
        self.station_list = np.array(self.station_df['station'])
        self.first_due_probs = {}
        for unit_type in self.unit_types:
            self.first_due_probs[unit_type] = np.zeros(len(self.station_list))


        for i,station_num in enumerate(self.station_list):
            self.station_dict[station_num] = station(station_num,self.df)
            for unit_type in self.unit_types:
                self.first_due_probs[unit_type][i] = self.station_dict[station_num].first_due_probs[unit_type]
                
    def __discretizer__(self):
        """
        Creates discrete approximations for all travel time distributions
        """
        
        for station_num in self.station_list:
            self.station_dict[station_num].discretizer(self.p)
            
            
    def build_distribution(self,p, weights):
        """
        After discretizing all of the individual station-level distributions, this builds an aggregate 
        travel time distribution based on discrete approximations.
        """

        #Create dictionaries to hold all of the distributions by unit type
        self.probs = {}
        self.times = {}
        
        self.p = p
        self.weights = np.array(weights)
        
        #Discretizing the individual station distribution approximations
        self.__discretizer__()
        
        for unit_type in self.unit_types:
            self.probs[unit_type] = []
            self.times[unit_type] = []

            for i,station_num in enumerate(self.station_list):
                
                #probability of station being first due    
                due_prob = self.first_due_probs[unit_type][i]

                #probability of station having unit available
                num_units = self.unit_counts[self.unit_counts['station'] == station_num][unit_type].values[0]
                
                #probability of a unit being out of service
                out_of_service = 1 - self.station_dict[station_num].send_given_available[unit_type]
                num_required = np.array(self.station_dict[station_num].num_required[unit_type])
                
                #Assumption is that only one unit will ever be out of service at a time
                send_given_due = num_units*out_of_service*np.sum(num_required[:(num_units-1)]) + \
                                 (1 - out_of_service*num_units)*np.sum(num_required[:num_units])
                
        

                #Probability of that station being first due AND sending a unit
                self.station_dict[station_num].send_probs[unit_type] = due_prob*send_given_due

                #Probablity of station not sending unit and being selected is probability of it being selected
                #minus the probability that it sends a unit and is selected
                self.station_dict[station_num].nsend_probs[unit_type] = due_prob - self.station_dict[station_num].send_probs[unit_type]

                #Appending the sent unit time estimates and probablities
                self.probs[unit_type] = np.concatenate([self.probs[unit_type], self.station_dict[station_num].send_probs[unit_type]*self.weights])
                send_times = self.station_dict[station_num].sent[unit_type]
                self.times[unit_type] =  np.concatenate([self.times[unit_type], send_times])

                #Appending the no sent unit time estimates and probabilities
                self.probs[unit_type] = np.concatenate([self.probs[unit_type], self.station_dict[station_num].nsend_probs[unit_type]*self.weights])
                nsend_times = self.station_dict[station_num].not_sent[unit_type]
                self.times[unit_type] =  np.concatenate([self.times[unit_type], nsend_times])


    def add_unit(self, station_num, unit_type,  delta):
        """
        Updates the number of units of a specified type at a specified station by delta
        Then recalculates the distribution approximation
        """
        self.unit_counts.loc[self.unit_counts['station'] == station_num, unit_type] += delta
        self.build_distribution(self.p, self.weights)
        self.build_lognorm()
        
    def time_draw(self, unit_type):
        """
        Draws an individual travel time
        """
        #First draw a station
        station_num = np.random.choice(self.station_list, p=self.first_due_probs[unit_type])

        #Give time a default value of nan
        time = np.nan

        #Then draw how many units are currently required in area
        num_req = np.array(self.station_dict[station_num].num_required[unit_type].index)
        p = np.array(self.station_dict[station_num].num_required[unit_type])
        num = int(np.random.choice(num_req, p=p))

        #If the first due has a unit not responding to an incident, draw whether it is sent
        num_units = self.unit_counts[self.unit_counts['station'] == station_num][unit_type].values[0]
        if num <= num_units:
            p = self.station_dict[station_num].send_given_available[unit_type]
            sends_unit = np.random.binomial(1,p)

            #If it sends a unit, draw from relevant distribution
            if sends_unit == 1:
                time = np.random.choice(self.station_dict[station_num].sent_full[unit_type])
        #If a time has not yet been assigned, that means a unit wasn't sent
        if np.isnan(time):
            time = np.random.choice(self.station_dict[station_num].not_sent_full[unit_type])

        return time

    def monte_carlo(self, niter=10000):
        """
        Runs a full monte carlo simulation for each unit type
        """
        self.mc_times = {}
        for unit_type in self.unit_types:
            self.mc_times[unit_type] = np.zeros(niter)
            for i in tqdm(range(niter)):
                self.mc_times[unit_type][i] = self.time_draw(unit_type)  
                
    def optimizer(self, unit_type, num_units, fun=None, times=None, temp_stations=None, revert_df=None, return_bool=True):
        """
        Simulates every possible combination of unit allocations
        """
        if fun is None:
            fun = self.mean_calc
        
        if times is None:
            times = []
        if revert_df is None:
            #If no revert dataframe is provided, reset the unit counts and then use that
            self.unit_reset()
            revert_df = self.unit_counts.copy()
        for station_num in self.station_list:
            self.add_unit(station_num, unit_type, 1)
            if num_units - 1 > 0:
                self.optimizer(unit_type,  num_units-1, fun, times, revert_df = self.unit_counts.copy(), return_bool=False)
            else:
                times.append(fun(unit_type))

            self.unit_reset(revert_df=revert_df)
        if return_bool:
            grid = np.meshgrid(*([self.station_list]*num_units))
            stations= np.reshape(grid,(num_units,len(self.station_list)**num_units), order='F').T
            self.simulated_times = times
            self.simulated_stations = stations
            idx = np.argmin(times)
            self.best = stations[idx]
            self.unit_reset()
            current = fun(unit_type)
            self.improvement = current - times[idx]

    def mean_calc_disc(self, unit_type):
        """
        Calculates the mean of the discretized distribution
        """
        return(self.probs[unit_type]@self.times[unit_type])
    
    def mean_calc(self, unit_type):
        """
        Calculates the mean of the discretized distribution
        """
        return(self.lognorm[unit_type].mean())  
    
    def quantile(self, unit_type,q=0.9):
        """
        Calculates the 90th percentile of the discretized distribution
        """
        return self.lognorm[unit_type].ppf(q)

    def quantile_disc(self, unit_type,q=90):
        """
        Calculates the 90th percentile of the discretized distribution
        """
        sort_idx = np.argsort(self.times[unit_type])
        sort_times = self.times[unit_type][sort_idx]
        cumsum = np.cumsum(self.probs[unit_type][sort_idx])
        idx = np.where(cumsum < q/100)[0][-1]
        return sort_times[idx]

    def frac_below_disc(self, unit_type, cutoff=240):
        """
        Calculates the fraction of incidents below a specified cutoff for the discretized distribution
        """
        sort_idx = np.argsort(self.times[unit_type])
        sort_times = self.times[unit_type][sort_idx]
        cumsum = np.cumsum(self.probs[unit_type][sort_idx])
        idx = np.where(sort_times < cutoff)[0][-1]
        #Needs to be negative because optimizer seeks minumum
        return -cumsum[idx]
    
    def frac_below(self, unit_type, cutoff=240):
        """
        Calculates the fraction of incidents below a specified cutoff for the lognormal fit
        """
        return -self.lognorm[unit_type].cdf(cutoff)
    
    
    def build_lognorm(self):
        """
        Builds a lognormal distribution based on the current unit counts for the department
        """
        self.lmean = {}
        self.lvar = {}
        self.transform_vec = {}
        self.lmean2 = {}
        self.lognorm = {}

        for unit_type in self.unit_types:
            self.lmean[unit_type] = []
            self.lmean2[unit_type] = []
            self.lvar[unit_type] = []
            self.transform_vec[unit_type] = []

        for station_num in self.station_list:
            stat = self.station_dict[station_num]
            stat.lmean_sent = {}
            stat.lmean_nsent = {}
            stat.lvar_sent = {}
            stat.lvar_nsent = {}
            stat.lmean2_sent = {}
            stat.lmean2_nsent = {}


            for unit_type in self.unit_types:
                stat.lmean_sent[unit_type] = np.mean(np.log(stat.sent_full[unit_type]))
                stat.lmean_nsent[unit_type] = np.mean(np.log(stat.not_sent_full[unit_type]))
                stat.lvar_sent[unit_type] = np.var(np.log(stat.sent_full[unit_type]),ddof=1)
                stat.lvar_nsent[unit_type] = np.var(np.log(stat.not_sent_full[unit_type]),ddof=1)
                stat.lmean2_sent[unit_type] = np.mean( np.log(stat.sent_full[unit_type])**2 )
                stat.lmean2_nsent[unit_type] = np.mean(np.log(stat.not_sent_full[unit_type])**2 )
                
                #Adding this information to the department level vectors
                self.lmean[unit_type].append(stat.lmean_sent[unit_type])
                self.lmean[unit_type].append(stat.lmean_nsent[unit_type])
                self.lmean2[unit_type].append(stat.lmean2_sent[unit_type])
                self.lmean2[unit_type].append(stat.lmean2_nsent[unit_type])
                self.lvar[unit_type].append(stat.lvar_sent[unit_type])
                self.lvar[unit_type].append(stat.lvar_nsent[unit_type])
                self.transform_vec[unit_type].append(stat.send_probs[unit_type])
                self.transform_vec[unit_type].append(stat.nsend_probs[unit_type])

        for unit_type in self.unit_types:
            #Transforming lists into numpy arrays
            self.transform_vec[unit_type] = np.array(self.transform_vec[unit_type])
            self.lmean[unit_type] = np.array(self.lmean[unit_type])
            self.lmean2[unit_type] = np.array(self.lmean2[unit_type])
            self.lvar[unit_type] = np.array(self.lvar[unit_type])

            #Calculating means and variances of the full distributions
            transformed_mean = self.transform_vec[unit_type].T@self.lmean[unit_type]  
            eh_2 = self.transform_vec[unit_type]@(self.lvar[unit_type]+self.lmean[unit_type]**2)
            transformed_var = eh_2 - transformed_mean**2  
            self.lnorm = lognorm(s=np.sqrt(transformed_var),scale=np.exp(transformed_mean))
            self.lognorm[unit_type] = lognorm(s=np.sqrt(transformed_var),scale=np.exp(transformed_mean))
            
            test_var = self.transform_vec[unit_type].T@self.lmean2[unit_type] - transformed_mean**2

    


class station:
    """
    This class holds station level attributes.
    """
    
    def __init__(self,station_num,df):
        #The station specific df, only contains incidents for which the station is first due
        self.df = df[df['address.first_due']==station_num].reset_index(drop=True)
        
        #The full dataframe that is passed into the station
        self.full_df = df
        
        self.station_num = station_num
        self.units = self.df[['unit_id', 'unit_type']].drop_duplicates().reset_index(drop=True)
        
        #Needs to be all of the unit types that are in df, not self.df because the station might not have
        #all unit types!
        self.unit_types = df['unit_type'].unique()
        
        self.first_due_prob_calc()
        self.time_maker()
        self.num_required()
        self.send_prob()
    
    
    def first_due_prob_calc(self):
        """
        Determines the probablity of the station being the first due for each unit type, weighted by number
        of units required.
        
        This is basically the geographic propensity of incidents of each type. 
        
        """
        
        
        self.first_due_probs = {}
        for unit_type in self.unit_types:
            unit_slice = self.full_df[self.full_df['unit_type'] == unit_type]
            self.first_due_probs[unit_type] = np.sum(unit_slice['address.first_due']==self.station_num)/len(unit_slice)



    def time_maker(self):
        """
        Making dictionaries that hold arrays for each unit type. These arrays are the list of all travel times
        corresponding to the unit type and whether it was sent. 
        """
        
        self.sent_full = {}
        self.not_sent_full = {}
        
    
        #Getting arrays for the travel 
        for unit_type in self.unit_types:
            self.sent_full[unit_type] = np.array(self.df[(self.df['from_first_due'] == True) & 
                                                    (self.df['unit_type']==unit_type)]['extended_data.response_duration'])
            self.not_sent_full[unit_type] = np.array(self.df[(self.df['from_first_due'] == False)& 
                                                    (self.df['unit_type']==unit_type)]['extended_data.response_duration'])
            
#         #If a station doesn't have a truck, use the engine travel times
#         if (len(self.sent_full['Truck/Aerial']) == 0):
#             self.sent_full['Truck/Aerial'] = self.sent_full['Engine']
                
            
    def num_required(self):
        """
        Making a dictionary that determines the distribution of number of incidents required for each unit type
        """
        
        self.num_required = {}
        
        for unit_type in self.unit_types:
            unit_slice = self.df[self.df['unit_type'] == unit_type]
            self.num_required[unit_type] = unit_slice['num_required'].value_counts()/len(unit_slice)
            
            
            
    def discretizer(self, p):
        """
        Makes discrete approximations of all the travel time distributions
        """
        
        
        self.sent = {}
        self.not_sent = {}
        
        for unit_type in self.unit_types:
            if len(self.sent_full[unit_type]) > 0:
                self.sent[unit_type] = np.nanpercentile(self.sent_full[unit_type],p)
            if len(self.not_sent_full[unit_type]) > 0:
                self.not_sent[unit_type] = np.nanpercentile(self.not_sent_full[unit_type],p)
                
    def send_prob(self, fill_missing=0.98):
        """
        Determines the probability of the station sending a unit given that it has at least one available. 
        """
        
        self.send_given_available = {}
        
        #Also initialze the full probablity of that station being available and sending a unit, but
        #this is calculated at the department level
        self.send_probs = {}
        self.nsend_probs = {}

    
        for unit_type in self.unit_types:
            unit_slice = self.df[(self.df['unit_type'] == unit_type) & (self.df['due_busy'] == False)]
            if len(unit_slice) > 0:
                self.send_given_available[unit_type] = np.sum(unit_slice['from_first_due'])/len(unit_slice)
            else:
                self.send_given_available[unit_type] = fill_missing
            
            
        
        
        