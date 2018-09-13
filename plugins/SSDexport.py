""" Plugin to generate SSD figures and save them in the project folder
Based on Leonor Inverno's SSD implementation
Author: Ander Okina """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, sim#, scr, tools
from bluesky.tools.aero import nm, ft
from bluesky.tools import geo, areafilter
import numpy as np
from datetime import datetime



from collections import Counter
from tempfile import TemporaryFile

#Plotting packages
import matplotlib.pyplot as plt
#from matplotlib.cbook import get_sample_data
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import shutil
from shapely.geometry import Polygon
from shapely.ops import unary_union

try:
    import pyclipper
except ImportError:
    print("Could not import pyclipper, RESO SSD will not function")

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global data
    data = Simulation()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SSDEXPORT',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 30, #If changed also change variable in flexMatrix!!

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        'update':          data.update,

        # The preupdate function is called before traffic is updated. Use this
        # function to provide settings that need to be used by traffic in the current
        # timestep. Examples are ASAS, which can give autopilot commands to resolve
        # a conflict.
        'preupdate':       data.preupdate,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         data.reset
        }

    stackfunctions = {
        # The command name for your function
        'SSDEXPORT': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'SSDEXPORT ON/OFF NL ON/OFF',

            # A list of the argument types your function accepts. For a description of this, see ...
            '[onoff, txt, onoff]',

            # The name of your function in this plugin
            data.initialize,

            # a longer help text of your function.
            'Generate flexibility data based on SSD polygons'],
        'MAP': [
            'MAP ON/OFF',
            'onoff',
            data.printMap,
            'Generate complexity map with collected flexibility data'
        ]
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

class Simulation():
    def __init__(self):
        self.time_stamp = 0
        self.active = False
        self.area = True #Activation of area
        self.print = False
        self.areaName = str('')
        self.flexibility_global = np.array([])
        self.dlat = 0
        self.dlon = 0
        self.coord = []
        self.min_lat = 0
        self.min_lon = 0
        self.map = False
        self.avg_flex = 0
        self.dalt = 3000*0.3048

    def update(self):

        if self.active== False: # only execute if plugin is turned on
            return

        self.time_stamp += 1




        if self.active == True:
            # Define heading and speed changes
            vmin = traf.asas.vmax #traf.spd+30 if traf.spd+30<450 else 450
            vmax = traf.asas.vmax #traf.spd-30 if traf.spd-30>250 else 250
            N_angle = 180

            angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
            xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
            SSD_lst = [list(map(list, np.flipud(xyc * vmax))), list(map(list, xyc * vmin))]

            # Outer circle: vmax
            SSD_outer = np.array(SSD_lst[0])
            x_SSD_outer = np.append(SSD_outer[:, 0], np.array(SSD_outer[0, 0]))
            y_SSD_outer = np.append(SSD_outer[:, 1], np.array(SSD_outer[0, 1]))
            # Inner circle: vmin
            SSD_inner = np.array(SSD_lst[1])
            x_SSD_inner = np.append(SSD_inner[:, 0], np.array(SSD_inner[0, 0]))
            y_SSD_inner = np.append(SSD_inner[:, 1], np.array(SSD_inner[0, 1]))
            # Initialize SSD variables with ntraf
            self.SSDVariables(traf.asas, traf.ntraf)
            # Construct ASAS
            self.constructSSD1(traf.asas, traf)
            self.visualizeSSD(x_SSD_outer,y_SSD_outer,x_SSD_inner,y_SSD_inner,traf.asas)


    def preupdate(self):
        return

    def reset(self):
        pass

    def flexMatrix(self,coords,t):
        #Generates the matrix in which the flexibility information is stored
        #Discretise coordinates so that flex[lat,lon,alt,iteration] = feas/(feas+unfeas)
        self.d = 1/40 #Discretise coordinates for every 1/100 of degree -> 0,6 nm
        self.nlat = (np.amax(coords[:,0]) - np.amin(coords[:,0]))/self.d
        self.nlon = (np.amax(coords[:,1]) - np.amin(coords[:,1]))/self.d
        self.min_lat = np.amin(coords[:,0])
        self.min_lon = np.amin(coords[:,1])
        nalt = int(50000/1000) # Altitude in 100ft intervals (flight levels)
        sim_length = t/30
        print('before')
        flex = np.ones((int(self.nlat)+1,int(self.nlon)+1,nalt+1,int(sim_length)))
        print('middle')
        flex[:] = np.NaN #Only actual data will be number, otherwise NaN
        print('flex created')
        return flex


    #It's only called once
    def initialize(self,*args):

        if not args:
            return True, "SSDSEQ is currently " + ("ON" if self.active else "OFF")

        self.active = True if args[0]==True else False
        self.area = True if args[2]==True else False #Area of country, only process within
        self.areaName = str(args[1])


        if self.area == True:
            #Generate area of NL that will be used to filter traffic
            self.coord = np.array([54.96964, 5.00976, 51.45904, 1.99951, 50.71375, 6.0424, 52.2058, 7.09716, 53.2963, 7.2509, 54.9922, 6.5478])
            areafilter.defineArea(self.areaName, 'POLY', self.coord)
            self.flexibility_global = self.flexMatrix(np.reshape(self.coord,(6,2)),24*60*60) #24 hours
            stack.stack('AREA '+str(self.areaName))

        return True

    def printMap(self, *args):
        #Prints the map with the available information of flexibility
        self.maps = True if args[0]==True else False
        if self.maps:
            #Save information for processing
            flex_map = np.nanmean(self.flexibility_global,axis=3) #Mean for all time steps discarding NaN
            if os.path.exists('flexibility3d.npy'): #If there's a file already, remove it
                os.remove('flexibility3d.npy')
            np.save('flexibility3d.npy', flex_map)
            flex_map = np.nanmean(flex_map,axis=2) #Mean for all altitudes discarding NaN
            if os.path.exists('flexibility2d.npy'): #If there's a file already, remove it
                os.remove('flexibility2d.npy')
            np.save('flexibility2d.npy', flex_map)
            x,y = self.discreteCoord(self.coord[0::2], self.coord[1::2])
            airspace = np.resize(np.stack((x,y)),(1,2*len(x)))
            np.save('airspace.npy', airspace)
            #fig, ax0 = plt.subplots(1)
            print(self.time_stamp) ####### Check time stamp and size of x in plot

            #c = ax0.pcolor(flex_map)
            #ax0.plot(self.coord[0::2],self.coord[1::2],'k')
            #ax0.set_title('Complexity map')
            #plt.colorbar(c)
            #fig.tight_layout()
            #plt.show()
            self.maps = False
            stack.stack('ECHO Data saved')


    def visualizeSSD(self, x_SSD_outer,y_SSD_outer,x_SSD_inner,y_SSD_inner,asas):
        #Generate the SSD velocity obstacles and convert them to polygons
        #so that their areas can be computed

        #Obtain list of aircraft inside the defined area
        inNL = areafilter.checkInside(self.areaName, traf.lat, traf.lon, traf.alt)
        #Obtain the indices of lat and lon of aircraft for flexibiity matrix
        inlat, inlon = self.discreteCoord(traf.lat, traf.lon)
        #print('Aircraft within NL: '+str(np.sum(inNL)))
        for i in range(traf.ntraf):
            if inNL[i]: #If aircraft in interest area (NL airspace)

                N_angle = 180
                #Define new ARV taking into consideration the heading constraints and the current heading of each aircraft
###### To try: create semicircle and then use with ARV
                asas.trncons = 90 #Turning angle constraint
                trn_cons = np.radians(asas.trncons)
                angles2 = np.arange(np.radians(traf.hdg[i])-trn_cons, np.radians(traf.hdg[i])+trn_cons, 2*trn_cons/N_angle)
                # Put points of unit-circle in a (180x2)-array (CW)
                xyc2 = np.transpose(np.reshape(np.concatenate((np.sin(angles2), np.cos(angles2))), (2, len(angles2))))
                #For tupple
                inner_semicircle = (tuple(map(tuple , xyc2 * asas.vmin)))
                outer_semicircle = tuple(map(tuple, np.flipud(xyc2 * asas.vmax)))
                new_circle_tup = inner_semicircle + outer_semicircle
                #For list
                inner_semicircle = list(map(list , xyc2 * asas.vmin))
                outer_semicircle = list(map(list, np.flipud(xyc2 * asas.vmax)))
                new_circle_lst = inner_semicircle + outer_semicircle
                sem_circ_pol = Polygon(new_circle_lst)
                tot_area = sem_circ_pol.area

                if traf.asas.ARV[i]:
                    for j in range(len(traf.asas.ARV[i])):
                        FRV_1 = np.array(traf.asas.ARV[i][j])
                        x_FRV1 = np.append(FRV_1[:,0] , np.array(FRV_1[0,0]))
                        y_FRV1 = np.append(FRV_1[:,1] , np.array(FRV_1[0,1]))
                        FRV1 = FRV_1.tolist()
                        if j>0:
                            un_FRV = un_FRV.union(Polygon(FRV1))
                        elif j==0:
                            un_FRV = Polygon(FRV1)
                    free_area = un_FRV.area

                    fig, ax = plt.subplots()
                    if type(un_FRV)==Polygon and self.print:
                        x,y = un_FRV.exterior.xy
                        plt.plot(x,y, color ='red')
                        ax.fill(x, y, color = '#C0C0C0') #gray
                        x,y = sem_circ_pol.exterior.xy
                        plt.plot(x,y,color = 'black')
                        plt.title((traf.id[i]))
                        plt.show()
                    elif self.print:
                        for k in range(len(list(un_FRV))):
                            x,y = un_FRV[k].exterior.xy
                            plt.plot(x,y, color = 'red')
                            ax.fill(x, y, color = '#C0C0C0') #gray
                            x,y = sem_circ_pol.exterior.xy
                            plt.plot(x,y,color = 'black')
                        plt.title((traf.id[i]))
                        plt.show()


                if traf.asas.FRV[i] and 0==1:
                    for j in range(len(traf.asas.FRV[i])):
                        FRV_2 = np.array(traf.asas.FRV[i][j])
                        x_FRV2 = np.append(FRV_2[:,0] , np.array(FRV_2[0,0]))
                        y_FRV2 = np.append(FRV_2[:,1] , np.array(FRV_2[0,1]))


                        FRV2 = FRV_2.tolist()
                        if j>0:
                            un_FRV2 = un_FRV2.union(Polygon(FRV2))
                        elif j==0:
                            un_FRV2 = Polygon(FRV2)

                    conf_area = un_FRV2.area

                    if type(un_FRV2)==Polygon and self.print and 1==0: #If not polygon -> multipolygon
                        x,y = un_FRV2.exterior.xy
                        plt.plot(x,y, color = '000000')
                        ax.fill(x, y, color = '#C0C0C0') #gray
                        plt.title(traf.id[i])
                        #plt.show()
                    elif self.print and 1==0:
                        for k in range(len(list(un_FRV2))):
                            x,y = un_FRV2[k].exterior.xy
                            plt.plot(x,y,color = '000000')
                            ax.fill(x, y, color = '#C0C0C0') #gray
                        plt.title(traf.id[i])
                        #plt.show()
                #print('ARV= '+str(free_area)+', FRV= '+str(conf_area)+', total = '+str(free_area+conf_area))

                if traf.asas.ARV[i]:
                    flex = free_area/tot_area
                    self.flexibility_global[inlat[i],inlon[i],int(round(traf.alt[i]/1000)),self.time_stamp] = flex
                else:
                    flex = 1 #No conflict area -> full flexibility
                    self.flexibility_global[inlat[i],inlon[i],self.time_stamp] = 1
                #print('Flexibility of aircraft '+traf.id[i]+' is: '+str(flex))
        #Compute average flexibility for timestep
        #self.avg_flex = np.mean(self.flexibility_global[:,:,self.time_stamp])

        return

    def discreteCoord(self,lat,lon):
        in_lat = ((lat-self.min_lat)/self.d) #Indice in matrix of latitude
        in_lon = ((lon-self.min_lon)/self.d) #Indice in matrix of longitude
        return in_lat.astype(int), in_lon.astype(int)


    def SSDVariables(self,asas, ntraf):
        """ Initialize variables for SSD """
        # Need to do it here, since ASAS.reset doesn't know ntraf

        #Forbidden Reachable Velocity regions
        asas.FRV          = [None] * ntraf

        #Allowed Reachable Velocity regions
        asas.ARV          = [None] * ntraf
        asas.ARV_min        = [None] * ntraf #NEW
        asas.ARV_tla        = [None] * ntraf #NEW

        # For calculation purposes
        asas.ARV_calc     = [None] * ntraf
        asas.ARV_calc_min = [None]* ntraf
        asas.ARV_calc_glb = [None]* ntraf
        asas.ARV_calc_dlos = [None]* ntraf
        asas.ARV_calc_dcpa = [None]* ntraf

        #Stores which layer of resolution each aircraft chose
        asas.reso_layer = [None]* ntraf
        asas.inrange      = [None] * ntraf

        # Stores resolution vector, also used in visualization
        asas.asasn        = np.zeros(ntraf, dtype=np.float32)
        asas.asase        = np.zeros(ntraf, dtype=np.float32)

        #Say if ac is in a LoS
        asas.los = [False]*ntraf

        # Area calculation
        asas.FRV_area     = np.zeros(ntraf, dtype=np.float32)

        asas.ARV_area     = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_min = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_tla = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_calc = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_calc_min = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_calc_glb = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_calc_dcpa = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_dlos = np.zeros(ntraf, dtype=np.float32)
        asas.layers_area = [None]*len(asas.layers_dict)



    def constructSSD1(self,asas, traf):
        """ Calculates the FRV and ARV of the SSD """

        #N = 0
        # Parameters - from ASAS
        N_angle = 180
        vmin    = asas.vmin             # [m/s] Defined in asas.py
        vmax    = asas.vmax             # [m/s] Defined in asas.py
        hsep    = asas.R                # [m] Horizontal separation (5 NM)
        margin  = asas.mar              # [-] Safety margin for evasion
        hsepm   = hsep * margin         # [m] Horizontal separation with safety margin
        alpham  = 0.4999 * np.pi        # [rad] Maximum half-angle for VO
        betalos = np.pi / 4             # [rad] Minimum divertion angle for LOS (45 deg seems optimal)
        adsbmax = 40. * nm              # [m] Maximum ADS-B range
        beta    = np.pi/4 + betalos/2

        #From traf
        lat     = traf.lat
        lon     = traf.lon
        ntraf   = traf.ntraf
        vs     = traf.vs

        #A default priocode must be defined for this CR method, otherwise it won't work with the predefined one
        if asas.priocode not in asas.strategy_dict:
            asas.priocode = "SRS1"

        # # Use velocity limits for the ring-shaped part of the SSD
        # Discretize the circles using points on circle
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
        # Put points of unit-circle in a (180x2)-array (CW)
        xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
        # Map them into the format pyclipper wants. Outercircle CCW, innercircle CW
        circle_tup = (tuple(map(tuple, np.flipud(xyc * vmax))), tuple(map(tuple , xyc * vmin)))
        circle_lst = [list(map(list, np.flipud(xyc * vmax))), list(map(list , xyc * vmin))]

        # If no traffic
        if ntraf == 0:
            return

        # If only one aircraft
        elif ntraf == 1:
            # Map them into the format ARV wants. Outercircle CCW, innercircle CW
            asas.ARV[0] = circle_lst
            asas.ARV_min[0] = circle_lst
            asas.ARV_tla[0] = circle_lst

            asas.FRV[0] = []
            asas.FRV_min[0] = []
            asas.FRV_tla[0]     = []
            asas.FRV_dlos[0]    = []

            asas.ARV_calc[0] = circle_lst
            asas.ARV_calc_min[0] = circle_lst
            asas.ARV_calc_glb[0] = circle_lst
            asas.ARV_calc_dlos[0] = circle_lst
            asas.ARV_calc_dcpa[0] = circle_lst

            # Calculate areas and store in asas
            asas.FRV_area[0] = 0

            asas.ARV_area[0] = 1
            asas.ARV_area_min[0] = 1
            asas.ARV_area_tla[0] = 1
            asas.ARV_area_dlos[0] = 1
            asas.ARV_area_calc[0] = 1
            asas.ARV_area_calc_min[0] = 1
            asas.ARV_area_calc_glb[0] = 1
            asas.ARV_area_calc_dcpa[0] = 1
            return

        # Function qdrdist_matrix needs 4 vectors as input (lat1,lon1,lat2,lon2)
        # To be efficient, calculate all qdr and dist in one function call
        # Example with ntraf = 5:   ind1 = [0,0,0,0,1,1,1,2,2,3]
        #                           ind2 = [1,2,3,4,2,3,4,3,4,4]
        # This way the qdrdist is only calculated once between every aircraft
        # To get all combinations, use this function to get the indices
        ind1, ind2 = self.qdrdist_matrix_indices(ntraf)

        # Get absolute bearing [deg] and distance [nm]
        # Not sure abs/rel, but qdr is defined from [-180,180] deg, w.r.t. North

        [qdr, dist] = geo.qdrdist_matrix(lat[ind1], lon[ind1], lat[ind2], lon[ind2])
        vs = self.qdrvs_matrix(vs[ind1],vs[ind2])

        # Put result of function from matrix to ndarray
        qdr  = np.reshape(np.array(qdr), np.shape(ind1))
        dist = np.reshape(np.array(dist), np.shape(ind1))
        vs  = np.reshape(np.array(vs), np.shape(ind1))
        # SI-units from [deg] to [rad]
        qdr  = np.deg2rad(qdr)
        # Get distance from [nm] to [m]
        dist = dist * nm

        # In LoS the VO can't be defined, act as if dist is on edge
        dist[dist < hsepm] = hsepm

        # Calculate vertices of Velocity Obstacle (CCW)
        # These are still in relative velocity space, see derivation in appendix
        # Half-angle of the Velocity obstacle [rad]
        # Include safety margin
        alpha = np.arcsin(hsepm / dist)
        # Limit half-angle alpha to 89.982 deg. Ensures that VO can be constructed
        alpha[alpha > alpham] = alpham
        #Take the cosinus of alpha to calculate the maximum length of the VO's legs
        cosalpha = np.cos(alpha)

        #construct with CS1
        self.CS1(asas, traf, ind1, ind2, adsbmax, dist, qdr, vs, cosalpha, xyc, circle_tup, circle_lst, beta, hsepm)

    def CS1(self,asas, traf, ind1, ind2, adsbmax, dist, qdr, vs, cosalpha, xyc, circle_tup, circle_lst, beta, hsepm):

        # Relevant info from traf and ASAS
        gsnorth = traf.gsnorth
        gseast  = traf.gseast
        ntraf = traf.ntraf
        vmax = asas.vmax
        vmin = asas.vmin

        # Local variables, will be put into asas later
        FRV_loc          = [None] * traf.ntraf

        ARV_loc          = [None] * traf.ntraf


        # Consider every aircraft
        for i in range(ntraf):
            # Calculate SSD only for aircraft in conflict (See formulas appendix)
            if 1==1:


                # SSD for aircraft i
                # Get indices that belong to aircraft i
                ind = np.where(np.logical_or(ind1 == i,ind2 == i))[0]

                # The i's of the other aircraft
                i_other = np.delete(np.arange(0, ntraf), i)
                # Aircraft that are within ADS-B range
                ac_adsb = np.where(dist[ind] < adsbmax)[0]
                ac_alt1 = np.where(vs[ind] >= 0)[0]
                ac_joint = np.concatenate([ac_adsb,ac_alt1])
                ac_adsb =  [item for item, count in Counter(ac_joint).items() if count > 1] #Only for A/C in ads-b range and VS converging
                # Now account for ADS-B range in indices of other aircraft (i_other)
                ind = ind[ac_adsb]
                i_other = i_other[ac_adsb]
                asas.inrange[i]  = i_other

                # VO from 2 to 1 is mirror of 1 to 2. Only 1 to 2 can be constructed in
                # this manner, so need a correction vector that will mirror the VO
                fix = np.ones(np.shape(i_other))
                fix[i_other < i] = -1


                drel_x, drel_y = fix*dist[ind]*np.sin(qdr[ind]), fix*dist[ind]*np.cos(qdr[ind])
                drel = np.dstack((drel_x,drel_y))

                cosalpha_i = cosalpha[ind]

                # Make a clipper object
                pc = pyclipper.Pyclipper()

                N_angle = 180
                #Define new ARV taking into consideration the heading constraints and the current heading of each aircraft

                asas.trncons = 90 #Turning angle constraint
                trn_cons = np.radians(asas.trncons)
                angles2 = np.arange(np.radians(traf.hdg[i])-trn_cons, np.radians(traf.hdg[i])+trn_cons, 2*trn_cons/N_angle)
                # Put points of unit-circle in a (180x2)-array (CW)
                xyc2 = np.transpose(np.reshape(np.concatenate((np.sin(angles2), np.cos(angles2))), (2, len(angles2))))
                #For tupple
                inner_semicircle = (tuple(map(tuple , xyc2 * vmin)))
                outer_semicircle = tuple(map(tuple, np.flipud(xyc2 * vmax)))
                new_circle_tup = inner_semicircle + outer_semicircle
                #For list
                inner_semicircle = [list(map(list , xyc2 * vmin))]
                outer_semicircle = [list(map(list, np.flipud(xyc2 * vmax)))]
                new_circle_lst = inner_semicircle + outer_semicircle


                if asas.trncons < 180:

                    # Add circles (ring-shape) to clipper as subject
                    pc.AddPath(pyclipper.scale_to_clipper(new_circle_tup), pyclipper.PT_SUBJECT, True)
                else:
                    #consider the whole SSD
                    pc.AddPaths(pyclipper.scale_to_clipper(circle_tup), pyclipper.PT_SUBJECT, True)

                # Add each other other aircraft to clipper as clip
                for j in range(np.shape(i_other)[0]):

                    ## Debug prints
                    ##print(traf.id[i] + " - " + traf.id[i_other[j]])
                    ## print(dist[ind[j]])
                    # Scale VO when not in LOS
                    if dist[ind[j]] > hsepm:

                        dist_mod = dist[ind[j]] #the value (not array) of the distance is needed for future computations


                        #direction of the VO's bisector
                        nd = drel[0,j,:]/dist_mod

                        R_pz = asas.R*asas.mar

                        R = np.array([[np.sqrt(1-(R_pz/dist_mod)**2), R_pz/dist_mod], [-R_pz/dist_mod, np.sqrt(1-(R_pz/dist_mod)**2)] ])

                        n_t1 = np.matmul(nd, R) #Direction of leg2
                        n_t2 = np.matmul(nd, np.transpose(R)) #Direction of leg1

                        #VO points
                        v_other = [gseast[i_other[j]],gsnorth[i_other[j]]]
                        legs_length = 10*vmax/cosalpha_i[j]
                        VO_points = np.array([v_other, np.add(n_t2*legs_length, v_other), np.add( n_t1* legs_length, v_other)])

                        # Normally VO shall be added of this other a/c
                        VO = pyclipper.scale_to_clipper(tuple(map(tuple, VO_points)))

                    else:
                        # Pair is in LOS
                        asas.los[i] = True
                        #In case two aircraft are in LoS, consider a samller RPZ
                        #in order to guarantee they get out of the LoS ASAP

                        dist_mod = dist[ind[j]] #the value (not array) of the distance is needed for future computations

                        R_pz = dist_mod*0.80

                        #direction of the VO's bisector
                        nd = drel[0,j,:]/dist_mod

                        R = np.array([[np.sqrt(1-(R_pz/dist_mod)**2), R_pz/dist_mod], [-R_pz/dist_mod, np.sqrt(1-(R_pz/dist_mod)**2)] ])

                        n_t1 = np.matmul(nd, R) #Direction of leg2
                        n_t2 = np.matmul(nd, np.transpose(R)) #Direction of leg1

                        #VO points
                        v_other = [gseast[i_other[j]],gsnorth[i_other[j]]]
                        legs_length = 10*vmax/cosalpha_i[j]
                        VO_points = np.array([v_other, np.add(n_t2*legs_length, v_other), np.add( n_t1* legs_length, v_other)])

                        # Normally VO shall be added of this other a/c
                        VO = pyclipper.scale_to_clipper(tuple(map(tuple, VO_points)))


                    # Add scaled VO to clipper
                    pc.AddPath(VO, pyclipper.PT_CLIP, True)

                # Execute clipper command
                FRV = pyclipper.scale_from_clipper(pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))
                ARV = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                # Scale back
                ARV = pyclipper.scale_from_clipper(ARV)


                # Check multi exteriors, if this layer is not a list, it means it has no exteriors
                # In that case, make it a list, such that its format is consistent with further code

                if len(ARV) == 0:
                    ARV_loc[i] = []
                    FRV_loc[i] = new_circle_lst
                else:
                    #Then:
                    if len(FRV) == 0:
                        FRV_loc[i] = []
                    else:
                        if not type(FRV[0][0]) == list:
                            FRV = [FRV]
                        FRV_loc[i] = FRV


                    if not type(ARV[0][0]) == list:
                        ARV = [ARV]
                    ARV_loc[i] = ARV





        #Storing the results into asas
        asas.FRV          = FRV_loc

        asas.ARV          = ARV_loc

        #The layers list
        asas.layers = [None, asas.ARV, None, None, None, None, None, None]

        return

    def qdrdist_matrix_indices(self,ntraf):
        """ This function gives the indices that can be used in the lon/lat-vectors """
        # The indices will be n*(n-1)/2 long
        # Only works for n >= 2, which is logical...
        # This is faster than np.triu_indices :)
        tmp_range = np.arange(ntraf - 1, dtype=np.int32)
        ind1 = np.repeat(tmp_range,(tmp_range + 1)[::-1])
        ind2 = np.ones(ind1.shape[0], dtype=np.int32)
        inds = np.cumsum(tmp_range[1:][::-1] + 1)
        np.put(ind2, inds, np.arange(ntraf * -1 + 3, 1))
        ind2 = np.cumsum(ind2, out=ind2)
        return ind1, ind2

    def qdrvs_matrix(self,vs1,vs2):
        #Realtive vertical speed: Check if 2 goes towards 1
        dvs = -vs1+vs2
        return dvs
