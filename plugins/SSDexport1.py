""" Plugin to generate SSD figures and save them in the project folder
Based on Leonor Inverno's SSD implementation
Author: Ander Okina """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, sim#, scr, tools
from bluesky.tools.aero import nm, ft
from bluesky.tools import geo, areafilter
import numpy as np
from datetime import datetime
from bluesky.traffic.asas import SSD
import time
from matplotlib import path


from collections import Counter

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
        'plugin_name':     'SSDEXPORT_test',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 45, #If changed also change variable in flexMatrix!!

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
            'SSDEXPORT ON/OFF NAME TIME',

            # Plugin on/off, name of area, area on/off
            '[onoff, txt, float]',

            # The name of your function in this plugin
            data.initialize,

            # a longer help text of your function.
            'Generate flexibility data based on SSD polygons. Input ON/OFF, name of area for simulation and time length of dataset (hours)'],
        'MAP': [
            'MAP ON/OFF',
            'onoff',
            data.printMap,
            'Save flexibility matrix with collected flexibility data'
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
        self.print = False
        self.scnName = str('')
        self.flexibility_global = np.array([])
        self.dlat = 0
        self.dlon = 0
        self.coord = []
        self.max_lat = 0
        self.min_lon = 0
        self.map = False
        self.avg_flex = 0
        self.dalt = 3000*0.3048
        self.ACcount = np.array([])
        self.cells = 0

    def update(self):

        if self.active== False: # only execute if plugin is turned on
            return

        self.time_stamp += 1




        if self.active == True:
            # Define heading and speed changes
            self.vmin = 130
            self.vmax = 300
            N_angle = 180

            angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
            xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
            SSD_lst = [list(map(list, np.flipud(xyc * self.vmax))), list(map(list, xyc * self.vmin))]

            # Outer circle: vmax
            SSD_outer = np.array(SSD_lst[0])
            x_SSD_outer = np.append(SSD_outer[:, 0], np.array(SSD_outer[0, 0]))
            y_SSD_outer = np.append(SSD_outer[:, 1], np.array(SSD_outer[0, 1]))
            # Inner circle: vmin
            SSD_inner = np.array(SSD_lst[1])
            x_SSD_inner = np.append(SSD_inner[:, 0], np.array(SSD_inner[0, 0]))
            y_SSD_inner = np.append(SSD_inner[:, 1], np.array(SSD_inner[0, 1]))
            # Initialize SSD variables with ntraf
            # Construct ASAS
            t = time.time()
            self.visualizeSSD(x_SSD_outer,y_SSD_outer,x_SSD_inner,y_SSD_inner)
            el = time.time()-t
            print('Time elapsed = '+str(el))


    def preupdate(self):
        return

    def reset(self):
        pass

    def flexMatrix(self,coords,t):
        #Generates the matrix in which the flexibility information is stored
        #Discretise coordinates so that flex[lat,lon,alt,iteration] = feas/(feas+unfeas)
        self.d = 1/10 #Discretise coordinates for every 1/20 of degree -> 3 nm
        self.nlat = np.int((np.amax(coords[:,0]) - np.amin(coords[:,0]))/self.d)+1
        self.nlon = np.int((np.amax(coords[:,1]) - np.amin(coords[:,1]))/self.d)+1
        self.max_lat = np.amax(coords[:,0])
        self.min_lon = np.amin(coords[:,1])
        self.nalt = int(50000/10000) # Altitude in 10000ft intervals (flight levels)
        sim_length = t/45+3
        flex = np.ones((self.nlon,self.nlat,self.nalt+1,int(sim_length)),dtype=np.float32)
        flex[:] = np.NaN #Only actual data will be number, otherwise NaN
        return flex


    #It's only called once
    def initialize(self,*args):

        if not args:
            return True, "SSDSEQ is currently " + ("ON" if self.active else "OFF")

        self.active = True if args[0]==True else False
        self.scnName = str(args[1])
        #Length of flexibility matrix's time axis
        if args[2]:
            self.t_sim = args[2]
        else:
            self.t_sim = 2 #2 hours as default

        #Generate area of EHAA FIR that will be used to filter traffic
        self.coord = np.array([ 50.76666667,   6.08333333,  51.83333333,   6.        ,\
        51.83333333,   6.        ,  52.26666667,   7.08333333,\
        52.26666667,   7.08333333,  53.33333333,   7.21666667,\
        53.33333333,   7.21666667,  53.58333333,   6.58333333,\
        53.58333333,   6.58333333,  55.        ,   6.5       ,\
        55.        ,   6.5       ,  55.        ,   5.        ,\
        55.        ,   5.        ,  51.5       ,   2.        ,\
        51.5       ,   2.        ,  51.25      ,   4.        ,\
        51.25      ,   4.        ,  51.46666667,   4.55      ,\
        51.46666667,   4.55      ,  51.16666667,   5.83333333,\
        51.16666667,   5.83333333,  50.83333333,   5.66666667,\
        50.83333333,   5.66666667,  50.76666667,   6.08333333,\
        50.76666667,   6.08333333,  50.76666667,   6.08333333])
        areafilter.defineArea('NL', 'POLY', self.coord,50000,100)

        if self.active:
            self.ACcount = np.zeros(int(self.t_sim*60*60/30)+5)
            self.flexibility_global = self.flexMatrix(np.reshape(self.coord,(int(len(self.coord)/2),2)),self.t_sim*60*60)
            print(np.shape(self.flexibility_global))
            self.grid = self.createGrid()
            np.save('selgrid.npy',self.grid)
        stack.stack('AREA NL')

        return True

    def createGrid(self):
        #Make grid of cells only inside the dutch airspace, get rid of the rest
        polygon = [[self.coord[1::2][i],self.coord[0::2][i]] for i in range(len(self.coord[0::2]))]
        p = path.Path(polygon)
        #Generate list of latitude and longitude indices
        lats = np.arange(len(self.flexibility_global[0,:,0,0]))
        lons = np.arange(len(self.flexibility_global[:,0,0,0]))
        #Genrate list of altitudes
        alts = np.arange(len(self.flexibility_global[0,0,:,0]))
        #Create the grid with all cells
        gridx,gridy,gridz = np.meshgrid(lons,lats,alts)
        grid0 = np.vstack([gridx.ravel(),gridy.ravel(),gridz.ravel()]) #0>lon,1>lat
        #Convert to actual coordinates
        latsn, lonsn = self.coord2latlon(grid0[1],grid0[0]) #lat and lon
        points = np.array([[lonsn[i]+self.d/2,latsn[i]-self.d/2] for i in range(len(latsn))])
        #Select points within polygon
        inside = p.contains_points(points) #Bool mask for points inside
        #Select grid points that are inside the polygon
        gridx = grid0[0][inside]
        gridy = grid0[1][inside]
        gridz = grid0[2][inside]
        grid = np.vstack([gridx.ravel(),gridy.ravel(),gridz.ravel()])


        np.save('grid_pts.npy',np.array([latsn[inside],lonsn[inside]]))
        np.save('grid.npy',grid)
        self.cells = len(gridx)
        print(np.shape(grid))
        return grid

    def printMap(self, *args):
        #Prints the map with the available information of flexibility
        stack.stack('ECHO Saving flexibility data...')
        self.maps = True if args[0]==True else False
        if self.maps:
            #Save information for processing
            #flex_map = np.nanmean(self.flexibility_global,axis=3) #Mean for all time steps discarding NaN
            #if os.path.exists('flexibility3d.npy'): #If there's a file already, remove it
            #    os.remove('flexibility3d.npy')
            #np.save('flexibility3d.npy', flex_map)
            #stack.stack('ECHO 3D array saved')
            #flex_map = np.nanmean(flex_map,axis=2) #Mean for all altitudes discarding NaN
            if os.path.exists('flexibility_'+self.scnName+'.npy'): #If there's a file already, remove it
                os.remove('flexibility_'+self.scnName+'.npy')
            np.save('flexibility_'+self.scnName+'.npy', self.flexibility_global)
            x,y = self.discreteCoord(self.coord[0::2], self.coord[1::2])
            airspace = np.resize(np.stack((y,x)),(1,2*len(x)))
            np.save('airspace.npy', airspace)
            #fig, ax0 = plt.subplots(1)
            np.save('ACcount.npy',self.ACcount)

            self.maps = False
            stack.stack('ECHO Data saved')


    def visualizeSSD(self, x_SSD_outer,y_SSD_outer,x_SSD_inner,y_SSD_inner):
        #Generate the SSD velocity obstacles and convert them to polygons
        #so that their areas can be computed
        inNL = areafilter.checkInside('NL', traf.lat, traf.lon, traf.alt)
        #Obtain list of aircraft inside the defined area
        #Obtain the indices of lat and lon of aircraft for flexibiity matrix
        self.ACcount[self.time_stamp-1] = np.sum(inNL)
        latcell, loncell = self.coord2latlon(self.grid[1],self.grid[0])
        FRV_all,ARV_all = self.constructSSD(traf, latcell, loncell)
        #Generate the maximum and minimum speed circles Polygon
        N_angle = 180
        vmax = self.vmax
        vmin = self.vmin
        # # Use velocity limits for the ring-shaped part of the SSD
        # Discretize the circles using points on circle
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
        # Put points of unit-circle in a (180x2)-array (CW)
        xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
        # Map them into the format pyclipper wants. Outercircle CCW, innercircle CW
        circle_lst_out = list(map(list, np.flipud(xyc * vmax)))
        circle_lst_in = list(map(list , xyc * vmin))
        circ_out_pol = Polygon(circle_lst_out)
        circ_in_pol = Polygon(circle_lst_in)
        circ_diff = circ_out_pol.difference(circ_in_pol)
        tot_area = circ_out_pol.area - circ_in_pol.area
        EHAMiy,EHAMix = self.discreteCoord(52.3,4.8) #Index of EHAM coordinates on grid
        print(EHAMiy,EHAMix)

        flex = np.ones(len(FRV_all))
        for i in range(len(FRV_all)): #Evaluate the VO of each grid point
            if i%50==0:
                print('Cell2 '+str(i))
            FRV = FRV_all[i]
            ARV = ARV_all[i]

            if ARV:
                for j in range(len(ARV)):
                    FRV_1 = np.array(ARV[j])
                    x_FRV1 = np.append(FRV_1[:,0] , np.array(FRV_1[0,0]))
                    y_FRV1 = np.append(FRV_1[:,1] , np.array(FRV_1[0,1]))
                    FRV1 = FRV_1.tolist()
                    if j>0:
                        un_FRV = un_FRV.union(Polygon(FRV1))
                    elif j==0:
                        un_FRV = Polygon(FRV1)
                un_FRV = un_FRV.intersection(circ_diff) #Intersect ARV with complete solution space
                free_area = un_FRV.area

                flex[i] = free_area/tot_area
                if flex[i]>1:
                    #Wrong return of ARV, no aircraft in conflict -> flex=1
                    self.flexibility_global[self.grid[0][i],self.grid[1][i],self.grid[2][i],self.time_stamp-1] = 1
                else:
                    self.flexibility_global[self.grid[0][i],self.grid[1][i],self.grid[2][i],self.time_stamp-1] = flex[i]

            else: #No velocity obstacles -> Full flexibility
                flex[i] = 1
                self.flexibility_global[self.grid[0][i],self.grid[1][i],self.grid[2][i],self.time_stamp-1] = 1

            if self.grid[0][i]==EHAMix and self.grid[1][i]==EHAMiy:
                #print(self.grid[0][i],self.grid[1][i])
                #print(EHAMix,EHAMiy)
                if self.print:
                    fig, ax = plt.subplots()
                    if type(un_FRV)==Polygon:
                        x,y = un_FRV.exterior.xy
                        plt.plot(x,y, color ='red')
                        ax.fill(x, y, color = '#C0C0C0') #gray
                        x,y = circ_out_pol.exterior.xy
                        plt.plot(x,y,color = 'black')
                        x,y = circ_in_pol.exterior.xy
                        plt.plot(x,y,color = 'black')
                        ax.fill(x, y, color = 'white') #gray
                        plt.title(str(self.grid[0][i])+', '+str(self.grid[1][i])+', '+str(self.grid[2][i]))
                        plt.show()
                    else: #Multipolygon
                        for k in range(len(list(un_FRV))):
                            x,y = un_FRV[k].exterior.xy
                            plt.plot(x,y, color = 'red')
                            ax.fill(x, y, color = '#C0C0C0') #gray
                            x,y = circ_out_pol.exterior.xy
                            plt.plot(x,y,color = 'green')
                            x,y = circ_in_pol.exterior.xy
                            plt.plot(x,y,color = 'black')
                            ax.fill(x, y, color = 'white') #gray
                        plt.title(str(self.grid[0][i])+', '+str(self.grid[1][i])+', '+str(self.grid[2][i]))
                        plt.show()

        # flexplot = np.nanmean(self.flexibility_global[:,:,:,self.time_stamp-1],axis=2)
        # a = plt.scatter(flexplot)
        # plt.show()
        # #plt.clim(0,1)
        # plt.colorbar(a,label='Flexibility');

        #print('Flex of '+str(self.grid[0][i])+','+str(self.grid[1][i])+','+str(self.grid[2][i])+' = '+str(self.flexibility_global[self.grid[0][i],self.grid[1][i],int(round(self.grid[2][i]/(5000*0.3048))),self.time_stamp-1]))
        return

    def discreteCoord(self,lat,lon):
        in_lat = ((self.max_lat-lat)/self.d) #Indice in matrix of latitude -> from up to down
        in_lon = ((lon-self.min_lon)/self.d) #Indice in matrix of longitude
        return in_lat.astype(int), in_lon.astype(int)

    def coord2latlon(self,iny,inx):
        lat = self.max_lat-iny*self.d
        lon = inx*self.d+self.min_lon
        return np.array(lat),np.array(lon)


    def constructSSD(self, traf, latcell, loncell):
        """ Calculates the FRV and ARV of the SSD """

        #Forbidden Reachable Velocity regions
        FRV          = [None]*self.cells

        #Allowed Reachable Velocity regions
        ARV          = [None]*self.cells


        altcell = self.grid[2]*5000*0.3048 #[m]

        N = 0
        # Parameters
        N_angle = 180                   # [-] Number of points on circle (discretization)
        vmin    = self.vmin             # [m/s] Defined in asas.py
        vmax    = self.vmax             # [m/s] Defined in asas.py
        hsep    = 5*1852                # [m] Horizontal separation (5 NM)
        margin  = 0.5                   # [-] Safety margin for evasion
        hsepm   = hsep * margin         # [m] Horizontal separation with safety margin
        alpham  = 0.4999 * np.pi        # [rad] Maximum half-angle for VO
        betalos = np.pi / 4             # [rad] Minimum divertion angle for LOS (45 deg seems optimal)
        adsbmax = 100. * nm             # [m] Maximum ADS-B range
        beta    =  np.pi/4 + betalos/2
        tau = 15*60                     # 15 min to LoS, for rounding off of VO

        # Relevant info from traf
        gsnorth = traf.gsnorth
        gseast  = traf.gseast
        lat     = traf.lat
        lon     = traf.lon
        alt     = traf.alt
        ntraf   = traf.ntraf
        hdg     = traf.hdg
        gs      = traf.tas
        vs      = traf.vs



        # Local variables, will be put into asas later
        FRV_loc          = [None] * self.cells
        ARV_loc          = [None] * self.cells
        FRV_locr         = [None] * self.cells
        ARV_locr         = [None] * self.cells
        # For calculation purposes
        ARV_calc_loc     = [None] * self.cells
        FRV_area_loc     = np.zeros(self.cells, dtype=np.float32)
        ARV_area_loc     = np.zeros(self.cells, dtype=np.float32)

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
            ARV_loc[0] = circle_lst
            FRV_loc[0] = []
            return

        # Function qdrdist_matrix needs 4 vectors as input (lat1,lon1,lat2,lon2)
        # To be efficient, calculate all qdr and dist in one function call
        # Example with ntraf = 5:   ind1 = [0,0,0,0,1,1,1,2,2,3]
        #                           ind2 = [1,2,3,4,2,3,4,3,4,4]
        # This way the qdrdist is only calculated once between every aircraft
        # To get all combinations, use this function to get the indices
        ind1, ind2 = np.meshgrid(np.arange(ntraf),np.arange(self.cells))
        indx = np.vstack([ind1.ravel(),ind2.ravel()])
        ind1 = indx[0]
        ind2 = indx[1]
        # Get absolute bearing [deg] and distance [nm]
        # Not sure abs/rel, but qdr is defined from [-180,180] deg, w.r.t. North
        [qdr, dist] = geo.qdrdist_matrix(latcell[ind2]-0.5, loncell[ind2]+0.5, lat[ind1], lon[ind1])
        #print(str(np.amin(dist))+' between '+str(latcell[ind2[np.where(np.amin(dist))]])+','+str(loncell[ind2[np.where(np.amin(dist))]])+' and '+str(lon[ind1[np.where(np.amin(dist))]])+','+str(lat[ind1[np.where(np.amin(dist))]]))
        #print(len(np.where(dist<75)))


        # ##########Debugging code###########
        # #Save data of distance of aircraft to each grid point
        # qdr2,dist2 = geo.qdrdist_matrix(lat[0],lon[0],latcell,loncell)
        # np.save('grid_comp.npy', np.array([latcell,loncell]))
        # np.save('grid_ind.npy',np.array([self.grid[0],self.grid[1]]))
        # np.save('distance.npy',dist2)
        # np.save('track.npy',qdr2)
        # print(lat[0],lon[0])

        # Put result of function from matrix to ndarray
        qdr  = np.reshape(np.array(qdr), np.shape(ind1))
        dist = np.reshape(np.array(dist), np.shape(ind1))
        # SI-units from [deg] to [rad]
        qdr  = np.deg2rad(qdr)
        # Get distance from [nm] to [m]
        dist = dist * nm
        #Altitude difference between cells and traffic (center of cell in terms of altitude)
        dalt = (altcell[ind2]+5000*0.3048) - alt[ind1] #[m]

        #Heading from [deg] to [rad]
        hdg = np.deg2rad(hdg)

        #Compute horizontal tau (dist/dist_rate)
        rdot = gs[ind1]*np.cos(qdr-hdg[ind1]) #[m/s]
        tau_h = dist/rdot #[s]

        #Compute vertical tau (alt/alt_range)
        hdot = vs[ind1]*np.sign(dalt) #[m/s]
        tau_v = (np.absolute(dalt))/hdot #[s]

        # In LoS the VO can't be defined, act as if dist is on edge
        dist[dist < hsepm] = hsepm

        # Calculate vertices of Velocity Obstacle (CCW)
        # These are still in relative velocity space, see derivation in appendix
        # Half-angle of the Velocity obstacle [rad]
        # Include safety margin
        alpha = np.arcsin(hsepm / dist)
        # Limit half-angle alpha to 89.982 deg. Ensures that VO can be constructed
        alpha[alpha > alpham] = alpham
        cosalpha = np.cos(alpha)
        # Relevant sin/cos/tan
        sinqdr = np.sin(qdr)
        cosqdr = np.cos(qdr)
        tanalpha = np.tan(alpha)
        cosqdrtanalpha = cosqdr * tanalpha
        sinqdrtanalpha = sinqdr * tanalpha

        # Relevant x1,y1,x2,y2 (x0 and y0 are zero in relative velocity space)
        x1 = (sinqdr + cosqdrtanalpha) * 2 * vmax
        x2 = (sinqdr - cosqdrtanalpha) * 2 * vmax
        y1 = (cosqdr - sinqdrtanalpha) * 2 * vmax
        y2 = (cosqdr + sinqdrtanalpha) * 2 * vmax

        # Loop over every grid cell
        t = []
        for i in range(self.cells):
            if i%50==0:
                print('Cell '+str(i))
            # Calculate SSD only for aircraft in conflict (See formulas appendix)
            # SSD for aircraft i
            # Get indices that belong to cell i
            ind = np.where(ind2 == i)[0]
            # Check whether there are any aircraft in the vicinity
            if len(ind) == 0:
                # No aircraft in the vicinity
                # Map them into the format ARV wants. Outercircle CCW, innercircle CW
                ARV_loc[i] = circle_lst
                FRV_loc[i] = []

            else:
                # The i's of the other aircraft
                i_other = np.arange(ntraf)
                cosalpha_i = cosalpha[ind]


                #### Filter aircraft to be taken into account ####
                #Aircraft with a positive tau smaller than a value or aircraft closer than 10nm
                ac_adsb = np.where(np.logical_or(np.logical_and(tau_h[ind]<4000,tau_h[ind]>0),dist[ind] < 6*nm))[0]
                #Aircraft with positive vertical tau smaller than 50 or with altitude difference smaller than 1500ft
                ac_alt1 = np.where(np.logical_or(np.logical_and(tau_v[ind]<800,tau_v[ind]>0),dalt[ind] < 1500*ft))[0]
                #concatenate both arrays and count which index happens more than once (both vertical and horizontally relevant)
                ac_joint = np.concatenate([ac_adsb,ac_alt1])
                ac_adsb =  [item for item, count in Counter(ac_joint).items() if count > 1] #Only for A/C in ads-b range AND VS converging

                # Now account for ADS-B range in indices of other aircraft (i_other)
                ind = ind[ac_adsb]
                i_other = i_other[ac_adsb]
                # VO from 2 to 1 is mirror of 1 to 2. Only 1 to 2 can be constructed in
                # this manner, so need a correction vector that will mirror the VO
                fix = np.ones(np.shape(i_other))
                #fix[i_other < i] = -1

                drel_x, drel_y = fix*dist[ind]*np.sin(qdr[ind]), fix*dist[ind]*np.cos(qdr[ind])
                drel = np.dstack((drel_x,drel_y))

                # Get vertices in an x- and y-array of size (ntraf-1)*3x1
                x = np.concatenate((gseast[i_other],
                                    x1[ind] * fix + gseast[i_other],
                                    x2[ind] * fix + gseast[i_other]))
                y = np.concatenate((gsnorth[i_other],
                                    y1[ind] * fix + gsnorth[i_other],
                                    y2[ind] * fix + gsnorth[i_other]))
                # Reshape [(ntraf-1)x3] and put arrays in one array [(ntraf-1)x3x2]
                x = np.transpose(x.reshape(3, np.shape(i_other)[0]))
                y = np.transpose(y.reshape(3, np.shape(i_other)[0]))
                xy = np.dstack((x,y))

                # Make a clipper object
                #pc = pyclipper.Pyclipper()
                pcr = pyclipper.Pyclipper()
                # Add circles (ring-shape) to clipper as subject
                #pc.AddPaths(pyclipper.scale_to_clipper(circle_tup), pyclipper.PT_SUBJECT, True)
                pcr.AddPaths(pyclipper.scale_to_clipper(circle_tup), pyclipper.PT_SUBJECT, True)


                # Add each other aircraft to clipper as clip
                for j in range(np.shape(i_other)[0]):
                    # Scale VO when not in LOS

                    dist_mod = dist[ind[j]]

                    if dist_mod > hsepm:

                        #direction of the VO's bisector
                        nd = drel[0,j,:]/dist_mod
                        R = np.array([[np.sqrt(1-(hsepm/dist_mod)**2), hsepm/dist_mod], [-hsepm/dist_mod, np.sqrt(1-(hsepm/dist_mod)**2)] ])
                        n_t1 = np.matmul(nd, R) #Direction of leg2
                        n_t2 = np.matmul(nd, np.transpose(R)) #Direction of leg1

                        #VO points
                        v_other = [gseast[i_other[j]],gsnorth[i_other[j]]]
                        legs_length = 2*vmax/cosalpha_i[j]
                        VO_points = np.array([v_other, np.add(n_t2*legs_length, v_other), np.add( n_t1* legs_length, v_other)])

                        #take only the farthest 2 vertices of the VO and make a tupple
                        vertexes = tuple(map(tuple,VO_points[1:,:]))
                        try:
                            VO_r1 = self.roundoff(tau, hsepm, dist_mod, VO_points[0,:], nd, n_t1, n_t2, vertexes, xyc)
                            VO_r = pyclipper.scale_to_clipper(VO_r1)
                        except:
                            #If error using roudoff, use normal velocity obstacle
                            VO_r = pyclipper.scale_to_clipper(tuple(map(tuple,xy[j,:,:])))
                    else:
                        # Pair is in LOS, instead of triangular VO, use darttip
                        qdr_los = qdr[ind[j]] + np.pi

                        # Length of inner-leg of darttip
                        leg = 1.1 * vmax / np.cos(beta) * np.array([1,1,1,0])
                        # Angles of darttip
                        angles_los = np.array([qdr_los + 2 * beta, qdr_los, qdr_los - 2 * beta, 0.])
                        # Calculate coordinates (CCW)
                        x_los = leg * np.sin(angles_los)
                        y_los = leg * np.cos(angles_los)
                        # Put in array of correct format
                        xy_los = np.vstack((x_los,y_los)).T
                        # Scale darttip
                        #VO = pyclipper.scale_to_clipper(tuple(map(tuple,xy_los)))
                        VO_r = pyclipper.scale_to_clipper(tuple(map(tuple,xy_los)))

                    # Add scaled VO to clipper
                    #pc.AddPath(VO, pyclipper.PT_CLIP, True)
                    pcr.AddPath(VO_r, pyclipper.PT_CLIP, True)

                # Execute clipper command
                #FRV = pyclipper.scale_from_clipper(pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))
                #ARV = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

                #Execute clipper command for rounded VO
                FRVr = pyclipper.scale_from_clipper(pcr.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))
                ARVr = pyclipper.scale_from_clipper(pcr.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))


                # Scale back
                #ARV = pyclipper.scale_from_clipper(ARV)

                # Check if ARV or FRV is empty
                if len(ARVr) == 0:
                    # No aircraft in the vicinity
                    # Map them into the format ARV wants. Outercircle CCW, innercircle CW
                    # ARV_loc[i] = []
                    # FRV_loc[i] = circle_lst
                    if len(ARVr) ==0: #indeed, there's only the possibility to have ARV_3/5 equal to zero if the original ARV is zero
                        ARV_locr[i] = []
                        FRV_locr[i] = circle_lst
                elif len(FRVr) == 0:
                    # Should not happen with one a/c or no other a/c in the vicinity.
                    # These are handled earlier. Happens when RotA has removed all
                    # Map them into the format ARV wants. Outercircle CCW, innercircle CW
                    # ARV_loc[i] = circle_lst
                    # FRV_loc[i] = []
                    FRV_locr[i] = []
                    ARV_locr[i] = circle_lst

                else:

                    #Then/Except:
                    if FRVr:
                        #Calcular tambem ARVs
                        if not type(ARVr[0][0]) == list:
                            ARVr = [ARVr]
                        ARV_locr[i] = ARVr
                        if not type(FRVr[0][0]) == list:
                            FRVr = [FRVr]
                        FRV_locr[i] = FRVr
                    else:
                        FRV_locr[i] = []
                        ARV_locr[i] = circle_lst

                    # # Check multi exteriors, if this layer is not a list, it means it has no exteriors
                    # # In that case, make it a list, such that its format is consistent with further code
                    # if not type(FRV[0][0]) == list:
                    #     FRV = [FRV]
                    # if not type(ARV[0][0]) == list:
                    #     ARV = [ARV]
                    # # Store in asas
                    # FRV_loc[i] = FRV
                    # ARV_loc[i] = ARV



        # If sequential approach, the local should go elsewhere
        FRV          = FRV_locr
        ARV          = ARV_locr

        return FRV,ARV



    def constructSSD2(self,grid, traf, priocode = "RS1"):
        """ Calculates the FRV and ARV of the SSD """
        #Forbidden Reachable Velocity regions
        FRV          = [None]

        #Allowed Reachable Velocity regions
        ARV          = [None]

        latown,lonown = self.coord2latlon(grid[1],grid[0])

        altown = grid[2]*5000*0.3048 #[m]

        N = 0
        # Parameters
        N_angle = 180                   # [-] Number of points on circle (discretization)
        vmin    = self.vmin             # 180kts - [m/s] Defined in asas.py
        vmax    = self.vmax             # 350kts - [m/s] Defined in asas.py
        hsep    = 5*1852                # [m] Horizontal separation (5 NM)
        margin  = 1                     # [-] Safety margin for evasion
        hsepm   = hsep * margin         # [m] Horizontal separation with safety margin
        alpham  = 0.4999 * np.pi        # [rad] Maximum half-angle for VO
        betalos = np.pi / 4             # [rad] Minimum divertion angle for LOS (45 deg seems optimal)
        adsbmax = 40. * nm              # [m] Maximum ADS-B range
        beta    =  np.pi/4 + betalos/2
        if priocode == "RS7" or priocode == "RS8":
            adsbmax /= 2

        # Relevant info from traf
        gsnorth = traf.gsnorth
        vs      = traf.vs
        gseast  = traf.gseast
        lat     = traf.lat
        lon     = traf.lon
        ntraf   = traf.ntraf
        hdg     = traf.hdg
        gs      = traf.tas
        alt     = traf.alt

        # Local variables, will be put into asas later
        FRV_loc          = [None]
        ARV_loc          = [None]
        # For calculation purposes
        ARV_calc_loc     = [None]
        FRV_area_loc     = np.zeros(traf.ntraf, dtype=np.float32)
        ARV_area_loc     = np.zeros(traf.ntraf, dtype=np.float32)

        # # Use velocity limits for the ring-shaped part of the SSD
        # Discretize the circles using points on circle
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
        # Put points of unit-circle in a (180x2)-array (CW)
        xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
        # Map them into the format pyclipper wants. Outercircle CCW, innercircle CW
        circle_tup = (tuple(map(tuple, np.flipud(xyc * vmax))), tuple(map(tuple , xyc * vmin)))
        circle_lst = [list(map(list, np.flipud(xyc * vmax))), list(map(list , xyc * vmin))]

        # Function qdrdist_matrix needs 4 vectors as input (lat1,lon1,lat2,lon2)
        # To be efficient, calculate all qdr and dist in one function call
        # Example with ntraf = 5:   ind1 = [0,0,0,0,1,1,1,2,2,3]
        #                           ind2 = [1,2,3,4,2,3,4,3,4,4]
        # This way the qdrdist is only calculated once between every aircraft
        # To get all combinations, use this function to get the indices
        ind1 = np.arange(ntraf) #ind1-traf, ind2-grid

        # Get absolute bearing [deg] and distance [nm]
        # Not sure abs/rel, but qdr is defined from [-180,180] deg, w.r.t. North
        [qdr, dist] = geo.qdrdist_matrix(latown, lonown, lat[ind1], lon[ind1])
        # Put result of function from matrix to ndarray
        qdr  = np.reshape(np.array(qdr), np.shape(ind1))
        dist = np.reshape(np.array(dist), np.shape(ind1))
        #Compute altitude difference between all indexes
        dalt = altown - alt[ind1] #[m]
        # SI-units from [deg] to [rad]
        qdr  = np.deg2rad(qdr)
        # Get distance from [nm] to [m]
        dist = dist * nm
        #Convert heading to radian
        hdg = np.deg2rad(hdg)

        #Compute horizontal tau (dist/dist_rate)
        rdot = gs[ind1]*np.cos(qdr-hdg[ind1]) #[m/s]
        tau_h = dist/rdot #[s]

        #Compute vertical tau (alt/alt_range)
        hdot = vs[ind1]*np.sign(dalt) #[m/s]
        tau_v = (np.absolute(dalt))/hdot #[s]

        # In LoS the VO can't be defined, act as if dist is on edge
        dist[dist < hsepm] = hsepm

        # Calculate vertices of Velocity Obstacle (CCW)
        # These are still in relative velocity space, see derivation in appendix
        # Half-angle of the Velocity obstacle [rad]
        # Include safety margin
        alpha = np.arcsin(hsepm / dist)
        # Limit half-angle alpha to 89.982 deg. Ensures that VO can be constructed
        alpha[alpha > alpham] = alpham
        # Relevant sin/cos/tan
        sinqdr = np.sin(qdr)
        cosqdr = np.cos(qdr)
        tanalpha = np.tan(alpha)
        cosqdrtanalpha = cosqdr * tanalpha
        sinqdrtanalpha = sinqdr * tanalpha

        # Relevant x1,y1,x2,y2 (x0 and y0 are zero in relative velocity space)
        x1 = (sinqdr + cosqdrtanalpha) * 2 * vmax
        x2 = (sinqdr - cosqdrtanalpha) * 2 * vmax
        y1 = (cosqdr - sinqdrtanalpha) * 2 * vmax
        y2 = (cosqdr + sinqdrtanalpha) * 2 * vmax
        # Calculate SSD for all aircraft
        # SSD for aircraft i
        # Get indices that belong to aircraft i
        ind = ind1
        # Check whether there are any aircraft in the vicinity
        if len(ind) == 0:
            # No aircraft in the vicinity
            # Map them into the format ARV wants. Outercircle CCW, innercircle CW
            ARV_loc = circle_lst
            FRV_loc = []
            ARV_calc_loc = ARV_loc[i]
        else:
            # The i's of the other aircraft
            i_other = ind1
            #### Filter aircraft to be taken into account ####
            #Aircraft with a positive tau smaller than a value or aircraft closer than 10nm
            ac_adsb = np.where(np.logical_or(np.logical_and(tau_h[ind]<1E50,tau_h[ind]>0),dist[ind] < 6*nm))[0]
            #Aircraft with positive vertical tau smaller than 50 or with altitude difference smaller than 1500ft
            ac_alt1 = np.where(np.logical_or(np.logical_and(tau_v[ind]<1E50,tau_v[ind]>0),dalt[ind] < 1500*ft))[0]
            #concatenate both arrays and count which index happens more than once (both vertical and horizontally relevant)
            ac_joint = np.concatenate([ac_adsb,ac_alt1])
            ac_adsb =  [item for item, count in Counter(ac_joint).items() if count > 1] #Only for A/C in ads-b range AND VS converging

            # Now account for ADS-B range in indices of other aircraft (i_other)
            ind = ind[ac_adsb]
            i_other = i_other[ac_adsb]
            # VO from 2 to 1 is mirror of 1 to 2. Only 1 to 2 can be constructed in
            # this manner, so need a correction vector that will mirror the VO
            fix = np.ones(np.shape(i_other))
            #fix[i_other < i] = -1
            # Relative bearing [deg] from [-180,180]
            # (less required conversions than rad in RotA)
            fix_ang = np.zeros(np.shape(i_other))
            #fix_ang[i_other < i] = 180.


            # Get vertices in an x- and y-array of size (ntraf-1)*3x1
            x = np.concatenate((gseast[i_other],
                                x1[ind] * fix + gseast[i_other],
                                x2[ind] * fix + gseast[i_other]))
            y = np.concatenate((gsnorth[i_other],
                                y1[ind] * fix + gsnorth[i_other],
                                y2[ind] * fix + gsnorth[i_other]))
            # Reshape [(ntraf-1)x3] and put arrays in one array [(ntraf-1)x3x2]
            x = np.transpose(x.reshape(3, np.shape(i_other)[0]))
            y = np.transpose(y.reshape(3, np.shape(i_other)[0]))
            xy = np.dstack((x,y))

            # Make a clipper object
            pc = pyclipper.Pyclipper()

            # Add circles (ring-shape) to clipper as subject
            pc.AddPaths(pyclipper.scale_to_clipper(circle_tup), pyclipper.PT_SUBJECT, True)

            # Add each other other aircraft to clipper as clip
            for j in range(np.shape(i_other)[0]):
                ## Debug prints
                ## print(traf.id[i] + " - " + traf.id[i_other[j]])
                ## print(dist[ind[j]])
                # Scale VO when not in LOS
                if dist[ind[j]] > hsepm:
                    # Normally VO shall be added of this other a/c
                    VO = pyclipper.scale_to_clipper(tuple(map(tuple,xy[j,:,:])))
                else:
                    # Pair is in LOS, instead of triangular VO, use darttip
                    # Check if bearing should be mirrored
                    qdr_los = qdr[ind[j]]
                    # Length of inner-leg of darttip
                    leg = 1.1 * vmax / np.cos(beta) * np.array([1,1,1,0])
                    # Angles of darttip
                    angles_los = np.array([qdr_los + 2 * beta, qdr_los, qdr_los - 2 * beta, 0.])
                    # Calculate coordinates (CCW)
                    x_los = leg * np.sin(angles_los)
                    y_los = leg * np.cos(angles_los)
                    # Put in array of correct format
                    xy_los = np.vstack((x_los,y_los)).T
                    # Scale darttip
                    VO = pyclipper.scale_to_clipper(tuple(map(tuple,xy_los)))
                # Add scaled VO to clipper
                pc.AddPath(VO, pyclipper.PT_CLIP, True)


            # Execute clipper command
            FRV = pyclipper.scale_from_clipper(pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))

            ARV = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

            # Scale back
            ARV = pyclipper.scale_from_clipper(ARV)

            # Check if ARV or FRV is empty
            if len(ARV) == 0:
                # No aircraft in the vicinity
                # Map them into the format ARV wants. Outercircle CCW, innercircle CW
                ARV_loc = []
                FRV_loc = circle_lst
                ARV_calc_loc = []

            elif len(FRV) == 0:
                # Should not happen with one a/c or no other a/c in the vicinity.
                # These are handled earlier. Happens when RotA has removed all
                # Map them into the format ARV wants. Outercircle CCW, innercircle CW
                ARV_loc = circle_lst
                FRV_loc = []
                ARV_calc_loc = circle_lst
            else:
                # Check multi exteriors, if this layer is not a list, it means it has no exteriors
                # In that case, make it a list, such that its format is consistent with further code
                if not type(FRV[0]) == list:
                    FRV = [FRV]
                if not type(ARV[0]) == list:
                    ARV = [ARV]
                # Store in asas
                FRV_loc = FRV
                ARV_loc = ARV

        return FRV_loc,ARV_loc


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
            #asas.ARV_min[0] = circle_lst
            #asas.ARV_tla[0] = circle_lst

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
                # Aircraft that are within ADS-B range and in converging vertical speeds
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

    def qdrdist_matrix_indices(self,ntraf,grid):
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

    def roundoff(self, tau, R_pz, dist_mod, xy, nd, n_t1, n_t2, vertexes, xyc):

        r_tc = R_pz/tau
        v_tc = np.add(np.array(dist_mod/tau*nd), xy)  #circle's center

        point1 = r_tc*np.array([-n_t1[1],n_t1[0]]) + v_tc #intersection of leg2 with the circle
        point2 = r_tc*np.array([n_t2[1],-n_t2[0]]) + v_tc #intersection of leg1 with the circle

        legs_points = [[point1[0],point1[1]], [point2[0],point2[1]], list(vertexes[0]), list(vertexes[1])]

        #Define the circle's coordinates
        circle_lst = [list(map(list, np.flipud(xyc * r_tc)))]
        circle1 = np.array(circle_lst[0])
        circle1 = np.add(circle1, v_tc) #add center of circle

        legs = pyclipper.Pyclipper()
        legs.AddPath(pyclipper.scale_to_clipper(legs_points), pyclipper.PT_SUBJECT, True)

        circle_cut = tuple(map(tuple, circle1))
        legs.AddPath(pyclipper.scale_to_clipper(circle_cut), pyclipper.PT_CLIP, True)

        union = pyclipper.scale_from_clipper(legs.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))
        union = np.array(union[0])
        #PUT THE UNION IN TUPLE SO IT CAN BE ADDED AS AN OBJECT TO A PYCLIPPER OBJECT
        VO_round = tuple(map(tuple, union))

        return VO_round
