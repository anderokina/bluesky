""" The plugin for the computation of the flexibility metric """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, settings, navdb, traf, sim#, scr, tools
import numpy as np
import copy
import pandas as pd
from bluesky.traffic.asas import StateBasedCD as cd
from bluesky import vincenti as vct
from bluesky.traffic import traffic as tr
from bluesky.tools import geo
from bluesky.tools.aero import nm
from bluesky.tools import plotter

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {

        'plugin_name':     'FLEXIBILITY',
        'plugin_type':     'sim',
        'update':          update1,
        'update_interval': 0.5,
        'reset':         reset
        }

    stackfunctions = {
        # The command name for your function
        'FLEXDISPLAY': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'FLEXDISPLAY ON/OFF',

            # A list of the argument types your function accepts. For a description of this, see ...
            '[onoff]',

            # The name of your function in this plugin
            flexdisplay,

            # a longer help text of your function.
            'Activate the display of the flexibility parameter in a plot']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

safe_h = 2.5 #nm
safe_v = 500 #ft
t_horizon = 200 #seconds
np.set_printoptions(threshold=np.inf)


#def algorithm():

    #for i in traf.id:
        #compute flexibility of each traffic
        #use other traffic
        #lat = traf.lat[i]
        #lon = traf.long[i]
        #Obtain all the reachable points within the given "disc"
        #Simulation characteristics:
        #hdg_max = 80
        #spd_max = 20
        #dhdg = 5
        #dspd = 2.5

        #for n in traf.id:
            #inner loop to compare with other traffics
            #pass
        #compare with SUA
        #extract flex for flight i
    #keep flex for all flights in a log
    #compute instant flexibility for all traffic
    #if a flight leaves the airspace compute trajectory flexibility

def update():
    ownship = np.array([], dtype=([('lat','f'), ('lon','f'),('alt','f'),('vs','f'),('id','U'),('spd','f'),('hdg','f')]))
    feas_t = np.zeros(len(traf.id))
    for i in range(len(traf.id)): #For each aircraft, compare to the rest of aircraft
        hdg,spd = computeDisk(traf.lat[i],traf.lon[i],traf.tas[i],traf.hdg[i])
        hdg = np.array(hdg)
        spd = np.array(spd)
        states = np.array(np.meshgrid(hdg,spd)).T.reshape(-1,2)
        feas = np.empty(len(spd)*len(hdg))
        ownship['lat'] = traf.lat[i]
        ownship['lon'] = traf.lon[i]
        ownship['alt'] = traf.alt[i]
        ownship['vs'] = traf.vs[i]
        ownship['id'] = traf.id[i]
        for j in range(len(states)):
            ownship['spd'] = states[j][1]
            ownship['hdg'] = states[j][0]
            feasible = detect(ownship, traf, safe_h, safe_v, t_horizon)
            if any(feasible):
                feas[j] = 1


def update1():
    flex_t = np.zeros(len(traf.id))

    total_inst_flex = []

    for i in range(len(traf.id)):
        hdg,spd = computeDisk(traf.lat[i],traf.lon[i],traf.tas[i],traf.hdg[i])
        hdg = np.array(hdg)
        spd = np.array(spd)
        states = np.array(np.meshgrid(hdg,spd)).T.reshape(-1,2)
        feas = []
        own = np.array([(traf.hdg[i],traf.tas[i],traf.lat[i],traf.lon[i])], dtype = [('hdg','f'),('spd','f'),('lat','f'),('lon','f')])
        for j in range(len(traf.id)): #This loop should be vectorized
            # [i for i in range(x) if i not in s], where s = set([i]) #Generates a list of numbers (indexes for traffic) where ownship (i) is not included -> Avoid using the if
            #Ideally with list of indexes for intruder traffics, evaluate all at the same time in a vectorised way
            if j!=i: #Don't evaluate ownship as intruder
                intr = np.array([(traf.hdg[j],traf.tas[j],traf.lat[j],traf.lon[j])], dtype = [('hdg','f'),('spd','f'),('lat','f'),('lon','f')])
                feas.append(np.array(positionFeasibility(own, intr, states)))
        feas = np.array(feas)
        feasi = 1*feas #bool to int
        feas_t = np.array([np.prod(feasi[:,n]) for n in range(len(feasi[0,:])-1)]) #Check feasibility of each state for all Aircraft
        flex_t[i] = np.count_nonzero(feas_t == 1)/len(feas_t[:])
    flex = np.mean(flex_t)
    print(flex)
    #Plot real time the instant flexibility in the sector
    #plotter.init()
    #plotter.plot(total_inst_flex)


def update2(): #Same as before, but with traffics and all states together in array: lat,lon,hdg,spd
    flex_t = np.zeros(len(traf.id))

    total_inst_flex = []

    for i in range(len(traf.id)):
        hdg,spd = computeDisk(traf.lat[i],traf.lon[i],traf.tas[i],traf.hdg[i])
        hdg = np.array(hdg)
        spd = np.array(spd)
        lat = traf.lat
        lon = traf.lon
        states = np.array(np.meshgrid(lat,lon,hdg,spd)).T.reshape(-1,4)
        feas = []
        own = np.array([(traf.hdg[i],traf.tas[i],traf.lat[i],traf.lon[i])], dtype = [('hdg','f'),('spd','f'),('lat','f'),('lon','f')])
        for j in range(len(traf.id)): #This loop should be vectorized
            # [i for i in range(x) if i not in s], where s = set([i]) #Generates a list of numbers (indexes for traffic) where ownship (i) is not included -> Avoid using the if
            #Ideally with list of indexes for intruder traffics, evaluate all at the same time in a vectorised way
            if j!=i: #Don't evaluate ownship as intruder
                intr = np.array([(traf.hdg[j],traf.tas[j],traf.lat[j],traf.lon[j])], dtype = [('hdg','f'),('spd','f'),('lat','f'),('lon','f')])
                feas.append(np.array(positionFeasibility(own, intr, states)))
        feas = np.array(feas)
        feasi = 1*feas #bool to int
        feas_t = np.array([np.prod(feasi[:,n]) for n in range(len(feasi[0,:])-1)]) #Check feasibility of each state for all Aircraft
        flex_t[i] = np.count_nonzero(feas_t == 1)/len(feas_t[:])
    flex = np.mean(flex_t)
    #Plot real time the instant flexibility in the sector
    #plotter.init()
    #plotter.plot(total_inst_flex)




def preupdate():
    pass


def reset():
    pass

### Other functions of your plugin
def flexdisplay():
    pass
    #Activate the live plotting of the flexibility metric
    #Idea: for one aircraft/sector wide number/already computed trajectory flex

def createList(initial, delta, max):
    #Generate a list from the maximum and resolution (for speed/heading lists)
    ins = initial - max
    list = [ins]
    while  ins <= initial + max:
        prev = ins
        ins = prev + delta
        list.append(ins)
    return list

def discreteGrid(p0, hdg, spd):
    #Create a list of the reachable points from the initial position
    dist_list = [spd[i]*t_horizon for i in range(len(spd))]
    lat_list = []
    lon_list = []
    for i in range(len(dist_list)):
        dis = dist_list[i]
        for n in range(len(hdg)):
            pt = vct.vinc_pt(p0[0],p0[1],hdg[n],dis)
            lat_list.append(pt[0])
            lon_list.append(pt[1])
    return [lat_list,lon_list]


def computeDisk(lat,lon,spd,hdg):
    #This function will compute the reachable points of a flight given simulation parameters
    p0 = np.array([lat,lon])

    dhdg = 5 #º
    dspd = 2.5 #kts
    hdg_max = 80 #º
    spd_max = 30 #kts

    #create discrete heading and speed lists
    hdg_list = createList(hdg, dhdg, hdg_max)
    spd_list = createList(spd, dspd, spd_max)

    #make a grid of the disk with the heading and speed lists and the initial position
    grid = discreteGrid(p0,hdg_list,spd_list)
    return [hdg_list,spd_list]

def positionFeasibility(own, intr, state):
    #This function provides the trajectory feasibility with a 2 aircraft conflict
    #Set protection areas of each aircraft
    safe_h = 2.5 #nm
    safe_v = 500 #ft
    #Compute relative velocity
    ownspd = np.array(state[:,1])
    ownhdg = np.array(state[:,0])
    v_own = np.array([ownspd*np.sin(np.radians(ownhdg)),ownspd*np.cos(np.radians(ownhdg))])
    intspd = intr['spd']
    inthdg = intr['hdg']
    v_int = np.array([intspd*np.sin(np.radians(inthdg)),intspd*np.cos(np.radians(inthdg))])

    dy = 60.*(intr['lat']-own['lat'])*1852.
    dx = 60.*(intr['lon']-own['lon'])*np.cos(np.radians(0.5*(own['lat']+intr['lat'])))*1852.

    d = np.sqrt(dx*dx+dy*dy)

    brg = np.arctan2(dx,dy)%(2.*np.pi)

    vxa = ownspd*np.sin(np.radians(ownhdg))
    vya = ownspd*np.cos(np.radians(ownhdg))
    vxb = intr['spd']*np.sin(np.radians(intr['hdg']))
    vyb = intr['spd']*np.cos(np.radians(intr['hdg']))

    vx = vxb-vxa
    vy = vyb-vya

    tcpa = -(dx*vx+dy*vy)/(vx*vx+vy*vy)
    tcpa[tcpa < 0] = np.inf #If negative tcpa, set it as infinity

    xcpa = dx+vx*tcpa
    ycpa = dy+vy*tcpa
    cpadist = np.sqrt(xcpa*xcpa+ycpa*ycpa)


    Rpaz  = 5.*1852

    time_feas = t_horizon>tcpa #cpa within time horizon
    pos_feas = cpadist>Rpaz #cpa closer than protection zone
    feas = (pos_feas*time_feas) #cpa within time horizon and protection zone (0-conflict, 1-clear)
    return np.logical_not(feas)

def positionFeasibility2(state): #same as before but with all states in array
    #This function provides the trajectory feasibility with a 2 aircraft conflict
    #Set protection areas of each aircraft
    safe_h = 2.5 #nm
    safe_v = 500 #ft
    #Compute relative velocity
    ownspd = np.array(state[:,1])
    ownhdg = np.array(state[:,0])
    v_own = np.array([state[:,3]*np.sin(np.radians(state[:,2])),state[:,3]*np.cos(np.radians(state[:,2]))])
    intspd = intr['spd']
    inthdg = intr['hdg']
    v_int = np.array([intspd*np.sin(np.radians(inthdg)),intspd*np.cos(np.radians(inthdg))])

    dy = 60.*(intr['lat']-own['lat'])*1852.
    dx = 60.*(intr['lon']-own['lon'])*np.cos(np.radians(0.5*(own['lat']+intr['lat'])))*1852.

    d = np.sqrt(dx*dx+dy*dy)

    brg = np.arctan2(dx,dy)%(2.*np.pi)

    vxa = ownspd*np.sin(np.radians(ownhdg))
    vya = ownspd*np.cos(np.radians(ownhdg))
    vxb = intr['spd']*np.sin(np.radians(intr['hdg']))
    vyb = intr['spd']*np.cos(np.radians(intr['hdg']))

    vx = vxb-vxa
    vy = vyb-vya

    tcpa = -(dx*vx+dy*vy)/(vx*vx+vy*vy)
    tcpa[tcpa < 0] = np.inf #If negative tcpa, set it as infinity

    xcpa = dx+vx*tcpa
    ycpa = dy+vy*tcpa
    cpadist = np.sqrt(xcpa*xcpa+ycpa*ycpa)


    Rpaz  = 5.*1852

    time_feas = t_horizon>tcpa #cpa within time horizon
    pos_feas = cpadist>Rpaz #cpa closer than protection zone
    feas = (pos_feas*time_feas) #cpa within time horizon and protection zone (0-conflict, 1-clear)
    return np.logical_not(feas)


def SUAFeasibility(own,SUA):
    #This function provides the feasibility with a SUA
    pass

def conflictDetection(traf1,traf2,spd,hdg):
    traf1['spd'] = spd
    traf1['hdg'] = hdg
    safe_h = 2.5 #nm
    safe_v = 500 #ft
    ownship = Traffic()
    ownship.lat = traf1['lat']
    ownship.lon = traf1['lon']
    ownship.tas = traf1['spd']
    ownship.hdg = traf1['hdg']
    intruder = Traffic()
    intruder.lat = traf2['lat']
    intruder.lon = traf2['lon']
    intruder.tas = traf2['spd']
    intruder.hdg = traf2['hdg']
    confpairs, lospairs, inconf, tcpamax, \
        qdr, dist, tcpa, tLOS = cd.detect(ownship, intruder, safe_h, safe_v, t_horizon)
    return inconf
