import numpy as np
from math import *
import matplotlib.pyplot as plt
import vincenti as vct
from bluesky.traffic.asas import StateBasedCD as cd #Import conflict detection algorithm
from bluesky.traffic.traffic import Traffic


#Simulation parameters for computaiton of reachable points
hdg_max = 80 #º
spd_max = 30 #kts
dhdg = 5 #º
dspd = 2.5 #kts
t_horizon = 100/3600 #hours

def setMaxHeading(maxHeading):
    global hdg_max
    hdg_max = maxHeading

def setHeadingResolution(dHeading):
    global dhdg
    dhdg = dHeading

def setMaxSpeed(maxSpeed):
    global spd_max
    spd_max = maxSpeed

def setSpeedResolution(dSpeed):
    global dspd
    dspd = dSpeed

def setTimeHorizon(dTime):
    global t_horizon
    t_horizon = dTime

#The algorithm is as follows:

#for i in traf.id:

    #Obtain all the reachable points within the given "disc"

    #Simulation characteristics as per R. Klomp - to be tuned
    #hdg_max = 80
    #spd_max = 20
    #dhdg = 5
    #dspd = 2.5

    #for n in traf.id:
        #inner loop to compare with other traffics and select feasible reachable points
        #pass
    #compare reachable points with SUA - to examine method
    #compute flexibility (robustness) for flight i
#keep point based flex for all flights in a list (maybe with the point in which it was computed)
#if a flight leaves the airspace compute trajectory flexibility
#at the end of the simulation compute sector based flexibility



def createList(initial, delta, max):
    #Generate a list from the maximum and resolution (for speed/heading lists)
    ins = initial - max
    list = [ins]
    while  ins <= initial + max:
        prev = ins
        ins = prev + delta
        list.append(ins)
    return list


def advanceTraffic(traf):
    heading = traf['hdg']
    speed = traf['spd']
    time_step = 5/3600 #hours
    a = speed*time_step*1852 #meters
    point = vct.vinc_pt(traf['lat'],traf['lon'], heading, a)
    traf['lat'] = point[0]
    traf['lon'] = point[1]
    return traf


def addDistance(hor, ver, lat, lon):
    #Add cartesian distances to the latitude and longitude
    ver = ver*1.852*1000 #to meters
    hor = hor*1.852*1000

    latf = lat + (180/pi)*(ver/6378137)
    lonf = lon + (180/pi)*(hor/6378137)/cos(radians(lat))
    return [latf,lonf]


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


def getBearing(long1, lat1, long2, lat2):

    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])
    bearing = atan2(sin(long2-long1)*cos(lat2), cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(long2-long1))
    bearing = degrees(bearing)
    return ((bearing + 360) % 360)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371/1.852 # Radius of earth in nautical miles
    return c * r


def computeDisk(traf):
    #This function will compute the reachable points of a flight given simulation parameters
    p0 = np.array([traf['lat'][0],traf['lon'][0]])

    #create discrete heading and speed lists
    hdg_list = createList(traf['hdg'][0], dhdg, hdg_max)
    spd_list = createList(traf['spd'][0], dspd, spd_max)

    #make a grid of the disk with the heading and speed lists and the initial position
    grid = discreteGrid(p0,hdg_list,spd_list)
    return [hdg_list,spd_list]


def positionFeasibility(own, intr, spd, hdg):
    #This function provides the trajectory feasibility with a 2 aircraft conflict

    #Set protection areas of each aircraft
    safe_h = 2.5 #nm
    safe_v = 500 #ft
    #Compute relative velocity
    ownspd = spd
    ownhdg = hdg
    v_own = np.array([ownspd*sin(radians(ownhdg)),ownspd*cos(radians(ownhdg))])
    intspd = intr['spd']
    inthdg = intr['hdg']
    v_int = np.array([intspd*sin(radians(inthdg)),intspd*cos(radians(inthdg))])

    dy = 60.*(intr['lat']-own['lat'])*1852.
    dx = 60.*(intr['lon']-own['lon'])*cos(radians(0.5*(own['lat']+intr['lat'])))*1852.

    d = sqrt(dx*dx+dy*dy)

    brg = atan2(dx,dy)%(2.*pi)

    vxa = ownspd*sin(radians(ownhdg))
    vya = ownspd*cos(radians(ownhdg))
    vxb = intr['spd']*sin(radians(intr['hdg']))
    vyb = intr['spd']*cos(radians(intr['hdg']))

    vx = vxb-vxa
    vy = vyb-vya

    tcpa = -(dx*vx+dy*vy)/(vx*vx+vy*vy)

    xcpa = dx+vx*tcpa
    ycpa = dy+vy*tcpa
    cpadist = sqrt(xcpa*xcpa+ycpa*ycpa)

    Rpaz  = 5.*1852

    if cpadist>Rpaz: #no conflict within the given time window
        #print('no conflict, min dist is '+str(cpadist/1852)+' nm')
        return 1
    elif cpadist<Rpaz and tcpa>t_horizon: #conflict but outside window
        return 1
    else:
        vrel    = sqrt(vx*vx+vy*vy)
        dspaz   = sqrt(Rpaz*Rpaz-cpadist*cpadist)
        dtpaz   = dspaz/vrel
        tinhor  = tcpa - dtpaz
        touthor = tcpa + dtpaz
        #print('dcpa '+str(cpadist/1852)+' nm, tcpa '+str(tcpa)+', t_in '+str(tinhor)+', t_out '+str(touthor))
        return 0


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


def main():
    #Made up traffic for testing purposes
    traf = np.array([(300.,400.,51.0121,3.02195)], dtype=[('hdg', 'f'),('spd', 'f'),('lat', 'f'),('lon', 'f')])
    trafint = np.array([(030.,350.,50.8701,1.993114)], dtype=[('hdg', 'f'),('spd', 'f'),('lat', 'f'),('lon', 'f')])
    a = computeDisk(traf)
    hdg = a[0] #lists of heading and speed possibilities
    spd = a[1]
    print(len(hdg))
    feasible = np.zeros((len(hdg),len(spd)))
    for i in range(len(hdg)):
        for n in range(len(spd)):
                feasible[i,n] = conflictDetection(traf,trafint,spd[n],hdg[i]) #true or false feasibility
    print(feasible)
    flexibility = np.count_nonzero(feasible == 1)/(len(hdg)*len(spd))
    print(flexibility)


def mainLoop():
    traf = np.array([(300.,400.,51.0121,3.02195)], dtype=[('hdg', 'f'),('spd', 'f'),('lat', 'f'),('lon', 'f')])
    trafint = np.array([(030.,350.,50.8701,1.993114)], dtype=[('hdg', 'f'),('spd', 'f'),('lat', 'f'),('lon', 'f')])
    a = computeDisk(traf)
    hdg = a[0] #lists of heading and speed possibilities
    spd = a[1]
    flexibility = []
    distance = []
    lat_o = []
    lon_o = []
    lat_i = []
    lon_i = []
    for step in range(100):
        feasible = np.zeros((len(hdg),len(spd)))
        for i in range(len(hdg)):
            for n in range(len(spd)):
                    feasible[i,n] = positionFeasibility(traf,trafint,spd[n],hdg[i]) #true or false feasibility
        flexibility.append(np.count_nonzero(feasible == 1)/(len(hdg)*len(spd)))
        distance.append(haversine(traf['lat'],traf['lon'],trafint['lat'],trafint['lon'])) #in nm
        lat_o.append(traf['lat'][0])
        lon_o.append(traf['lon'][0])
        lat_i.append(trafint['lat'][0])
        lon_i.append(trafint['lon'][0])
        #print('flex '+str(flexibility[step])+', dist '+str(distance[step]))
        traf = advanceTraffic(traf)
        #print('Ownship: lat '+str(traf['lat'])+',lon '+str(traf['lon']))
        trafint = advanceTraffic(trafint)
        #print('Intruder: lat '+str(trafint['lat'])+',lon '+str(trafint['lon']))


    flex_traj_mean = np.mean(np.array(flexibility))
    plt.figure(1)
    plt.plot(lon_o,lat_o,'ro')
    plt.plot(lon_o[0],lat_o[0],'g^')
    plt.plot(lon_i,lat_i,'bo')
    plt.plot(lon_i[0],lat_i[0],'g^')

    plt.figure(2)
    x_coordinate = [i for i in range(len(flexibility)) ]
    plt.plot(x_coordinate,flexibility)
    plt.plot([x_coordinate[0],x_coordinate[-1]],[flex_traj_mean,flex_traj_mean],'r--')

    plt.show()

main()
