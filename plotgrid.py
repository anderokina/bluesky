import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plotEHLE():
    #EHLE DEPS
    napx,napy = m(nap[1::2],nap[0::2])
    arnx, arny = m(arn[1::2],arn[0::2])
    eelx, eely = m(eel[1::2],eel[0::2])
    kolx, koly = m(kol[1::2],kol[0::2])
    rimx, rimy = m(rim[1::2],rim[0::2])
    woox, wooy = m(woo[1::2],woo[0::2])
    depehle = m.plot(napx,napy,color='cyan',linewidth='0.6',label='EHLE Departure',alpha=0.7)
    m.plot(arnx,arny,color='cyan',linewidth='0.6',alpha=0.7)
    m.plot(eelx,eely,color='cyan',linewidth='0.6',alpha=0.7)
    m.plot(kolx,koly,color='cyan',linewidth='0.6',alpha=0.7)
    m.plot(rimx,rimy,color='cyan',linewidth='0.6',alpha=0.7)
    m.plot(woox,wooy,color='cyan',linewidth='0.6',alpha=0.7)
    #EHLE ARRS
    lugx, lugy = m(lug[1::2],lug[0::2])
    nikx, niky = m(nik[1::2],nik[0::2])
    rknx, rkny = m(rkn[1::2],rkn[0::2])
    redx, redy = m(red[1::2],red[0::2])
    molx, moly = m(mol[1::2],mol[0::2])
    toplex, topley = m(tople[1::2],tople[0::2])
    lamx, lamy = m(lam[1::2],lam[0::2])
    putx, puty = m(put[1::2],put[0::2])
    helx, hely = m(hel[1::2],hel[0::2])
    inkx, inky = m(ink[1::2],ink[0::2])
    sonx, sony = m(son[1::2],son[0::2])
    m.plot(lugx,lugy,color='magenta',linewidth='0.6',label='EHLE Arrival',alpha=0.7)
    m.plot(nikx,niky,color='magenta',linewidth='0.6',alpha=0.7)
    m.plot(rknx,rkny,color='magenta',linewidth='0.6',alpha=0.7)
    m.plot(redx,redy,color='magenta',linewidth='0.6',alpha=0.7)
    m.plot(molx,moly,color='magenta',linewidth='0.6',alpha=0.7)
    m.plot(toplex,topley,color='magenta',linewidth='0.6',alpha=0.7)
    m.plot(lamx,lamy,color='magenta',linewidth='0.6',alpha=0.7)
    m.plot(putx,puty,color='magenta',linewidth='0.6',alpha=0.7)
    m.plot(helx,hely,color='magenta',linewidth='0.6',alpha=0.7)
    m.plot(inkx,inky,color='magenta',linewidth='0.6',alpha=0.7)
    m.plot(sonx,sony,color='magenta',linewidth='0.6',alpha=0.7)

def plotEHAM36L():
    #EHAM DEPS
    spyx,spyy = m(spy[1::2],spy[0::2])
    berx, bery = m(ber[1::2],ber[0::2])
    gorx, gory = m(gor[1::2],gor[0::2])
    kudx, kudy = m(kud[1::2],kud[0::2])
    lopx, lopy = m(lop[1::2],lop[0::2])
    arnamx, arnamy = m(arnam[1::2],arnam[0::2])
    m.plot(spyx,spyy,color='green',linewidth='0.8',label='EHAM Departure',alpha=0.7)
    m.plot(berx, bery,color='green',linewidth='0.8',alpha=0.7)
    m.plot(gorx, gory,color='green',linewidth='0.8',alpha=0.7)
    m.plot(kudx, kudy,color='green',linewidth='0.8',alpha=0.7)
    m.plot(lopx, lopy,color='green',linewidth='0.8',alpha=0.7)
    m.plot(arnamx, arnamy,color='green',linewidth='0.8',alpha=0.7)

def plotEHAMarr():
    topx,topy = m(top[1::2],top[0::2])
    molamx,molamy = m(molam[1::2],molam[0::2])
    lamamx,lamamy = m(lamam[1::2],lamam[0::2])
    redamx,redamy = m(redam[1::2],redam[0::2])
    denx,deny = m(den[1::2],den[0::2])
    helamx,helamy = m(helam[1::2],helam[0::2])
    putamx,putamy = m(putam[1::2],putam[0::2])
    rek1x,rek1y = m(rek1[1::2],rek1[0::2])
    rek2x,rek2y = m(rek2[1::2],rek2[0::2])
    norx,nory = m(nor[1::2],nor[0::2])
    eel1x,eel1y = m(eel1[1::2],eel1[0::2])
    eel2x,eel2y = m(eel2[1::2],eel2[0::2])
    m.plot(topx,topy,color='blue',linewidth='0.8', label='EHAM Arrival',alpha=0.7)
    m.plot(molamx,molamy,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(lamamx,lamamy,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(redamx,redamy,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(denx,deny,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(helamx,helamy,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(putamx,putamy,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(rek1x,rek1y,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(rek2x,rek2y,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(norx,nory,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(eel1x,eel1y,color='blue',linewidth='0.8',alpha=0.7)
    m.plot(eel2x,eel2y,color='blue',linewidth='0.8',alpha=0.7)

try: #Define EHLE Departures
    ## Coordinates for departure routes to EHLE
    #NAPRO
    nap = np.array([52.46286400, 5.53193000, 52.230389 , 5.528803, 52.109833, 5.565403, 52.061544, 5.675425, 51.975525, 5.835936, 51.855833, 6.058889])
    #ARNEM
    arn = np.array([52.46286400, 5.53193000, 52.230389 , 5.528803, 52.096447, 6.076603])
    #EEL
    eel = np.array([52.46286400, 5.53193000, 52.739403, 5.270489, 52.782453, 5.361492, 53.105469, 6.054969, 53.16390600, 6.66667500, 53.213333, 7.216944])
    #KOLAG
    kol = np.array([52.46286400, 5.53193000, 52.54028100, 4.85378100, 52.748750, 4.358931, 52.975219, 3.685389, 53.015225, 3.564464, 53.043472, 3.254881])
    #RIMBU
    rim = np.array([52.46286400, 5.53193000, 52.33476100, 5.09216100, 52.071344, 3.839831, 51.924067, 3.171836, 51.809567, 2.666908, 51.706278, 2.162114])
    #WOODY
    woo = np.array([52.46286400, 5.53193000, 52.33476100, 5.09216100, 51.924175, 4.767331, 51.834572, 4.697325, 51.405661, 4.366481])
except:
    pass

try: #Define EHAM departures
    ## Coordinates for departure routes from EHAM
    THR36L = [52.32861, 4.708889]
    EH012 = [52.42028,4.7175]
    EH013 = [52.54472,4.729444]
    EH034 = [52.51778,4.576667]
    EH036 = [52.25694,4.847778]
    EH045 = [52.33889,4.824167]
    EH047 = [52.43111,4.718333]
    EH072 = [52.02056,4.843056]
    EH084 = [52.39528,4.715278]
    EH087 = [52.48,4.772222]
    EH088 = [52.4875,4.915278]
    EH090 = [52.46722,4.534444]
    EH091 = [52.42278,4.488889]
    EH094 = [52.46611,4.661389]
    #SPY
    spy = np.concatenate((np.array(THR36L+EH084+EH012+EH047+EH013),np.array([52.586936,4.950803,52.739403,5.270489,53.105469,6.054969,53.21529200,6.78516400,53.213333 ,7.216944])))
    #BERGI
    ber = np.concatenate((np.array(THR36L+EH084+EH012+EH047+EH094+EH034),np.array([52.748750,4.358931,52.975219 ,3.685389,53.100892,3.303508])))
    #GORLO
    gor = np.concatenate((np.array(THR36L+EH084+EH012+EH047+EH094+EH090+EH091),np.array([52.314181,4.156200,51.924067,3.171836,51.908239 ,2.322450])))
    #KUDAD
    kud = np.concatenate((np.array(THR36L+EH084+EH045+EH036+EH072),np.array([51.924175,4.767331,51.405661,4.366481])))
    #LOPIK
    lop = np.concatenate((np.array(THR36L+EH084+EH045+EH036),np.array([52.097478 ,5.054633,51.930828,5.129156,51.265556,5.421014])))
    #ARNEM
    arnam = np.concatenate((np.array(THR36L+EH084+EH012+EH047+EH087+EH088),np.array([52.33476100 ,5.09216100,52.096447,6.076603,52.023611 ,6.764167])))
except:
    pass

try: #EHAM Arrivals
    #TOPPA
    top = np.array([53.402592 ,3.561467,52.760756,3.746642,52.571647,3.776447,52.525511,3.967350])
    #MOLIX
    molam = np.array([52.822000, 3.068669,52.681433 ,3.468547,52.571647,3.776447,52.525511,3.967350])
    #LAMSO
    lamam = np.array([52.732897 ,2.994356, 52.58527778, 3.41388889, 52.525511,3.967350])
    #REDFA
    redam = np.array([52.114586, 2.487947,52.447925,3.421064,52.525511,3.967350])
    #DENUT
    den = np.array([51.236111,3.657500, 51.72272200,3.85818100, 51.912764 ,4.132594])
    #HELEN
    helam = np.array([51.235314 ,3.869711, 51.72272200,3.85818100, 51.912764 ,4.132594])
    #PUTTY
    putam = np.array([51.365853 ,4.337614,51.72272200,3.85818100, 51.912764 ,4.132594])
    #REKKEN1
    rek1 = np.array([52.13319700,6.76387800, 52.490456,6.024022, 52.511214,5.569081])
    #REKKEN2
    rek2 = np.array([52.13319700,6.76387800,52.330894 ,6.745586, 52.715319,6.709544, 52.511214,5.569081])
    #NORKU
    nor = np.array([52.215556,6.976389,52.330894 ,6.745586,52.468611,6.467222,52.490456,6.024022, 52.511214,5.569081])
    #EELDE1
    eel1 = np.array([53.16390600,6.66667500, 52.511214,5.569081])
    #EELDE2
    eel2 = np.array([53.16390600,6.66667500,52.715319,6.709544, 52.511214,5.569081])

except:
    pass

try: #Define EHLE arrivals
    ##Coordinates for arrival routes to EHLE
    #LUGUM
    lug = np.array([53.307222, 7.067817, 53.21529200, 6.78516400, 52.678128, 5.243264, 52.46286400, 5.53193000])
    #NIK
    nik = np.array([51.16249800, 4.18916700, 51.28972222, 4.15527778, 51.530314 ,4.086594, 51.74132500,4.24348600, 51.89758600,4.55434700, 52.33476100, 5.09216100, 52.46286400, 5.53193000])
    #RKN
    rkn = np.array([52.13319700, 6.76387800, 52.246469, 6.250142, 52.385350, 5.602169, 52.46286400, 5.53193000])
    #REDFA
    red = np.array([52.114586, 2.487947, 52.367750, 3.857308, 52.350153, 4.575042, 52.33476100, 5.09216100, 52.46286400, 5.53193000])
    #MOLIX
    mol = np.array([52.822000, 3.068669, 52.614911, 4.215736, 52.54028100, 4.85378100, 52.46286400, 5.53193000])
    #TOPPA
    tople = np.array([52.54028100, 4.85378100, 52.760756 ,3.746642, 52.975219 ,3.685389, 53.402592 ,3.561467])
    #LAMSO
    lam = np.array([52.54028100, 4.85378100, 52.732897 ,2.994356])
    #PUTTY
    put = np.array([51.74132500,4.24348600, 51.365853 ,4.337614])
    #HELEN
    hel = np.array([51.530314 ,4.086594, 51.235314 ,3.869711])
    #INKET
    ink = np.array([51.89758600,4.55434700, 51.814628, 4.771961])
    #SONEB
    son = np.array([52.246469 ,6.250142, 52.023611 ,6.764167])
except:
    pass



#Generate dutch airspace boundary coordinate list
coord = np.array([ 50.76666667,   6.08333333,  51.83333333,   6.        ,\
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

coords = np.reshape(coord,(int(len(coord)/2),2))
d = 1/10 #Discretise coordinates for every 1/10 of degree -> 6 nm
nlat = (np.amax(coords[:,0]) - np.amin(coords[:,0]))/d
lats = np.amax(coords[:,0])-np.arange(nlat)*d
nlon = (np.amax(coords[:,1]) - np.amin(coords[:,1]))/d
lons = np.amin(coords[:,1])+np.arange(nlon)*d


interest = np.array([[6.65,5.05,5.52],[53.15,52.3,52.45]])

lon, lat = np.meshgrid(lons, lats)
lat_0 = (lats[:].mean())

#Create map on which results will be displayed
m = Basemap(projection = 'lcc',\
        lon_0 = 0, lat_0 = 50, lat_ts = lat_0,\
        llcrnrlat=50.7,urcrnrlat=55,\
        llcrnrlon=1.5,urcrnrlon= 8,\
        resolution='l')
grid = np.load('grid_pts.npy')
grid[0] = grid[0]-0.5*d
grid[1] = grid[1]+0.5*d
xg,yg = m(grid[1],grid[0])
x, y = m(coords[:,1], coords[:,0])
intx, inty = m(interest[0],interest[1])


fig = plt.figure(figsize=(12,8),num='Grid')
m.shadedrelief(scale=0.5)
plotEHAM36L()
plotEHAMarr()
plotEHLE()
m.plot(x, y, color = 'blue', markersize=5, linewidth=1)
m.scatter(xg,yg,marker='.',color='black',facecolors='none')
m.scatter(intx, inty, marker = 'x', color='red')
plt.title('Grid points of discretised airspace')
plt.show()
