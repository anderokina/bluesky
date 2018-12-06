import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import gridspec
from scipy.stats import stats
from scipy import stats as stt
from scipy.signal import savgol_filter

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


def coord2latlon(iny,inx):
    d = 1/10
    lat = 55.-iny*d
    lon = inx*d+2.
    return np.array(lat),np.array(lon)

def discreteCoord(lat,lon):
    d = 1/10
    in_lat = ((55.-lat)/d) #Indice in matrix of latitude -> from up to down
    in_lon = ((lon-2.)/d) #Indice in matrix of longitude
    return in_lat.astype(int), in_lon.astype(int)

#Select which scenario to process
traf_ehle = '45'
traf_peak = '5'

#Load baseline scenario data:
flex4d1 = np.load('Results/Grid/flexibility_7-9-'+str(traf_peak)+'.npy')
flex3d1 = np.nanmean(flex4d1,axis=3)
flex4d2 = np.load('Results/Grid/flexibility_31-8-'+str(traf_peak)+'.npy')
flex3d2 = np.nanmean(flex4d2,axis=3)
flex4d3 = np.load('Results/Grid/flexibility_29-6-'+str(traf_peak)+'.npy')
flex3d3 = np.nanmean(flex4d3,axis=3)
flex4d4 = np.load('Results/Grid/flexibility_06-7-'+str(traf_peak)+'.npy')
flex3d4 = np.nanmean(flex4d4,axis=3)
flex4d5 = np.load('Results/Grid/flexibility_15-6-'+str(traf_peak)+'.npy')
flex3d5 = np.nanmean(flex4d5,axis=3)
flex4d6 = np.load('Results/Grid/flexibility_28-6-'+str(traf_peak)+'.npy')
flex3d6 = np.nanmean(flex4d6,axis=3)

#Load Lelystad scenario data
flex4d1s = np.load('Results/Grid/flexibility_7-9-'+str(traf_peak)+'_'+traf_ehle+'.npy')
flex3d1s = np.nanmean(flex4d1s,axis=3)
flex4d2s = np.load('Results/Grid/flexibility_31-8-'+str(traf_peak)+'_'+traf_ehle+'.npy')
flex3d2s = np.nanmean(flex4d2s,axis=3)
flex4d3s = np.load('Results/Grid/flexibility_29-6-'+str(traf_peak)+'_'+traf_ehle+'.npy')
flex3d3s = np.nanmean(flex4d3s,axis=3)
flex4d4s = np.load('Results/Grid/flexibility_06-7-'+str(traf_peak)+'_'+traf_ehle+'.npy')
flex3d4s = np.nanmean(flex4d4s,axis=3)
flex4d5s = np.load('Results/Grid/flexibility_15-6-'+str(traf_peak)+'_'+traf_ehle+'.npy')
flex3d5s = np.nanmean(flex4d5s,axis=3)
flex4d6s = np.load('Results/Grid/flexibility_28-6-'+str(traf_peak)+'_'+traf_ehle+'.npy')
flex3d6s = np.nanmean(flex4d6s,axis=3)

#Perform substraction of data
dflex1 = flex3d1s - flex3d1
dflex1 = np.nan_to_num(dflex1)
dflex2 = flex3d2s - flex3d2
dflex2 = np.nan_to_num(dflex2)
dflex3 = flex3d3s - flex3d3
dflex3 = np.nan_to_num(dflex3)
dflex4 = flex3d4s - flex3d4
dflex4 = np.nan_to_num(dflex4)
dflex5 = flex3d5s - flex3d5
dflex5 = np.nan_to_num(dflex5)
dflex6 = flex3d6s - flex3d6
dflex6 = np.nan_to_num(dflex6)

#Perform substraction of data
dflex4_1 = flex4d1s - flex4d1
dflex4_1 = np.nan_to_num(dflex4_1)
dflex4_2 = flex4d2s - flex4d2
dflex4_2 = np.nan_to_num(dflex4_2)
dflex4_3 = flex4d3s - flex4d3
dflex4_3 = np.nan_to_num(dflex4_3)
dflex4_4 = flex4d4s - flex4d4
dflex4_4 = np.nan_to_num(dflex4_4)
dflex4_5 = flex4d5s - flex4d5
dflex4_5 = np.nan_to_num(dflex4_5)
dflex4_6 = flex4d6s - flex4d6
dflex4_6 = np.nan_to_num(dflex4_6)

#Concatenate all data, leaving last 4 time steps out as they contain empty data
flex_original = np.concatenate([flex4d1[:,:,:,:-4],flex4d2[:,:,:,:-4],flex4d3[:,:,:,:-4],flex4d4[:,:,:,:-4],flex4d5[:,:,:,:-4],flex4d6[:,:,:,:-4]],axis=3)
flex_traffic = np.concatenate([flex4d1s[:,:,:,:-4],flex4d2s[:,:,:,:-4],flex4d3s[:,:,:,:-4],flex4d4s[:,:,:,:-4],flex4d5s[:,:,:,:-4],flex4d6s[:,:,:,:-4]],axis=3)

#Perform mean along time of concatenated data
flex_original_3 = np.nanmean(flex_original,axis=3)
flex_traffic_3 = np.nanmean(flex_traffic,axis=3)

#Identify the points with the highest difference in flexibility overall
flex_diff = flex_traffic_3-flex_original_3
flex_diff = np.nan_to_num(flex_diff)
valn_l = np.partition(flex_diff.flatten(), -30)[-30] #30th largest value
ww_l = np.where(flex_diff>valn_l) #indexes of where the 30 largest values are
ww_l = np.array([[ww_l[0][i],ww_l[1][i],ww_l[2][i]] for i in range(len(ww_l[0]))])
#ww_un = np.unique(ww_l,axis=0) #Select unique values

#Analyse where the biggest differences are (points with highest difference)

num = 2 #Number of highest points to be selected

valn_1 = np.partition(dflex1.flatten(), -num)[-num] #20th largest value
ww_1 = np.where(dflex1>valn_1) #indexes of where the 20 largest values are
ww_1 = np.array([[ww_1[0][i],ww_1[1][i],ww_1[2][i]] for i in range(len(ww_1[0]))])

valn_2 = np.partition(dflex2.flatten(), -num)[-num] #20th largest value
ww_2 = np.where(dflex2>valn_2) #indexes of where the 20 largest values are
ww_2 = np.array([[ww_2[0][i],ww_2[1][i],ww_2[2][i]] for i in range(len(ww_2[0]))])

valn_3 = np.partition(dflex3.flatten(), -num)[-num] #20th largest value
ww_3 = np.where(dflex3>valn_3) #indexes of where the 20 largest values are
ww_3 = np.array([[ww_3[0][i],ww_3[1][i],ww_3[2][i],] for i in range(len(ww_3[0]))])

valn_4 = np.partition(dflex4.flatten(), -num)[-num] #20th largest value
ww_4 = np.where(dflex4>valn_4) #indexes of where the 20 largest values are
ww_4 = np.array([[ww_4[0][i],ww_4[1][i],ww_4[2][i]] for i in range(len(ww_4[0]))])

valn_5 = np.partition(dflex5.flatten(), -num)[-num] #20th largest value
ww_5 = np.where(dflex5>valn_5) #indexes of where the 20 largest values are
ww_5 = np.array([[ww_5[0][i],ww_5[1][i],ww_5[2][i]] for i in range(len(ww_5[0]))])

valn_6 = np.partition(dflex6.flatten(), -num)[-num] #20th largest value
ww_6 = np.where(dflex6>valn_6) #indexes of where the 20 largest values are
ww_6 = np.array([[ww_6[0][i],ww_6[1][i],ww_6[2][i]] for i in range(len(ww_6[0]))])

#Extra points to always analyse: conflicting points
indlat,indlon = discreteCoord(np.array([53.15,52.3,52.45,52.51]),np.array([6.65,5.05,5.52,4.82]))
ww_extra = np.array([np.repeat(indlon,6),np.repeat(indlat,6),np.array([0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5])])
ww_extra = np.array([[ww_extra[0][i],ww_extra[1][i],ww_extra[2][i]] for i in range(len(ww_extra[2]))])
print(ww_extra)
#Join all arrays and select unique points
ww_joint = np.concatenate([ww_1,ww_2,ww_3,ww_4,ww_5,ww_6])#,ww_extra])
#ww_joint = ww_extra
ww_un = np.unique(ww_joint,axis=0)

ww_sp = np.array([[25,28,29,30,34,38],[26,26,26,26,25,30]]) #Points identified and analysed (wave 3,traf 45)
lat,lon = coord2latlon(ww_sp[1],ww_sp[0])
print(lat,lon)

ww_una = np.array([np.repeat(ww_sp[0],6),np.repeat(ww_sp[1],6),np.array([0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5])])
ww_una = np.array([[ww_una[0][i],ww_una[1][i],ww_una[2][i]] for i in range(len(ww_una[2]))])
print(ww_una)

#Convert indexes to coordinates to show on map where this happens
wwlat, wwlon = coord2latlon(ww_un[:,1]+0.5*1/10,ww_un[:,0]-0.5*1/10)
wwlata, wwlona = coord2latlon(ww_una[:,1]+0.5*1/10,ww_una[:,0]-0.5*1/10)
ww_slat,ww_slon = coord2latlon(ww_extra[:,1],ww_extra[:,0])


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


lon, lat = np.meshgrid(lons, lats)
lat_0 = (lats[:].mean())


#Create map on which results will be displayed
m = Basemap(projection = 'lcc',\
        lon_0 = 0, lat_0 = 50, lat_ts = lat_0,\
        llcrnrlat=50.7,urcrnrlat=55,\
        llcrnrlon=1.5,urcrnrlon= 8,\
        resolution='l')

amsx,amsy = m(4.7683, 52.3105)
lelx,lely = m(5.5204, 52.4552)

xg,yg = m(wwlona,wwlata)
xsg, ysg = m(ww_slon,ww_slat)

x, y = m(coords[:,1], coords[:,0])

dlfex = [dflex1,dflex2,dflex3,dflex4,dflex5]

fig = plt.figure(figsize=(12,8),num='Conflict_pts')
m.shadedrelief(scale=0.5)
#m.pcolormesh(lon, lat, np.transpose(np.nanmean(dlfex[i],axis=2)), latlon=True, cmap='hot')
m.plot(x, y, color = 'blue', markersize=5, linewidth=1)
a = m.scatter(xg,yg,marker='.',c=ww_una[:,2]*100,facecolors='none')
m.colorbar(a,label='Altitude [FL]')
m.scatter(xsg,ysg,marker='x',color='red')
plotEHAM36L()
plotEHAMarr()
plotEHLE()
m.scatter(amsx,amsy,facecolors='none', edgecolors='b')
m.scatter(lelx,lely,facecolors='none', edgecolors='b')
plt.text(amsx,amsy,'EHAM')
plt.text(lelx,lely,'EHLE')

for i in range(4):
    plt.annotate('['+str(i+1)+']',(xsg[6*i],ysg[6*i]))

for i in range(6):
    plt.annotate(str(i+1),(xg[6*i],yg[6*i]))
plt.title('Grid points of discretised airspace')
plt.tight_layout()
plt.show()

#Plot the flexibility boxplots of the 3 hand-selected points
#Concatenate all data, leaving last 4 time steps out as they contain empty data
flex_original = np.concatenate([flex4d1[:,:,:,:-4],flex4d2[:,:,:,:-4],flex4d3[:,:,:,:-4],flex4d4[:,:,:,:-4],flex4d5[:,:,:,:-4],flex4d6[:,:,:,:-4]],axis=3)
flex_traffic = np.concatenate([flex4d1s[:,:,:,:-4],flex4d2s[:,:,:,:-4],flex4d3s[:,:,:,:-4],flex4d4s[:,:,:,:-4],flex4d5s[:,:,:,:-4],flex4d6s[:,:,:,:-4]],axis=3)

#Select flexibility data for wanted points
flex_sel = flex_original[[i[0] for i in ww_extra],[i[1] for i in ww_extra],[i[2] for i in ww_extra],:]
flex_sel_t = flex_traffic[[i[0] for i in ww_extra],[i[1] for i in ww_extra],[i[2] for i in ww_extra],:]

print(ww_extra)

#Plot the data in boxplots
plt.figure(figsize=(10,12),num='distribution')
plt.subplot(411)
plt.title('Original traffic / '+str(traf_ehle)+' flt/h EHLE')
plt.ylabel('Northeast point [1]')
bp1 = plt.boxplot([flex_sel[0],flex_sel_t[0],flex_sel[1],flex_sel_t[1],flex_sel[2],flex_sel_t[2],flex_sel[3],flex_sel_t[3],flex_sel[4],flex_sel_t[4],flex_sel[5],flex_sel_t[5]],patch_artist=True)
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL'+str(ww_extra[0][2]*100),'FL'+str(ww_extra[1][2]*100),'FL'+str(ww_extra[2][2]*100),'FL'+str(ww_extra[3][2]*100),'FL'+str(ww_extra[4][2]*100),'FL'+str(ww_extra[5][2]*100)])
plt.subplot(412)
plt.ylabel('East of Schiphol [2]')
bp2 = plt.boxplot([flex_sel[6],flex_sel_t[6],flex_sel[7],flex_sel_t[7],flex_sel[8],flex_sel_t[8],flex_sel[9],flex_sel_t[9],flex_sel[10],flex_sel_t[10],flex_sel[11],flex_sel_t[11]],patch_artist=True)
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL'+str(ww_extra[0][2]*100),'FL'+str(ww_extra[1][2]*100),'FL'+str(ww_extra[2][2]*100),'FL'+str(ww_extra[3][2]*100),'FL'+str(ww_extra[4][2]*100),'FL'+str(ww_extra[5][2]*100)])
plt.subplot(413)
plt.ylabel('Lelystad airport [3]')
bp3 = plt.boxplot([flex_sel[12],flex_sel_t[12],flex_sel[13],flex_sel_t[13],flex_sel[14],flex_sel_t[14],flex_sel[15],flex_sel_t[15],flex_sel[16],flex_sel_t[16],flex_sel[17],flex_sel_t[17]],patch_artist=True)
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL'+str(ww_extra[0][2]*100),'FL'+str(ww_extra[1][2]*100),'FL'+str(ww_extra[2][2]*100),'FL'+str(ww_extra[3][2]*100),'FL'+str(ww_extra[4][2]*100),'FL'+str(ww_extra[5][2]*100)])
plt.subplot(414)
plt.ylabel('North of Schiphol [4]')
bp4 = plt.boxplot([flex_sel[18],flex_sel_t[18],flex_sel[19],flex_sel_t[19],flex_sel[20],flex_sel_t[20],flex_sel[21],flex_sel_t[21],flex_sel[22],flex_sel_t[22],flex_sel[23],flex_sel_t[23]],patch_artist=True)
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL'+str(ww_extra[0][2]*100),'FL'+str(ww_extra[1][2]*100),'FL'+str(ww_extra[2][2]*100),'FL'+str(ww_extra[3][2]*100),'FL'+str(ww_extra[4][2]*100),'FL'+str(ww_extra[5][2]*100)])

colors = ['white','grey','white','grey','white','grey','white','grey','white','grey','white','grey']
for bplot in (bp1, bp2, bp3, bp4):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
# plt.subplot(322)
# plt.title(str(traf_ehle)+' flt/h EHLE')
# plt.boxplot([flex_sel_t[0],flex_sel_t[1],flex_sel_t[2],flex_sel_t[3],flex_sel_t[4],flex_sel_t[5]])
# plt.xticks([1, 2,3,4,5,6], ['FL'+str(ww_extra[0][2]*100),'FL'+str(ww_extra[1][2]*100),'FL'+str(ww_extra[2][2]*100),'FL'+str(ww_extra[3][2]*100),'FL'+str(ww_extra[4][2]*100),'FL'+str(ww_extra[5][2]*100)])
# plt.subplot(324)
# plt.boxplot([flex_sel_t[6],flex_sel_t[7],flex_sel_t[8],flex_sel_t[9],flex_sel_t[10],flex_sel_t[11]])
# plt.xticks([1, 2,3,4,5,6], ['FL'+str(ww_extra[0][2]*100),'FL'+str(ww_extra[1][2]*100),'FL'+str(ww_extra[2][2]*100),'FL'+str(ww_extra[3][2]*100),'FL'+str(ww_extra[4][2]*100),'FL'+str(ww_extra[5][2]*100)])
# plt.subplot(326)
# plt.boxplot([flex_sel_t[12],flex_sel_t[13],flex_sel_t[14],flex_sel_t[15],flex_sel_t[16],flex_sel_t[17]])
# plt.xticks([1, 2,3,4,5,6], ['FL'+str(ww_extra[0][2]*100),'FL'+str(ww_extra[1][2]*100),'FL'+str(ww_extra[2][2]*100),'FL'+str(ww_extra[3][2]*100),'FL'+str(ww_extra[4][2]*100),'FL'+str(ww_extra[5][2]*100)])
plt.tight_layout()
plt.savefig('Results/Grid/distribution.png',bbox_inches='tight')
plt.show()

#Select flexibility data for wanted points
flex_sel = flex_original[[i[0] for i in ww_una],[i[1] for i in ww_una],[i[2] for i in ww_una],:]
flex_sel_t = flex_traffic[[i[0] for i in ww_una],[i[1] for i in ww_una],[i[2] for i in ww_una],:]

#Plot the data in boxplots
plt.figure(figsize=(14,8),num='distribution_all')
plt.subplot(231)
plt.title('Point 1')
bp1 = plt.boxplot([flex_sel[0],flex_sel_t[0],flex_sel[1],flex_sel_t[1],flex_sel[2],flex_sel_t[2],flex_sel[3],flex_sel_t[3],flex_sel[4],flex_sel_t[4],flex_sel[5],flex_sel_t[5]],patch_artist=True)
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL0','FL100','FL200','FL300','FL400','FL500'])
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')
plt.subplot(232)
plt.title('Point 2')
bp2 = plt.boxplot([flex_sel[6],flex_sel_t[6],flex_sel[7],flex_sel_t[7],flex_sel[8],flex_sel_t[8],flex_sel[9],flex_sel_t[9],flex_sel[10],flex_sel_t[10],flex_sel[11],flex_sel_t[11]],patch_artist=True)
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL0','FL100','FL200','FL300','FL400','FL500'])
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')
plt.subplot(233)
plt.title('Point 3')
bp3 = plt.boxplot([flex_sel[12],flex_sel_t[12],flex_sel[13],flex_sel_t[13],flex_sel[14],flex_sel_t[14],flex_sel[15],flex_sel_t[15],flex_sel[16],flex_sel_t[16],flex_sel[17],flex_sel_t[17]],patch_artist=True)
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL0','FL100','FL200','FL300','FL400','FL500'])
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')
plt.subplot(234)
plt.title('Point 4')
bp4 = plt.boxplot([flex_sel[18],flex_sel_t[18],flex_sel[19],flex_sel_t[19],flex_sel[20],flex_sel_t[20],flex_sel[21],flex_sel_t[21],flex_sel[22],flex_sel_t[22],flex_sel[23],flex_sel_t[23]],patch_artist=True)
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL0','FL100','FL200','FL300','FL400','FL500'])
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')
plt.subplot(235)
plt.title('Point 5')
bp5 = plt.boxplot([flex_sel[24],flex_sel_t[24],flex_sel[25],flex_sel_t[25],flex_sel[26],flex_sel_t[26],flex_sel[27],flex_sel_t[27],flex_sel[28],flex_sel_t[28],flex_sel[29],flex_sel_t[29]],patch_artist=True)
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL0','FL100','FL200','FL300','FL400','FL500'])
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')
plt.subplot(236)
plt.title('Point 6')
bp6 = plt.boxplot([flex_sel[30],flex_sel_t[30],flex_sel[31],flex_sel_t[31],flex_sel[32],flex_sel_t[32],flex_sel[33],flex_sel_t[33],flex_sel[34],flex_sel_t[34],flex_sel[35],flex_sel_t[35]],patch_artist=True)
plt.xticks([1.5, 3.5,5.5,7.5,9.5,11.5], ['FL0','FL100','FL200','FL300','FL400','FL500'])
plt.plot([2.5,2.5],[1,0], color='black')
plt.plot([4.5,4.5],[1,0], color='black')
plt.plot([6.5,6.5],[1,0], color='black')
plt.plot([8.5,8.5],[1,0], color='black')
plt.plot([10.5,10.5],[1,0], color='black')

for bplot in (bp1, bp2, bp3, bp4, bp5, bp6):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
# plt.subplot(267)
# plt.title('Point 1')
# plt.ylabel(str(traf_ehle)+' flt/h EHLE')
# plt.boxplot([flex_sel_t[0],flex_sel_t[1],flex_sel_t[2],flex_sel_t[3],flex_sel_t[4],flex_sel_t[5]])
# plt.xticks([1, 2,3,4,5,6], [' ','FL100',' ','FL300',' ','FL500'])
# plt.subplot(268)
# plt.title('Point 2')
# plt.boxplot([flex_sel_t[6],flex_sel_t[7],flex_sel_t[8],flex_sel_t[9],flex_sel_t[10],flex_sel_t[11]])
# plt.xticks([1, 2,3,4,5,6], [' ','FL100',' ','FL300',' ','FL500'])
# plt.subplot(269)
# plt.title('Point 3')
# plt.boxplot([flex_sel_t[12],flex_sel_t[13],flex_sel_t[14],flex_sel_t[15],flex_sel_t[16],flex_sel_t[17]])
# plt.xticks([1, 2,3,4,5,6], [' ','FL100',' ','FL300',' ','FL500'])
# plt.subplot(2,6,10)
# plt.title('Point 4')
# plt.boxplot([flex_sel_t[18],flex_sel_t[19],flex_sel_t[20],flex_sel_t[21],flex_sel_t[22],flex_sel_t[23]])
# plt.xticks([1, 2,3,4,5,6], [' ','FL100',' ','FL300',' ','FL500'])
# plt.subplot(2,6,11)
# plt.title('Point 5')
# plt.boxplot([flex_sel_t[24],flex_sel_t[25],flex_sel_t[26],flex_sel_t[27],flex_sel_t[28],flex_sel_t[29]])
# plt.xticks([1, 2,3,4,5,6], [' ','FL100',' ','FL300',' ','FL500'])
# plt.subplot(2,6,12)
# plt.title('Point 6')
# plt.boxplot([flex_sel_t[30],flex_sel_t[31],flex_sel_t[32],flex_sel_t[33],flex_sel_t[34],flex_sel_t[35]])
# plt.xticks([1, 2,3,4,5,6], [' ','FL100',' ','FL300',' ','FL500'])

plt.tight_layout()
plt.show()


#Select flexibility data for wanted points
print(ww_un)
flex_sel = flex_original[[i[0] for i in ww_un],[i[1] for i in ww_un],[i[2] for i in ww_un],:]
flex_sel_t = flex_traffic[[i[0] for i in ww_un],[i[1] for i in ww_un],[i[2] for i in ww_un],:]

flex_sel_list = []
flex_sel_t_list = []
for i in range(len(flex_sel)):
    flex_sel_list.append(flex_sel[i])
    flex_sel_t_list.append(flex_sel_t[i])

#Plot the data in boxplots
plt.figure(figsize=(12,8),num='distribution_t')
plt.subplot(211)
plt.ylabel('Original traffic')
plt.boxplot(flex_sel_list)
#plt.xticks(np.arange(len(flex_sel)).tolist(),[str(wwlat[i])+'N, '+str(wwlon[i])+'E, FL'+str(ww_un[i][2]*100) for i in range(len(flex_sel))])
plt.subplot(212)
plt.ylabel(str(traf_ehle)+' flt/h EHLE')
plt.boxplot(flex_sel_t_list)
#plt.xticks(np.arange(len(flex_sel)).tolist(),[str(wwlat[i])+'N, '+str(wwlon[i])+'E, FL'+str(ww_un[i][2]*100) for i in range(len(flex_sel))])
plt.show()



if 1==0:
    for i in range(5):#range(len(ww_un[:,0])):
        print(i)
        #Box plots of flexibility in each conflictive point
        #Original vs extra traffic boxes to compare
        plt.figure(figsize=(12,8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1.5, 2])
        #plt.title('Flexibility distribution at '+str(wwlat[i])+'N, '+str(wwlon[i])+'E, FL'+str(ww_un[i][2]*100))
        plt.suptitle('Flexibility distribution at '+str(wwlat[i])+'N, '+str(wwlon[i])+'E, FL'+str(ww_un[i][2]*100))
        plt.subplot(gs[0])
        flex_orig = np.concatenate([flex4d1[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d2[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d3[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d4[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d5[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d6[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4]])
        flex_avg = np.nanmean(flex_orig)
        flex_orig_z = stats.zscore(flex_orig)
        flex_traf = np.concatenate([flex4d1s[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d2s[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d3s[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d4s[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d5s[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4],flex4d6s[ww_un[i][0],ww_un[i][1],ww_un[i][2],:-4]])
        flex_traf_z = stats.zscore(flex_traf)
        flex_diff = flex_traf-flex_orig
        flex_diff_z = stats.zscore(flex_diff)
        #Flatten arrays for easier handling and remove NaN values
        flex_orig = flex_orig.flatten()
        flex_orig = flex_orig[~np.isnan(flex_orig)]
        flex_traf = flex_traf.flatten()
        flex_traf = flex_traf[~np.isnan(flex_traf)]
        flex_diff = flex_diff.flatten()
        flex_diff = flex_diff[~np.isnan(flex_diff)]
        #Make boxplots
        print(np.shape(flex_orig_z))
        plt.boxplot([flex_orig,flex_traf])
        plt.xticks([1, 2], ['Original', str(traf_ehle)+' extra flights'])
        plt.subplot(gs[1])
        plt.boxplot([flex_diff])
        plt.xticks([1],['Difference'])
        plt.subplot(gs[2])
        m.shadedrelief(scale=0.5)
        m.plot(x,y,color='blue', markersize=5, linewidth=1)
        m.plot(xg[i],yg[i],color = 'black',marker='x',markersize=5)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        plt.figure(figsize=(12,8))
        ax1 = plt.subplot(221)
        plt.hist(flex_orig, np.linspace(0,1,num=50))
        plt.subplot(223, sharex = ax1)
        plt.boxplot(flex_orig, vert=False)
        ax2 = plt.subplot(222)
        plt.hist(flex_orig_z,np.linspace(np.amin(flex_orig_z),np.amax(flex_orig_z),num=50))
        plt.subplot(224, sharex = ax2)
        plt.boxplot(flex_orig_z,vert=False)
        plt.show()


    #Perform anova analisys to see correlation of data from different days in the same point
    #f,p = stats.f_oneway(dflex4_1[ww_un[i][0],ww_un[i][1],ww_un[i][2],:].flatten(), dflex4_2[ww_un[i][0],ww_un[i][1],ww_un[i][2],:].flatten(), dflex4_3[ww_un[i][0],ww_un[i][1],ww_un[i][2],:].flatten(), dflex4_4[ww_un[i][0],ww_un[i][1],ww_un[i][2],:].flatten(), dflex4_5[ww_un[i][0],ww_un[i][1],ww_un[i][2],:].flatten(), dflex4_6[ww_un[i][0],ww_un[i][1],ww_un[i][2],:].flatten())
    #print('F-value: '+str(f%8)+'. P-value: '+str(p%8))
