import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.ndimage.filters import minimum_filter, maximum_filter



flex = np.load('flexibility2d.npy')
coord = np.array([54.96964, 5.00976, 51.45904, 1.99951, 50.71375, 6.0424, 52.2058, 7.09716, 53.2963, 7.2509, 54.9922, 6.5478,54.96964, 5.00976])
coords = np.reshape(coord,(7,2))
d = 1/20 #Discretise coordinates for every 1/100 of degree -> 0,6 nm
nlat = (np.amax(coords[:,0]) - np.amin(coords[:,0]))/d
lats = np.amax(coords[:,0])-np.arange(nlat)*d
nlon = (np.amax(coords[:,1]) - np.amin(coords[:,1]))/d
lons = np.amin(coords[:,1])+np.arange(nlon)*d

lon, lat = np.meshgrid(lons, lats)
fig = plt.figure(figsize=(10, 8))
lat_0 = (lats[:].mean())
m = Basemap(projection = 'lcc',\
        lon_0 = 0, lat_0 = 50, lat_ts = lat_0,\
        llcrnrlat=50,urcrnrlat=55,\
        llcrnrlon=0,urcrnrlon= 8,\
        resolution='l')
#m.shadedrelief(scale=0.5)
m.pcolormesh(lon, lat, np.transpose(flex), latlon=True, cmap='RdBu_r')
plt.clim(0, 1)
m.drawcoastlines(linewidth=2,color='lightgray')
m.drawcountries(linewidth=2,color='lightgray')
#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='coral',lake_color='aqua')
x, y = m(coords[:,1], coords[:,0])
m.plot(x, y, color = 'black', markersize=5, linewidth=1)

plt.title('Flexibility')
plt.colorbar(label='Flexibility %');
plt.show()
