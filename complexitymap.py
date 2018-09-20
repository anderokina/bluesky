import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Load data files of flexibility and airspace

flex = np.load('flexibility2d.npy')
flex3d = np.load('flexibility3d.npy')
coord = np.load('airspace.npy')

#fig, ax0 = plt.subplots(1)
#c = ax0.pcolor((flex))
#ax0.plot(coord[0,1::2],coord[0,0::2],'k')
#ax0.set_title('Flexibility map')
#plt.colorbar(c)
#fig.tight_layout()
#plt.show()

ax = sns.heatmap((flex))
ax.plot(coord[0,1::2],coord[0,0::2],'k')
plt.show()

## Plot two different altitudes side by side


alt1 = [0,15000]
flex_alt1 = flex3d[:,:,int(round(alt1[0]/1000)):int(round(alt1[1]/1000))]
alt2 = [30000,35000]
flex_alt2 = flex3d[:,:,int(round(alt2[0]/1000)):int(round(alt2[1]/1000))]

flex_alt1_mean = np.nanmean(flex_alt1,axis=2)
flex_alt2_mean = np.nanmean(flex_alt2,axis=2)


plt.subplot(2, 1, 1)
sns.heatmap(flex_alt1_mean)
plt.title('Flexibility '+str(alt1)+' ft.')

plt.subplot(2, 1, 2)
sns.heatmap(flex_alt2_mean)
plt.title('Flexibility '+str(alt2)+' ft.')
plt.tight_layout()

plt.show()
