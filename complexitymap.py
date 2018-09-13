import numpy as np
import matplotlib.pyplot as plt

#Load data files of flexibility and airspace

flex = np.load('flexibility2d.npy')
flex3d = np.load('flexibility3d.npy')
print(flex)
coord = np.load('airspace.npy')

fig, ax0 = plt.subplots(1)
c = ax0.pcolor(np.transpose(flex))
#ax0.plot(coord[0,1::2],coord[0,0::2],'k')
ax0.set_title('Flexibility map')
plt.colorbar(c)
fig.tight_layout()
plt.show()

import seaborn as sns
ax = sns.heatmap(np.transpose(flex))
plt.show()
