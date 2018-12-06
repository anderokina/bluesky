import numpy as np
import matplotlib.pyplot as plt

#Select which scenario to process
traf_peak = 'grid1'
traf_ehle = '15'

#Load flexibility scenario data for the previous data

if traf_ehle != '':
    flex4d = np.load('Results/EHAM'+str(traf_peak)+'/eham_'+str(traf_peak)+'_'+traf_ehle+'_ehle.npy')
else:
    flex4d = np.load('Results/EHAM'+str(traf_peak)+'/eham_'+str(traf_peak)+'.npy')

#Compute relevant flexibility data for each grid position
#Maximum, minimum, stdDev, average
flex_max = np.nanmax(flex4d,axis=3)
flex_min = np.nanmin(flex4d,axis=3)
flex_d = flex_max-flex_min
flex_std = np.nanstd(flex4d,axis=3)
flex_avg = np.nanmean(flex4d,axis=3)

#Select relevant points based on previous data
#Minimum value lower than certain threshold
min_p = np.where(flex_min<0.05)
min_data = flex4d[min_p[0],min_p[1],min_p[2],:]
print('Minimum values: '+str(np.shape(min_data)))

#Difference between max and min higher than certain value
diff_p = np.where(flex_d>0.99)
diff_data = flex4d[diff_p[0],diff_p[1],diff_p[2],:]
print('Difference values: '+str(np.shape(diff_data)))
#Get rid of NaNs
diff_data = diff_data[:,:-4].tolist()

#Plot the data from the interesting points in a boxplot
plt.boxplot(diff_data)
plt.show()

#Average lower than threshold
avg_p = np.where(flex_avg<0.45)
avg_data = flex4d[avg_p[0],avg_p[1],avg_p[2],:]
print('Average values: '+str(np.shape(avg_data)))

#Get rid of NaNs
avg_data = avg_data[:,:-4].tolist()

#Plot the data from the interesting points in a boxplot
plt.boxplot(avg_data)
plt.show()
