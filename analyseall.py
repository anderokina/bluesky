import numpy as np
import matplotlib.pyplot as plt


traf_day = ['7-9','31-8','29-6','06-7','15-6','28-6']
traf_peak = ['1','3','5']
traf_ehle = ['','_15','_25','_45']
traf_ehle_n = np.array([0,15,25,45])

#Values to compute
flexmax = np.zeros((len(traf_day),len(traf_peak),len(traf_ehle)))
flexmin = np.zeros((len(traf_day),len(traf_peak),len(traf_ehle)))
flex_avg = np.zeros((len(traf_day),len(traf_peak),len(traf_ehle)))
flex_std = np.zeros((len(traf_day),len(traf_peak),len(traf_ehle)))
flex_red = np.zeros((len(traf_day),len(traf_peak)))

for i in range(len(traf_day)):
    for j in range(len(traf_peak)):
        for k in range(len(traf_ehle)):
            flex = np.load('Results/Grid/flexibility_'+traf_day[i]+'-'+traf_peak[j]+traf_ehle[k]+'.npy')
            flexmax[i,j,k] = np.nanmax(flex)
            flexmin[i,j,k] = np.nanmin(flex)
            flex_avg[i,j,k] = np.nanmean(flex)
            flex_std[i,j,k] = np.nanstd(flex)
        flex_red[i,j] = (flex_avg[i,j,0]-flex_avg[i,j,-1])/45 #Reduction of flexibility for each added traffic

print(flex_red*100)

#Choose which parameter to plot
flex_c = 100*flex_avg

plt.figure(figsize=(12,8), num = 'Average_flex')
plt.subplot(321) #7-9
plt.plot(traf_ehle_n,flex_c[0,0,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[0,1,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[0,2,:], marker = 'x')
plt.title('Average flexibility on '+traf_day[0])
plt.legend(['In wave','Out wave','In+out wave'])
plt.xlabel('Lelystad traffic (flt/h)')
plt.ylabel('Flexibility (%)')
plt.xticks([0, 15, 25, 45], ['0', '15', '25', '45'])
plt.subplot(322) #31-8
plt.plot(traf_ehle_n,flex_c[1,0,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[1,1,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[1,2,:], marker = 'x')
plt.title('Average flexibility on '+traf_day[1])
plt.xlabel('Lelystad traffic (flt/h)')
plt.ylabel('Flexibility (%)')
plt.xticks([0, 15, 25, 45], ['0', '15', '25', '45'])
plt.subplot(323) #29-6
plt.plot(traf_ehle_n,flex_c[2,0,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[2,1,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[2,2,:], marker = 'x')
plt.title('Average flexibility on '+traf_day[2])
plt.xlabel('Lelystad traffic (flt/h)')
plt.ylabel('Flexibility (%)')
plt.xticks([0, 15, 25, 45], ['0', '15', '25', '45'])
plt.subplot(324)
plt.plot(traf_ehle_n,flex_c[3,0,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[3,1,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[3,2,:], marker = 'x')
plt.title('Average flexibility on '+traf_day[3])
plt.xlabel('Lelystad traffic (flt/h)')
plt.ylabel('Flexibility (%)')
plt.xticks([0, 15, 25, 45], ['0', '15', '25', '45'])

plt.subplot(325)
plt.plot(traf_ehle_n,flex_c[4,0,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[4,1,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[4,2,:], marker = 'x')
plt.title('Average flexibility on '+traf_day[4])
plt.xlabel('Lelystad traffic (flt/h)')
plt.ylabel('Flexibility (%)')
plt.xticks([0, 15, 25, 45], ['0', '15', '25', '45'])

plt.subplot(326)
plt.plot(traf_ehle_n,flex_c[5,0,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[5,1,:], marker = 'x')
plt.plot(traf_ehle_n,flex_c[5,2,:], marker = 'x')
plt.title('Average flexibility on '+traf_day[5])
plt.xlabel('Lelystad traffic (flt/h)')
plt.ylabel('Flexibility (%)')
plt.xticks([0, 15, 25, 45], ['0', '15', '25', '45'])

plt.tight_layout()
plt.show()

flex1 = np.load('Results/Grid/flexibility_7-9-3.npy')
flex2 = np.load('Results/Grid/flexibility_7-9-3_45.npy')

flex1 = flex1.flatten()
flex2 = flex2.flatten()
flex1 = flex1[~np.isnan(flex1)]
flex2 = flex2[~np.isnan(flex2)]

plt.figure()
bp1 = plt.boxplot([flex1,flex2],patch_artist=True)
plt.xticks([1,2],['Original','45 flt/h EHLE'])
plt.title('Flexibility distribution of the whole scenario')


plt.tight_layout()
plt.show()
