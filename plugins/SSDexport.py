""" Plugin to generate SSD figures and save them in the project folder """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, sim#, scr, tools
import numpy as np
from datetime import datetime


#Plotting packages
import matplotlib.pyplot as plt
#from matplotlib.cbook import get_sample_data
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code
    global data
    data = Simulation()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'SSDEXPORT',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': traf.asas.dtasas,

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
        'EXPORT': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'EXPORT FLT',

            # A list of the argument types your function accepts. For a description of this, see ...
            '[txt]',

            # The name of your function in this plugin
            data.initialize,

            # a longer help text of your function.
            'Print something to the bluesky console based on the flag passed to MYFUN.']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

class Simulation():
    def __init__(self):
        self.time_stamp = 0
        self.active = False
        #self.filename = "simulation_data.txt"
        self.save_SSD = True
        self.show_resolayer = True


    def update(self):

        if self.active== False: # only execute if plugin is turned on
            return

        self.time_stamp += 1

        print('Got to update!!!')

        #if traf.asas.inconf.any():
        #    self.update_txtfile()

        if self.save_SSD == True:
            # SSD - CIRCLES OF VMAX AND VMIN
            vmin = traf.asas.vmin
            vmax = traf.asas.vmax
            N_angle = 180

            angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
            xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
            SSD_lst = [list(map(list, np.flipud(xyc * vmax))), list(map(list, xyc * vmin))]

            # Outer circle: vmax
            SSD_outer = np.array(SSD_lst[0])
            x_SSD_outer = np.append(SSD_outer[:, 0], np.array(SSD_outer[0, 0]))
            y_SSD_outer = np.append(SSD_outer[:, 1], np.array(SSD_outer[0, 1]))
            # Inner circle: vmin
            SSD_inner = np.array(SSD_lst[1])
            x_SSD_inner = np.append(SSD_inner[:, 0], np.array(SSD_inner[0, 0]))
            y_SSD_inner = np.append(SSD_inner[:, 1], np.array(SSD_inner[0, 1]))
            print('On to visualize')
            self.visualizeSSD(x_SSD_outer,y_SSD_outer,x_SSD_inner,y_SSD_inner)

        #stack.stack('ECHO This is an update.')
        #stack.stack('ECHO The current time stamp is {time_stamp} seconds'.format(time_stamp = self.time_stamp))

    def preupdate(self):
        return

    def reset(self):
        pass

    #It's only called once
    def initialize(self,*args):

        if not args:
            return True, "SSDSEQ is currently " + ("ON" if self.active else "OFF")

        self.active = True

        if self.active== True and len(args)== 1:
            #If the txt file is already created, it should be deleted before a new simulation

            #stack.stack('ECHO The current time stamp is {time_stamp} seconds'.format(time_stamp = self.time_stamp))

            stack.stack('SSD {flight}'.format(flight = args[0]))
            #stack.stack('SYN SUPER {no_ac}'.format(no_ac = args[1]))
            #stack.stack('RESO SEQSSD')
            self.save_SSD = True

            timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')
            #self.filename     = "ASASDATA_%s_%s.log" % (stack.get_scenname(), timestamp)
            #self.path_to_file = "output_smallscn/" + self.filename

            #A new scenario was uploaded so the time stamp has to be set to 0
            #self.time_stamp = 0

        return True, 'My plugin received an o%s flag.' % ('n' if self.active else 'ff')

    def visualizeSSD(self, x_SSD_outer,y_SSD_outer,x_SSD_inner,y_SSD_inner):
        ''' VISUALIZING SSD'''
        print(traf.asas.cr_name)
        for i in range(traf.ntraf):
            if traf.asas.cr_name == "SSD":
                print('In!')
                #v_own = np.array([traf.gseast[i], traf.gsnorth[i]])

                #------------------------------------------------------------------------------

                #PLOTS
                fig, ax = plt.subplots()

                line1, = ax.plot(x_SSD_outer, y_SSD_outer, color = '#000000', label="Velocity limits")
                ax.plot(x_SSD_inner, y_SSD_inner, color = '#404040')

                if traf.asas.ARV[i]:
                    for j in range(len(traf.asas.ARV[i])):
                        FRV_1 = np.array(traf.asas.ARV[i][j])
                        x_FRV1 = np.append(FRV_1[:,0] , np.array(FRV_1[0,0]))
                        y_FRV1 = np.append(FRV_1[:,1] , np.array(FRV_1[0,1]))
                        plt.plot(x_FRV1, y_FRV1, '-', color = '#C0C0C0', alpha = 0.2) #grey
                        ax.fill(x_FRV1, y_FRV1, color = '#C0C0C0', alpha = 0.2) #grey


                if traf.asas.FRV[i]:
                    for j in range(len(traf.asas.FRV[i])):
                        FRV_1 = np.array(traf.asas.FRV[i][j])
                        x_FRV1 = np.append(FRV_1[:,0] , np.array(FRV_1[0,0]))
                        y_FRV1 = np.append(FRV_1[:,1] , np.array(FRV_1[0,1]))
                        plt.fill(x_FRV1, y_FRV1, color = '#808080') #grey
                        #plt.plot(x_FRV1, y_FRV1, '-', color = '#FF0000', alpha=0.5) #red
                        #plt.fill(x_FRV1, y_FRV1, color = '#FF0000', alpha= 0.5) #red


                """
                if traf.asas.FRV_5[i]:
                    for j in range(len(traf.asas.FRV_5[i])):
                        FRV_1_5 = np.array(traf.asas.FRV_5[i][j])
                        x_FRV1_5 = np.append(FRV_1_5[:,0] , np.array(FRV_1_5[0,0]))
                        y_FRV1_5 = np.append(FRV_1_5[:,1] , np.array(FRV_1_5[0,1]))
                        plt.plot(x_FRV1_5, y_FRV1_5, '-', color = '#FFFF33')
                        plt.fill(x_FRV1_5, y_FRV1_5, color = '#FFFF33')


                if traf.asas.FRV_3[i]:
                    for j in range(len(traf.asas.FRV_3[i])):
                        FRV_1_3 = np.array(traf.asas.FRV_3[i][j])
                        x_FRV1_3 = np.append(FRV_1_3[:,0] , np.array(FRV_1_3[0,0]))
                        y_FRV1_3 = np.append(FRV_1_3[:,1] , np.array(FRV_1_3[0,1]))
                        plt.plot(x_FRV1_3, y_FRV1_3, '-r')
                        plt.fill(x_FRV1_3, y_FRV1_3, 'r')
                """

                if self.show_resolayer == True:
                    no_layer = traf.asas.reso_layer[i] #layer number
                    if not no_layer == 0:
                        layer = traf.asas.layers[no_layer][i]
                        if len(layer)>0:
                            for j in range(len(layer)):
                                FRV_1_5 = np.array(layer[j])
                                x_FRV1_5 = np.append(FRV_1_5[:,0] , np.array(FRV_1_5[0,0]))
                                y_FRV1_5 = np.append(FRV_1_5[:,1] , np.array(FRV_1_5[0,1]))
                                plt.plot(x_FRV1_5, y_FRV1_5, '-', color = '#000000') #limited in black

                vown = traf.gs[i]*0.92
                hdg = np.radians(traf.hdg[i])
                vownx = vown*np.sin(hdg)
                vowny = vown*np.cos(hdg)

                ax.arrow(x=0,y=0, dx=vownx, dy=vowny, color = '#00CC00', head_width=15, overhang=0.5, zorder=10)
                sol_point, = ax.plot(traf.asas.asase[i], traf.asas.asasn[i], 'd', color = '#000099', label='Solution')


                """ Legend """

                #For color coding
                #red_patch = mpatches.Patch(color = '#FF0000', label= r'$t_{LoS} \leq 3\ mins$')
                #gray_patch = mpatches.Patch(color = '#808080', label=r'$t_{LoS} > 5\ mins$') #dark grey patch for FRV
                #yellow_patch = mpatches.Patch(color = '#FFFF33', label= r'$ 3 \ mins < t_{LoS} \leq 5\ mins$') #dark grey patch for FRV
                #white_patch = mpatches.Patch(label='ARV', color = '#C0C0C0', alpha = 0.2)
                #vel_line = mlines.Line2D([], [], color = '#00CC00',linestyle='-', linewidth=1.5, label='Velocity vector')
                #layer_line = mlines.Line2D([], [], color = '#000000',linestyle='-', linewidth=1.5, label='Selected layer: CS' + str(traf.asas.reso_layer[i]))
                #plt.legend(handles=[gray_patch, yellow_patch, red_patch, white_patch, line1, vel_line, sol_point, layer_line], loc=1, borderaxespad=0., bbox_to_anchor=(1.30, 1))

                plt.axis('equal')
                plt.axis('off')
                plt.savefig("/Users/anderokina/Documents/GitHub/bluesky/figures"+ str(traf.id[i])+"_"+str(self.time_stamp) +"s"+".jpg",bbox_inches = 'tight')
                plt.close()

        return
