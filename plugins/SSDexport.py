""" Plugin to generate SSD figures and save them in the project folder """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, sim#, scr, tools
from bluesky.tools.aero import nm
from bluesky.tools import geo
import numpy as np
from datetime import datetime
from bluesky.traffic.asas import SeqSSD_faster as SSDfun

#Plotting packages
import matplotlib.pyplot as plt
#from matplotlib.cbook import get_sample_data
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os

try:
    import pyclipper
except ImportError:
    print("Could not import pyclipper, RESO SSD will not function")

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
        'SSDEXPORT': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'SSDEXPORT ON/OFF',

            # A list of the argument types your function accepts. For a description of this, see ...
            '[onoff]',

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
        self.show_resolayer = False


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
            # Initialize SSD variables with ntraf
            self.SSDVariables(traf.asas, traf.ntraf)
            # Construct ASAS
            self.constructSSD1(traf.asas, traf)
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

        self.active = True if args[0]==True else False

        if self.active== True and len(args)== 1:
            #If the txt file is already created, it should be deleted before a new simulation

            #stack.stack('ECHO The current time stamp is {time_stamp} seconds'.format(time_stamp = self.time_stamp))

            #stack.stack('SSD {flight}'.format(flight = args[0]))
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

        for i in range(traf.ntraf):
            if 1==1:
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
                        plt.plot(x_FRV1, y_FRV1, '-', color = '#000000') #grey
                        ax.fill(x_FRV1, y_FRV1, color = '#C0C0C0') #grey


                if traf.asas.FRV[i]:
                    for j in range(len(traf.asas.FRV[i])):
                        FRV_1 = np.array(traf.asas.FRV[i][j])
                        x_FRV1 = np.append(FRV_1[:,0] , np.array(FRV_1[0,0]))
                        y_FRV1 = np.append(FRV_1[:,1] , np.array(FRV_1[0,1]))
                        plt.fill(x_FRV1, y_FRV1, color = '#FF0000') #red
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
                plt.savefig(os.getcwd()+"/figures/"+ str(traf.id[i])+"_"+str(self.time_stamp) +"s"+".png",format = 'png',bbox_inches = 'tight')
                plt.close()

                #Process the picture.....

                #Delete the picture
                #os.remove("/Users/anderokina/Documents/GitHub/bluesky/figures"+ str(traf.id[i])+"_"+str(self.time_stamp) +"s"+".jpg")



        return

    def SSDVariables(self,asas, ntraf):
        """ Initialize variables for SSD """
        # Need to do it here, since ASAS.reset doesn't know ntraf

        #Forbidden Reachable Velocity regions
        asas.FRV          = [None] * ntraf

        #Allowed Reachable Velocity regions
        asas.ARV          = [None] * ntraf
        asas.ARV_min        = [None] * ntraf #NEW
        asas.ARV_tla        = [None] * ntraf #NEW

        # For calculation purposes
        asas.ARV_calc     = [None] * ntraf
        asas.ARV_calc_min = [None]* ntraf
        asas.ARV_calc_glb = [None]* ntraf
        asas.ARV_calc_dlos = [None]* ntraf
        asas.ARV_calc_dcpa = [None]* ntraf

        #Stores which layer of resolution each aircraft chose
        asas.reso_layer = [None]* ntraf
        asas.inrange      = [None] * ntraf

        # Stores resolution vector, also used in visualization
        asas.asasn        = np.zeros(ntraf, dtype=np.float32)
        asas.asase        = np.zeros(ntraf, dtype=np.float32)

        #Say if ac is in a LoS
        asas.los = [False]*ntraf

        # Area calculation
        asas.FRV_area     = np.zeros(ntraf, dtype=np.float32)

        asas.ARV_area     = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_min = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_tla = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_calc = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_calc_min = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_calc_glb = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_calc_dcpa = np.zeros(ntraf, dtype=np.float32)
        asas.ARV_area_dlos = np.zeros(ntraf, dtype=np.float32)
        asas.layers_area = [None]*len(asas.layers_dict)



    def constructSSD1(self,asas, traf):
        """ Calculates the FRV and ARV of the SSD """

        #N = 0
        # Parameters - from ASAS
        N_angle = 180
        vmin    = asas.vmin             # [m/s] Defined in asas.py
        vmax    = asas.vmax             # [m/s] Defined in asas.py
        hsep    = asas.R                # [m] Horizontal separation (5 NM)
        margin  = asas.mar              # [-] Safety margin for evasion
        hsepm   = hsep * margin         # [m] Horizontal separation with safety margin
        alpham  = 0.4999 * np.pi        # [rad] Maximum half-angle for VO
        betalos = np.pi / 4             # [rad] Minimum divertion angle for LOS (45 deg seems optimal)
        adsbmax = 40. * nm              # [m] Maximum ADS-B range
        beta    =  np.pi/4 + betalos/2

        #From traf
        lat     = traf.lat
        lon     = traf.lon
        ntraf   = traf.ntraf

        #A default priocode must be defined for this CR method, otherwise it won't work with the predefined one
        if asas.priocode not in asas.strategy_dict:
            asas.priocode = "SRS1"

        # # Use velocity limits for the ring-shaped part of the SSD
        # Discretize the circles using points on circle
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / N_angle)
        # Put points of unit-circle in a (180x2)-array (CW)
        xyc = np.transpose(np.reshape(np.concatenate((np.sin(angles), np.cos(angles))), (2, N_angle)))
        # Map them into the format pyclipper wants. Outercircle CCW, innercircle CW
        circle_tup = (tuple(map(tuple, np.flipud(xyc * vmax))), tuple(map(tuple , xyc * vmin)))
        circle_lst = [list(map(list, np.flipud(xyc * vmax))), list(map(list , xyc * vmin))]

        # If no traffic
        if ntraf == 0:
            return

        # If only one aircraft
        elif ntraf == 1:
            # Map them into the format ARV wants. Outercircle CCW, innercircle CW
            asas.ARV[0] = circle_lst
            asas.ARV_min[0] = circle_lst
            asas.ARV_tla[0] = circle_lst

            asas.FRV[0] = []
            asas.FRV_min[0] = []
            asas.FRV_tla[0]     = []
            asas.FRV_dlos[0]    = []

            asas.ARV_calc[0] = circle_lst
            asas.ARV_calc_min[0] = circle_lst
            asas.ARV_calc_glb[0] = circle_lst
            asas.ARV_calc_dlos[0] = circle_lst
            asas.ARV_calc_dcpa[0] = circle_lst

            # Calculate areas and store in asas
            asas.FRV_area[0] = 0

            asas.ARV_area[0] = 1
            asas.ARV_area_min[0] = 1
            asas.ARV_area_tla[0] = 1
            asas.ARV_area_dlos[0] = 1
            asas.ARV_area_calc[0] = 1
            asas.ARV_area_calc_min[0] = 1
            asas.ARV_area_calc_glb[0] = 1
            asas.ARV_area_calc_dcpa[0] = 1
            return

        # Function qdrdist_matrix needs 4 vectors as input (lat1,lon1,lat2,lon2)
        # To be efficient, calculate all qdr and dist in one function call
        # Example with ntraf = 5:   ind1 = [0,0,0,0,1,1,1,2,2,3]
        #                           ind2 = [1,2,3,4,2,3,4,3,4,4]
        # This way the qdrdist is only calculated once between every aircraft
        # To get all combinations, use this function to get the indices
        ind1, ind2 = self.qdrdist_matrix_indices(ntraf)

        # Get absolute bearing [deg] and distance [nm]
        # Not sure abs/rel, but qdr is defined from [-180,180] deg, w.r.t. North

        [qdr, dist] = geo.qdrdist_matrix(lat[ind1], lon[ind1], lat[ind2], lon[ind2])

        # Put result of function from matrix to ndarray
        qdr  = np.reshape(np.array(qdr), np.shape(ind1))
        dist = np.reshape(np.array(dist), np.shape(ind1))
        # SI-units from [deg] to [rad]
        qdr  = np.deg2rad(qdr)
        # Get distance from [nm] to [m]
        dist = dist * nm

        # In LoS the VO can't be defined, act as if dist is on edge
        dist[dist < hsepm] = hsepm

        # Calculate vertices of Velocity Obstacle (CCW)
        # These are still in relative velocity space, see derivation in appendix
        # Half-angle of the Velocity obstacle [rad]
        # Include safety margin
        alpha = np.arcsin(hsepm / dist)
        # Limit half-angle alpha to 89.982 deg. Ensures that VO can be constructed
        alpha[alpha > alpham] = alpham
        #Take the cosinus of alpha to calculate the maximum length of the VO's legs
        cosalpha = np.cos(alpha)

        #construct with CS1
        self.CS1(asas, traf, ind1, ind2, adsbmax, dist, qdr, cosalpha, xyc, circle_tup, circle_lst, beta, hsepm)


        """
        if asas.priocode == "SRS1":
            srs1(asas, traf, ind1, ind2, adsbmax, dist, qdr, cosalpha, xyc, circle_tup, circle_lst, beta, hsepm)
        """

    def CS1(self,asas, traf, ind1, ind2, adsbmax, dist, qdr, cosalpha, xyc, circle_tup, circle_lst, beta, hsepm):

        # Relevant info from traf and ASAS
        gsnorth = traf.gsnorth
        gseast  = traf.gseast
        ntraf = traf.ntraf
        vmax = asas.vmax
        vmin = asas.vmin

        # Local variables, will be put into asas later
        FRV_loc          = [None] * traf.ntraf

        ARV_loc          = [None] * traf.ntraf


        # Consider every aircraft
        for i in range(ntraf):
            # Calculate SSD only for aircraft in conflict (See formulas appendix)
            if 1==1:


                # SSD for aircraft i
                # Get indices that belong to aircraft i
                ind = np.where(np.logical_or(ind1 == i,ind2 == i))[0]

                # The i's of the other aircraft
                i_other = np.delete(np.arange(0, ntraf), i)
                # Aircraft that are within ADS-B range
                ac_adsb = np.where(dist[ind] < adsbmax)[0]
                # Now account for ADS-B range in indices of other aircraft (i_other)
                ind = ind[ac_adsb]
                i_other = i_other[ac_adsb]
                asas.inrange[i]  = i_other

                # VO from 2 to 1 is mirror of 1 to 2. Only 1 to 2 can be constructed in
                # this manner, so need a correction vector that will mirror the VO
                fix = np.ones(np.shape(i_other))
                fix[i_other < i] = -1


                drel_x, drel_y = fix*dist[ind]*np.sin(qdr[ind]), fix*dist[ind]*np.cos(qdr[ind])
                drel = np.dstack((drel_x,drel_y))

                cosalpha_i = cosalpha[ind]

                # Make a clipper object
                pc = pyclipper.Pyclipper()

                N_angle = 180
                #Define new ARV taking into consideration the heading constraints and the current heading of each aircraft

                asas.trncons = 90 #Turning angle constraint
                trn_cons = np.radians(asas.trncons)
                angles2 = np.arange(np.radians(traf.hdg[i])-trn_cons, np.radians(traf.hdg[i])+trn_cons, 2*trn_cons/N_angle)
                # Put points of unit-circle in a (180x2)-array (CW)
                xyc2 = np.transpose(np.reshape(np.concatenate((np.sin(angles2), np.cos(angles2))), (2, len(angles2))))
                #For tupple
                inner_semicircle = (tuple(map(tuple , xyc2 * vmin)))
                outer_semicircle = tuple(map(tuple, np.flipud(xyc2 * vmax)))
                new_circle_tup = inner_semicircle + outer_semicircle
                #For list
                inner_semicircle = [list(map(list , xyc2 * vmin))]
                outer_semicircle = [list(map(list, np.flipud(xyc2 * vmax)))]
                new_circle_lst = inner_semicircle + outer_semicircle


                if asas.trncons < 180:

                    # Add circles (ring-shape) to clipper as subject
                    pc.AddPath(pyclipper.scale_to_clipper(new_circle_tup), pyclipper.PT_SUBJECT, True)
                else:
                    #consider the whole SSD
                    pc.AddPaths(pyclipper.scale_to_clipper(circle_tup), pyclipper.PT_SUBJECT, True)

                # Add each other other aircraft to clipper as clip
                for j in range(np.shape(i_other)[0]):

                    ## Debug prints
                    ##print(traf.id[i] + " - " + traf.id[i_other[j]])
                    ## print(dist[ind[j]])
                    # Scale VO when not in LOS
                    if dist[ind[j]] > hsepm:

                        dist_mod = dist[ind[j]] #the value (not array) of the distance is needed for future computations


                        #direction of the VO's bisector
                        nd = drel[0,j,:]/dist_mod

                        R_pz = asas.R*asas.mar

                        R = np.array([[np.sqrt(1-(R_pz/dist_mod)**2), R_pz/dist_mod], [-R_pz/dist_mod, np.sqrt(1-(R_pz/dist_mod)**2)] ])

                        n_t1 = np.matmul(nd, R) #Direction of leg2
                        n_t2 = np.matmul(nd, np.transpose(R)) #Direction of leg1

                        #VO points
                        v_other = [gseast[i_other[j]],gsnorth[i_other[j]]]
                        legs_length = 10*vmax/cosalpha_i[j]
                        VO_points = np.array([v_other, np.add(n_t2*legs_length, v_other), np.add( n_t1* legs_length, v_other)])

                        # Normally VO shall be added of this other a/c
                        VO = pyclipper.scale_to_clipper(tuple(map(tuple, VO_points)))

                    else:
                        # Pair is in LOS
                        asas.los[i] = True
                        #In case two aircraft are in LoS, consider a samller RPZ
                        #in order to guarantee they get out of the LoS ASAP

                        dist_mod = dist[ind[j]] #the value (not array) of the distance is needed for future computations

                        R_pz = dist_mod*0.80

                        #direction of the VO's bisector
                        nd = drel[0,j,:]/dist_mod

                        R = np.array([[np.sqrt(1-(R_pz/dist_mod)**2), R_pz/dist_mod], [-R_pz/dist_mod, np.sqrt(1-(R_pz/dist_mod)**2)] ])

                        n_t1 = np.matmul(nd, R) #Direction of leg2
                        n_t2 = np.matmul(nd, np.transpose(R)) #Direction of leg1

                        #VO points
                        v_other = [gseast[i_other[j]],gsnorth[i_other[j]]]
                        legs_length = 10*vmax/cosalpha_i[j]
                        VO_points = np.array([v_other, np.add(n_t2*legs_length, v_other), np.add( n_t1* legs_length, v_other)])

                        # Normally VO shall be added of this other a/c
                        VO = pyclipper.scale_to_clipper(tuple(map(tuple, VO_points)))


                    # Add scaled VO to clipper
                    pc.AddPath(VO, pyclipper.PT_CLIP, True)

                # Execute clipper command
                FRV = pyclipper.scale_from_clipper(pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO))
                ARV = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                # Scale back
                ARV = pyclipper.scale_from_clipper(ARV)


                # Check multi exteriors, if this layer is not a list, it means it has no exteriors
                # In that case, make it a list, such that its format is consistent with further code

                if len(ARV) == 0:
                    ARV_loc[i] = []
                    FRV_loc[i] = new_circle_lst
                else:
                    #Then:
                    if len(FRV) == 0:
                        FRV_loc[i] = []
                    else:
                        if not type(FRV[0][0]) == list:
                            FRV = [FRV]
                        FRV_loc[i] = FRV


                    if not type(ARV[0][0]) == list:
                        ARV = [ARV]
                    ARV_loc[i] = ARV





        #Storing the results into asas
        asas.FRV          = FRV_loc

        asas.ARV          = ARV_loc

        #The layers list
        asas.layers = [None, asas.ARV, None, None, None, None, None, None]

        return

    def qdrdist_matrix_indices(self,ntraf):
        """ This function gives the indices that can be used in the lon/lat-vectors """
        # The indices will be n*(n-1)/2 long
        # Only works for n >= 2, which is logical...
        # This is faster than np.triu_indices :)
        tmp_range = np.arange(ntraf - 1, dtype=np.int32)
        ind1 = np.repeat(tmp_range,(tmp_range + 1)[::-1])
        ind2 = np.ones(ind1.shape[0], dtype=np.int32)
        inds = np.cumsum(tmp_range[1:][::-1] + 1)
        np.put(ind2, inds, np.arange(ntraf * -1 + 3, 1))
        ind2 = np.cumsum(ind2, out=ind2)
        return ind1, ind2
