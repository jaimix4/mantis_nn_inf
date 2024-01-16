# Author: Jaime Caballero
# 09-09-2022

import numpy as np
from cherab.core.atomic import deuterium, helium
from cherab.openadas import OpenADAS

adas = OpenADAS()

hplanck = 6.62607015e-34 # m2kg/s
cspeed = 2.99792458e8    # m/s

# Here I need to add another information that might be usefull
# Also a read me file explaining the order, units and description 
# of all the parameters output by this, given a certain order

class Deuterium:

    def __init__(self):
        
        #####

        # DEFINE RATES

        self.SCD = adas.ionisation_rate(deuterium, 0)
        self.ADC = adas.recombination_rate(deuterium, 1)
        self.CCD = adas.thermal_cx_rate(deuterium, 0, deuterium, 1) # here it is just from 0 to 1, or from  0 to 1 and 1 to 0 or it is the same?

        self.PLT = adas.line_radiated_power_rate(deuterium, 0)
        self.PRB = adas.continuum_radiated_power_rate(deuterium, 1)

        # DEFINE PECs as List
        # Maybe this is better with a dictionary - CHECK
        # definetely make a dictionary instead of this

        self.PECexc = [adas.impact_excitation_pec(deuterium, 0, (3, 2)), adas.impact_excitation_pec(deuterium, 0, (4, 2)), adas.impact_excitation_pec(deuterium, 0, (5, 2)), adas.impact_excitation_pec(deuterium, 0, (7, 2))]
        self.PECrec = [adas.recombination_pec(deuterium, 0, (3, 2)), adas.recombination_pec(deuterium, 0, (4, 2)), adas.recombination_pec(deuterium, 0, (5, 2)), adas.recombination_pec(deuterium, 0, (7, 2))]

        # DEFINE wavelengths as list as well
        # Maybe this is better with a dictionary - CHECK

        self.PEC_wavelengths = [adas.wavelength(deuterium, 0, (3,2)), adas.wavelength(deuterium, 0, (4,2)), adas.wavelength(deuterium, 0, (5,2)), adas.wavelength(deuterium, 0, (7,2)) ]


    # Function for computing the rate (outputs) for a given combination of $T_e$, $n_e$, $n_o$

    def compute_rates(self, Te_ne_no):
        
        # INPUT: just an array with Te, ne, no in that order
        #specify units!!! TO DO
        
        # OUTPUT: array with Irate, Rrate, CXrate, Pexc, Prec in that order
        #specify units!!! TO DO
        

        Te = Te_ne_no[0]
        ne = Te_ne_no[1]
        no = Te_ne_no[2]
        
        # Computing the rates
        # SCD, ADC, CCD in particles.m^3/s
        # Irate, Rrate, CXrate in particles/m^3*s    
        
        Irate  =  ne * no * self.SCD(ne,Te)
        
        Rrate  =  ne**2 * self.ADC(ne,Te)
        
        CXrate =  ne * no * self.CCD(ne,Te)

        # PLT, PRB in W.m^3
        # Pexc, Prec in W/m^3 

        Pexc   =  ne * no * self.PLT(ne,Te)
        
        Prec   =  ne**2 * self.PRB(ne,Te)
        
        return np.array([Irate, Rrate, CXrate, Pexc, Prec])


    # this function takes a combinatio of Te_ne_no, and computes the total brightness
    # of the selected input balmer lines 


    def compute_emissivites(self, Te_ne_no, balmer_lines = [3,4,5,7]):
        
        # INPUT: - array with Te, ne, no in that order
        #        - array with transitions, list high energy state 
        #          example:
        #                  if 3->2, 4->2 and 5->2 wanted, then
        #                  the input array should be [3,4,5] 
        #specify units!!! TO DO
        
        # OUTPUT: - array with total brightness for each input line
        #specify units!!! TO DO
        
        Te = Te_ne_no[0]
        ne = Te_ne_no[1]
        no = Te_ne_no[2]
        
        emi_output = np.zeros(len(balmer_lines))
        
        for idx, upper in enumerate(balmer_lines):
            
            #extracting PEC coefficients
            
            #PECexc = adas.impact_excitation_pec(deuterium, 0, (upper, 2))
            #PECrec = adas.recombination_pec(deuterium, 0, (upper, 2))
        
        
            #total brightness
            # I think this is in Watts/m3, it might be better to work with photons
            # 
            Br = ne**2 * self.PECrec[idx](ne,Te) + ne*no * self.PECexc[idx](ne,Te)
            
            # with this get the energy and compute photons
            #
            wavelen = self.PEC_wavelengths[idx]
            E = (hplanck*cspeed)/(wavelen*1e-9) # energy of photon for this balmer line
            
            # emissivity in #ph/s*m3
            emi_output[idx] = Br/E
        
        
        return emi_output

    def compute_emissivites_ratios(self, Te_ne_no, balmer_lines = [3,4,5,7], ratio = [5, 3]):
        
        # INPUT: - array with Te, ne, no in that order
        #        - array with transitions, list high energy state 
        #          example:
        #                  if 3->2, 4->2 and 5->2 wanted, then
        #                  the input array should be [3,4,5] 
        #specify units!!! TO DO
        
        # OUTPUT: - array with total brightness for each input line
        #specify units!!! TO DO
        
        Te = Te_ne_no[0]
        ne = Te_ne_no[1]
        no = Te_ne_no[2]

        
        emi_output = np.zeros(len(balmer_lines))
        
        for idx, upper in enumerate(balmer_lines):
            
            #extracting PEC coefficients
            
            #PECexc = adas.impact_excitation_pec(deuterium, 0, (upper, 2))
            #PECrec = adas.recombination_pec(deuterium, 0, (upper, 2))
        
        
            #total brightness
            # I think this is in Watts/m3, it might be better to work with photons
            # 
            Br = ne**2 * self.PECrec[idx](ne,Te) + ne*no * self.PECexc[idx](ne,Te)
            Br_rec = ne**2 * self.PECrec[idx](ne,Te)
            # with this get the energy and compute photons
            #
            wavelen = self.PEC_wavelengths[idx]
            E = (hplanck*cspeed)/(wavelen*1e-9) # energy of photon for this balmer line
            
            # emissivity in #ph/s*m3
            emi_output[idx] = Br/E
            Br_rec = Br_rec/E

            if upper == ratio[0]:
                ratio_first = Br_rec/emi_output[idx]
            elif upper == ratio[1]:
                ratio_second = Br_rec/emi_output[idx]

        ratio_first_second = emi_output[balmer_lines.index(ratio[0])]/emi_output[balmer_lines.index(ratio[1])]
        # print(balmer_lines.index(ratio[0]))
        # print('hh')
        # print(balmer_lines.index(ratio[1]))
        
        return emi_output, ratio_first, ratio_second, ratio_first_second


    def compute_emissivites_ratio_B3rec(self, Te_ne_no, balmer_lines = [3,4,5,7], ratio = [5, 3]):
        
        # INPUT: - array with Te, ne, no in that order
        #        - array with transitions, list high energy state 
        #          example:
        #                  if 3->2, 4->2 and 5->2 wanted, then
        #                  the input array should be [3,4,5] 
        #specify units!!! TO DO
        
        # OUTPUT: - array with total brightness for each input line
        #specify units!!! TO DO
        
        Te = Te_ne_no[0]
        ne = Te_ne_no[1]
        no = Te_ne_no[2]

        
        emi_output = np.zeros(len(balmer_lines))
        
        for idx, upper in enumerate(balmer_lines):
            
            #extracting PEC coefficients
            
            #PECexc = adas.impact_excitation_pec(deuterium, 0, (upper, 2))
            #PECrec = adas.recombination_pec(deuterium, 0, (upper, 2))
        
        
            #total brightness
            # I think this is in Watts/m3, it might be better to work with photons
            # 
            Br_rec = ne**2 * self.PECrec[idx](ne,Te)
            Br = Br_rec + ne*no * self.PECexc[idx](ne,Te)
            
            # with this get the energy and compute photons
            #
            wavelen = self.PEC_wavelengths[idx]
            E = (hplanck*cspeed)/(wavelen*1e-9) # energy of photon for this balmer line
            
            # emissivity in #ph/s*m3
            emi_output[idx] = Br/E
            Br_rec = Br_rec/E

            if upper == 3:
                ratio_B3rec = Br_rec/emi_output[idx]

        # print(balmer_lines.index(ratio[0]))
        # print('hh')
        # print(balmer_lines.index(ratio[1]))
        
        return emi_output, ratio_B3rec


        # HELIUM CLASS TO BE DONE, SAME AS DEUTERIUM