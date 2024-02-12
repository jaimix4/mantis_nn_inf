import numpy as np
import pandas as pd
from CRM_ADAS import Deuterium
from scipy.stats import qmc
import subprocess
import tempfile
import os 

class data_gen:

    def __init__(self, Te_range, ne_range, no_range, pressure_limit, num_samples, Te_range_scale = 'log', Irate_limit = [5e20, 1e25], Rrate_limit = [1e19, 1e30], Brec3_limit = [0.03, 0.95]):

        # Define CRM_ADAS object

        self.pd_created = False

        self.deu_crm = Deuterium()
        
        # add 
        self.lower_bounds = np.array([Te_range[0], ne_range[0], no_range[0]])
        self.upper_bounds = np.array([Te_range[-1], ne_range[-1], no_range[-1]])

        # for log scale on Te, ne, no 
        # self.lower_bounds_log = self.lower_bounds.copy() # np.log10(self.lower_bounds)
        # self.upper_bounds_log = self.upper_bounds.copy() #np.log10(self.upper_bounds)

        self.lower_bounds_log = np.log10(self.lower_bounds)
        self.upper_bounds_log = np.log10(self.upper_bounds)

        # for Te, log ne, no
        # self.lower_bounds_log[1:] = np.log10(self.lower_bounds[1:])
        # self.upper_bounds_log[1:] = np.log10(self.upper_bounds[1:])

        # self.lower_bounds_log[1:] = np.log10(self.lower_bounds[1:])
        # self.upper_bounds_log[1:] = np.log10(self.upper_bounds[1:])

        #print(self.lower_bounds_log)

        # Latin hypercube sampler 

        sampler = qmc.LatinHypercube(d=3, strength=1)
        sample_unit_space = sampler.random(n = int(num_samples))

        samples_log10 = qmc.scale(sample_unit_space, self.lower_bounds_log, self.upper_bounds_log)

        #print(samples_log10.shape)
        self.samples_real_space = samples_log10.copy()

        # rescaling to real space

        #self.samples_real_space[:,:] = np.power(np.ones(self.samples_real_space[:,:].shape)*10, self.samples_real_space[:,:])
       # self.samples_real_space[:, 1:] = np.power(np.ones(self.samples_real_space[:, 1:].shape)*10, self.samples_real_space[:, 1:])
        
        self.samples_real_space = np.power(np.ones(self.samples_real_space.shape)*10, self.samples_real_space)

        # print(self.samples_real_space)
       
        # Delete values that are not in the pressure range
        # pressure limit of 2*Te*ne

        print('number of samples before applying pressure limit: {}'.format(self.samples_real_space.shape[0]))

        # self.samples_real_space = self.samples_real_space[self.samples_real_space[:,0]*self.samples_real_space[:,1]*2 < pressure_limit]

        print('number of samples after applying pressure limit: {}'.format(self.samples_real_space.shape[0]))

        # Generate pandas dataframe
        # 'emi_3-2','emi_4-2','emi_5-2','emi_7-2','Te','ne','no',' nHe ',' nHe+', 'Irate','Rrate','CXrate','Pexc','Prec', '728/668', '706/668', 'He728','He706', 'He668', 'Brec3/B3'
        # '   0   ','   1   ','   2   ','   3   ','4 ',' 5','6 ','  7  ','  8  ',  '  9 ','  10 ',' 11   ',' 12 ','  13 ','  14    ','  15   ','  16  ','  17 ','  18   ','  19   '    
        param_num = 20

        self.data = np.zeros((self.samples_real_space.shape[0], param_num))

        # computing full data from CRM_ADAS module
        for idx, Te_ne_no in enumerate(self.samples_real_space):

            # Te, ne, no
            self.data[idx, 4:7] = Te_ne_no

            # nHe
            self.data[idx, 7] = (np.exp(np.random.uniform(np.log(0.01), np.log(50)))/100) * Te_ne_no[2]

            # rates 
            self.data[idx, 9:14] = self.deu_crm.compute_rates(Te_ne_no)

            # emissivities
            self.data[idx, :4], self.data[idx, -1] = self.deu_crm.compute_emissivites_ratio_B3rec(Te_ne_no)

            # sum of recombination and ionization rates
            # gone, stupid idea
            #self.data[idx, -1] = self.data[idx, 8] + self.data[idx, 9]


        # Applying the Irate limit and Rrate limit
        # Irate limit: 1e18 - 1e25
        # Rrate limit: 1e17 - ...

        Irate_lower_limit = Irate_limit[0]
        Irate_upper_limit = Irate_limit[-1]
        Rrate_lower_limit = Rrate_limit[0]

        # B3rec limit
        B3rec_lower_limit = Brec3_limit[0]
        B3rec_upper_limit = Brec3_limit[-1]


        # index 8 is Irate and index 9 is Rrate
        # take in consideration that for large number of samples
        # the total number of samples after cleansing with at least 
        # this specific Irate and Rrate is 50% to 60% of the total number of samples
        # inputed in the beginning
        #self.data = self.data[[a and b and c for a, b, c in zip(self.data[:,8] >= Irate_lower_limit, self.data[:,8] <= Irate_upper_limit, self.data[:,9] >= Rrate_lower_limit)]]
        
        # B3rec limit
        # self.data = self.data[[a and b for a, b in \
        #     zip(self.data[:,-2] >= B3rec_lower_limit, \
        #         self.data[:,-2] <= B3rec_upper_limit)]]

        self.data = self.data[[a and b and c and d for a, b, c, d in \
            zip(self.data[:,-1] >= B3rec_lower_limit, \
                self.data[:,-1] <= B3rec_upper_limit, \
                    self.data[:,10] >= Rrate_lower_limit, \
                        self.data[:,9] >= Irate_lower_limit)]]

        
        # Putting all values of Irate < 1e19 to 1e19
        # adjusting values of Irate 
        #self.data[:,8][self.data[:,-2] < 0.95] = 1e16
        
        # neutral density 
        # self.data[:,6][self.data[:,8] < 1e19] = 1e21
        # self.data[:,8][self.data[:,8] < 1e19] = 1e19

        # self.data[:,9][self.data[:,9] < 1e19] = 1e17

        print('\n number of samples after applying Irate (rec fraction), Rrate limits: {}'.format(self.data.shape[0]))

        # FOR BIG DATASETS THIS CAN BE CHANGED TO PARQUET FILES
        # BUT I HAVE NO IDEA HOW C CAN READ PARQUET FILES SO... 

        # introducing He ratios using the c file with the goto model
        # nam_tempfile_input = 'temp_cfile_absHe.csv'
        # nam_tempfile_output = 'temp_cfile_out_absHe.csv'
        # np.savetxt(nam_tempfile_input, self.data[:, 4:6], delimiter=',')

        fd, temp_input_path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)  # Close the file descriptor returned by mkstemp
        np.savetxt(temp_input_path, self.data[:, 4:6], delimiter=',')

        fd, temp_output_path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)  # Close the file descriptor returned by mkstemp

        # with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_input:
        #     np.savetxt(temp_input, self.data[:, 4:6], delimiter=',')
        #     temp_input_path = temp_input.name  # Save the path for later use

        # with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_output:
        #     temp_output_path = temp_output.name  # Save the path for later use

        print('\nOutput of the c file: ')
        subprocess.run(['./emissions_jaime', temp_input_path, temp_output_path])
        print('\n c file for He ratios run done. Name of temporary file for data: {} \n this file can be ignored or deleted'.format(temp_output_path))

        arr = np.loadtxt(temp_output_path, delimiter=",", dtype=np.float64)

        # nHe+\
        log_arr = np.random.uniform(low=np.log10(1e16), high=np.log10(1e18), size=self.data[:, 8].shape[0])
        self.data[:, 8] = np.power(10, log_arr) #np.random.uniform(low=1e16, high=1e18, size=self.data[:, 8].shape[0]) #(np.exp(np.random.uniform(np.log(0.01), np.log(50)))/100) * self.data[:, 6]

        # He728
        self.data[:, 16] = (self.data[:,5] * self.data[:,7] * arr[:, 2]) + (self.data[:,5] * self.data[:,8] * arr[:, 5])
        # He706
        self.data[:, 17] = (self.data[:,5] * self.data[:,7] * arr[:, 3]) + (self.data[:,5] * self.data[:,8] * arr[:, 6])
        # He668
        self.data[:, 18] = (self.data[:,5] * self.data[:,7] * arr[:, 4]) + (self.data[:,5] * self.data[:,8] * arr[:, 7])

        # 728/706 
        self.data[:, 14] = self.data[:, 16] / self.data[:, 17] #arr[:, 2]/arr[:, 3]
        # 728/668
        self.data[:, 15] = self.data[:, 16] / self.data[:, 18] 

        # remove temproary files
        os.remove(temp_input_path)
        os.remove(temp_output_path)

    def get_data_pd(self):

        self.df_data = pd.DataFrame(self.data, columns=['emi_3-2', 'emi_4-2', 'emi_5-2', 'emi_7-2', 'Te', 'ne', 'no', 'nHe', 'nHe+', 'Irate', 'Rrate', 'CXrate', 'Pexc', 'Prec', '728/706', '728/668', 'He728', 'He706', 'He668', 'Brec3/B3'])
        self.pd_created = True
        return self.df_data

    def save_data_pd(self, filename, filetype = 'parquet'):

        if self.pd_created == False:
            self.get_data_pd()
        if filetype == 'parquet':
            #self.df_data.to_parquet(filename, index=False)
            # save a compress version of parquet file 
            self.df_data.to_parquet(filename, index=False, compression='gzip')
        elif filetype == 'csv':
            self.df_data.to_csv(filename, index=False)


# Te_range = [0.200001, 80.0] # eV
# ne_range = [1e18, 2.5e20] # m^-3
# no_range = [1e15, 1e20] # m^-3

# pressure_limit = 3e21 # eV * m^-3 -> 2*Te*ne

# num_samples = 12 

# data = data_gen(Te_range, ne_range, no_range, pressure_limit, num_samples)

# df = data.get_data_pd()

# print(df.head())

# data.save_data_pd('raw_datasets/new_data_{}.csv'.format(num_samples))


