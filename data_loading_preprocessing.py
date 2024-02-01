import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
import yaml
import pyarrow.parquet as pq
from pathlib import Path

from tensorflow.keras.utils import Sequence


def verify_yaml(yalmfile_path):

    with open(yalmfile_path, "r") as yamlfile:
        data_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read successful")

    if data_config.get('inputs_prepro').get('mean') == None or data_config.get('inputs_prepro').get('scale') == None or \
        data_config.get('outputs_prepro').get('min') == None or data_config.get('outputs_prepro').get('scale') == None or \
            data_config.get('dataset').get('x_train_len') == None or data_config.get('inputs_labels').get('index') == None: # or data_config['inputs_prepro']['scale'] == None or data_config['outputs_prepro']['min'] == None or data_config['outputs_prepro']['scale'] == None:

        print('No full preprocessing, appending it')

        inputs_labels = data_config['inputs_labels']['value']
        inputs_prepro = data_config['inputs_prepro']['value']

        outputs_labels = data_config['outputs_labels']['value']
        outputs_prepro = data_config['outputs_prepro']['value']

        data_to_dump = get_scalers(inputs_labels, inputs_prepro, StandardScaler(), outputs_labels, outputs_prepro,\
                     MinMaxScaler(feature_range=data_config['min_max_scale']['value']), data_folder=data_config['dataset']['value'])

        data_config['inputs_prepro']['mean'] = [float(i) for i in data_to_dump[0]]
        data_config['inputs_prepro']['scale'] = [float(i) for i in data_to_dump[1]]
        data_config['outputs_prepro']['min'] = [float(i) for i in data_to_dump[2]]
        data_config['outputs_prepro']['scale'] = [float(i) for i in data_to_dump[3]]
        data_config['dataset']['x_train_len'] = data_to_dump[4]

        index_inputs_list = []

        for lbl in data_config.get('inputs_labels').get('value'):

            if lbl == 'emi_3-2':

                index_inputs_list.append(0)

            elif lbl == 'emi_4-2':

                index_inputs_list.append(1)

            elif lbl == 'emi_5-2':

                index_inputs_list.append(2)

            elif lbl == 'emi_7-2':

                index_inputs_list.append(3)

            elif lbl == '728/668':

                index_inputs_list.append(4)

            elif lbl == '728/706':

                index_inputs_list.append(5)

            # elif lbl == 'Te':
            #
            #     index_inputs_list.append('Te')
            #
            # elif lbl == 'ne':
            #
            #     index_inputs_list.append('ne')


        data_config['inputs_labels']['index'] = index_inputs_list


        with open(yalmfile_path,'w') as out_yamlfile:
            yaml.safe_dump(data_config, out_yamlfile)


    else:

        print('full preprocessing preset')

    return data_config

def grab_data_preprocessing(inputs_labels, inputs_prepro, input_scaler, outputs_labels, outputs_prepro, output_scaler, data_folder = 'dataset_exp4/adasH_gotoHe_crm_LHS_IRrate_'):


    # inputs/outputs_normalization its a list with the type of preproccessing
    # 0 -> log10
    # 1 -> non
    # example - inputs = 'emi_3-2', 'emi_4-2', 'emi_5-2'
    #         - inputs_normalization = 0, 0, 0


    # grab data from csv
    if Path(data_folder + 'TRAIN.parquet').is_file():

        data_set_train = pq.read_table(data_folder + 'TRAIN.parquet').to_pandas()
        data_set_val = pq.read_table(data_folder + 'VAL.parquet').to_pandas()
        data_set_test = pq.read_table(data_folder + 'TEST.parquet').to_pandas()

        data_set_train.to_parquet(data_folder + 'TRAIN.parquet', index=False, compression='gzip')
        data_set_val.to_parquet(data_folder + 'VAL.parquet', index=False, compression='gzip')
        data_set_test.to_parquet(data_folder + 'TEST.parquet', index=False, compression='gzip')

        print("----CREATED PARQUET FILES, to replace csv size and slowness of files-----")

    else:

        data_set_train = pd.read_csv(data_folder + 'TRAIN.csv')
        data_set_val = pd.read_csv(data_folder + 'VAL.csv')
        data_set_test = pd.read_csv(data_folder + 'TEST.csv')


        data_set_train.to_parquet(data_folder + 'TRAIN.parquet', index=False, compression='gzip')
        data_set_val.to_parquet(data_folder + 'VAL.parquet', index=False, compression='gzip')
        data_set_test.to_parquet(data_folder + 'TEST.parquet', index=False, compression='gzip')

        print("----CREATED PARQUET FILES, to replace csv size and slowness of files-----")


    data_set_train = pd.read_csv(data_folder + 'TRAIN.csv')

    data_set_val = pd.read_csv(data_folder + 'VAL.csv')

    data_set_test = pd.read_csv(data_folder + 'TEST.csv')


    # # shuffle data
    # data_shuffled = data_set_all.sample(frac=0.10)

    # assign all inputs to X_full
    X_train = data_set_train[inputs_labels].values
    X_val = data_set_val[inputs_labels].values
    X_test = data_set_test[inputs_labels].values

    # assign all outputs to y_full
    y_train = data_set_train[outputs_labels].values
    y_val = data_set_val[outputs_labels].values
    y_test = data_set_test[outputs_labels].values



    # preprocessing of inputs
    for col, treatment in enumerate(inputs_prepro):

        if treatment == 'log10':

            X_train[:, col] = np.log10(X_train[:, col])
            X_val[:, col] = np.log10(X_val[:, col])
            X_test[:, col] = np.log10(X_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            X_train[:, col] = np.power(X_train[:, col], 1/treatment)
            X_val[:, col] = np.power(X_val[:, col], 1/treatment)
            X_test[:, col] = np.power(X_test[:, col], 1/treatment)

        else:

            print("No scaling apply to input data")

            pass

    # preprocessing of outputs
    for col, treatment in enumerate(outputs_prepro):

        if treatment == 'log10':

            y_train[:, col] = np.log10(y_train[:, col])
            y_val[:, col] = np.log10(y_val[:, col])
            y_test[:, col] = np.log10(y_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            y_train[:, col] = np.power(y_train[:, col], 1/treatment)
            y_val[:, col] = np.power(y_val[:, col], 1/treatment)
            y_test[:, col] = np.power(y_test[:, col], 1/treatment)

        else:

            print("No scaling apply to output data")

            pass

    # perform scaling
    # maybe just scaled on X_train ??????????

    input_scaler.fit(np.vstack((X_train, X_val, X_test)))


    output_scaler.fit(np.vstack((y_train, y_val, y_test)))


    X_train = input_scaler.transform(X_train)
    X_val = input_scaler.transform(X_val)
    X_test = input_scaler.transform(X_test)

    y_train = output_scaler.transform(y_train)
    y_val = output_scaler.transform(y_val)
    y_test = output_scaler.transform(y_test)



    return X_train, y_train, X_val, y_val, X_test, y_test, input_scaler.transform, input_scaler.inverse_transform, output_scaler.transform, output_scaler.inverse_transform

######################################
# function to load data for training #
######################################

def grab_data_preprocessing_2(inputs_labels, inputs_prepro, input_scaler, outputs_labels, outputs_prepro, output_scaler, data_folder = 'dataset_exp4/adasH_gotoHe_crm_LHS_IRrate_'):


    # inputs/outputs_normalization its a list with the type of preproccessing
    # 0 -> log10
    # 1 -> non
    # example - inputs = 'emi_3-2', 'emi_4-2', 'emi_5-2'
    #         - inputs_normalization = 0, 0, 0
    

    if Path(data_folder + 'TRAIN.parquet').is_file():

        data_set_train = pq.read_table(data_folder + 'TRAIN.parquet').to_pandas()
        data_set_val = pq.read_table(data_folder + 'VAL.parquet').to_pandas()
        data_set_test = pq.read_table(data_folder + 'TEST.parquet').to_pandas()

        print("----PARQUET FILES FOUND-----")

    else:

        data_set_train = pd.read_csv(data_folder + 'TRAIN.csv')
        data_set_val = pd.read_csv(data_folder + 'VAL.csv')
        data_set_test = pd.read_csv(data_folder + 'TEST.csv')


        data_set_train.to_parquet(data_folder + 'TRAIN.parquet', index=False, compression='gzip')
        data_set_val.to_parquet(data_folder + 'VAL.parquet', index=False, compression='gzip')
        data_set_test.to_parquet(data_folder + 'TEST.parquet', index=False, compression='gzip')

        print("----CREATED PARQUET FILES, to replace csv size and slowness of files-----")


    # # shuffle data
    # data_shuffled = data_set_all.sample(frac=0.10)

    # assign all inputs to X_full
    X_train = data_set_train[inputs_labels].values
    X_val = data_set_val[inputs_labels].values
    X_test = data_set_test[inputs_labels].values

    # assign all outputs to y_full
    y_train = data_set_train[outputs_labels].values
    y_val = data_set_val[outputs_labels].values
    y_test = data_set_test[outputs_labels].values


    X_train_real = X_train.copy()
    y_train_real = y_train.copy()


    # preprocessing of inputs
    for col, treatment in enumerate(inputs_prepro):

        if treatment == 'log10':

            X_train[:, col] = np.log10(X_train[:, col])
            X_val[:, col] = np.log10(X_val[:, col])
            X_test[:, col] = np.log10(X_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            X_train[:, col] = np.power(X_train[:, col], 1/treatment)
            X_val[:, col] = np.power(X_val[:, col], 1/treatment)
            X_test[:, col] = np.power(X_test[:, col], 1/treatment)

        else:

            print("No scaling apply to input data")

            pass

    # preprocessing of outputs
    for col, treatment in enumerate(outputs_prepro):

        if treatment == 'log10':

            y_train[:, col] = np.log10(y_train[:, col])
            y_val[:, col] = np.log10(y_val[:, col])
            y_test[:, col] = np.log10(y_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            y_train[:, col] = np.power(y_train[:, col], 1/treatment)
            y_val[:, col] = np.power(y_val[:, col], 1/treatment)
            y_test[:, col] = np.power(y_test[:, col], 1/treatment)

        else:

            print("No scaling apply to output data")

            pass

    # perform scaling
    # maybe just scaled on X_train ??????????

    input_scaler.fit(np.vstack((X_train, X_val, X_test)))

    # forcing the minimum onn output

    # y_something = np.vstack((y_train, y_val, y_test))

    # y_something[:,3][y_something[:, -1] >= 0.95] = np.nan
     
    # fake_arr = np.zeros((2, 6))

    # fake_arr[0, :] = np.nanmin(y_something, axis = 0) # min Te
    # fake_arr[1, :] = np.nanmax(y_something, axis = 0) # min Te


    # print(fake_arr)

    output_scaler.fit(np.vstack((y_train, y_val, y_test)))

    X_train = input_scaler.transform(X_train)
    X_val = input_scaler.transform(X_val)
    X_test = input_scaler.transform(X_test)

    y_train = output_scaler.transform(y_train)
    y_val = output_scaler.transform(y_val)
    y_test = output_scaler.transform(y_test)



    return X_train_real, y_train_real, X_train, y_train, X_val, y_val, X_test, y_test, input_scaler.transform, input_scaler.inverse_transform, output_scaler.transform, output_scaler.inverse_transform

def grab_data_preprocessing_crazy_nn(inputs_labels, inputs_prepro, input_scaler, outputs_labels, outputs_prepro, output_scaler, \
    data_folder = 'dataset_exp4/adasH_gotoHe_crm_LHS_IRrate_', rec3_class = 0.95):


    # inputs/outputs_normalization its a list with the type of preproccessing
    # 0 -> log10
    # 1 -> non
    # example - inputs = 'emi_3-2', 'emi_4-2', 'emi_5-2'
    #         - inputs_normalization = 0, 0, 0
    

    if Path(data_folder + 'TRAIN.parquet').is_file():

        data_set_train = pq.read_table(data_folder + 'TRAIN.parquet').to_pandas()
        data_set_val = pq.read_table(data_folder + 'VAL.parquet').to_pandas()
        data_set_test = pq.read_table(data_folder + 'TEST.parquet').to_pandas()

        print("----PARQUET FILES FOUND-----")

    else:

        data_set_train = pd.read_csv(data_folder + 'TRAIN.csv')
        data_set_val = pd.read_csv(data_folder + 'VAL.csv')
        data_set_test = pd.read_csv(data_folder + 'TEST.csv')


        data_set_train.to_parquet(data_folder + 'TRAIN.parquet', index=False, compression='gzip')
        data_set_val.to_parquet(data_folder + 'VAL.parquet', index=False, compression='gzip')
        data_set_test.to_parquet(data_folder + 'TEST.parquet', index=False, compression='gzip')

        print("----CREATED PARQUET FILES, to replace csv size and slowness of files-----")


    # # shuffle data
    # data_shuffled = data_set_all.sample(frac=0.10)

    # assign all inputs to X_full
    X_train = data_set_train[inputs_labels].values
    X_val = data_set_val[inputs_labels].values
    X_test = data_set_test[inputs_labels].values

    # assign all outputs to y_full
    y_train = data_set_train[outputs_labels].values
    y_val = data_set_val[outputs_labels].values
    y_test = data_set_test[outputs_labels].values


    X_train_real = X_train.copy()
    y_train_real = y_train.copy()


    # preprocessing of inputs
    for col, treatment in enumerate(inputs_prepro):

        if treatment == 'log10':

            X_train[:, col] = np.log10(X_train[:, col])
            X_val[:, col] = np.log10(X_val[:, col])
            X_test[:, col] = np.log10(X_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            X_train[:, col] = np.power(X_train[:, col], 1/treatment)
            X_val[:, col] = np.power(X_val[:, col], 1/treatment)
            X_test[:, col] = np.power(X_test[:, col], 1/treatment)

        else:

            print("No scaling apply to input data")

            pass

    # preprocessing of outputs
    for col, treatment in enumerate(outputs_prepro):

        if treatment == 'log10':

            y_train[:, col] = np.log10(y_train[:, col])
            y_val[:, col] = np.log10(y_val[:, col])
            y_test[:, col] = np.log10(y_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            y_train[:, col] = np.power(y_train[:, col], 1/treatment)
            y_val[:, col] = np.power(y_val[:, col], 1/treatment)
            y_test[:, col] = np.power(y_test[:, col], 1/treatment)

        else:

            print("No scaling apply to output data")

            pass

    # perform scaling
    # maybe just scaled on X_train ??????????

    input_scaler.fit(np.vstack((X_train, X_val, X_test)))

    # forcing the minimum onn output

    y_something = np.vstack((y_train, y_val, y_test))

    # forcing minimum of Irate to minimum on D3rec regime -> {0, 0.95}
    y_something[:,3][y_something[:, -1] >= rec3_class] = np.nan
    #  forcing maximum of Rate to maximum on D3rec regime -> {0, 0.95}
    y_something[:,4][y_something[:, -1] >= rec3_class] = np.nan

    # here you can also just for a value, but lets keep it like this 
    # until dataset is fully explored 
     
    fake_arr = np.zeros((2, 6))

    fake_arr[0, :] = np.nanmin(y_something, axis = 0) # min Te
    fake_arr[1, :] = np.nanmax(y_something, axis = 0) # min Te


    print(fake_arr)

    output_scaler.fit(fake_arr)


    X_train = input_scaler.transform(X_train)
    X_val = input_scaler.transform(X_val)
    X_test = input_scaler.transform(X_test)

    y_train = output_scaler.transform(y_train)
    y_val = output_scaler.transform(y_val)
    y_test = output_scaler.transform(y_test)



    return X_train_real, y_train_real, X_train, y_train, X_val, y_val, X_test, y_test, input_scaler.transform, input_scaler.inverse_transform, output_scaler.transform, output_scaler.inverse_transform

######################################
# function to load data for training #
######################################

def get_scalers_crazy_nn(inputs_labels, inputs_prepro, input_scaler, outputs_labels, outputs_prepro, output_scaler, \
    data_folder = 'dataset_exp4/adasH_gotoHe_crm_LHS_IRrate_', rec3_class = 0.95):


    # inputs/outputs_normalization its a list with the type of preproccessing
    # 0 -> log10
    # 1 -> non
    # example - inputs = 'emi_3-2', 'emi_4-2', 'emi_5-2'
    #         - inputs_normalization = 0, 0, 0


    # grab data from csv
    # data_set_train = pd.read_csv(data_folder + 'TRAIN.CSV')

    # data_set_val = pd.read_csv(data_folder + 'VAL.CSV')

    # data_set_test = pd.read_csv(data_folder + 'TEST.CSV')

    if Path(data_folder + 'TRAIN.parquet').is_file():

        data_set_train = pq.read_table(data_folder + 'TRAIN.parquet').to_pandas()
        data_set_val = pq.read_table(data_folder + 'VAL.parquet').to_pandas()
        data_set_test = pq.read_table(data_folder + 'TEST.parquet').to_pandas()

        print("----PARQUET FILES FOUND-----")

    else:

        data_set_train = pd.read_csv(data_folder + 'TRAIN.csv')
        data_set_val = pd.read_csv(data_folder + 'VAL.csv')
        data_set_test = pd.read_csv(data_folder + 'TEST.csv')


        data_set_train.to_parquet(data_folder + 'TRAIN.parquet', index=False, compression='gzip')
        data_set_val.to_parquet(data_folder + 'VAL.parquet', index=False, compression='gzip')
        data_set_test.to_parquet(data_folder + 'TEST.parquet', index=False, compression='gzip')

        print("----CREATED PARQUET FILES, to replace csv size and slowness of files-----")


    # # shuffle data
    # data_shuffled = data_set_all.sample(frac=0.10)

    # assign all inputs to X_full
    X_train = data_set_train[inputs_labels].values
    X_val = data_set_val[inputs_labels].values
    X_test = data_set_test[inputs_labels].values

    # assign all outputs to y_full
    y_train = data_set_train[outputs_labels].values
    y_val = data_set_val[outputs_labels].values
    y_test = data_set_test[outputs_labels].values



    # preprocessing of inputs
    for col, treatment in enumerate(inputs_prepro):

        if treatment == 'log10':

            X_train[:, col] = np.log10(X_train[:, col])
            X_val[:, col] = np.log10(X_val[:, col])
            X_test[:, col] = np.log10(X_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            X_train[:, col] = np.power(X_train[:, col], 1/treatment)
            X_val[:, col] = np.power(X_val[:, col], 1/treatment)
            X_test[:, col] = np.power(X_test[:, col], 1/treatment)

        else:

            print("No scaling apply to input data")

            pass

    # preprocessing of outputs
    for col, treatment in enumerate(outputs_prepro):

        if treatment == 'log10':

            y_train[:, col] = np.log10(y_train[:, col])
            y_val[:, col] = np.log10(y_val[:, col])
            y_test[:, col] = np.log10(y_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            y_train[:, col] = np.power(y_train[:, col], 1/treatment)
            y_val[:, col] = np.power(y_val[:, col], 1/treatment)
            y_test[:, col] = np.power(y_test[:, col], 1/treatment)

        else:

            print("No scaling apply to output data")

            pass

    # perform scaling
    # maybe just scaled on X_train ??????????

    input_scaler.fit(np.vstack((X_train, X_val, X_test)))

    # output_scaler.fit(np.vstack((y_train, y_val, y_test)))

    y_something = np.vstack((y_train, y_val, y_test))

    # forcing minimum of Irate to minimum on D3rec regime -> {0, 0.95}
    y_something[:,3][y_something[:, -1] >= rec3_class] = np.nan
    # forcing maximum of Rate to maximum on D3rec regime -> {0, 0.95}
    y_something[:,4][y_something[:, -1] >= rec3_class] = np.nan

    # here you can also just for a value, but lets keep it like this 
    # until dataset is fully explored 
     
    fake_arr = np.zeros((2, 6))

    fake_arr[0, :] = np.nanmin(y_something, axis = 0) # min Te
    fake_arr[1, :] = np.nanmax(y_something, axis = 0) # min Te

    print(fake_arr)

    output_scaler.fit(fake_arr)

    return list(input_scaler.mean_), list(input_scaler.scale_), list(output_scaler.data_min_), list(output_scaler.scale_), X_train.shape[0]


def get_scalers(inputs_labels, inputs_prepro, input_scaler, outputs_labels, outputs_prepro, output_scaler, data_folder = 'dataset_exp4/adasH_gotoHe_crm_LHS_IRrate_'):


    # inputs/outputs_normalization its a list with the type of preproccessing
    # 0 -> log10
    # 1 -> non
    # example - inputs = 'emi_3-2', 'emi_4-2', 'emi_5-2'
    #         - inputs_normalization = 0, 0, 0


    # grab data from csv
    # data_set_train = pd.read_csv(data_folder + 'TRAIN.CSV')

    # data_set_val = pd.read_csv(data_folder + 'VAL.CSV')

    # data_set_test = pd.read_csv(data_folder + 'TEST.CSV')

    if Path(data_folder + 'TRAIN.parquet').is_file():

        data_set_train = pq.read_table(data_folder + 'TRAIN.parquet').to_pandas()
        data_set_val = pq.read_table(data_folder + 'VAL.parquet').to_pandas()
        data_set_test = pq.read_table(data_folder + 'TEST.parquet').to_pandas()

        print("----PARQUET FILES FOUND-----")

    else:

        data_set_train = pd.read_csv(data_folder + 'TRAIN.csv')
        data_set_val = pd.read_csv(data_folder + 'VAL.csv')
        data_set_test = pd.read_csv(data_folder + 'TEST.csv')


        data_set_train.to_parquet(data_folder + 'TRAIN.parquet', index=False, compression='gzip')
        data_set_val.to_parquet(data_folder + 'VAL.parquet', index=False, compression='gzip')
        data_set_test.to_parquet(data_folder + 'TEST.parquet', index=False, compression='gzip')

        print("----CREATED PARQUET FILES, to replace csv size and slowness of files-----")


    # # shuffle data
    # data_shuffled = data_set_all.sample(frac=0.10)

    # assign all inputs to X_full
    X_train = data_set_train[inputs_labels].values
    X_val = data_set_val[inputs_labels].values
    X_test = data_set_test[inputs_labels].values

    # assign all outputs to y_full
    y_train = data_set_train[outputs_labels].values
    y_val = data_set_val[outputs_labels].values
    y_test = data_set_test[outputs_labels].values



    # preprocessing of inputs
    for col, treatment in enumerate(inputs_prepro):

        if treatment == 'log10':

            X_train[:, col] = np.log10(X_train[:, col])
            X_val[:, col] = np.log10(X_val[:, col])
            X_test[:, col] = np.log10(X_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            X_train[:, col] = np.power(X_train[:, col], 1/treatment)
            X_val[:, col] = np.power(X_val[:, col], 1/treatment)
            X_test[:, col] = np.power(X_test[:, col], 1/treatment)

        else:

            print("No scaling apply to input data")

            pass

    # preprocessing of outputs
    for col, treatment in enumerate(outputs_prepro):

        if treatment == 'log10':

            y_train[:, col] = np.log10(y_train[:, col])
            y_val[:, col] = np.log10(y_val[:, col])
            y_test[:, col] = np.log10(y_test[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            y_train[:, col] = np.power(y_train[:, col], 1/treatment)
            y_val[:, col] = np.power(y_val[:, col], 1/treatment)
            y_test[:, col] = np.power(y_test[:, col], 1/treatment)

        else:

            print("No scaling apply to output data")

            pass

    # perform scaling
    # maybe just scaled on X_train ??????????

    input_scaler.fit(np.vstack((X_train, X_val, X_test)))

    output_scaler.fit(np.vstack((y_train, y_val, y_test)))


    return list(input_scaler.mean_), list(input_scaler.scale_), list(output_scaler.data_min_), list(output_scaler.scale_), X_train.shape[0]

#################################################
# function to decode or pass data to real space #
#################################################

def data_decoder(set, preproccesing, inverse_transform):

    if set.ndim == 1 and set.shape[0] == len(preproccesing):

        set = np.reshape(set, (1, set.size))

    decoded = inverse_transform(set)

    non_scale = decoded.copy()

    # preprocessing of inputs
    for col, treatment in enumerate(preproccesing):

        if treatment == 'log10':

            decoded[:, col] = 10**decoded[:, col]

        elif isinstance(treatment, int) or isinstance(treatment, float):

            decoded[:, col] = np.power(decoded[:, col], treatment)

        else:

            pass

    return decoded, non_scale


######################################################
# function to encode or pass data to processed space #
######################################################

def data_encoder(set, preproccesing, transform):

    if set.ndim == 1 and set.shape[0] == len(preproccesing):

        set = np.reshape(set, (1, set.size))


    # preprocessing of inputs
    for col, treatment in enumerate(preproccesing):

        if treatment == 'log10':

            set[:, col] = np.log10(set[:, col])

        elif isinstance(treatment, int) or isinstance(treatment, float):

            set[:, col] = np.power(set[:, col], 1/treatment)

        else:

            pass

    non_scale = set.copy()

    encoded = transform(non_scale.copy())

    return encoded, non_scale

##################################################
# data generator for training, just training set #
##################################################

class MyDataGenerator(Sequence):
    def __init__(self, x, y, batch_size, gau_noise, \
        inputs_prepro, transform_input, outputs_prepro, transform_output):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.gau_noise = gau_noise
        self.indexes = np.arange(len(self.x))

        self.inputs_prepro = inputs_prepro
        self.transform_input = transform_input

        self.outputs_prepro = outputs_prepro
        self.transform_output = transform_output

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        # Shuffle indexes at the end of each epoch
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_x = self.x[self.indexes[start:end]]
        batch_y = self.y[self.indexes[start:end]]
        
        # Apply Gaussian noise to the batch input data
        #noise = np.random.normal(loc=0, scale=self.stddev, size=batch_x.shape)

        batch_x =  \
            data_encoder(batch_x + batch_x * self.gau_noise * np.random.randn(batch_x.shape[0], batch_x.shape[1]), \
                self.inputs_prepro, self.transform_input)[0]

        batch_y =  \
            data_encoder(batch_y, \
                self.outputs_prepro, self.transform_output)[0]

        return batch_x, batch_y

##################################################
class MyDataGenerator_CrazyNN(Sequence):
    def __init__(self, x, y, batch_size, gau_noise, \
        inputs_prepro, transform_input, outputs_prepro, transform_output, min_max_scale, rec3_class):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.gau_noise = gau_noise
        self.indexes = np.arange(len(self.x))

        self.inputs_prepro = inputs_prepro
        self.transform_input = transform_input

        self.outputs_prepro = outputs_prepro
        self.transform_output = transform_output

        self.min_max_scale = min_max_scale
        self.rec3_class = rec3_class

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        # Shuffle indexes at the end of each epoch
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_x = self.x[self.indexes[start:end]]
        batch_y = self.y[self.indexes[start:end]]

        # Here this is not taking into account that I can use a different 
        # MinMaxScale for the the inputs and outputs,
        # this needs to be corrected in the future
        batch_y_classification = batch_y[:, -1].copy()*self.min_max_scale[-1]
        batch_y_classification[batch_y_classification < self.min_max_scale[-1]*self.rec3_class] = 1
        batch_y_classification[batch_y_classification >= self.min_max_scale[-1]*self.rec3_class] = 0
        batch_y_classification.astype(int)
        
        # Apply Gaussian noise to the batch input data
        #noise = np.random.normal(loc=0, scale=self.stddev, size=batch_x.shape)

        # normal version 
        # batch_x =  \
        #     data_encoder(batch_x + batch_x * self.gau_noise * np.random.randn(batch_x.shape[0], batch_x.shape[1]), \
        #         self.inputs_prepro, self.transform_input)[0]

        # trick to apply same noise to last two inputs on batch_x
        #batch_x[:, :2] = batch_x[:, :2] + batch_x[:, :2] * self.gau_noise[:2] * np.random.randn(batch_x.shape[0], 2)
        # random_arr = np.random.randn(batch_x.shape[0], 1).reshape(-1)
        # batch_x[:, -2] = batch_x[:, -2] + batch_x[:, -2] * self.gau_noise[-2] * random_arr
        # batch_x[:, -1] = batch_x[:, -1] + batch_x[:, -1] * self.gau_noise[-2] * random_arr

        # GAUSSIAN NOISE
        #################################################################################################################
        # noise from Da
        # batch_x[:, 0] = batch_x[:, 0] + batch_x[:, 0] * self.gau_noise[0] * np.random.randn(batch_x.shape[0], 1).reshape(-1) #random_arr
        # # noise from Db
        # batch_x[:, 1] = batch_x[:, 1] + batch_x[:, 1] * self.gau_noise[1] * np.random.randn(batch_x.shape[0], 1).reshape(-1) #random_arr
        # # noise from 728 
        # random_arr_728 = np.random.randn(batch_x.shape[0], 1).reshape(-1)
        # # noise from 706
        # random_arr_706 = np.random.randn(batch_x.shape[0], 1).reshape(-1)
        # # noise from 668
        # random_arr_668 = np.random.randn(batch_x.shape[0], 1).reshape(-1)

        # # noise from He728/706
        # batch_x[:, -2] = batch_x[:, -2] * (1 + self.gau_noise[-2]*random_arr_728)/(1 + self.gau_noise[-2]*random_arr_706)
        
        # # noise from He728/668
        # batch_x[:, -1] = batch_x[:, -1] * (1 + self.gau_noise[-1]*random_arr_728)/(1 + self.gau_noise[-1]*random_arr_668)
        #################################################################################################################


        # SYSTEMATIC ERROR, done "right"
        #################################################################################################################
        # single_noise = np.random.randn(batch_x.shape[0], 1).reshape(-1)
        # # Da
        # batch_x[:, 0] = batch_x[:, 0] + batch_x[:, 0] * self.gau_noise[0] * single_noise
        # # Dg
        # batch_x[:, 1] = batch_x[:, 1] + batch_x[:, 1] * self.gau_noise[1] * single_noise
        # # He728/706
        # batch_x[:, -2] = batch_x[:, -2] * (1 + self.gau_noise[-2]*single_noise)/(1 + self.gau_noise[-2]*single_noise)
        # # He728/668
        # batch_x[:, -1] = batch_x[:, -1] * (1 + self.gau_noise[-1]*single_noise)/(1 + self.gau_noise[-1]*single_noise)
        #################################################################################################################


        # SYSTEMATIC ERROR, done "wrong"
        #################################################################################################################
        single_noise = np.random.randn(batch_x.shape[0], 1).reshape(-1)
        # Da
        batch_x[:, 0] = batch_x[:, 0] + batch_x[:, 0] * self.gau_noise[0] * single_noise
        # Dg
        batch_x[:, 1] = batch_x[:, 1] + batch_x[:, 1] * self.gau_noise[1] * single_noise
        # He728/706
        batch_x[:, -2] = batch_x[:, -2] + batch_x[:, -2] * self.gau_noise[-2] * single_noise
        # He728/668
        batch_x[:, -1] = batch_x[:, -1] + batch_x[:, -1] * self.gau_noise[-1] * single_noise
        #################################################################################################################


        # GAUSSIAN NOISE
        #################################################################################################################
        # noise from Da
        # batch_x[:, 0] = batch_x[:, 0] + batch_x[:, 0] * 0.01 * np.random.randn(batch_x.shape[0], 1).reshape(-1) #random_arr
        # # noise from Db
        # batch_x[:, 1] = batch_x[:, 1] + batch_x[:, 1] * 0.01 * np.random.randn(batch_x.shape[0], 1).reshape(-1) #random_arr
        # # noise from 728 
        # random_arr_728 = np.random.randn(batch_x.shape[0], 1).reshape(-1)
        # # noise from 706
        # random_arr_706 = np.random.randn(batch_x.shape[0], 1).reshape(-1)
        # # noise from 668
        # random_arr_668 = np.random.randn(batch_x.shape[0], 1).reshape(-1)

        # # noise from He728/706
        # batch_x[:, -2] = batch_x[:, -2] * (1 + 0.01 * random_arr_728)/(1 + 0.01 * random_arr_706)
        
        # # noise from He728/668
        # batch_x[:, -1] = batch_x[:, -1] * (1 + 0.01 * random_arr_728)/(1 + 0.01 * random_arr_668)
        ################################################################################################################

        batch_x =  \
            data_encoder(batch_x, \
                self.inputs_prepro, self.transform_input)[0]


        batch_y_regression =  \
            data_encoder(batch_y, \
                self.outputs_prepro, self.transform_output)[0]

        batch_y_regression = batch_y_regression[:, :-1]

        return batch_x, np.concatenate([batch_y_regression, batch_y_classification.reshape(-1, 1)], axis=1)


