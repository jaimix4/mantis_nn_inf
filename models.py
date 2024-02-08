import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
import pickle
from keras.losses import Huber
import keras.backend as K
import sys

# new very useful 
# class for creating a pre processing layer
class Pre_processingLayer(Layer):

    def __init__(self, inputs_prepo, inputs_mean_params, inputs_scale_params, **kwargs):

        super(Pre_processingLayer, self).__init__(**kwargs)
        self.inputs_prepo = tf.constant(inputs_prepo, dtype=tf.float32)
        self.inputs_mean_params = tf.constant(inputs_mean_params, dtype=tf.float32)
        self.inputs_scale_params = tf.constant(inputs_scale_params, dtype=tf.float32)

    def call(self, inputs):

        # applying compression if needed
        return ((tf.pow(inputs, 1/self.inputs_prepo) - self.inputs_mean_params)/self.inputs_scale_params)*inputs/inputs

    def get_config(self):
        config = super(Pre_processingLayer, self).get_config()
        config.update({
            'inputs_prepo': self.inputs_prepo.numpy().tolist(),
            'inputs_mean_params': self.inputs_mean_params.numpy().tolist(),
            'inputs_scale_params': self.inputs_scale_params.numpy().tolist(),
        })
        return config

# new very useful 
# class for creating a post processing layer
class Post_processingLayer(Layer):

    def __init__(self, outputs_prepo, outputs_min_params, outputs_scale_params, **kwargs):

        super(Post_processingLayer, self).__init__(**kwargs)
        self.outputs_prepo = tf.constant(outputs_prepo, dtype=tf.float32)
        self.outputs_min_params = tf.constant(outputs_min_params, dtype=tf.float32)
        self.outputs_scale_params = tf.constant(outputs_scale_params, dtype=tf.float32)

    def call(self, inputs):

        # applying compression if needed
        return tf.pow(inputs/self.outputs_scale_params + self.outputs_min_params, self.outputs_prepo)*inputs/inputs

    def get_config(self):
        config = super(Post_processingLayer, self).get_config()
        config.update({
            'outputs_prepo': self.outputs_prepo.numpy().tolist(),
            'outputs_min_params': self.outputs_min_params.numpy().tolist(),
            'outputs_scale_params': self.outputs_scale_params.numpy().tolist(),
            #'name': self.name
        })
        return config

# function to load model from folder  
def load_model(folder_name):
    
    # load config file 
    with open(folder_name + '/config_params.pkl', 'rb') as file:
        config_params = pickle.load(file)

    print('### config parameters ###')
    print(config_params)
    print('#########################')

    # create model 
    if config_params['model'] == 'MLP_BNN_model_Denseflipout_2':

        n_inputs = len(config_params['inputs_labels'])

        n_outputs = len(config_params['outputs_labels'])

        num_layers = config_params['num_layers']

        num_nodes = config_params['num_nodes']

        last_bayes_layer = config_params['last_bayes_layer']

        # check if taking out that X_train.shape[0] is fine
        #kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] * 1.0)
        kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (1e4 * 1.0)

        model = MLP_BNN_model_book_2(n_inputs, n_outputs, \
                                        num_layers, num_nodes, last_bayes_layer, \
                                        act_fn = config_params['activation_fn'], \
                                        l2_reg = config_params['l2_regularizer'], \
                                        kernel_div = kernel_divergence_fn)

        model_dummy = tf.keras.models.load_model(folder_name)

        model.set_weights(model_dummy.get_weights())

        return model, config_params

    elif config_params['model'] == 'MLP_BNN_model_Denseflipout_multi_output':

        n_inputs = len(config_params['inputs_labels']) 

        # ignoring the last output since this one goes in the classification layer
        n_outputs = len(config_params['outputs_labels']) - 1

        num_layers = config_params['num_layers']

        num_nodes = config_params['num_nodes']

        last_bayes_layer = config_params['last_bayes_layer']

        try:
            batch_norm = config_params['batch_norm']
        except:
            batch_norm = True

        # check if taking out that X_train.shape[0] is fine
        #kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] * 1.0)
        kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (1e4 * 1.0)

        model = MLP_BNN_model_book_regression_classification(n_inputs, n_outputs, \
                                        num_layers, num_nodes, last_bayes_layer, \
                                        act_fn = config_params['activation_fn'], \
                                        l2_reg = config_params['l2_regularizer'], \
                                        kernel_div = kernel_divergence_fn,\
                                        n_class = 2, \
                                        batch_norm = batch_norm, \
                                        custom_loss_for_inference=False)
        
        model.summary()
        # keras.losses.custom_loss = custom_loss

        model_dummy = tf.keras.models.load_model(folder_name, custom_objects={'custom_loss': custom_loss})

        model.set_weights(model_dummy.get_weights())

        return model, config_params

        # this could be code better for the very near future

    else:

        print('model not implemented yet')


# new very useful 
def load_model_from_wandb(run_id, model_artifact_name):
    
    # load config file 
    import wandb

    api = wandb.Api()
    run = api.run(run_id)

    config_params = run.config
    # load config file
    print('### config parameters ###')
    print(config_params)
    print('#########################')

    # create model 
    if config_params['model'] == 'MLP_BNN_model_Denseflipout_2':

        n_inputs = len(config_params['inputs_labels'])

        n_outputs = len(config_params['outputs_labels'])

        num_layers = config_params['num_layers']

        num_nodes = config_params['num_nodes']

        last_bayes_layer = config_params['last_bayes_layer']

        # check if taking out that X_train.shape[0] is fine
        #kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] * 1.0)
        kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (1e4 * 1.0)

        model = MLP_BNN_model_book_2(n_inputs, n_outputs, \
                                        num_layers, num_nodes, last_bayes_layer, \
                                        act_fn = config_params['activation_fn'], \
                                        l2_reg = config_params['l2_regularizer'], \
                                        kernel_div = kernel_divergence_fn)

        # get model from wandb 
        run = wandb.init()
        artifact = run.use_artifact(model_artifact_name, type='model')
        artifact_dir = artifact.download()
        wandb.finish()
        model_dummy = tf.keras.models.load_model(artifact_dir)

        # pass weights to model
        model.set_weights(model_dummy.get_weights())

        return model, config_params

    # TO DO MODEL CRAZY NN HERE
        

    else:

        print('model not implemented yet')

def MLP_nn_model(n_inputs, n_outputs, num_layers, num_nodes, l2_reg, act_fn, gau_noise):

    model = keras.Sequential()
    model.add(layers.GaussianNoise(gau_noise, input_shape=(n_inputs,)))
    model.add(layers.BatchNormalization())
    #model.add(layers.Dense(num_nodes, input_dim = n_inputs, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=l2(1e-7)))
    for x in range(num_layers):
      model.add(layers.Dense(num_nodes, kernel_initializer='he_uniform', activation=act_fn, kernel_regularizer=l2(l2_reg)))
    model.add(layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')

    return model

def MLP_nn_model_MCD(n_inputs, n_outputs, num_layers, num_nodes, dropout_layers, l2_reg, act_fn, dropout_p, gau_noise):


    inputs = keras.Input(shape=(n_inputs,))
    x = layers.GaussianNoise(gau_noise)(inputs, training = False)
    x = layers.BatchNormalization()(x)
    for i in range(num_layers):
        x = layers.Dense(num_nodes, kernel_initializer='he_uniform', activation= act_fn, kernel_regularizer=l2(l2_reg))(x)
        x = layers.Dropout(dropout_p)(x, training = True)
    outputs = layers.Dense(n_outputs)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mse', optimizer='adam')

    return model


def MLP_BNN_model_book_3(n_inputs, n_outputs, num_layers, num_nodes, last_bayes_layer, act_fn, l2_reg, gau_noise_H, gau_noise_He, kernel_div):


    H_input = keras.Input(shape=(2,))
    gau_H = layers.GaussianNoise(gau_noise_H)(H_input, training = False)

    He_input = keras.Input(shape=(2,))
    gau_He = layers.GaussianNoise(gau_noise_He)(He_input, training = False)

    full_input = tf.keras.layers.Concatenate()([gau_H, gau_He])

    x = layers.Dense(num_nodes, kernel_initializer='he_uniform', activation= act_fn, kernel_regularizer=l2(l2_reg))(full_input)
    x = layers.BatchNormalization()(x)


    for ii in range(num_layers - 2):
        x = layers.Dense(num_nodes, kernel_initializer='he_uniform', activation= act_fn, kernel_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)

    for ii in range(1):
        x = tfp.layers.DenseFlipout(last_bayes_layer, activation=act_fn, kernel_divergence_fn=kernel_div)(x)

    outputs = layers.Dense(n_outputs)(x)

    model = keras.Model(inputs=full_input, outputs=outputs)

    model.compile(loss='mse', optimizer='adam')

    return model


#kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] *1.0)

# new, it does not work 
def MLP_BNN_model_book_prepost(n_inputs, n_outputs, num_layers, num_nodes, last_bayes_layer, act_fn, l2_reg, gau_noise, kernel_div, \
  inputs_prepo, inputs_mean_params, inputs_scale_params, outputs_prepo, outputs_min_params, outputs_scale_params):


    input_single = Input(shape=(n_inputs,))
    x = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)
    x = Dense(num_nodes, activation=act_fn, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    for i in range(num_layers - 2):
            x = Dense(num_nodes, activation=act_fn, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_reg))(x)
            x = BatchNormalization()(x)
    tfp.layers.DenseFlipout(last_bayes_layer, activation=act_fn, kernel_divergence_fn=kernel_div)(x)
    x = Dense(n_outputs)(x)
    output_single = Post_processingLayer(outputs_prepo, outputs_min_params, outputs_scale_params)(x)

    model = Model(input_single, output_single)
    model.compile(loss='mse', optimizer='adam')

    return model


# CUSTOM LOSS FUNCTION dummy
# this custom_loss function is used only to set it in an already train network
# for exporting, the actual loss function is the one inside the model definition
def custom_loss(y_true, y_pred):

    
    regression_loss_Te = K.abs(y_pred[:, 0] - y_true[:, 0])*(4.0*K.pow(K.ones_like(y_true[:, 0])*0.5, y_true[:, 0]) + 1)#(5 - 0.4*y_true[:, 0])

    regression_loss_ne = K.abs(y_pred[:, 1] - y_true[:, 1])*(4.0*K.pow(K.ones_like(y_true[:, 1])*0.5, y_true[:, 1]) + 1)#*(5 - 0.4*y_true[:, 1])

    regression_loss_no = K.abs(y_pred[:, 2] - y_true[:, 2])*(4.0*K.pow(K.ones_like(y_true[:, 2])*0.5, y_true[:, 2]) + 1)*y_true[:, 5]

    regression_loss_Irate = K.abs(y_pred[:, 3] - y_true[:, 3])*(4.0*K.pow(K.ones_like(y_true[:, 3])*0.5, y_true[:, 3]) + 1)*y_true[:, 5]
    
    regression_loss_Rrate = K.abs(y_pred[:, 4] - y_true[:, 4])*(4.0*K.pow(K.ones_like(y_true[:, 4])*0.5, y_true[:, 4]) + 1)#*(5 - 0.4*y_true[:, 4])

    loss_no_rec = (regression_loss_no + regression_loss_Irate)*(y_true[:, 5])

    loss_rec = (regression_loss_Te + regression_loss_ne + regression_loss_Rrate)*(1 + (2/3)*( 1 - y_true[:, 5]))

    # regression_loss = (regression_loss_Te + regression_loss_ne + \
    #     regression_loss_no + regression_loss_Irate + \
    #         regression_loss_Rrate)

    regression_loss = loss_no_rec + loss_rec

    class_loss = K.sparse_categorical_crossentropy(y_true[:, 5], y_pred[:, 5:])

    return regression_loss + class_loss

def MLP_BNN_model_book_regression_classification(n_inputs, n_outputs, num_layers, num_nodes, \
    last_bayes_layer, act_fn, l2_reg, kernel_div, n_class, batch_norm, custom_loss_for_inference = True):

    # input layer 
    input_single = Input(shape=(n_inputs,))
    x = Dense(num_nodes, activation=act_fn, kernel_initializer='he_uniform', \
        kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(input_single)
    if batch_norm:
        x = BatchNormalization()(x)
    for i in range(num_layers - 2):
        x = Dense(num_nodes, activation=act_fn, kernel_initializer='he_uniform', \
            kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        if batch_norm:
            x = BatchNormalization()(x)
        # x = layers.Dropout(0.1)(x)
    def relu_advanced(x):
        return K.relu(x, max_value=5.0)
    # last bayesian layer with just kernel as random variable
    # x = tfp.layers.DenseFlipout(last_bayes_layer, activation=relu_advanced, kernel_divergence_fn=kernel_div)(x)
    # last bayesian layer with kernel and bias as random variables
    x = tfp.layers.DenseFlipout(last_bayes_layer, activation='relu', kernel_divergence_fn=kernel_div, bias_prior_fn = tfp.layers.default_multivariate_normal_fn, \
     bias_divergence_fn = kernel_div)(x)

    # standard ouput layer:
    # output_regression = Dense(n_outputs, activation='linear')(x)
    # dense_flipout output layer:
    output_regression = tfp.layers.DenseFlipout(n_outputs, kernel_divergence_fn=kernel_div, activation='linear', name = 'out_regression')(x) #,\
        #bias_prior_fn = tfp.layers.default_multivariate_normal_fn, bias_divergence_fn = kernel_div)(x)
    output_classification = Dense(n_class, activation='softmax', name = 'out_classification')(x)

    model_output = concatenate([output_regression, output_classification])

    def custom_loss(y_true, y_pred):

        
        regression_loss_Te = K.abs(y_pred[:, 0] - y_true[:, 0])*(2.0*K.pow(K.ones_like(y_true[:, 0])*0.5, y_true[:, 0]) + 1) \
            *(y_true[:, 5] + (1 - y_true[:, 5])*(1/1))

        regression_loss_ne = K.abs(y_pred[:, 1] - y_true[:, 1])*(2.0*K.pow(K.ones_like(y_true[:, 1])*0.5, y_true[:, 1]) + 1)\
            *(y_true[:, 5] + (1 - y_true[:, 5])*(1/1))

        # this output is ignore if it is in the non-ionized regime
        regression_loss_no = K.abs(y_pred[:, 2] - y_true[:, 2])*y_true[:, 5]*(2.0*K.pow(K.ones_like(y_true[:, 2])*0.5, y_true[:, 2]) + 1)

        # this output is ignore if it is in the non-ionized regime (ignores low values)
        regression_loss_Irate = K.abs(y_pred[:, 3] - y_true[:, 3])*y_true[:, 5]*(4.0*K.pow(K.ones_like(y_true[:, 3])*0.5, y_true[:, 3]) + 1)
        
        # this output is ignore if it is in the non-ionized regime (ignores high values)
        regression_loss_Rrate = K.abs(y_pred[:, 4] - y_true[:, 4])*y_true[:, 5]*(5.0*K.pow(K.ones_like(y_true[:, 4])*0.5, y_true[:, 4]) + 1)

        # doing MAE of losses 
        regression_loss = K.mean(regression_loss_Te +  regression_loss_ne + regression_loss_no + regression_loss_Irate + regression_loss_Rrate, axis = -1)

        # classification loss
        class_loss = K.sparse_categorical_crossentropy(y_true[:, 5], y_pred[:, 5:]) 

        # pressure loss
        physics_loss = K.mean(K.abs(y_pred[:, 0]*y_pred[:, 1] - y_true[:, 0]*y_true[:, 1]))

        # sum of all losses 
        return regression_loss + class_loss  + physics_loss 

    model = Model(inputs = input_single, outputs = model_output)

    if custom_loss_for_inference:
        model.compile(loss=custom_loss, optimizer='adam')
    else:
        model.compile(loss='mae', optimizer='adam')

    return model


def MLP_BNN_model_book_2(n_inputs, n_outputs, num_layers, num_nodes, last_bayes_layer, act_fn, l2_reg, kernel_div):

    #kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] *1.0)

    model = keras.Sequential()
    model.add(keras.Input(shape=(n_inputs,)))
    
    #model.add(layers.GaussianNoise(gau_noise*0, input_shape=(n_inputs,)))
    #model.add(layers.BatchNormalization())
    for x in range(num_layers - 1):
      #model.add(layers.Dense(num_nodes, input_dim = n_inputs, kernel_initializer='he_uniform', activation=act_fn, kernel_regularizer=l2(l2_reg)))
      model.add(layers.Dense(num_nodes, kernel_initializer='he_uniform', activation=act_fn, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
      model.add(layers.BatchNormalization())
    for x in range(1):
      model.add(tfp.layers.DenseFlipout(last_bayes_layer, activation=act_fn, kernel_divergence_fn=kernel_div))
    model.add(layers.Dense(n_outputs, activation='linear'))
    model.compile(loss='mae', optimizer='adam')

    return model

# new, it is only use for trying things 
def MLP_BNN_model_book_2_dummy(n_inputs, n_outputs, num_layers, num_nodes, last_bayes_layer, act_fn, l2_reg, gau_noise, kernel_div):

    #kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] *1.0)

    model = keras.Sequential()
    model.add(keras.Input(shape=(n_inputs,)))
    
    #model.add(layers.GaussianNoise(gau_noise*0, input_shape=(n_inputs,)))
    #model.add(layers.BatchNormalization())
    for x in range(num_layers - 1):
      model.add(layers.Dense(num_nodes, kernel_initializer='he_uniform', activation=act_fn, kernel_regularizer=l2(l2_reg)))
      model.add(layers.BatchNormalization())
    for x in range(1):
      #model.add(tfp.layers.DenseFlipout(last_bayes_layer, activation=act_fn, kernel_divergence_fn=kernel_div))
      model.add(layers.Dense(num_nodes, kernel_initializer='he_uniform', activation=act_fn, kernel_regularizer=l2(l2_reg)))
    model.add(layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')

    return model


def MLP_BNN_model_book(n_inputs, n_outputs, num_layers, num_nodes, act_fn, gau_noise, kernel_div):


    #kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] *1.0)

    #kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] *1.0)

    model = keras.Sequential()
    model.add(layers.GaussianNoise(gau_noise, input_shape=(n_inputs,)))
    model.add(layers.BatchNormalization())
    #model.add(layers.Dense(num_nodes, input_dim = n_inputs, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=l2(1e-7)))
    for x in range(num_layers):
      model.add(tfp.layers.DenseFlipout(num_nodes, activation=act_fn, kernel_divergence_fn=kernel_div))
    model.add(layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')

    return model


class BNN_ensemble_DNN():

    def __init__(self, model_path):

        self.model_path = model_path

        self.model_bnn, self.config_params = load_model(self.model_path)

    def model_bnn(self):
        return self.model_bnn

    def config_params(self):

        return self.config_params

    def ensemble_DNN(self, num_networks = 8, include_prepro = True, save_model = True):

        num_networks = num_networks
  
        n_inputs = len(self.config_params['inputs_labels'])
        n_outputs = len(self.config_params['outputs_labels'])
        num_layers = self.config_params['num_layers']
        num_nodes = self.config_params['num_nodes']
        last_bayes_layer = self.config_params['last_bayes_layer']

        inputs_prepo = self.config_params['inputs_prepro']
        inputs_mean_params = self.config_params['inputs_mean_params']
        inputs_scale_params = self.config_params['inputs_scale_params']

        outputs_prepo = self.config_params['outputs_prepro']
        outputs_min_params = self.config_params['outputs_min_params']
        outputs_scale_params = self.config_params['outputs_scale_params']

        act_fn = self.config_params['activation_fn']

        networks_list = []

        networks_input_list = []

        #input_single = Input(shape=(n_inputs,))
        # input_single_prepro = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)

        # NETWORK 1
        for i in range(num_networks):

            input_single = Input(shape=(n_inputs,))
            # POSSIBLE TO PUT PREPROCESSING HERE WITH NO INPUT LAYER 
            # AND THEN CREATE A BIG INPUT LAYER FOR ALL THE NETWORKS
            x = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)
            x = Dense(num_nodes, activation=act_fn)(x)
            x = BatchNormalization()(x)
            for i in range(num_layers - 2):
                    x = Dense(num_nodes, activation=act_fn)(x)
                    x = BatchNormalization()(x)
            x = Dense(last_bayes_layer, activation=act_fn)(x)
            x = Dense(n_outputs, activation = 'linear')(x)
            output_single = Post_processingLayer(outputs_prepo, outputs_min_params, outputs_scale_params)(x)

            # HERE ADD POST PREOCESSING LAYER 
            
            networks_list.append(output_single)
            networks_input_list.append(input_single)

        merged_models = concatenate(networks_list)

        model_merged = Model(inputs=networks_input_list, outputs=merged_models)

        model_merged.summary()

        for idx in range(num_networks*2, len(model_merged.layers) - (1 + num_networks), num_networks):

            idx_og_model = int(idx/num_networks - 2)

            for idx_layer in range(idx, idx+num_networks):

                if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

                    model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

                else:

                    sampled_weights = []
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
                    model_merged.layers[idx_layer].set_weights(sampled_weights)

        # for idx in range(num_networks*2, len(model_merged.layers) - (1 + num_networks), num_networks):

        #     idx_og_model = int(idx/num_networks - 2)

        #     for idx_layer in range(idx, idx+num_networks):

        #         if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

        #             model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

        #         else:

        #             sampled_weights = []
        #             sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
        #             sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
        #             model_merged.layers[idx_layer].set_weights(sampled_weights)

        if save_model:
            
            model_merged.save(self.model_path + '_ensemble_DNN_{}'.format(num_networks))

        return model_merged

    def ensemble_DNN_single_input(self, num_networks = 8, include_prepro = True, save_model = True):

        num_networks = num_networks
  
        n_inputs = len(self.config_params['inputs_labels'])
        n_outputs = len(self.config_params['outputs_labels'])
        num_layers = self.config_params['num_layers']
        num_nodes = self.config_params['num_nodes']
        last_bayes_layer = self.config_params['last_bayes_layer']

        inputs_prepo = self.config_params['inputs_prepro']
        inputs_mean_params = self.config_params['inputs_mean_params']
        inputs_scale_params = self.config_params['inputs_scale_params']

        outputs_prepo = self.config_params['outputs_prepro']
        outputs_min_params = self.config_params['outputs_min_params']
        outputs_scale_params = self.config_params['outputs_scale_params']

        act_fn = self.config_params['activation_fn']

        networks_list = []

        networks_input_list = []

        input_single = Input(shape=(n_inputs,))
        input_single_prepro = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)

        # NETWORK 1
        for i in range(num_networks):

            # input_single = Input(shape=(n_inputs,))
            # POSSIBLE TO PUT PREPROCESSING HERE WITH NO INPUT LAYER 
            # AND THEN CREATE A BIG INPUT LAYER FOR ALL THE NETWORKS
            #x = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)
            x = Dense(num_nodes, activation=act_fn)(input_single_prepro)
            #x = Dense(num_nodes, activation=act_fn)(x)
            x = BatchNormalization()(x)
            for i in range(num_layers - 2):
                    x = Dense(num_nodes, activation=act_fn)(x)
                    x = BatchNormalization()(x)
            x = Dense(last_bayes_layer, activation=act_fn)(x)
            x = Dense(n_outputs, activation = 'linear')(x)
            output_single = Post_processingLayer(outputs_prepo, outputs_min_params, outputs_scale_params)(x)

            # HERE ADD POST PREOCESSING LAYER 
            
            networks_list.append(output_single)
            #networks_input_list.append(input_single)

        merged_models = concatenate(networks_list)

        model_merged = Model(inputs=input_single, outputs=merged_models)

        model_merged.summary()

        print(model_merged)

        idx_og_model = 0

        for idx in range(2, len(model_merged.layers) - (1 + num_networks), num_networks):

            #idx_og_model = int((idx + 1)/num_networks - 1)

            for idx_layer in range(idx, idx+num_networks):

                if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

                    model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

                else:

                    sampled_weights = []
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
                    model_merged.layers[idx_layer].set_weights(sampled_weights)
            
            idx_og_model += 1
        # for idx in range(num_networks*2, len(model_merged.layers) - (1 + num_networks), num_networks):

        #     idx_og_model = int(idx/num_networks - 2)

        #     for idx_layer in range(idx, idx+num_networks):

        #         if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

        #             model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

        #         else:

        #             sampled_weights = []
        #             sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
        #             sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
        #             model_merged.layers[idx_layer].set_weights(sampled_weights)

        if save_model:
            
            model_merged.save(self.model_path + '_ensemble_DNN_{}'.format(num_networks))

        return model_merged


    def load_test_set(self):

        from pathlib import Path
        import pyarrow.parquet as pq

        data_folder = self.config_params['dataset']
        print("Getting data from folder: ", data_folder)

        if Path(data_folder + 'TEST.parquet').is_file():

          data_set_test = pq.read_table(data_folder + 'TEST.parquet').to_pandas()

          print("----PARQUET FILES FOUND-----")

        inputs_labels = self.config_params['inputs_labels']
        outputs_labels = self.config_params['outputs_labels']

        X_test = data_set_test[inputs_labels].values
        y_test = data_set_test[outputs_labels].values

        return X_test, y_test


class BNN_crazy_ensemble_DNN():

    def __init__(self, model_path):

        self.model_path = model_path

        self.model_bnn, self.config_params = load_model(self.model_path)

    def model_bnn(self):
        return self.model_bnn

    def config_params(self):

        return self.config_params

    def ensemble_DNN(self, num_networks = 8, include_prepro = True, save_model = True):

        num_networks = num_networks
  
        n_inputs = len(self.config_params['inputs_labels'])
        # get rid of one output that goes in other layer 
        n_outputs = len(self.config_params['outputs_labels']) - 1
        num_layers = self.config_params['num_layers']
        num_nodes = self.config_params['num_nodes']
        last_bayes_layer = self.config_params['last_bayes_layer']

        inputs_prepo = self.config_params['inputs_prepro']
        inputs_mean_params = self.config_params['inputs_mean_params']
        inputs_scale_params = self.config_params['inputs_scale_params']

        outputs_prepo = self.config_params['outputs_prepro']
        outputs_min_params = self.config_params['outputs_min_params']
        outputs_scale_params = self.config_params['outputs_scale_params']

        act_fn = self.config_params['activation_fn']

        networks_list = []

        networks_input_list = []

        #input_single = Input(shape=(n_inputs,))
        # input_single_prepro = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)

        # NETWORK 1
        for i in range(num_networks):

            input_single = Input(shape=(n_inputs,))
            # POSSIBLE TO PUT PREPROCESSING HERE WITH NO INPUT LAYER 
            # AND THEN CREATE A BIG INPUT LAYER FOR ALL THE NETWORKS
            x = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)
            x = Dense(num_nodes, activation=act_fn)(x)
            x = BatchNormalization()(x)
            for i in range(num_layers - 2):
                    x = Dense(num_nodes, activation=act_fn)(x)
                    x = BatchNormalization()(x)
            x = Dense(last_bayes_layer, activation=act_fn)(x)
            x = Dense(n_outputs, activation = 'linear')(x)
            output_single = Post_processingLayer(outputs_prepo, outputs_min_params, outputs_scale_params)(x)

            # HERE ADD POST PREOCESSING LAYER 
            
            networks_list.append(output_single)
            networks_input_list.append(input_single)

        merged_models = concatenate(networks_list, name = 'output')

        model_merged = Model(inputs=networks_input_list, outputs=merged_models)

        model_merged.summary()

        for idx in range(num_networks*2, len(model_merged.layers) - (1 + num_networks), num_networks):

            idx_og_model = int(idx/num_networks - 2)

            for idx_layer in range(idx, idx+num_networks):

                if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

                    model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

                else:

                    sampled_weights = []
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
                    model_merged.layers[idx_layer].set_weights(sampled_weights)

        # for idx in range(num_networks*2, len(model_merged.layers) - (1 + num_networks), num_networks):

        #     idx_og_model = int(idx/num_networks - 2)

        #     for idx_layer in range(idx, idx+num_networks):

        #         if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

        #             model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

        #         else:

        #             sampled_weights = []
        #             sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
        #             sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
        #             model_merged.layers[idx_layer].set_weights(sampled_weights)

        if save_model:
            
            model_merged.save(self.model_path + '_ensemble_DNN_{}'.format(num_networks))

        return model_merged

    def ensemble_DNN_single_input(self, num_networks = 8, include_prepro = True, save_model = True):

        num_networks = num_networks
  
        n_inputs = len(self.config_params['inputs_labels'])
        # get rid of one output 
        n_outputs = len(self.config_params['outputs_labels']) - 1
        num_layers = self.config_params['num_layers']
        num_nodes = self.config_params['num_nodes']
        last_bayes_layer = self.config_params['last_bayes_layer']

        inputs_prepo = self.config_params['inputs_prepro']
        inputs_mean_params = self.config_params['inputs_mean_params']
        inputs_scale_params = self.config_params['inputs_scale_params']

        outputs_prepo = self.config_params['outputs_prepro']
        outputs_min_params = self.config_params['outputs_min_params']
        outputs_scale_params = self.config_params['outputs_scale_params']

        # knowing how the output will be, we modify this to have the right number of nodes
        # the two last are the classification outcome 
        outputs_prepo[-1] = 1
        outputs_min_params[-1] = 0.0
        outputs_scale_params[-1] = 1.0
        # no add one more 
        outputs_prepo.append(1)
        outputs_min_params.append(0.0)
        outputs_scale_params.append(1.0)

        act_fn = self.config_params['activation_fn']

        networks_list = []

        networks_input_list = []

        input_single = Input(shape=(n_inputs,), name = 'input')
        input_single_prepro = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)

        # code for figuring out if self.model_bnn has batch normalization layers
        batch_norm_present = False
        for idx in range(len(self.model_bnn.layers)):
            if self.model_bnn.layers[idx].__class__.__name__ == 'BatchNormalization':
                batch_norm_present = True


        print('batch norm present')
        print(batch_norm_present)

        # NETWORK 1
        for i in range(num_networks):

            # input_single = Input(shape=(n_inputs,))
            # POSSIBLE TO PUT PREPROCESSING HERE WITH NO INPUT LAYER 
            # AND THEN CREATE A BIG INPUT LAYER FOR ALL THE NETWORKS
            #x = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)
            x = Dense(num_nodes, activation=act_fn)(input_single_prepro)
            #x = Dense(num_nodes, activation=act_fn)(x)
            if batch_norm_present:
                x = BatchNormalization()(x)
            for i in range(num_layers - 2):
                    x = Dense(num_nodes, activation=act_fn)(x)
                    if batch_norm_present:
                        x = BatchNormalization()(x)
            # layer that branches both outputs
            x = Dense(last_bayes_layer, activation=act_fn)(x)

            # the two outputs layers 
            # REGRESSION
            output_regression = Dense(n_outputs, activation = 'linear')(x)
            # omit the last node here

            # CLASSIFICATION
            output_classification = Dense(2, activation='softmax')(x)

            # CONCATENATE BOTH OUTPUTS
            model_output = concatenate([output_regression, output_classification])


            print(outputs_prepo)

            # POST PROCESSING LAYER
            total_output = Post_processingLayer(outputs_prepo, outputs_min_params, outputs_scale_params)(model_output)
            
            networks_list.append(total_output)
            #networks_input_list.append(input_single)

        merged_models = concatenate(networks_list, name = 'output')

        model_merged = Model(inputs=input_single, outputs=merged_models)

        model_merged.compile(loss='mse', optimizer='adam')

        model_merged.summary()

        print(model_merged.layers)

        idx_og_model = 1

        #for idx in range(2, len(model_merged.layers) - (1 + num_networks), num_networks):
        some_number = 1
        if batch_norm_present:
            some_number = 0

        for idx in range(2, 2*(num_layers - 1)*num_networks + num_networks + 2 - some_number*num_networks*(num_layers - 1), num_networks):

            #idx_og_model = int((idx + 1)/num_networks - 1)

            for idx_layer in range(idx, idx+num_networks):

                if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

                    model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

                else:

                    sampled_weights = []
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
                    model_merged.layers[idx_layer].set_weights(sampled_weights)
                    print('denseflipout transmitted')
            
            idx_og_model += 1

        # print(model_merged.layers[idx])
        # print('layers')
        # print('idx if model og now')
        # print(idx_og_model)
            
        print("UNTIL HERE FINE")

        # putting weigths on multioutput layer
        for idx in range(2*(num_layers - 1)*num_networks + num_networks + 2 - some_number*num_networks*(num_layers - 1), \
            (2*(num_layers - 1)*num_networks + num_networks + 2) + 2*num_networks - some_number*num_networks*(num_layers - 1), 2):

            if self.model_bnn.layers[idx_og_model].__class__.__name__ == 'DenseFlipout':

                # this is for output bayesian
                sampled_weights = []
                sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
                sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
                model_merged.layers[idx].set_weights(sampled_weights)

            else:

                model_merged.layers[idx].set_weights(self.model_bnn.layers[idx_og_model].get_weights())


            model_merged.layers[idx+1].set_weights(self.model_bnn.layers[idx_og_model+1].get_weights())

            print(model_merged.layers[idx])


        if save_model:
            
            model_merged.save(self.model_path + '_ensemble_DNN_{}'.format(num_networks))
            # also save model as .h5
            model_merged.save(self.model_path + '_ensemble_DNN_{}_HDF5.h5'.format(num_networks), save_format='h5')
            # also 
            tf.saved_model.save(model_merged, self.model_path + '_ensemble_DNN_{}_mantis'.format(num_networks))

        return model_merged
    
    def load_set_pre_process(self, set_label = 'TEST'):

        from pathlib import Path
        import pyarrow.parquet as pq

        data_folder = self.config_params['dataset']
        print("Getting data from folder: ", data_folder)

        if Path(data_folder + set_label + '.parquet').is_file():

          data_set_test = pq.read_table(data_folder + set_label + '.parquet').to_pandas()

          print("----PARQUET FILES FOUND-----")

        inputs_labels = self.config_params['inputs_labels']
        outputs_labels = self.config_params['outputs_labels']

        X_test = data_set_test[inputs_labels].values
        y_test = data_set_test[outputs_labels].values

        # make this self.config_params['inputs_prepro'] a numpy float array
        self.config_params['inputs_prepro'] = np.array(self.config_params['inputs_prepro'], dtype = np.float32)
        self.config_params['inputs_mean_params'] = np.array(self.config_params['inputs_mean_params'], dtype = np.float32)
        self.config_params['inputs_scale_params'] = np.array(self.config_params['inputs_scale_params'], dtype = np.float32)

        self.config_params['outputs_prepro'] = np.array(self.config_params['outputs_prepro'], dtype = np.float32)
        self.config_params['outputs_min_params'] = np.array(self.config_params['outputs_min_params'], dtype = np.float32)
        self.config_params['outputs_scale_params'] = np.array(self.config_params['outputs_scale_params'], dtype = np.float32)

        X_test = (np.power(X_test, 1/self.config_params['inputs_prepro']) - self.config_params['inputs_mean_params'])/self.config_params['inputs_scale_params']
        y_test = np.power(y_test/self.config_params['outputs_scale_params'] + self.config_params['outputs_min_params'], self.config_params['outputs_prepro'])

        return X_test, y_test


    def load_test_set(self):

        from pathlib import Path
        import pyarrow.parquet as pq

        data_folder = self.config_params['dataset']
        print("Getting data from folder: ", data_folder)

        if Path(data_folder + 'TEST.parquet').is_file():

          data_set_test = pq.read_table(data_folder + 'TEST.parquet').to_pandas()

          print("----PARQUET FILES FOUND-----")

        inputs_labels = self.config_params['inputs_labels']
        outputs_labels = self.config_params['outputs_labels']

        X_test = data_set_test[inputs_labels].values
        y_test = data_set_test[outputs_labels].values

        return X_test, y_test

    def ensemble_DNN_single_input_improved(self, num_networks = 8, include_prepro = True, save_model = True):

        num_networks = num_networks
  
        n_inputs = len(self.config_params['inputs_labels'])
        # get rid of one output 
        n_outputs = len(self.config_params['outputs_labels']) - 1
        num_layers = self.config_params['num_layers']
        num_nodes = self.config_params['num_nodes']
        last_bayes_layer = self.config_params['last_bayes_layer']

        inputs_prepo = self.config_params['inputs_prepro']
        inputs_mean_params = self.config_params['inputs_mean_params']
        inputs_scale_params = self.config_params['inputs_scale_params']

        outputs_prepo = self.config_params['outputs_prepro']
        outputs_min_params = self.config_params['outputs_min_params']
        outputs_scale_params = self.config_params['outputs_scale_params']

        # knowing how the output will be, we modify this to have the right number of nodes
        # the two last are the classification outcome 
        outputs_prepo[-1] = 1
        outputs_min_params[-1] = 0.0
        outputs_scale_params[-1] = 1.0
        # no add one more 
        outputs_prepo.append(1)
        outputs_min_params.append(0.0)
        outputs_scale_params.append(1.0)

        act_fn = self.config_params['activation_fn']

        networks_list = []

        networks_input_list = []

        input_single = Input(shape=(n_inputs,), name = 'input')
        input_single_prepro = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)

        # NETWORK 1
        for i in range(num_networks):

            # input_single = Input(shape=(n_inputs,))
            # POSSIBLE TO PUT PREPROCESSING HERE WITH NO INPUT LAYER 
            # AND THEN CREATE A BIG INPUT LAYER FOR ALL THE NETWORKS
            #x = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)
            x = Dense(num_nodes, activation=act_fn)(input_single_prepro)
            #x = Dense(num_nodes, activation=act_fn)(x)
            x = BatchNormalization()(x)
            for i in range(num_layers - 2):
                    x = Dense(num_nodes, activation=act_fn)(x)
                    x = BatchNormalization()(x)
            # layer that branches both outputs
            x = Dense(last_bayes_layer, activation=act_fn)(x)

            # the two outputs layers 
            # REGRESSION
            output_regression = Dense(n_outputs, activation = 'linear')(x)
            # omit the last node here

            # CLASSIFICATION
            output_classification = Dense(2, activation='softmax')(x)

            # CONCATENATE BOTH OUTPUTS
            model_output = concatenate([output_regression, output_classification])


            print(outputs_prepo)

            # POST PROCESSING LAYER
            total_output = Post_processingLayer(outputs_prepo, outputs_min_params, outputs_scale_params)(model_output)
            
            networks_list.append(total_output)
            #networks_input_list.append(input_single)

        merged_models = concatenate(networks_list, name = 'output')

        model_merged = Model(inputs=input_single, outputs=merged_models)

        model_merged.compile(loss='mae', optimizer='adam')

        model_merged.summary()

        print(model_merged.layers)

        idx_og_model = 1

        #for idx in range(2, len(model_merged.layers) - (1 + num_networks), num_networks):
        for idx in range(2, 2*(num_layers - 1)*num_networks + num_networks + 2, num_networks):

            #idx_og_model = int((idx + 1)/num_networks - 1)

            for idx_layer in range(idx, idx+num_networks):

                if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

                    model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

                else:

                    sampled_weights = []
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
                    model_merged.layers[idx_layer].set_weights(sampled_weights)
                    print('denseflipout transmitted')
            
            idx_og_model += 1

        # print(model_merged.layers[idx])
        # print('layers')
        # print('idx if model og now')
        # print(idx_og_model)

        # putting weigths on multioutput layer
        for idx in range(2*(num_layers - 1)*num_networks + num_networks + 2, \
            (2*(num_layers - 1)*num_networks + num_networks + 2) + 2*num_networks, 2):

            model_merged.layers[idx].set_weights(self.model_bnn.layers[idx_og_model].get_weights())
            model_merged.layers[idx+1].set_weights(self.model_bnn.layers[idx_og_model+1].get_weights())

            print(model_merged.layers[idx])


        if save_model:
            
            model_merged.save(self.model_path + '_ensemble_DNN_{}'.format(num_networks))
            # also save model as .h5
            model_merged.save(self.model_path + '_ensemble_DNN_{}_HDF5.h5'.format(num_networks), save_format='h5')
            # also 
            tf.saved_model.save(model_merged, self.model_path + '_ensemble_DNN_{}_mantis'.format(num_networks))

        return model_merged

# PREPROCESSING LAYER FOR ACCEPTING ABSOLUTE EMISSIONS
# DIRECTLY FROM TOMOGRAPHIC INVERSIONS 
class Pre_processingLayer_ABS_EMIS(Layer):

    def __init__(self, inputs_prepo, inputs_mean_params, inputs_scale_params, **kwargs):

        super(Pre_processingLayer_ABS_EMIS, self).__init__(**kwargs)
        self.inputs_prepo = tf.constant(inputs_prepo, dtype=tf.float32)
        self.inputs_mean_params = tf.constant(inputs_mean_params, dtype=tf.float32)
        self.inputs_scale_params = tf.constant(inputs_scale_params, dtype=tf.float32)

    def call(self, inputs):

        # applying compression if needed

        ratios_He_728_706 = inputs[:, 2]/inputs[:, 3]
        ratios_He_706_668 = inputs[:, 2]/inputs[:, 4]

        inputs_with_ratio = tf.stack([inputs[:, 0], inputs[:,1], ratios_He_728_706, ratios_He_706_668], axis=-1)

        return ((tf.pow(inputs_with_ratio, 1/self.inputs_prepo) - self.inputs_mean_params)/self.inputs_scale_params)


    def get_config(self):
        config = super(Pre_processingLayer_ABS_EMIS, self).get_config()
        config.update({
            'inputs_prepo': self.inputs_prepo.numpy().tolist(),
            'inputs_mean_params': self.inputs_mean_params.numpy().tolist(),
            'inputs_scale_params': self.inputs_scale_params.numpy().tolist(),
        })
        return config

class Post_processingLayer_WITH_MEAN(Layer):

    def __init__(self, outputs_prepo, outputs_min_params, outputs_scale_params, num_networks, **kwargs):

        super(Post_processingLayer_WITH_MEAN, self).__init__(**kwargs)
        self.outputs_prepo = tf.constant(outputs_prepo, dtype=tf.float32)
        self.outputs_min_params = tf.constant(outputs_min_params, dtype=tf.float32)
        self.outputs_scale_params = tf.constant(outputs_scale_params, dtype=tf.float32)
        self.num_networks = num_networks

    def call(self, inputs):

        # do mean of all 

        # Reshape the inputs so that the second dimension corresponds to the number of networks
        reshaped_inputs = tf.reshape(inputs, [-1, self.num_networks, 7])

        # Compute the mean across the second dimension
        mean_inputs = tf.reduce_mean(reshaped_inputs, axis=1)

        # applying compression if needed
        return tf.pow(mean_inputs/self.outputs_scale_params + self.outputs_min_params, self.outputs_prepo)

    def get_config(self):
        config = super(Post_processingLayer_WITH_MEAN, self).get_config()
        config.update({
            'outputs_prepo': self.outputs_prepo.numpy().tolist(),
            'outputs_min_params': self.outputs_min_params.numpy().tolist(),
            'outputs_scale_params': self.outputs_scale_params.numpy().tolist(),
            'num_networks': self.num_networks
            #'name': self.name
        })
        return config
    
class BNN_GPU_tune_ensemble_DNN():

    def __init__(self, model_path):

        self.model_path = model_path

        self.model_bnn, self.config_params = load_model(self.model_path)

    def model_bnn(self):
        return self.model_bnn

    def config_params(self):

        return self.config_params


    def ensemble_DNN_single_input(self, num_networks = 8, include_prepro = True, save_model = True):

        num_networks = num_networks
  
        n_inputs = len(self.config_params['inputs_labels'])
        # get rid of one output 
        n_outputs = len(self.config_params['outputs_labels']) - 1
        num_layers = self.config_params['num_layers']
        num_nodes = self.config_params['num_nodes']
        last_bayes_layer = self.config_params['last_bayes_layer']

        inputs_prepo = self.config_params['inputs_prepro']
        inputs_mean_params = self.config_params['inputs_mean_params']
        inputs_scale_params = self.config_params['inputs_scale_params']

        outputs_prepo = self.config_params['outputs_prepro']
        outputs_min_params = self.config_params['outputs_min_params']
        outputs_scale_params = self.config_params['outputs_scale_params']

        # knowing how the output will be, we modify this to have the right number of nodes
        # the two last are the classification outcome 
        outputs_prepo[-1] = 1
        outputs_min_params[-1] = 0.0
        outputs_scale_params[-1] = 1.0
        # no add one more 
        outputs_prepo.append(1)
        outputs_min_params.append(0.0)
        outputs_scale_params.append(1.0)

        act_fn = self.config_params['activation_fn']

        networks_list = []

        networks_input_list = []

        # HERE YOU DECLARE AS INPUTS ALL THE ABSOLUTE EMISSIONS Da, Dg, He728, He706, He668
        n_inputs_abs_MANTIS = 5
        input_single = Input(shape=(n_inputs_abs_MANTIS,), name = 'input')
        # HERE THE PREPROCESSING LAYER THAT DOES THE DIVISION AND ALL OF THAT 
        input_single_prepro = Pre_processingLayer_ABS_EMIS(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)
        # code for figuring out if self.model_bnn has batch normalization layers
        batch_norm_present = False
        for idx in range(len(self.model_bnn.layers)):
            if self.model_bnn.layers[idx].__class__.__name__ == 'BatchNormalization':
                batch_norm_present = True


        print('batch norm present')
        print(batch_norm_present)

        # NETWORK 1
        for i in range(num_networks):

            # input_single = Input(shape=(n_inputs,))
            # POSSIBLE TO PUT PREPROCESSING HERE WITH NO INPUT LAYER 
            # AND THEN CREATE A BIG INPUT LAYER FOR ALL THE NETWORKS
            #x = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)
            x = Dense(num_nodes, activation=act_fn)(input_single_prepro)
            #x = Dense(num_nodes, activation=act_fn)(x)
            if batch_norm_present:
                x = BatchNormalization()(x)
            for i in range(num_layers - 2):
                    x = Dense(num_nodes, activation=act_fn)(x)
                    if batch_norm_present:
                        x = BatchNormalization()(x)
            # layer that branches both outputs
            x = Dense(last_bayes_layer, activation=act_fn)(x)

            # the two outputs layers 
            # REGRESSION
            output_regression = Dense(n_outputs, activation = 'linear')(x)
            # omit the last node here

            # CLASSIFICATION
            output_classification = Dense(2, activation='softmax')(x)

            # CONCATENATE BOTH OUTPUTS
            model_output = concatenate([output_regression, output_classification])

            print(outputs_prepo)
            # ########################################
            # # This for post_processing every output
            # # POST PROCESSING LAYER
            # total_output = Post_processingLayer(outputs_prepo, outputs_min_params, outputs_scale_params)(model_output)
            
            # networks_list.append(total_output)
            # #networks_input_list.append(input_single)
            # ########################################

            ########################################
            # This for taking the average of the outputs in pre process form
            networks_list.append(model_output)
            ########################################

        # ########################################
        # # This for post_processing every output
        # merged_models = concatenate(networks_list, name = 'output')
        # ########################################
            
        ########################################
        # This for taking the average of the outputs in pre process form
        merged_models_n_networks = concatenate(networks_list)
        
        merged_models = Post_processingLayer_WITH_MEAN(outputs_prepo, outputs_min_params, \
                outputs_scale_params, num_networks, name = 'output')(merged_models_n_networks)
        ########################################


        model_merged = Model(inputs=input_single, outputs=merged_models)

        model_merged.compile(loss='mse', optimizer='adam')

        model_merged.summary()

        print(model_merged.layers)

        idx_og_model = 1

        #for idx in range(2, len(model_merged.layers) - (1 + num_networks), num_networks):
        some_number = 1
        if batch_norm_present:
            some_number = 0

        for idx in range(2, 2*(num_layers - 1)*num_networks + num_networks + 2 - some_number*num_networks*(num_layers - 1), num_networks):

            #idx_og_model = int((idx + 1)/num_networks - 1)

            for idx_layer in range(idx, idx+num_networks):

                if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

                    model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

                else:

                    sampled_weights = []
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
                    model_merged.layers[idx_layer].set_weights(sampled_weights)
                    print('denseflipout transmitted')
            
            idx_og_model += 1

        # print(model_merged.layers[idx])
        # print('layers')
        # print('idx if model og now')
        # print(idx_og_model)
            
        print("UNTIL HERE FINE")

        # putting weigths on multioutput layer
        for idx in range(2*(num_layers - 1)*num_networks + num_networks + 2 - some_number*num_networks*(num_layers - 1), \
            (2*(num_layers - 1)*num_networks + num_networks + 2) + 2*num_networks - some_number*num_networks*(num_layers - 1), 2):

            if self.model_bnn.layers[idx_og_model].__class__.__name__ == 'DenseFlipout':

                # this is for output bayesian
                sampled_weights = []
                sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
                sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
                model_merged.layers[idx].set_weights(sampled_weights)

            else:

                model_merged.layers[idx].set_weights(self.model_bnn.layers[idx_og_model].get_weights())


            model_merged.layers[idx+1].set_weights(self.model_bnn.layers[idx_og_model+1].get_weights())

            print(model_merged.layers[idx])


        if save_model:
            
            model_merged.save(self.model_path + '_GPU_opt_ensemble_DNN_{}'.format(num_networks))
            # also save model as .h5
            model_merged.save(self.model_path + '_GPU_opt_ensemble_DNN_{}_HDF5.h5'.format(num_networks), save_format='h5')
            # also 
            tf.saved_model.save(model_merged, self.model_path + '_GPU_opt_ensemble_DNN_{}_mantis'.format(num_networks))

        return model_merged
    
    def load_set_pre_process(self, set_label = 'TEST'):

        from pathlib import Path
        import pyarrow.parquet as pq

        data_folder = self.config_params['dataset']
        print("Getting data from folder: ", data_folder)

        if Path(data_folder + set_label + '.parquet').is_file():

          data_set_test = pq.read_table(data_folder + set_label + '.parquet').to_pandas()

          print("----PARQUET FILES FOUND-----")

        inputs_labels = self.config_params['inputs_labels']
        outputs_labels = self.config_params['outputs_labels']

        X_test = data_set_test[inputs_labels].values
        y_test = data_set_test[outputs_labels].values

        # make this self.config_params['inputs_prepro'] a numpy float array
        self.config_params['inputs_prepro'] = np.array(self.config_params['inputs_prepro'], dtype = np.float32)
        self.config_params['inputs_mean_params'] = np.array(self.config_params['inputs_mean_params'], dtype = np.float32)
        self.config_params['inputs_scale_params'] = np.array(self.config_params['inputs_scale_params'], dtype = np.float32)

        self.config_params['outputs_prepro'] = np.array(self.config_params['outputs_prepro'], dtype = np.float32)
        self.config_params['outputs_min_params'] = np.array(self.config_params['outputs_min_params'], dtype = np.float32)
        self.config_params['outputs_scale_params'] = np.array(self.config_params['outputs_scale_params'], dtype = np.float32)

        X_test = (np.power(X_test, 1/self.config_params['inputs_prepro']) - self.config_params['inputs_mean_params'])/self.config_params['inputs_scale_params']
        y_test = np.power(y_test/self.config_params['outputs_scale_params'] + self.config_params['outputs_min_params'], self.config_params['outputs_prepro'])

        return X_test, y_test


    def load_test_set(self):

        from pathlib import Path
        import pyarrow.parquet as pq

        data_folder = self.config_params['dataset']
        print("Getting data from folder: ", data_folder)

        if Path(data_folder + 'TEST.parquet').is_file():

          data_set_test = pq.read_table(data_folder + 'TEST.parquet').to_pandas()

          print("----PARQUET FILES FOUND-----")

        inputs_labels = self.config_params['inputs_labels']
        outputs_labels = self.config_params['outputs_labels']

        X_test = data_set_test[inputs_labels].values
        y_test = data_set_test[outputs_labels].values

        return X_test, y_test

    def ensemble_DNN_single_input_improved(self, num_networks = 8, include_prepro = True, save_model = True):

        num_networks = num_networks
  
        n_inputs = len(self.config_params['inputs_labels'])
        # get rid of one output 
        n_outputs = len(self.config_params['outputs_labels']) - 1
        num_layers = self.config_params['num_layers']
        num_nodes = self.config_params['num_nodes']
        last_bayes_layer = self.config_params['last_bayes_layer']

        inputs_prepo = self.config_params['inputs_prepro']
        inputs_mean_params = self.config_params['inputs_mean_params']
        inputs_scale_params = self.config_params['inputs_scale_params']

        outputs_prepo = self.config_params['outputs_prepro']
        outputs_min_params = self.config_params['outputs_min_params']
        outputs_scale_params = self.config_params['outputs_scale_params']

        # knowing how the output will be, we modify this to have the right number of nodes
        # the two last are the classification outcome 
        outputs_prepo[-1] = 1
        outputs_min_params[-1] = 0.0
        outputs_scale_params[-1] = 1.0
        # no add one more 
        outputs_prepo.append(1)
        outputs_min_params.append(0.0)
        outputs_scale_params.append(1.0)

        act_fn = self.config_params['activation_fn']

        networks_list = []

        networks_input_list = []

        input_single = Input(shape=(n_inputs,), name = 'input')
        input_single_prepro = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)

        # NETWORK 1
        for i in range(num_networks):

            # input_single = Input(shape=(n_inputs,))
            # POSSIBLE TO PUT PREPROCESSING HERE WITH NO INPUT LAYER 
            # AND THEN CREATE A BIG INPUT LAYER FOR ALL THE NETWORKS
            #x = Pre_processingLayer(inputs_prepo, inputs_mean_params, inputs_scale_params)(input_single)
            x = Dense(num_nodes, activation=act_fn)(input_single_prepro)
            #x = Dense(num_nodes, activation=act_fn)(x)
            x = BatchNormalization()(x)
            for i in range(num_layers - 2):
                    x = Dense(num_nodes, activation=act_fn)(x)
                    x = BatchNormalization()(x)
            # layer that branches both outputs
            x = Dense(last_bayes_layer, activation=act_fn)(x)

            # the two outputs layers 
            # REGRESSION
            output_regression = Dense(n_outputs, activation = 'linear')(x)
            # omit the last node here

            # CLASSIFICATION
            output_classification = Dense(2, activation='softmax')(x)

            # CONCATENATE BOTH OUTPUTS
            model_output = concatenate([output_regression, output_classification])


            print(outputs_prepo)

            # POST PROCESSING LAYER
            total_output = Post_processingLayer(outputs_prepo, outputs_min_params, outputs_scale_params)(model_output)
            
            networks_list.append(total_output)
            #networks_input_list.append(input_single)

        merged_models = concatenate(networks_list, name = 'output')

        model_merged = Model(inputs=input_single, outputs=merged_models)

        model_merged.compile(loss='mae', optimizer='adam')

        model_merged.summary()

        print(model_merged.layers)

        idx_og_model = 1

        #for idx in range(2, len(model_merged.layers) - (1 + num_networks), num_networks):
        for idx in range(2, 2*(num_layers - 1)*num_networks + num_networks + 2, num_networks):

            #idx_og_model = int((idx + 1)/num_networks - 1)

            for idx_layer in range(idx, idx+num_networks):

                if self.model_bnn.layers[idx_og_model].__class__.__name__ != 'DenseFlipout':

                    model_merged.layers[idx_layer].set_weights(self.model_bnn.layers[idx_og_model].get_weights())

                else:

                    sampled_weights = []
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].kernel_posterior.sample())
                    sampled_weights.append(self.model_bnn.layers[idx_og_model].bias_posterior.sample())
                    model_merged.layers[idx_layer].set_weights(sampled_weights)
                    print('denseflipout transmitted')
            
            idx_og_model += 1

        # print(model_merged.layers[idx])
        # print('layers')
        # print('idx if model og now')
        # print(idx_og_model)

        # putting weigths on multioutput layer
        for idx in range(2*(num_layers - 1)*num_networks + num_networks + 2, \
            (2*(num_layers - 1)*num_networks + num_networks + 2) + 2*num_networks, 2):

            model_merged.layers[idx].set_weights(self.model_bnn.layers[idx_og_model].get_weights())
            model_merged.layers[idx+1].set_weights(self.model_bnn.layers[idx_og_model+1].get_weights())

            print(model_merged.layers[idx])


        if save_model:
            
            model_merged.save(self.model_path + '_ensemble_DNN_{}'.format(num_networks))
            # also save model as .h5
            model_merged.save(self.model_path + '_ensemble_DNN_{}_HDF5.h5'.format(num_networks), save_format='h5')
            # also 
            tf.saved_model.save(model_merged, self.model_path + '_ensemble_DNN_{}_mantis'.format(num_networks))

        return model_merged



        



        


