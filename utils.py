import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from tensorflow import keras, shape
import tensorflow.keras.backend as K
from tensorflow.keras.layers import(LSTM, Dense, Input, Lambda, RepeatVector, TimeDistributed)
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model, Sequential



def set_train_test(df, batch_size = 4, time_steps = 16, retseq = True, train_ratio = 0.8,
    return_timestamp = False, scale = True, return_ordinal_index = False):

    number_of_features = df.shape[1]

    tempo_X_train_for_scaling = df.iloc[:int(df.shape[0] * train_ratio), :number_of_features]
    scaler = StandardScaler()
    scaler.fit(tempo_X_train_for_scaling)

    ordinal_index = np.arange(0, df.shape[0], 1)

    X = df.iloc[:-1, :number_of_features]
    X = X.iloc[:X.shape[0] - X.shape[0] % time_steps]

    stride = 1
    number_of_samples_to_generate = (X.shape[0] - time_steps) // stride

    X_1timestep = []
    y_1timestep = []
    timestamp = []
    ordinal_index_1timestep = []

    for i in range(number_of_samples_to_generate):
        X_1timestep.append(X[i: i + time_steps].values)

        if retseq:
            y_1timestep.append(X.iloc[i+1:i+time_steps+1].values.reshape(-1, number_of_features))
        else:
            y_1timestep.append(X.iloc[i + time_steps].values.reshape(-1, number_of_features))

        # Because I use timestamp for anomaly generation with VAE where I map X--> X,
        # I will slice the index according to X, not Y, therefore there exists the index -1
        # so that I won't slice the same index as Y
        if return_timestamp:
            timestamp.append(X.index[i+time_steps-1])
        
        ordinal_index_1timestep.append(ordinal_index[i+time_steps])

        if X[i: i+time_steps].shape[0] != time_steps:
            print('Shape mismatch')
    
    X_1timestep = np.array(X_1timestep)
    y_1timestep = np.array(y_1timestep)
    timestamp = np.array(timestamp)
    ordinal_index_1timestep = np.array(ordinal_index_1timestep)

    # Take care of residual part for batch training
    X_1timestep = X_1timestep[:X_1timestep.shape[0] - X_1timestep.shape[0] % batch_size]
    y_1timestep = y_1timestep[:y_1timestep.shape[0] - y_1timestep.shape[0] % batch_size]
    timestamp = timestamp[:timestamp.shape[0] - timestamp.shape[0] % batch_size]

    # Batch training for Stateful
    # Does not hurt non-Stateful models either
    X_batches = []
    y_batches = []
    ordinal_index_batches = []

    for i in range(X_1timestep.shape[0] - time_steps * batch_size): # -time_steps * batch_size prevents index going out of range
        for j in range(0, time_steps * batch_size, time_steps): # since we have i+j we do not lose any data while preventing case above
            X_batches.append(X_1timestep[i+j])
            y_batches.append(y_1timestep[i+j])
            ordinal_index_batches.append(ordinal_index_1timestep[i+j])

    X_1timestep = np.array(X_batches)
    y_1timestep = np.array(y_batches)
    ordinal_index_1timestep = np.array(ordinal_index_batches)
    
    train_size = int(X_1timestep.shape[0] * train_ratio)
    train_size = train_size - train_size % batch_size
    X_train = X_1timestep[:train_size]
    y_train = y_1timestep[:train_size]
    X_test = X_1timestep[train_size:]
    y_test = y_1timestep[train_size:]

    if scale:
        X_train = scaler.transform(X_train.reshape(-1, number_of_features)).reshape(-1, time_steps, number_of_features)
        if train_ratio != 1:
            X_test = scaler.transform(X_test.reshape(-1, number_of_features)).reshape(-1, time_steps, number_of_features)

        if retseq:
            y_train = scaler.transform(y_train.reshape(-1, number_of_features)).reshape(-1, time_steps, number_of_features)
            if train_ratio != 1:
                y_test = scaler.transform(y_test.reshape(-1, number_of_features)).reshape(-1, time_steps, number_of_features)

        else:
            y_train = scaler.transform(y_train.reshape(-1, number_of_features)).reshape(-1, 1, number_of_features)
            if train_ratio != 1:
                y_test = scaler.transform(y_test.reshape(-1, number_of_features)).reshape(-1, 1, number_of_features)

    objects_to_return = [X_train, y_train, X_test, y_test]

    if return_timestamp:
        objects_to_return += [timestamp[:train_size], scaler]

    if return_ordinal_index:
        objects_to_return.append(ordinal_index_1timestep[train_size:] + 1) # +1 because the indices in this array are created parallel to X but we need indices for y

    return objects_to_return


def create_lstm_model(batch_size, time_steps, number_of_features, return_sequences, stateful):
    lstm = Sequential()
    lstm.add(LSTM(number_of_features, batch_input_shape = (batch_size, time_steps, number_of_features),
                  return_sequences = return_sequences, stateful = stateful))
    lstm.add(Dense(number_of_features))
    return lstm


def create_lstm_vae_model(batch_size, time_steps, number_of_features, int_dim, latent_dim):
    def vae_sampling(args):
        z_mean, z_log_sigma = args
        batch_size = shape(z_mean)[0]
        latent_dim = shape(z_mean)[1]
        epsilon = K.random_normal(shape = (batch_size, latent_dim), mean = 0, stddev = 1)
        return z_mean + K.exp(z_log_sigma / 2) * epsilon
    
    # Encoder
    input_x = Input(shape = (time_steps, number_of_features))
    encoder_LSTM_int = LSTM(int_dim, return_sequences = True)(input_x)
    encoder_LSTM_latent = LSTM(latent_dim, return_sequences = False)(encoder_LSTM_int)

    z_mean = Dense(latent_dim)(encoder_LSTM_latent)
    z_log_sigma = Dense(latent_dim)(encoder_LSTM_latent)
    z_encoder_output = Lambda(vae_sampling, output_shape = (latent_dim,))([z_mean, z_log_sigma])

    encoder = Model(input_x, [z_mean, z_log_sigma, z_encoder_output])

    # Decoder
    decoder_input = Input(shape = (latent_dim))
    decoder_repeated = RepeatVector(time_steps)(decoder_input)
    decoder_LSTM_int = LSTM(int_dim, return_sequences = True)(decoder_repeated)
    decoder_LSTM = LSTM(number_of_features, return_sequences = True)(decoder_LSTM_int)
    decoder_dense1 = TimeDistributed(Dense(number_of_features * 2))(decoder_LSTM)
    decoder_output = TimeDistributed(Dense(number_of_features))(decoder_LSTM)
    decoder = Model(decoder_input, decoder_output)

    # VAE
    output = decoder(encoder(input_x)[2]) # this is the part encoder and decoder are connected together. Decoder takes the encoder output's[2] as input
    lstm_vae = keras.Model(input_x, output, name = 'lstm_vae')

    
    # Loss
    reconstruction_loss = mse(input_x, output)
    reconstruction_loss *= number_of_features
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    lstm_vae.add_loss(vae_loss)
    
    return encoder, decoder, lstm_vae #


    def all_metrics_together(y, y_hat):
        accuracy = metrics.accuracy_score(y, y_hat)
        recall = metrics.recall_score(y, y_hat)
        precision = metrics.precision_score(y, y_hat)
        f1 = metrics.f1_score(y, y_hat)

        df = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1': f1}, index = ['metric_value'])
        return df
