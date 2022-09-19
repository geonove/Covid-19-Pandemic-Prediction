import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import datetime
import calendar 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pdb # Our loved pdb


# Class for the generation of the training and validation sets
class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df,
                 label_columns=None):

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df

        # Work out the label column indices.
        self.label_columns = label_columns
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features_):
        inputs = features_[:, self.input_slice, :]
        labels = features_[:, self.labels_slice, :]
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.label_width,
            shuffle=False,
            batch_size=1)

        return ds.map(self.split_window)

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)


# Compute the day of the week from the date
def findDay(date):
    date = '20' + date
    born = datetime.datetime.strptime(date, '%Y%m%d').weekday()
    return (calendar.day_name[born])


# Given the file create csv for all single regions
def parse_all_data(filename):
    with open("regions.json") as f:
        reg_name = json.load(f)

    # Create a dataset with only selected columns
    df = pd.read_csv(filename)
    df2 = df[['codice_regione', 'nuovi_positivi', 'totale_ospedalizzati', 'dimessi_guariti', 'deceduti']]
    df2.columns = ['region', 'newinfections', 'hospitalized', 'recovered', 'deceased']
    df2.at[22, 'region'] = 4

    # Compute the data and create folder for csv
    data = df["data"].to_numpy()[-1]
    data = data[2:4] + data[5:7] + data[8:10]
    folder = f'data_regions_{data}'
    if not folder in os.listdir():
        os.mkdir(folder)

    # Loop over all regions and save csv files
    for reg in range(1,22):
        region_rows = [i for i in range(len(df2['region'])) if df2['region'][i] == reg]
        region_df = df2.iloc[region_rows, 1:]
        region_name = reg_name[str(reg)]
        region_df.to_csv(f'./data_regions_{data}/' + region_name + '.csv')
    
    return data


# Function to smooth the time-series
def smoothing(data, log=True):
    
    smoothed_data = savgol_filter(data, 7, 1)
    if log: smoothed_data = np.log(smoothed_data+1)

    return pd.DataFrame(smoothed_data)


# Given time-series compute its increments
def encode(data, predictor):
    if predictor in ('deceased','recovered'):
        return smoothing(data.diff()[1:].to_numpy()).diff()[1:].to_numpy()
    if predictor in ('newinfections','hospitalized'):
        return smoothing(data).diff()[2:].to_numpy()


# Given increments compute its time-series
def decode(data, iv1, iv2, predictor):
    current = np.cumsum(data)

    if predictor in ('deceased','recovered'):
        current += np.log(iv1-iv2)
        current = (np.exp(current) - 1)
        current = np.cumsum(current)
        current += iv1
        return current.tolist()

    if predictor in ('newinfections','hospitalized'):
        # Last day of the 28-day window
        current += np.log(iv1)
        current = (np.exp(current) - 1)
        return current.tolist()


# Given data create numpy data for time-series and increments for every predictor
def preprocess(dataset, predictors, initial_skip):

    # Read only the desired predictor (discard 2 for differentiation)
    region_data = pd.DataFrame(dataset[predictors].iloc[-initial_skip:, :].to_numpy(), columns=predictors)
    aux = region_data[:-2].copy()

    # Apply smoothing function (filter + log) and then compute the daily increment
    for column in aux.columns:
        aux[column] = encode(region_data[column],column)
    smoothed_region = aux.copy()

    # Apply smoothing function to the daily increments (only filter)
    for column in smoothed_region.columns:
        smoothed_region[column] = smoothing(smoothed_region[column], log=False)

    return region_data, smoothed_region


# Split all data in windows containing train and validation part
def data_split(region_data, smoothed_region, par, split=0.25):

    # percentage reserved to validation set
    val_size = int(len(region_data) * split / 7) * 7
    region_data = region_data.iloc[2:].reset_index().iloc[:,1:]

    # Original data (only for plot purposes)
    X_train_raw = region_data.iloc[:-val_size] if split else region_data
    X_valid_raw = region_data.iloc[-val_size:] if split else region_data

    # Training data
    X_train = smoothed_region.iloc[:- val_size] if split else smoothed_region
    X_valid = smoothed_region.iloc[- val_size:] if split else smoothed_region

    # Smoothed (filter + log) validation data to reconstruct the predictions from the daily increment
    X_valid_smooth = X_valid_raw.copy()
    for column in X_valid_raw.columns:
        X_valid_smooth[column] = smoothing(X_valid_raw[column], log=False).to_numpy()

    # Generation of the training dataset
    w1 = WindowGenerator(input_width=par["window"], label_width=par["telescope"] , shift=par["stride"] ,
                         train_df=X_train, val_df=X_valid, label_columns=[c for c in region_data])

    # Generation of another dataset needed only for prediction reconstruction
    w2 = WindowGenerator(input_width=par["window"], label_width=par["telescope"] , shift=par["stride"] ,
                         train_df=X_train, val_df=X_valid_smooth, label_columns=[c for c in region_data])

    return (X_train_raw, X_valid_raw), (X_train, X_valid), (w1, w2)


# Plotter used for train-valid split of time-series and increments
def plot_data(data_raw, data, predictors, region_name, save):
    
    # Plot of training data and original data
    for predictor in predictors:
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(data_raw[0][predictor], label='Train data')
        axs[0].plot(data_raw[1][predictor], label='Valid data')
        axs[0].legend()
        axs[0].set_title(f'Original data for {predictor} in {region_name}')

        axs[1].plot(data[0][predictor], label='Train data')
        axs[1].plot(data[1][predictor], label='Valid data')
        axs[1].legend()
        axs[1].set_title(f'Encoded data for {predictor} in {region_name}')
        
        fig.tight_layout() 
        if save: plt.savefig(f"images\data\{region_name}_{predictor}")

    plt.show()


# Train the model given the windows
def compile_and_fit(model, window_gen, epochs, pat):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=pat,
                                                      mode='min',
                                                      restore_best_weights=True)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    hist = model.fit(window_gen.train,
                     validation_data=window_gen.val ,
                     callbacks=[early_stopping],
                     epochs=epochs)
    return hist


# given the trained model generate predictions for the validation set
def postprocess(w, model, predictors):

    total_predictions = list()
    for e1, e2 in zip(w[0].val, w[1].val):
        output = model.predict(e1)
        for k, predictor in enumerate(predictors):
            prediction = decode(output[:, :, k], e2[0][0, -1, k].numpy(), e2[0][0, -2, k].numpy(), predictor)

            if len(total_predictions) < len(predictors):
                total_predictions.append(prediction)
            else:
                total_predictions[k] += prediction

    return total_predictions


# Visualize the results on the validation set
def plot_results(X_valid, predictions, region_name, par, save):

    for idx, predictor in enumerate(X_valid.columns):
        
        plt.figure()
        # Plot real validation data
        plt.plot(X_valid[predictor].to_numpy(), label="smooth data", color = "blue")

        # Plot predictions on validation data
        position = par["window"] + par["stride"] - par["telescope"] 
        predicted_days = list(range(position, position + len(predictions[idx])))
        plt.plot(predicted_days, predictions[idx], label='predictions', color='red')

        # Plot vertical lines every 7 days
        weeks = list(range(0, len(X_valid)+7, 7))
        plt.vlines(weeks, colors='gray', alpha=0.5,
                   ymin=min(X_valid[predictor]), 
                   ymax=max(X_valid[predictor]))

        plt.title(f"Predictions for {predictor} in {region_name}")
        plt.legend()
        if save: plt.savefig(f"images\pred\{region_name}_{predictor}")

    plt.show()


# Create predictions for the test-set (non available data)
def predict_future(dataset, predictors, model, window):
    
    data = dataset[predictors].iloc[- window - 2:]
    iv2 = data.iloc[-2]
    iv1 = data.iloc[-1]

    aux = data[:-2].copy()

    for column in data.columns:
        aux[column] = encode(data[column], column)

    output = model.predict(np.expand_dims(aux[predictors].to_numpy(), axis=0))

    total_predictions = list()
    for k, predictor in enumerate(predictors):
        prediction = decode(output[:, :, k], iv1[predictor], iv2[predictor], predictor)

        if len(total_predictions) < len(predictors):
            total_predictions.append(prediction)
        else:
            total_predictions[k] += prediction

    return np.array(total_predictions, int)


# Decide which data should be used 
def initialize_data(filename):
    
    # Last day of training
    last_day = "220504"
    generate_data = False

    # Creating the dataset from the csv file
    if filename is not None and generate_data: 
        last_day = parse_all_data(filename)

    # Discarded days
    n_weeks = 80
    initial_skip = 7*n_weeks+2

    return last_day, initial_skip


# Select regions and predictor and chose configuration
def initialize_model():

    # Chosen regions
    regions = list()
    regions.append("Lombardia")
    #regions.append("Lazio")
    #regions.append("Sicilia")

    # Chosen predictors
    predictors = list()
    predictors.append("newinfections")
    predictors.append("hospitalized")
    predictors.append("deceased")
    predictors.append("recovered")

    # Configuration for window generation and training

    # Predict all 7 days
    config_1 = {"telescope": 7, "stride": 7, "window": 28, "epochs": 200, "patience": 10, "save_par": 50}
    # Predict the 1st day
    config_2 = {"telescope": 1, "stride": 1, "window": 28, "epochs": 50,  "patience": 5,  "save_par": 20}
    # Train full data
    config_3 = {"telescope": 7, "stride": 7, "window": 21, "epochs": 30, "patience": 100, "save_par": 10}

    return regions, predictors, config_1


# Main function of the project
def main():

    # Initialize parameters for model and training
    last_day, initial_skip = initialize_data('dpc-covid19-ita-regioni.csv')
    regions, predictors, params = initialize_model()
    only_test = True

    # Loop over the single regions
    for region_name in regions:

        # Reading the dataset of the current region
        dataset = pd.read_csv(f'./data_regions_{last_day}/' + region_name + '.csv')
        # Preprocessing (smoothing and differentiation)
        region_data, smoothed_region = preprocess(dataset, predictors, initial_skip)
        # Train and Validation windows generation
        data_raw, data, w = data_split(region_data, smoothed_region, params, split = 0.35)
        # Plot original and encoded data
        plot_data(data_raw, data, predictors, region_name, params["save_par"])

        # Definition of the network
        tfkl = tf.keras.layers
        lstm_model = tf.keras.models.Sequential([
            tfkl.LSTM(256, return_sequences=False),
            tfkl.Dense(units = params["telescope"] * len(predictors)),
            tfkl.Reshape((params["telescope"] , len(predictors)))
        ])

        # Train, postprocess and plot of the results
        compile_and_fit(lstm_model, w[0], params["epochs"], params["patience"])
        predictions = postprocess(w, lstm_model, predictors)
        plot_results(data_raw[1], predictions, region_name, params, params["save_par"])

        # Predictions
        export_data = predict_future(dataset, predictors, lstm_model, params["window"])
        df = pd.DataFrame(export_data.T, columns=predictors)
        df.to_csv(f'predictions\{region_name}_predictions.csv')

    print("Predictions for 7 days after", findDay(last_day), last_day)

if __name__ == '__main__': main()
