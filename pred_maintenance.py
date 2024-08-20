import numpy as np
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt
import random
import time
import pickle

# consts
TRAIN, TEST, CREATE = False, True, False

if CREATE:

    # loading
    ds_train = pd.read_csv('turbofan_ds/PM_train.txt', sep=' ', header=None).dropna(axis=1)
    ds_train_raw = ds_train.copy(deep=True)
    ds_test = pd.read_csv('turbofan_ds/PM_test.txt', sep=' ', header=None).dropna(axis=1)
    ds_truth = pd.read_csv('turbofan_ds/PM_truth.txt', sep=' ', header=None).dropna(axis=1)

    # columns for test and train
    col_names = ['id', 'cycle', 'os1', 'os2', 'os3',
                 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    ds_train.columns = col_names
    ds_test.columns = col_names
    ds_train_raw.columns = col_names

    # adding ids for truth as ids are sorted in order
    ds_truth.columns = ['rul']
    ds_truth['id'] = ds_truth.index + 1

    # Creating time to failure, which is just the final cycle timestamp
    ds_train['ttf'] = ds_train.groupby(by='id')['cycle'].transform(max) - ds_train['cycle']

    # creating time to failure for test snapshots from truth ds
    total_life = ds_test.groupby('id')['cycle'].max().reset_index()['cycle'] + ds_truth['rul']
    temp = []
    for i, x in enumerate(ds_test['cycle']):
        c_id = ds_test.loc[i, 'id']
        temp.append(total_life[c_id - 1] - x)
    ds_test['ttf'] = temp

    # Turning into labels
    PERIOD = 15

    ds_train['label'] = ds_train['ttf'].transform(lambda x: 1 if x <= PERIOD else 0)
    ds_test['label'] = ds_test['ttf'].transform(lambda x: 1 if x <= PERIOD else 0)

    to_drop = []
    for col in ds_train.columns:
        if len(ds_train[col].unique()) == 1:
            to_drop.append(col)
    ds_train.drop(columns=to_drop, inplace=True)
    ds_test.drop(columns=to_drop, inplace=True)
    ds_train_raw.drop(columns=to_drop, inplace=True)

    # Data scaling
    def scale_func(a_min, a_max, l, r):
        def func(x):
            return (x - a_min) * (r - l) / (a_max - a_min) + l

        return func


    for ds in [ds_train, ds_test]:
        for i, col in enumerate(ds.columns[2:-2]):
            ds[str(col)] = ds[str(col)].transform(scale_func(ds[str(col)].min(), ds[str(col)].max(), 0, 1))

    # Generating TimeSequences for LSTM
    def gen_sequences(df, seq_size, seq_cols):
        start_padding = pd.DataFrame(np.zeros((seq_size - 1, df.shape[1])), columns=df.columns)
        df = pd.concat([start_padding, df])
        used_data = df[seq_cols].to_numpy(dtype=np.float32)
        output = []
        for start, stop in zip(range(0, used_data.shape[0] - seq_size), range(seq_size, used_data.shape[0])):
            output.append(used_data[start:stop, :])
        return np.array(output)


    # X and Y for training and testing
    SEQ_LENGTH = 50
    x_train = np.concatenate(
        [gen_sequences(ds_train[ds_train['id'] == id], SEQ_LENGTH, ds_train.columns[2:-2]) for id in
         ds_train['id'].unique()])
    y_train = np.concatenate([gen_sequences(ds_train[ds_train['id'] == id], SEQ_LENGTH, ['label']) for id in
                              ds_train['id'].unique()]).max(axis=1)

    x_test = np.concatenate([gen_sequences(ds_test[ds_test['id'] == id], SEQ_LENGTH, ds_test.columns[2:-2]) for id in
                             ds_test['id'].unique()])
    y_test = np.concatenate([gen_sequences(ds_test[ds_test['id'] == id], SEQ_LENGTH, ['label']) for id in
                             ds_test['id'].unique()]).max(axis=1)

    with open("pred_maintenance_data.pkl", "wb") as file:
        pickle.dump({"x_train": x_train,
                     "y_train": y_train,
                     "x_test": x_test,
                     "y_test": y_test,
                     "ds_train_raw": ds_train_raw},
                    file)

# Model Building
if TRAIN:
    num_features = x_train.shape[2]

    model = keras.models.Sequential([
        keras.layers.LSTM(input_shape=[SEQ_LENGTH, num_features], units=100, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(units=SEQ_LENGTH, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, epochs=10, batch_size=200, validation_split=0.05)
    model.save('models/pred_maintenance_1')

if TEST:
    with open("pred_maintenance_data.pkl", "rb") as file:
        data_dict = pickle.load(file)
        x_train, y_train, x_test, y_test, ds_train_raw = [data_dict[k] for k in data_dict.keys()]

    print("Data Loaded")
    model = keras.models.load_model("models/pred_maintenance_1")
    print("Model Loaded")
    while True:
        machine, timestamp = (input("Machine and day: ")).split(sep=', ')
        machine, timestamp = int(machine), int(timestamp)
        ind = len(ds_train_raw[ds_train_raw['id'] < machine]) + timestamp - 1
        predictions = model.predict(x_train[ind:ind+1])
        print(f"{predictions[0][0]//0.001/10}% chance of failure in next 15 days")
    # Code to make new plot for any desired machine from train_data

    # plt.rcParams["figure.figsize"] = (12.5, 7.5)
    # MACHINE_ID = 1
    # for i, col in enumerate(ds_train_raw.columns[4:]):
    #     plot_data = ds_train_raw[ds_train_raw['id'] == MACHINE_ID][col]
    #     plt.subplot(3, 5, i+1)
    #     plt.plot(range(1, len(plot_data)+1), plot_data, f"{random.choice(list('bgrcmk'))}.-")
    #     plt.title(f"{col} for machine {MACHINE_ID}")
    #     plt.xlabel("days")
    #     plt.ylabel(f"{col} reading")
    # plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.savefig(f"machine_{MACHINE_ID}_plot.png")
    # plt.show()
