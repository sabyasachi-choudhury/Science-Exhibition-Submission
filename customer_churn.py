import pandas as pd
from tensorflow import keras

ds = pd.read_csv('churn_dataset.csv', sep=',').drop(['customerID'], axis=1)
conv_dict = {
    'Churn': {'Yes': 1, 'No': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'PaperlessBilling': {'Yes': 1, 'No': 0}
}
for col in ds.columns:
    if col not in ['MonthlyCharges', 'TotalCharges', 'tenure']:
        if col not in conv_dict.keys():
            conv_dict[col] = {}
            for elem in ds[col].unique():
                conv_dict[col][elem] = len(conv_dict[col].keys())

drop_ind = []
for col in ['MonthlyCharges', 'TotalCharges', 'tenure']:
    for i, elem in enumerate(ds[col]):
        try:
            x = float(elem)
        except ValueError:
            drop_ind.append(i)
ds.drop(drop_ind, axis=0, inplace=True)

for col in ds.columns:
    if col not in ['MonthlyCharges', 'TotalCharges', 'tenure']:
        ds[col] = ds[col].apply(lambda x: conv_dict[col][x])
    else:
        ds[col] = ds[col].apply(lambda x: float(x))

train_x = ds.iloc[:6743, :-1].to_numpy()
train_y = ds.iloc[:6743, -1].to_numpy()
test_x = ds.iloc[6743:, :-1].to_numpy()
test_y = ds.iloc[6743:, -1].to_numpy()

TRAIN = False

if TRAIN:
    model = keras.models.Sequential([
        keras.layers.Dense(19, input_shape=[19], activation=keras.activations.elu),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=1000, epochs=500, validation_split=0.05)
    model.summary()
    model.save('models/churn_2')

else:
    for i in range(100):
        print(i, train_x[i], train_y[i])

    DATA_IND_TO_PREDICT = int(input("enter index of test data to predict"))

    model = keras.models.load_model("models/churn_1")
    predictions = model.predict(train_x[DATA_IND_TO_PREDICT:DATA_IND_TO_PREDICT+1])
    print(f"Probability of customer churning: {round(predictions[0][0]*1000)/10} %")