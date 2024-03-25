import pandas as pd
import numpy as np

# Load the train data
train_data = pd.read_excel('./Dataset - Copy/first.train.xlsx')

# This will fill missing values with the mean of the column
train_data = train_data.fillna(train_data.mean())

# Remove duplicates
train_data = train_data.drop_duplicates()

# List of all test data files
test_data_files = [
    './Dataset - Copy/first.test.xlsx',
    './Dataset - Copy/test.second.xlsx',
    './Dataset - Copy/third.test.xlsx',
    './Dataset - Copy/test.fourth.xlsx',
    './Dataset - Copy/test.fifth.xlsx',
    './Dataset - Copy/test.sixth.xlsx',
    './Dataset - Copy/test.seventh..xlsx',
    './Dataset - Copy/test.eight.xlsx',
    './Dataset - Copy/test.nine.xlsx',
    './Dataset - Copy/test.tenth.xlsx',
    './Dataset - Copy/test.eleventh.xlsx',
    './Dataset - Copy/test.tewelth.xlsx',
    './Dataset - Copy/test.thirteen.xlsx',
    './Dataset - Copy/test.fourteen.xlsx',
    './Dataset - Copy/test.fifteen.xlsx',
    './Dataset - Copy/test.sixteen.xlsx',
    './Dataset - Copy/test.seventeen.xlsx',
    './Dataset - Copy/test.eighteen.xlsx',
    './Dataset - Copy/test.nineteen.xlsx',
    './Dataset - Copy/test.twenty.xlsx',
    './Dataset - Copy/test.twentyone.xlsx',
    './Dataset - Copy/test.twentytwo.xlsx',
    './Dataset - Copy/test.twentythree.xlsx',
    './Dataset - Copy/test.twentyfour.xlsx',
    './Dataset - Copy/test.twentyfive.xlsx'
]

# Load and preprocess all test datasets
test_datasets = []
for file in test_data_files:
    data = pd.read_excel(file)
    data = data.fillna(data.mean())
    data = data.drop_duplicates()
    test_datasets.append(data)
# Split the data into features and target
X_train = train_data.rename(columns = {'f':'Floor Number'}, inplace = True)
X_train = train_data.drop(['id','timestamp','x', 'y','Labels','Actual_labels'], axis=1)
y_train = train_data[['x', 'y']]

X_tests = []
y_tests = []

for data in test_datasets:
    
    X_test = data.loc[:, 'WAP001':'WAP620'].copy()
    X_test['Floor Number'] = data['Floor Number']

    y_test = data[['x', 'y']]
    X_tests.append(X_test)
    y_tests.append(y_test)
from sklearn.neighbors import KNeighborsRegressor
# Train a KNN model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
#Make Predictions
predictions = []

for X_test in X_tests:
    prediction = pd.DataFrame(model.predict(X_test))
    predictions.append(prediction)
#calculate error
errors = []
ALEs = []

for i in range(len(y_tests)):
    error = np.sqrt((predictions[i][0] - y_tests[i]['x'])**2 + (predictions[i][1] - y_tests[i]['y'])**2)
    ALE = error.mean(axis=0)
    errors.append(error)
    ALEs.append(ALE)
print(ALEs)
ALEERROR = pd.DataFrame(ALEs, columns=['ALE'])
ALEERROR.to_excel('ALE_25_months.xlsx', index=False)
