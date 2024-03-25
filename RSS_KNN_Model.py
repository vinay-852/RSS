import pandas as pd
import numpy as np

# Load the train data
train_data = pd.read_excel('/workspaces/RSS/Dataset - Copy/first.test.xlsx')

# This will fill missing values with the mean of the column
train_data = train_data.fillna(train_data.mean())

# Remove duplicates
train_data = train_data.drop_duplicates()

# List of all test data files
test_data_files = [
    '/workspaces/RSS/Dataset - Copy/first.test.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.second.xlsx',
    '/workspaces/RSS/Dataset - Copy/third.test.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.fourth.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.fifth.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.sixth.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.seventh..xlsx',
    '/workspaces/RSS/Dataset - Copy/test.eight.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.nine.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.tenth.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.eleventh.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.tewelth.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.thirteen.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.fourteen.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.fifteen.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.sixteen.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.seventeen.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.eighteen.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.nineteen.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.twenty.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.twentyone.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.twentytwo.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.twentythree.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.twentyfour.xlsx',
    '/workspaces/RSS/Dataset - Copy/test.twentyfive.xlsx'
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
