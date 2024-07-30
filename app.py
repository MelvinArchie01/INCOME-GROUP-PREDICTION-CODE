import streamlit as st
import pandas as pd




df = pd.read_csv("C:\\Users\\user\\Documents\\RENEWABLE ENERGY FOLDER\\WorldBank Renewable Energy Consumption_WorldBank Renewable Energy Consumption.csv")
# app title
st.title('RENEWABLE ENERGY INCOME GROUP PREDICTION MODEL')

#creating a paragraph

st.write('''  Renewable Energy Consumption has declined over the years with countries shifting to other sources of energy
                                            ''')

 
st.write(df.head(5)) #printing the first 5 rows


#having user slider

num_rows = st.slider("Select the number of rows", min_value = 1, max_value = len(df), value = 5)
st.write("Here are the rows you have selected in the Dataset")
st.write(df.head(num_rows)) #st.write is the print function in python
st.write('The number of rows and columns in the dataset')
st.write(df.shape)
st.write("number of duplicates:", df[df.duplicated()])

#------------------------------------------------------------------------------------------------------------
if st.checkbox('check for duplicates'):
   st.write(df[df.duplicated()])

if st.checkbox('total number of duplicates'):
   st.write(df.duplicated().sum())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Check for duplicates
if st.checkbox('Check for Duplicates'):
    st.write(f'Duplicates in DataFrame: {df.duplicated().sum()}')

#changing the dtype from string to integers

df['Year'] = pd.to_datetime(df['Year'])
df['Year'] = df['Year'].dt.year
def clean_outliers(column):
  mean = df[column].mean()
  std = df[column].std()
  threshold = 3
  lower_limit = mean - (threshold * std)
  upper_limit = mean + (threshold * std)

  return df[(df[column]>=lower_limit) & (df[column]<=upper_limit)]

columns = ['Year', 'Energy Consump.']
for column in columns:
  new_df = clean_outliers(column)

  # Drop 'Country Code' column
new_df = df.drop('Country Code', axis=1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# Display the dataframe
st.write("The new Dataset with the Diagnosis Column encoded:",new_df.head(3))

# Split the data into features and target
X = new_df.drop('Income Group', axis = 1)
y = new_df['Income Group']


####################################################################################################
# Encode categorical variables
encoded_columns = ['Country Name', 'Income Group', 'Indicator Code', 'Indicator Name', 'Region']
le_dict = {col: LabelEncoder() for col in encoded_columns}

for column in encoded_columns:
    le_dict[column].fit(new_df[column])
    new_df[column] = le_dict[column].transform(new_df[column])

# Encode target variable
le_target = LabelEncoder()
new_df['Income Group'] = le_target.fit_transform(new_df['Income Group'])

X = new_df.drop('Income Group', axis=1)
y = new_df['Income Group']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth= 30, min_samples_leaf= 1, min_samples_split= 2)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_encoded = le.fit_transform(y_train)
model.fit(X_train, y_train_encoded)
y_train_encoded = le.fit_transform(y_train)
model.fit(X_train, y_train_encoded)
y_test_encoded = le.fit_transform(y_test)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_encoded, y_pred)
st.write("Accuracy:", accuracy)


st.sidebar.write("## Enter new data for prediction")

Country_Name = st.sidebar.selectbox("Country Name", le_dict['Country Name'].classes_)

Indicator_Code = st.sidebar.selectbox("Indicator Code", le_dict['Indicator Code'].classes_)
Indicator_Name = st.sidebar.selectbox("Indicator Name", le_dict['Indicator Name'].classes_)
Region= st.sidebar.selectbox("Region", le_dict['Region'].classes_)

Energy_Consumption = st.sidebar.number_input("Energy Consumption")
Year = st.sidebar.number_input("Year")
# Encode user input
encoded_input = [
    le_dict['Country Name'].transform([Country_Name])[0],
   
    le_dict['Indicator Code'].transform([Indicator_Code])[0],
    le_dict['Indicator Name'].transform([Indicator_Name])[0],
    le_dict['Region'].transform([Region])[0],
    Year,
    Energy_Consumption
]
 


income_group_map = {
    0: "High income",
    1: "Low income",
    2: "Lower middle income",
    3: "Upper middle income"
}


if st.sidebar.button('Income Group'):
    prediction = model.predict([encoded_input])[0]
    predicted_income_group = income_group_map[prediction]
    st.sidebar.write('Income Group:', predicted_income_group)














    



















