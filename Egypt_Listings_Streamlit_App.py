import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.tools import mpl_to_plotly
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.io as pio
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, minmax_scale, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import TransformedTargetRegressor
import pickle
from wael_lib import *
import ipywidgets as widgets
from IPython.display import display, clear_output
import streamlit as st

pyo.init_notebook_mode(connected=True)
pio.renderers.default = "notebook_connected"
pio.templates.default = 'plotly_dark'

data = pd.read_csv('egypt_real_estate_listings.csv')
df = data.copy()

# Replace missing values with "0" in column: 'price'
df = df.fillna({'price': "0"})
# Replace missing values with "unknown" in column: 'description'
df = df.fillna({'description': "unknown"})
# Replace missing values with "unknown,unknown" in column: 'location'
df = df.fillna({'location': "unknown,unknown"})
# Replace missing values with "unknown" in column: 'type'
df = df.fillna({'type': "unknown"})
# Replace missing values with "0" in column: 'size'
df = df.fillna({'size': "0"})
# Replace missing values with "0" in column: 'bedrooms'
df = df.fillna({'bedrooms': "0"})
# Replace missing values with "0" in column: 'bathrooms'
df = df.fillna({'bathrooms': "0"})
# Replace missing values with "0" in column: 'available_from'
df = df.fillna({'available_from': "0"})
# Replace missing values with "unknown" in column: 'payment_method'
df = df.fillna({'payment_method': "unknown"})
# Drop column: 'down_payment'
df = df.drop(columns=['down_payment'])

# Replace all instances of "," with "" in column: 'price'
df['price'] = df['price'].str.replace(",", "", case=False, regex=False)
# Derive column 'Size_sqm' from column: 'size'
df.insert(6, "Size_sqm", df["size"].str.split(" ").str[3])
# Drop column: 'size'
df = df.drop(columns=['size'])
# Rename column 'Size_sqm' to 'size_sqm'
df = df.rename(columns={'Size_sqm': 'size_sqm'})
# Derive column 'num_bedrooms' from column: 'bedrooms'
df.insert(7, "num_bedrooms", df["bedrooms"].str.split("+").str[0])
# Split text using string '+' in column: 'bedrooms'
loc_0 = df.columns.get_loc('bedrooms')
df_split = df['bedrooms'].str.split(pat='+', expand=True, n=1).add_prefix('bedrooms_')
df = pd.concat([df.iloc[:, :loc_0], df_split, df.iloc[:, loc_0:]], axis=1)
df = df.drop(columns=['bedrooms'])
# Replace all instances of "studio" with "1" in column: 'bedrooms_0'
df['bedrooms_0'] = df['bedrooms_0'].str.replace("studio", "1", case=False, regex=False)
# Rename column 'bedrooms_0' to 'bedrooms'
df = df.rename(columns={'bedrooms_0': 'bedrooms'})
# Rename column 'bedrooms_1' to 'with_maid'
df = df.rename(columns={'bedrooms_1': 'with_maid'})
# Replace all instances of "Maid" with "1" in column: 'with_maid'
df['with_maid'] = df['with_maid'].str.replace("Maid", "1", case=False, regex=False)
# Replace missing values with "0" in column: 'with_maid'
df = df.fillna({'with_maid': "0"})
# Drop column: 'num_bedrooms'
df = df.drop(columns=['num_bedrooms'])
# Split text using string ',' in column: 'location'
loc_0 = df.columns.get_loc('location')
df_split = df['location'].str.split(pat=',', expand=True).add_prefix('location_')
df = pd.concat([df.iloc[:, :loc_0], df_split, df.iloc[:, loc_0:]], axis=1)
df = df.drop(columns=['location'])
# Drop column: 'location_3'
df = df.drop(columns=['location_3'])
# Drop column: 'location_4'
df = df.drop(columns=['location_4'])
# Rename column 'location_0' to 'compound_name'
df = df.rename(columns={'location_0': 'compound_name'})
# Rename column 'location_1' to 'region'
df = df.rename(columns={'location_1': 'region'})
# Rename column 'location_2' to 'city'
df = df.rename(columns={'location_2': 'city'})
# Replace missing values with "0" in column: 'size_sqm'
df = df.fillna({'size_sqm': "0"})
# Replace missing values with "unknown" in column: 'city'
df = df.fillna({'city': "unknown"})
# Move 'price' column to the last position
columns = df.columns.tolist()
columns.append(columns.pop(columns.index('price')))
df = df[columns]
# Move 'available_from' column to the first position
columns = ['available_from'] + [col for col in df.columns if col != 'available_from']
df = df[columns]

df['available_from'] = pd.to_datetime(df['available_from'], format='%d %b %Y', errors='coerce')
# Replace missing values with the most common value of each column in: 'available_from'
df.fillna({'available_from': df['available_from'].mode()[0]}, inplace=True)
# Convert column 'size_sqm' to integer
df['size_sqm'] = df['size_sqm'].astype(str).str.replace(',', '')
df['size_sqm'] = df['size_sqm'].astype(int)
df['bedrooms'] = df['bedrooms'].astype(int)
#df['with_maid'] = df['with_maid'].astype(int)
#df['bathrooms'] = df['bathrooms'].astype(int)
df['price'] = df['price'].astype(int)
# Replace all instances of "+ " with "" in column: 'with_maid'
df['with_maid'] = df['with_maid'].astype(str).str.replace("+ ", "", case=False, regex=False)
# Replace all instances of " " with "" in column: 'with_maid'
df['with_maid'] = df['with_maid'].astype(str).str.replace(" ", "", case=False, regex=False)
df['with_maid'] = df['with_maid'].replace('', '0')
df['with_maid'] = df['with_maid'].astype(int)
# Replace all instances of "+" with "" in column: 'bathrooms'
df['bathrooms'] = df['bathrooms'].astype(str).str.replace("+", "", case=False, regex=False)
df['bathrooms'] = df['bathrooms'].fillna('0')
df['bathrooms'] = df['bathrooms'].replace('none', '0')
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce').fillna(0).astype(int)
# Remove leading and trailing whitespace
df['compound_name'] = df['compound_name'].str.strip()
df['type'] = df['type'].str.strip()
df['region'] = df['region'].str.strip()
df['city'] = df['city'].str.strip()
df['payment_method'] = df['payment_method'].str.strip()

df['price'] = df['price'].replace(0, df['price'].mean()).astype(int)
df['bathrooms'] = df['bathrooms'].replace(0, df['bathrooms'].mode()[0]).astype(int)
df['bedrooms'] = df['bedrooms'].replace(0, df['bedrooms'].mode()[0]).astype(int)
df['size_sqm'] = df['size_sqm'].replace(0, df['size_sqm'].mean()).astype(int)

numerical_cols = ['size_sqm', 'bedrooms', 'with_maid', 'bathrooms', 'price']

# Handle outliers in 'price'
wdh = WDataHandler()
df['price'] = wdh.handle_outliers(df, 'price')

df['size_sqm'] = wdh.handle_outliers(df, 'size_sqm')

df['price_per_sqm'] = df['price'] / df['size_sqm']
df['price_per_sqm'] = wdh.handle_outliers(df, 'price_per_sqm')
df['price_per_sqm'] = df['price_per_sqm'].astype(int)
px.box(df, y='price_per_sqm')

region_price_per_sqm = df.groupby(['region', 'city', 'type'])['price_per_sqm'].mean().reset_index().sort_values(by='price_per_sqm', ascending=False)

types_list = df['type'].unique().tolist()
region_list = df['region'].unique().tolist()

x = df[['region','size_sqm', 'payment_method', 'type']]

y = df['price_per_sqm']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

cat_cols = ['region','type','payment_method']
num_cols = ['size_sqm']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# Define model pipeline with TransformedTargetRegressor and Y scaling using StandardScaler
model = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(n_neighbors=5))
    ]),
    transformer=StandardScaler()
)
model.fit(x_train, y_train)
# Title and description
st.title("Egypt Real Estate Price Estimator")
st.markdown("Estimate property prices per square meter and explore price distribution in selected regions.")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

region_input = st.sidebar.selectbox("Region:", options=region_list, index=region_list.index('5th Settlement Compounds') if '5th Settlement Compounds' in region_list else 0)
size_input = st.sidebar.number_input("Size (sqm):", value=120, min_value=1)
type_input = st.sidebar.selectbox("Unit Type:", options=types_list, index=types_list.index('Apartment') if 'Apartment' in types_list else 0)
payment_method_input = st.sidebar.selectbox("Payment Method:", options=['Cash', 'Installments'], index=1)

# Predict button
if st.sidebar.button("Estimate Price"):
    sample_data = pd.DataFrame({
        'region': [region_input],
        'size_sqm': [size_input],
        'type': [type_input],
        'payment_method': [payment_method_input]
    })
    predicted_price = model.predict(sample_data)
    st.header("Estimated Price")
    st.markdown(f'<div style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 15px; text-align: center; font-size: 18px; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"><strong>Estimated Price per sqm</strong>: {predicted_price[0]:,.0f} EGP</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background-color: #2196F3; color: white; padding: 10px; border-radius: 15px; text-align: center; font-size: 18px; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"><strong>Estimated Total Price</strong>: {predicted_price[0] * size_input:,.0f} EGP</div>', unsafe_allow_html=True)

# Average prices dataframe
st.header(f"Average Prices per sqm in {region_input}")
average_prices = df[((df['region'].str.contains(region_input)) | (df['city'].str.contains(region_input))) & (df['type'].str.contains(type_input))].groupby('compound_name')['price_per_sqm'].mean().reset_index()

median_price = average_prices['price_per_sqm'].median()
mean_price = average_prices['price_per_sqm'].mean()
min_price = average_prices['price_per_sqm'].min()
max_price = average_prices['price_per_sqm'].max()
num_compounds = average_prices.count()[0]
highest_compounds = average_prices.sort_values(by='price_per_sqm', ascending=False).head(3)['compound_name'].tolist()
lowest_compounds = average_prices.sort_values(by='price_per_sqm', ascending=True).head(3)['compound_name'].tolist()
st.markdown(f"""
### Summary Statistics
- **Median**: {median_price:,.0f} EGP
- **Mean**: {mean_price:,.0f} EGP
- **Min**: {min_price:,.0f} EGP
- **Max**: {max_price:,.0f} EGP
- **Number of Compounds**: {num_compounds}
- **Highest Average Prices**: {', '.join(highest_compounds)}
- **Lowest Average Prices**: {', '.join(lowest_compounds)}
""")
st.dataframe(average_prices.sort_values(by='price_per_sqm', ascending=False).reset_index(drop=True))
# Charts
st.header("Charts")

# Highest average prices
fig1 = px.bar(average_prices.sort_values(by='price_per_sqm', ascending=False).head(10), 
              x='compound_name', y='price_per_sqm', 
              title=f'Highest Average Prices per sqm in {region_input}', 
              labels={'compound_name': 'Compound Name', 'price_per_sqm': 'Average Price per sqm (EGP)'})
st.plotly_chart(fig1)

# Lowest average prices
fig2 = px.bar(average_prices.sort_values(by='price_per_sqm', ascending=True).head(10), 
              x='compound_name', y='price_per_sqm', 
              title=f'Lowest Average Prices per sqm in {region_input}', 
              labels={'compound_name': 'Compound Name', 'price_per_sqm': 'Average Price per sqm (EGP)'})
st.plotly_chart(fig2)

# Q1 to Q3 average prices
q1 = average_prices['price_per_sqm'].quantile(0.25)
q3 = average_prices['price_per_sqm'].quantile(0.75)
q1_q3_properties = average_prices[(average_prices['price_per_sqm'] >= q1) & (average_prices['price_per_sqm'] <= q3)]
fig3 = px.bar(q1_q3_properties.sort_values(by='price_per_sqm'), 
              x='compound_name', y='price_per_sqm', 
              title=f'Average Prices per sqm in {region_input} (Q1 to Q3)', 
              labels={'compound_name': 'Compound Name', 'price_per_sqm': 'Average Price per sqm (EGP)'})
st.plotly_chart(fig3)

# Price distribution histogram
fig4 = px.histogram(df[((df['region'].str.contains(region_input)) | (df['city'].str.contains(region_input))) & (df['type'].str.contains(type_input))], x='price_per_sqm', nbins=30, 
                    title=f'Price Distribution per sqm in {region_input}', 
                    labels={'price_per_sqm': 'Price per sqm (EGP)'}, marginal='box')
st.plotly_chart(fig4)

# ECDF in probability scale
fig5 = px.ecdf(df[((df['region'].str.contains(region_input)) | (df['city'].str.contains(region_input))) & (df['type'].str.contains(type_input))], x='price_per_sqm', 
               title=f'Probability % of Price per sqm in {region_input}', 
               labels={'price_per_sqm': 'Price per sqm (EGP)'}, ecdfnorm='percent')
fig5.update_yaxes(title_text='Probability (%)')
st.plotly_chart(fig5)

# ECDF in complementary probability scale
fig6 = px.ecdf(df[((df['region'].str.contains(region_input)) | (df['city'].str.contains(region_input))) & (df['type'].str.contains(type_input))], x='price_per_sqm', 
               title=f'Complementary Probability % of Price per sqm in {region_input}', 
               labels={'price_per_sqm': 'Price per sqm (EGP)'}, ecdfnorm='percent', ecdfmode='complementary')
fig6.update_yaxes(title_text='Complementary Probability (%)')
st.plotly_chart(fig6)