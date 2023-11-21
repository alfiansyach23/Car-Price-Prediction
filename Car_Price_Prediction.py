import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import locale
import os

st.title('Car Price Prediction')
st.sidebar.header('Choose According to your Criteria')

project_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(project_dir, 'Car_Price_Prediction.png')
st.image(image_path, use_column_width=True)

def user_input_features():
    product_category_options = ['Compact', 'Luxury', 'Pick Up', 'Sedan', 'SUV']
    brand_options = sorted(['BMW', 'Ford', 'Mercedes-Benz', 'Chevrolet', 'Toyota', 'Audi', 'Porsche', 'Lexus', 'Jeep', 'Land', 'Nissan', 'Cadillac', 'RAM', 'GMC', 'Dodge', 'Kia', 
                     'Hyundai', 'Acura', 'Honda', 'Volkswagen', 'INFINITI', 'Lincoln', 'Jaguar', 'Volvo', 'Maserati', 'Bentley', 'Buick',
                     'Chrysler', 'Saturn'])
    transmission_options = ['Automatic', 'CVT', 'Manual']
    accident_options = ['None reported', 'At least 1 accident or damage reported']
    fuel_type_options = ['Diesel', 'E85 Flex Fuel', 'Hybrid', 'Hydrogen', 'Plug-In Hybrid']
    
    category_brand_mapping = {
    'Luxury': sorted(['Volvo', 'Porsche', 'Mercedes-Benz', 'Maserati', 'Lincoln', 'Lexus', 'Jaguar', 'INFINITI', 'Cadillac', 'Buick', 'Bentley', 'BMW', 'Audi', 'Acura']),
    'Sedan': sorted(['Toyota', 'Nissan', 'Kia', 'Hyundai', 'Honda', 'Ford', 'Dodge', 'Chrysler', 'Chevrolet']),
    'SUV': sorted(['Land', 'Jeep']),
    'Compact': sorted(['Volkswagen', 'Saturn']),
    'Pick Up': sorted(['RAM', 'GMC'])
    }

    brand_transmission_mapping = {
    'Acura': ['Automatic', 'Manual'],
    'Audi': ['Automatic', 'Manual'],
    'BMW': ['Automatic'],
    'Bentley': ['Automatic', 'Manual'],
    'Buick': ['Automatic', 'Manual'],
    'Cadillac': ['Automatic', 'Manual'],
    'Chevrolet': ['Automatic', 'Manual'],
    'Chrysler': ['Automatic', 'CVT', 'Manual'],
    'Dodge': ['Automatic', 'Manual'],
    'Ford': ['Automatic', 'CVT', 'Manual'],
    'GMC': ['Automatic', 'Manual'],
    'Honda': ['Automatic', 'CVT'],
    'Hyundai': ['Automatic', 'Manual'],
    'INFINITI': ['Manual'],
    'Jaguar': ['Automatic'],
    'Jeep': ['Automatic', 'Manual'],
    'Kia': ['Automatic', 'Manual'],
    'Land': ['Automatic'],
    'Lexus': ['Automatic', 'CVT'],
    'Lincoln': ['Automatic', 'CVT', 'Manual'],
    'Maserati': ['Automatic'],
    'Mercedes-Benz': ['Automatic', 'Manual'],
    'Nissan': ['Automatic'],
    'Porsche': ['Automatic'],
    'RAM': ['Automatic', 'Manual'],
    'Saturn': ['Automatic'],
    'Toyota': ['Automatic', 'CVT', 'Manual'],
    'Volkswagen': ['Automatic'],
    'Volvo': ['Automatic']
    }

    brand_fuel_type_mapping = {
    'Acura': {'Automatic': ['Hybrid'],
              'Manual': ['Hybrid']},
    'Audi': {'Automatic': ['Diesel', 'E85 Flex Fuel', 'Hybrid'],
             'Manual': ['Diesel', 'Hybrid']},
    'BMW': {'Automatic': ['Hybrid', 'Plug-In Hybrid']},
    'Bentley': {'Automatic': ['E85 Flex Fuel'],
                'Manual': ['E85 Flex Fuel']},
    'Buick': {'Automatic': ['E85 Flex Fuel'],
              'Manual': ['Hybrid']},
    'Cadillac': {'Automatic': ['Diesel', 'E85 Flex Fuel'],
                 'Manual': ['E85 Flex Fuel']},
    'Chevrolet': {'Automatic': ['Diesel', 'E85 Flex Fuel', 'Hybrid'],
                  'Manual': ['Diesel', 'E85 Flex Fuel']},
    'Chrysler': {'Automatic': ['E85 Flex Fuel'],
                 'CVT': ['Hybrid'],
                 'Manual': ['E85 Flex Fuel']},
    'Dodge': {'Automatic': ['Diesel', 'E85 Flex Fuel'],
              'Manual': ['E85 Flex Fuel']},
    'Ford': {'Automatic': ['Diesel', 'E85 Flex Fuel', 'Hybrid'],
             'CVT': ['Hybrid', 'Plug-In Hybrid'],
             'Manual': ['Diesel', 'E85 Flex Fuel']},
    'GMC': {'Automatic': ['Diesel', 'E85 Flex Fuel'],
            'Manual': ['Diesel', 'E85 Flex Fuel']},
    'Honda': {'Automatic': ['Plug-In Hybrid'],
              'CVT': ['Hybrid']},
    'Hyundai': {'Automatic': ['Hybrid', 'Plug-In Hybrid'],
                'Manual': ['Hybrid']},
    'INFINITI': {'Manual': ['Hybrid']},
    'Jaguar': {'Automatic': ['Diesel', 'Hybrid']},
    'Jeep': {'Automatic': ['E85 Flex Fuel', 'Hybrid', 'Plug-In Hybrid'],
             'Manual': ['E85 Flex Fuel']},
    'Kia': {'Automatic': ['Hybrid', 'Plug-In Hybrid'],
            'Manual': ['Hybrid', 'Plug-In Hybrid']},
    'Land': {'Automatic': ['Diesel', 'E85 Flex Fuel', 'Hybrid']},
    'Lexus': {'Automatic': ['Hybrid'],
              'CVT': ['Hybrid']},
    'Lincoln': {'Automatic': ['Hybrid'],
                'CVT': ['Plug-In Hybrid'],
                'Manual': ['E85 Flex Fuel']},
    'Maserati': {'Automatic': ['Hybrid']},
    'Mercedes-Benz': {'Automatic': ['Diesel', 'E85 Flex Fuel', 'Hybrid'],
                      'Manual': ['Diesel', 'E85 Flex Fuel']},
    'Nissan': {'Automatic': ['E85 Flex Fuel']},
    'Porsche': {'Automatic': ['Diesel', 'Plug-In Hybrid']},
    'RAM': {'Automatic': ['Diesel', 'E85 Flex Fuel', 'Hybrid'],
            'Manual': ['Diesel']},
    'Saturn': {'Automatic': ['Hybrid']},
    'Toyota': {'Automatic': ['E85 Flex Fuel', 'Hybrid', 'Plug-In Hybrid'],
               'CVT': ['Hybrid'],
               'Manual': ['E85 Flex Fuel']},
    'Volkswagen': {'Automatic': ['Diesel']},
    'Volvo': {'Automatic': ['Hybrid', 'Plug-In Hybrid']}
    }
    
    product_category = st.sidebar.selectbox('🛍️ Car Category', product_category_options)
    brand = st.sidebar.selectbox('🚗 Brand', category_brand_mapping[product_category])
    milage = st.sidebar.number_input('🛣️ Milage', 100, 399000, 100, 100)
    horse_power = st.sidebar.number_input('🐎 Horse Power (HP)', 76, 603, 76, 1)
    engine_capacity = st.sidebar.number_input('⚙️ Engine Capacity (Liter)', 1.5, 7.3, 1.5, 0.1)
    total_cylinder = st.sidebar.number_input('🔩 Cylinder', 2, 12, 2, 1)
    transmission = st.sidebar.selectbox('🔧 Transmission Types', brand_transmission_mapping[brand])
    accident = st.sidebar.selectbox('🚑 Accident', accident_options)
    fuel_type = st.sidebar.selectbox('⛽ Fuel Type', brand_fuel_type_mapping[brand][transmission])

    data = {'product_category': product_category,
            'brand': brand,
            'milage': milage,
            'horse_power': horse_power,
            'engine_capacity': engine_capacity,
            'total_cylinder': total_cylinder,
            'transmission': transmission,
            'accident': accident,
            'fuel_type': fuel_type
            }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

csv_file_path = os.path.join(project_dir, 'Car_Price_Prediction_Clean.csv')
price_raw = pd.read_csv(csv_file_path)
price = price_raw.drop(columns=['price'])
df = pd.concat([input_df, price], axis=0)
df_test = df[:1]

df['product_category'] = df['product_category'].astype(str)
df['brand'] = df['brand'].astype(str)
df['transmission'] = df['transmission'].astype(str)
df['accident'] = df['accident'].astype(str)

label_encode_columns = ['brand']
label_encoder = LabelEncoder()
for column in label_encode_columns:
    df[column] = label_encoder.fit_transform(df[column])

df['product_category'] = df['product_category'].replace({'Pick Up': 0, 'Compact': 1, 'SUV': 2, 'Sedan': 3, 'Luxury': 4})

transmission_order = {
    'Manual': 0,
    'Automatic': 1,
    'CVT': 2
}
df['transmission'] = df['transmission'].map(transmission_order)

df['accident'] = df['accident'].replace({'At least 1 accident or damage reported': 0, 'None reported': 1})

one_hot_encode_columns = ['fuel_type']
df_encoded = pd.get_dummies(df, columns=one_hot_encode_columns)
# df_encoded.drop(columns=['fuel_type_Gasoline'], inplace=True)

model_path = os.path.join(project_dir, 'Car_Price_Prediction_Sample.sav')
loaded_model = pickle.load(open(model_path))

locale.setlocale(locale.LC_ALL, 'id_ID')

if st.button('Predict'):
    predicted_price = loaded_model.predict(df_encoded)
    formatted_price = locale.format('%.2f', predicted_price[0], grouping=True)
    
    st.markdown('<div class="prediction-container">'
                f'<p>USD ($) Predicted Price: {formatted_price}</p>'
                '</div>', unsafe_allow_html=True)
