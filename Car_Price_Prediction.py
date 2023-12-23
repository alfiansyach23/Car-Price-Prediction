import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

import locale
import re
import pickle
import os

st.title('Car Price Prediction')
st.sidebar.header('Choose According to your Criteria')

project_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(project_dir, 'Car_Price_Prediction.png')
st.image(image_path, use_column_width=True)

def user_input_features():
    product_category_options = ['Compact', 'Luxury', 'Pick Up', 'SUV', 'Sedan']
    brand_options = sorted(['Volkswagen', 'Saturn', 'Volvo', 
                            'Porsche', 'Mercedes-Benz', 'Maserati', 'Lincoln', 'Lexus', 'Jaguar', 'INFINITI', 'Cadillac', 
                            'Buick', 'Bentley', 'BMW', 'Audi', 'Acura', 'RAM', 'GMC', 'Land', 'Jeep', 'Hummer', 
                            'Toyota', 'Nissan', 'Kia', 'Hyundai', 'Honda', 'Ford', 'Dodge', 'Chrysler', 'Chevrolet'])
    transmission_options = ['Automatic', 'CVT', 'Dual-Clutch', 'Manual']
    accident_options = ['None reported', 'At least 1 accident or damage reported']
    fuel_type_options = ['Diesel', 'E85 Flex Fuel', 'Hybrid', 'Hydrogen', 'Plug-In Hybrid']
    
    category_brand_mapping = {
    'Compact': sorted(['Volkswagen', 'Saturn']),
    'Luxury': sorted(['Volvo', 'Porsche', 'Mercedes-Benz', 'Maserati', 'Lincoln', 'Lexus', 'Jaguar', 'INFINITI', 
                      'Cadillac', 'Buick', 'Bentley', 'BMW', 'Audi', 'Acura']),
    'Pick Up': sorted(['RAM', 'GMC']),
    'SUV': sorted(['Land', 'Jeep', 'Hummer']),
    'Sedan': sorted(['Toyota', 'Nissan', 'Kia', 'Hyundai', 'Honda', 'Ford', 'Dodge', 'Chrysler', 'Chevrolet'])
    }

    brand_transmission_mapping = {
    'Acura': ['Automatic', 'CVT', 'Manual'],
    'Audi': ['Automatic', 'Dual-Clutch', 'Manual'],
    'BMW': ['Automatic', 'Manual'],
    'Bentley': ['Automatic', 'Manual'],
    'Buick': ['Automatic', 'CVT', 'Manual'],
    'Cadillac': ['Automatic', 'Manual'],
    'Chevrolet': ['Automatic', 'Manual'],
    'Chrysler': ['Automatic', 'CVT', 'Manual'],
    'Dodge': ['Automatic', 'Manual'],
    'Ford': ['Automatic', 'CVT', 'Manual'],
    'GMC': ['Automatic', 'Manual'],
    'Honda': ['Automatic', 'CVT', 'Manual'],
    'Hyundai': ['Automatic', 'Manual'],
    'INFINITI': ['Automatic', 'CVT', 'Manual'],
    'Jaguar': ['Automatic', 'Manual'],
    'Jeep': ['Automatic', 'Manual'],
    'Kia': ['Automatic', 'CVT', 'Manual'],
    'Land': ['Automatic', 'Manual'],
    'Lexus': ['Automatic', 'CVT', 'Manual'],
    'Lincoln': ['Automatic', 'CVT', 'Manual'],
    'Maserati': ['Automatic', 'Dual-Clutch', 'Manual'],
    'Mercedes-Benz': ['Automatic', 'Manual'],
    'Nissan': ['Automatic', 'CVT', 'Manual'],
    'Porsche': ['Automatic', 'Dual-Clutch', 'Manual'],
    'RAM': ['Automatic', 'Manual'],
    'Saturn': ['Automatic', 'Manual'],
    'Toyota': ['Automatic', 'CVT', 'Manual'],
    'Volkswagen': ['Automatic', 'Manual'],
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

    horse_power_unique = [70, 400, 235, 250, 316, 218, 325, 227, 455, 330, 281, 160, 262, 247, 300, 240, 315, 210, 283, 292, 201, 184, 140, 150, 280, 
                          174, 170, 220, 256, 115, 268, 200, 241, 228, 147, 276, 382, 139, 130, 132, 270, 236, 212, 385, 295, 381, 285, 203, 437, 219, 
                          106, 208, 282, 360, 266, 225, 306, 401, 180, 179, 275, 137, 273, 239, 121, 230, 159, 133, 278, 76, 260, 245, 301, 389, 310, 
                          302, 126, 182, 134, 151, 158, 176, 168, 305, 175, 152, 265, 148, 165, 243, 173, 169, 271, 224, 161, 172, 205, 571, 563, 370, 
                          390, 395, 410, 702, 350, 383, 335, 340, 416, 690, 355, 473, 348, 365, 500, 520, 443, 252, 248, 414, 303, 255, 440, 405, 415, 
                          420, 480, 550, 453, 540, 475, 345, 450, 261, 375, 434, 640, 605, 580, 379, 320, 502, 641, 430, 621, 572, 177, 253, 287, 125, 
                          485, 565, 332, 188, 290, 122, 317, 131, 222, 545, 185, 141, 284, 109, 215, 142, 291, 162, 286, 78, 181, 166, 108, 136, 362, 
                          503, 469, 523, 429, 402, 603, 435, 577, 496, 456, 258, 221, 630, 192, 463, 536, 449, 421, 612, 272, 329, 333, 451, 518, 163, 
                          191, 190, 493, 510, 197, 369, 211, 562, 710, 186, 167, 323, 244, 155, 187, 263, 232, 156, 178, 95, 543, 424, 454, 404, 394, 
                          444, 600, 425, 189, 118, 380, 311, 471, 409, 204, 308, 354, 467, 386, 557, 575, 296, 246, 237, 217, 560, 507, 573, 759, 740, 
                          729, 602, 199, 146, 206, 120, 512, 293, 202, 309, 470, 294, 495, 328, 298, 138, 226, 378, 344, 104, 198, 145, 242, 393, 143, 
                          304, 288, 397, 445, 193, 342, 277, 341, 171, 760, 412, 411, 207, 318, 195, 526, 460, 153, 123, 231, 650, 651, 611, 553, 671, 
                          769, 788, 483, 670, 660, 164, 101, 797, 645, 717, 707, 372, 154, 363, 314, 490, 505, 110, 319, 426, 535, 264, 312, 353, 334, 
                          366, 324, 638, 114, 668, 403, 321, 556, 1, 552, 582, 549, 542, 616, 567, 521, 528, 659, 626, 326, 617, 357, 407, 601, 313, 555, 
                          322, 462, 591, 349, 570, 525, 610, 715, 789, 279]

    engine_capacity_unique = [1, 2, 3, 4, 5, 6, 7, 8]
    total_cylinder_unique = [3, 4, 6, 5, 8, 12, 10, 2, 16]

    product_category = st.sidebar.selectbox('🛍️ Car Category', product_category_options)
    brand = st.sidebar.selectbox('🚗 Brand', category_brand_mapping[product_category])
    milage = st.sidebar.number_input('🛣️ Milage', min_value=100, max_value=405000, value=100)
    horse_power = st.sidebar.number_input('🐎 Horse Power (HP)', min(horse_power_unique), max(horse_power_unique))
    engine_capacity = st.sidebar.number_input('⚙️ Engine Capacity (Liter)', min(engine_capacity_unique), max(engine_capacity_unique))
    total_cylinder = st.sidebar.number_input('🔩 Cylinder', min(total_cylinder_unique), max(total_cylinder_unique))
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

csv_file_path = os.path.join(project_dir, 'Car_Price_Prediction.csv')
df = pd.read_csv(csv_file_path)

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.drop(columns=['fuel_type_Gasoline'], inplace=True)
X_test.drop(columns=['fuel_type_Gasoline'], inplace=True)

train_to_drop = X_train[(X_train['fuel_type_Diesel'] == 0) & 
                        (X_train['fuel_type_E85 Flex Fuel'] == 0) & 
                        (X_train['fuel_type_Hybrid'] == 0) & 
                        (X_train['fuel_type_Hydrogen'] == 0) & 
                        (X_train['fuel_type_Plug-In Hybrid'] == 0)].index

X_train.drop(index=train_to_drop, inplace=True)
y_train.drop(index=train_to_drop, inplace=True)

test_to_drop = X_test[(X_test['fuel_type_Diesel'] == 0) & 
                        (X_test['fuel_type_E85 Flex Fuel'] == 0) & 
                        (X_test['fuel_type_Hybrid'] == 0) & 
                        (X_test['fuel_type_Hydrogen'] == 0) & 
                        (X_test['fuel_type_Plug-In Hybrid'] == 0)].index

X_test.drop(index=test_to_drop, inplace=True)
y_test.drop(index=test_to_drop, inplace=True)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

gb_model_filtered = GradientBoostingRegressor(
    n_estimators = 67,
    min_samples_split = 7, 
    min_samples_leaf = 2, 
    max_features = 'log2', 
    max_depth = 5, 
    loss = 'huber',  
    learning_rate = 0.1,
    criterion = 'friedman_mse'
)

gb_model_filtered.fit(X_train, y_train)

df_new = pd.DataFrame(X_test, columns=['product_category', 'brand', 'milage', 'horse_power', 'engine_capacity', 
                                       'total_cylinder', 'transmission', 'accident', 'fuel_type_Diesel', 'fuel_type_E85 Flex Fuel', 
                                       'fuel_type_Hybrid', 'fuel_type_Hydrogen', 'fuel_type_Plug-In Hybrid'])

sample_predictions = gb_model_filtered.predict(X_test)
sample_predictions = list(sample_predictions)

if st.button('Predict'):
    sample_predictions = gb_model_filtered.predict(df_new)
    formatted_price = locale.format('%.2f', sample_predictions[0], grouping=True)
    
    st.markdown(
        f'<div class="prediction-container">'
        f'<p>USD ($) Predicted Price: {formatted_price}</p>'
        '</div>',
        unsafe_allow_html=True
    )
