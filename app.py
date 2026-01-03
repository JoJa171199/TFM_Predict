from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# 1. Cargar el modelo y los datos de contexto al arrancar
model = joblib.load('modelo_turbo_madrid.pkl')
features_names = joblib.load('features_list.pkl')
# Cargamos una tabla pequeña que relacione Barrio -> Renta, Criminalidad, etc.
df_context = pd.read_csv('data_context.csv') 

@app.post("/predict")
def predict(data: dict):
    # data llega de n8n: {"neighborhood": "Sol", "size_m2": 80, "rooms": 2, "bathrooms": 1}
    
    # 2. Inyección Automática de Variables Sociales
    barrio_data = df_context[df_context['neighborhood'] == data['neighborhood']].iloc[0]
    
    # 3. Construir el vector de entrada con las interacciones "Turbo"
    input_dict = {
        'size_m2': data['size_m2'],
        'rooms': data['rooms'],
        'bathrooms': data['bathrooms'],
        'Renta_neta_media_hogar': barrio_data['Renta_neta_media_hogar'],
        'Indice_Criminalidad': barrio_data['Indice_Criminalidad'],
        'Indice_Calidad_Aire': barrio_data['Indice_Calidad_Aire'],
        # Calculamos las interacciones que el modelo espera
        'Renta_x_Size': barrio_data['Renta_neta_media_hogar'] * data['size_m2'],
        'Renta_x_Rooms': barrio_data['Renta_neta_media_hogar'] * data['rooms']
    }
    
    df_input = pd.DataFrame([input_dict])[features_names]
    
    # 4. Predicción (recordando que usamos Log-Transform)
    prediction_log = model.predict(df_input)
    precio_final = np.expm1(prediction_log)[0]
    
    return {
        "precio_m2": round(float(precio_final), 2),
        "precio_total": round(float(precio_final * data['size_m2']), 2)
    }