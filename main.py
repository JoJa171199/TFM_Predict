from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="API Predicción Precios Madrid (LGBM)")

# 1. Cargar el modelo y la lista de features al inicio
# Usamos try/except para facilitar el debug en los logs de Render
try:
    model = joblib.load('modelo_turbo_madrid.pkl')
    features_names = joblib.load('features_list.pkl')
    print("✅ Modelo y Features cargados correctamente.")
    print(f"📋 Features esperadas: {features_names}")
except Exception as e:
    print(f"❌ Error fatal cargando archivos .pkl: {e}")
    model = None
    features_names = []

# 2. Definir el esquema EXACTO que envía n8n (Las 8 features + inputs raw)
class InputData(BaseModel):
    # Datos básicos (necesarios para cálculo final)
    size_m2: float
    
    # Las 8 Features que el modelo exige (calculadas previamente en n8n)
    rooms: float
    bathrooms: float
    # Nota: size_m2 ya está arriba
    Renta_neta_media_hogar: float
    Indice_Criminalidad: float
    Indice_Calidad_Aire: float
    Renta_x_Size: float
    Renta_x_Rooms: float

@app.get("/")
def home():
    return {"status": "online", "model": "LGBM Regressor Log-Scale"}

@app.post("/predict")
def predict(data: InputData):
    if not model:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")
    
    try:
        # 3. Mapear los datos entrantes al DataFrame en el ORDEN EXACTO del entrenamiento
        # Creamos un diccionario temporal para asegurar el orden
        input_payload = {
            'size_m2': data.size_m2,
            'rooms': data.rooms,
            'bathrooms': data.bathrooms,
            'Renta_neta_media_hogar': data.Renta_neta_media_hogar,
            'Indice_Criminalidad': data.Indice_Criminalidad,
            'Indice_Calidad_Aire': data.Indice_Calidad_Aire,
            'Renta_x_Size': data.Renta_x_Size,
            'Renta_x_Rooms': data.Renta_x_Rooms
        }
        
        # Crear DataFrame filtrando solo las columnas que el modelo conoce
        df_input = pd.DataFrame([input_payload])[features_names]
        
        # 4. Predicción (El modelo devuelve Logaritmo)
        prediction_log = model.predict(df_input)[0]
        
        # 5. Transformación Inversa (Log -> Euros)
        precio_m2_estimado = np.expm1(prediction_log)
        
        # 6. Cálculo del Total
        precio_total = precio_m2_estimado * data.size_m2
        
        return {
            "precio_m2_estimado": round(float(precio_m2_estimado), 2),
            "precio_total_estimado": round(float(precio_total), 2),
            "moneda": "EUR",
            "log_value_debug": float(prediction_log) # Para debug
        }
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        raise HTTPException(status_code=400, detail=str(e))
