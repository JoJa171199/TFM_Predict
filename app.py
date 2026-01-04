from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import unicodedata
import os

app = FastAPI(title="PropTech Valora API")

# Permitir conexiones externas (CORS) para que n8n o tu web no sean bloqueados
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UTILIDADES ---
def normalizar_texto(texto):
    """Elimina tildes, espacios y convierte a minúsculas para comparaciones robustas."""
    if not isinstance(texto, str): return ""
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode("utf-8")
    return texto.strip().lower()

# --- CARGA DE ACTIVOS (Al iniciar el servidor) ---
try:
    model = joblib.load('modelo_turbo_madrid.pkl')
    features_names = joblib.load('features_list.pkl')
    df_context = pd.read_csv('data_context.csv')
    # Normalizamos los nombres de los barrios en la base de datos una sola vez
    df_context['neighborhood_norm'] = df_context['neighborhood'].apply(normalizar_texto)
    print("✅ Activos cargados y barrios normalizados.")
except Exception as e:
    print(f"❌ Error crítico en la carga de archivos: {e}")

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    """Endpoint para que Render sepa que la API está viva."""
    return {"status": "online", "model": "Turbo Opción A"}

@app.post("/predict")
def predict(data: dict):
    try:
        # 1. Validación de entrada básica
        required = ['neighborhood', 'size_m2', 'rooms', 'bathrooms']
        for field in required:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Falta el campo: {field}")

        # 2. Búsqueda robusta del barrio
        barrio_buscado = normalizar_texto(data['neighborhood'])
        match = df_context[df_context['neighborhood_norm'] == barrio_buscado]

        if match.empty:
            raise HTTPException(status_code=404, detail=f"El barrio '{data['neighborhood']}' no se encuentra en la base de datos de Madrid.")
        
        barrio_data = match.iloc[0]

        # 3. Construcción del vector de entrada (Feature Engineering)
        input_dict = {
            'size_m2': float(data['size_m2']),
            'rooms': int(data['rooms']),
            'bathrooms': int(data['bathrooms']),
            'Renta_neta_media_hogar': float(barrio_data['Renta_neta_media_hogar']),
            'Indice_Criminalidad': float(barrio_data['Indice_Criminalidad']),
            'Indice_Calidad_Aire': float(barrio_data['Indice_Calidad_Aire']),
            # Interacciones calculadas igual que en el entrenamiento
            'Renta_x_Size': float(barrio_data['Renta_neta_media_hogar']) * float(data['size_m2']),
            'Renta_x_Rooms': float(barrio_data['Renta_neta_media_hogar']) * int(data['rooms'])
        }

        # Convertir a DataFrame asegurando el orden exacto de las columnas
        df_input = pd.DataFrame([input_dict])[features_names]

        # 4. Predicción y reversión del Logaritmo
        prediction_log = model.predict(df_input)
        precio_m2 = np.expm1(prediction_log)[0]
        precio_total = precio_m2 * data['size_m2']

        return {
            "success": True,
            "barrio_oficial": barrio_data['neighborhood'],
            "precio_m2_estimado": round(float(precio_m2), 2),
            "precio_total_estimado": round(float(precio_total), 2),
            "contexto": {
                "renta": round(barrio_data['Renta_neta_media_hogar'], 2),
                "seguridad": round(barrio_data['Indice_Criminalidad'], 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
