from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import unicodedata

app = FastAPI(title="PropTech Valora API - Madrid 2026")

# CORS: Permite que n8n o cualquier web consulte tu API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CARGA DE ACTIVOS ---
try:
    model = joblib.load('modelo_turbo_madrid.pkl')
    features_names = joblib.load('features_list.pkl')
    df_context = pd.read_csv('data_context.csv')
    print("✅ Activos cargados con éxito.")
except Exception as e:
    print(f"❌ Error al cargar activos: {e}")

# --- UTILIDADES ---
def normalizar(texto):
    """Estandariza nombres de barrios (sin tildes, minúsculas)."""
    if not isinstance(texto, str): return ""
    return "".join(c for c in unicodedata.normalize('NFD', texto.lower().strip())
                  if unicodedata.category(c) != 'Mn')

# Pre-normalizar la base de datos para rapidez
df_context['neighborhood_norm'] = df_context['neighborhood'].apply(normalizar)

@app.get("/")
def home():
    return {"message": "PropTech Valora API Online", "version": "1.0.0"}

@app.post("/predict")
def predict(data: dict):
    try:
        # 1. Obtener datos de entrada (desde n8n/Usuario)
        barrio_input = normalizar(data.get('neighborhood', ''))
        size = float(data.get('size_m2', 0))
        rooms = int(data.get('rooms', 0))
        bathrooms = int(data.get('bathrooms', 0))

        # 2. Buscar contexto social del barrio
        match = df_context[df_context['neighborhood_norm'] == barrio_input]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Barrio '{data['neighborhood']}' no encontrado.")
        
        social = match.iloc[0]

        # 3. Recrear variables del modelo (incluyendo interacciones del Modelo Turbo)
        renta = float(social['Renta_neta_media_hogar'])
        
        input_data = {
            'size_m2': size,
            'rooms': rooms,
            'bathrooms': bathrooms,
            'Renta_neta_media_hogar': renta,
            'Indice_Criminalidad': float(social['Indice_Criminalidad']),
            'Indice_Calidad_Aire': float(social['Indice_Calidad_Aire']),
            'Renta_x_Size': renta * size,
            'Renta_x_Rooms': renta * rooms
        }

        # Asegurar orden de columnas del modelo
        df_final = pd.DataFrame([input_data])[features_names]

        # 4. Predicción (Log-reversión)
        pred_log = model.predict(df_final)
        precio_m2 = np.expm1(pred_log)[0]
        precio_total = precio_m2 * size

        return {
            "success": True,
            "neighborhood": social['neighborhood'],
            "valuation": {
                "price_m2": round(float(precio_m2), 2),
                "total_price": round(float(precio_total), 2)
            },
            "socioeconomic": {
                "renta_media_hogar": renta,
                "indice_criminalidad": social['Indice_Criminalidad']
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
