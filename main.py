import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# 1. INICIALIZACIÓN DE LA APP
app = FastAPI(title="PropTech Valora API", description="API de valoración inmobiliaria para Madrid", version="1.0")

# 2. CARGA DE ARTEFACTOS (Se ejecuta una sola vez al arrancar)
print("Cargando modelos y datos...")

try:
    # Cargar el modelo entrenado
    model = joblib.load("modelo_turbo_madrid.pkl")
    
    # Cargar la lista de columnas para asegurar el orden exacto
    expected_features = joblib.load("features_list.pkl")
    
    # Cargar el contexto de barrios (convertimos a diccionario para búsqueda rápida)
    # El index será el nombre del barrio (en minúsculas para evitar errores)
    df_context = pd.read_csv("data_context.csv")
    df_context['neighborhood_norm'] = df_context['neighborhood'].str.lower().str.strip()
    context_dict = df_context.set_index('neighborhood_norm').to_dict(orient='index')
    
    print("✅ Sistema cargado correctamente.")

except Exception as e:
    print(f"❌ Error crítico al cargar archivos: {e}")
    raise e

# 3. DEFINICIÓN DEL FORMATO DE ENTRADA (Input del Usuario)
class PropertyInput(BaseModel):
    neighborhood: str
    size_m2: float
    rooms: int
    bathrooms: int

# 4. ENDPOINT DE PREDICCIÓN
@app.post("/predict")
def predict_price(data: PropertyInput):
    # A. Normalizar nombre del barrio
    barrio_key = data.neighborhood.lower().strip()
    
    # B. Buscar contexto del barrio
    if barrio_key not in context_dict:
        raise HTTPException(status_code=404, detail=f"Barrio '{data.neighborhood}' no encontrado en la base de datos.")
    
    ctx = context_dict[barrio_key]
    
    # C. Obtener variables base
    renta = ctx['Renta_neta_media_hogar']
    criminalidad = ctx['Indice_Criminalidad']
    aire = ctx['Indice_Calidad_Aire']
    
    # D. Ingeniería de Features (Calculadas en tiempo real)
    # Asumimos interacción directa (multiplicación) basada en los nombres
    renta_x_size = renta * data.size_m2
    renta_x_rooms = renta * data.rooms
    
    # E. Construir el DataFrame de entrada (Una sola fila)
    input_data = {
        'size_m2': data.size_m2,
        'rooms': data.rooms,
        'bathrooms': data.bathrooms,
        'Renta_neta_media_hogar': renta,
        'Indice_Criminalidad': criminalidad,
        'Indice_Calidad_Aire': aire,
        'Renta_x_Size': renta_x_size,
        'Renta_x_Rooms': renta_x_rooms
    }
    
    # Convertir a DataFrame y reordenar columnas según features_list.pkl
    df_input = pd.DataFrame([input_data])
    
    # Asegurar que el orden sea EXACTAMENTE el que espera el modelo
    try:
        df_input = df_input[expected_features]
    except KeyError as e:
        missing = set(expected_features) - set(df_input.columns)
        raise HTTPException(status_code=500, detail=f"Faltan columnas calculadas: {missing}")

    # F. Predicción
    prediction = model.predict(df_input)[0]
    
    return {
        "barrio": ctx['neighborhood'], # Devolvemos el nombre oficial del CSV
        "precio_estimado": round(prediction, 2),
        "moneda": "EUR",
        "factores_contexto": {
            "renta_zona": renta,
            "seguridad": criminalidad,
            "calidad_aire": aire
        }
    }

# 5. PUNTO DE ENTRADA PARA DESARROLLO LOCAL
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
