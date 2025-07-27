from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import io
import os
import tempfile
from typing import Optional, Dict, Any
import logging
from ....services.ml_service import MLService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API to predict whether a customer will buy again",
    description="API - Prueba Técnica",
    version="2.0.0"
)

ml_service = MLService()

@app.get("/")
async def root():
    """Endpoint raíz de la API."""
    return {
        "message": "API to predict whether a customer will buy again", 
        "version": "2.0.0",
        "available_endpoints": [
            "/preprocess - Preprocesamiento de datos",
            "/feature-engineering - Ingeniería de características",
            "/train - Entrenamiento de modelo",
            "/predict - Predicciones",
            "/status - Estado del servicio",
            "/models - Lista de modelos disponibles"
        ]
    }

@app.post("/preprocess")
async def preprocess_data(
    file: UploadFile = File(..., description="Archivo CSV con datos raw"),
    save_processed_data: bool = Form(True, description="Guardar datos procesados"),
    output_filename: str = Form("processed_data.csv", description="Nombre del archivo de salida")
) -> Dict[str, Any]:
    """
    Endpoint 1: Preprocesamiento de datos.
    
    Args:
        file: Archivo CSV con datos transaccionales raw
        save_processed_data: Si guardar los datos procesados
        output_filename: Nombre del archivo de salida
        
    Returns:
        Dict con resultados del preprocesamiento
    """
    try:
        logger.info(f"Iniciando preprocesamiento de archivo: {file.filename}")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="El archivo debe ser CSV")
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_input_path = temp_file.name
        
        output_path = None
        if save_processed_data:
            os.makedirs("data/processed", exist_ok=True)
            output_path = f"data/processed/{output_filename}"
        
        results = ml_service.preprocess_data(temp_input_path, output_path)
        
        os.unlink(temp_input_path)
        
        if results['status'] == 'success':
            return JSONResponse(content=results, status_code=200)
        else:
            raise HTTPException(status_code=500, detail=results['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/feature-engineering")
async def feature_engineering(
    file: UploadFile = File(..., description="Archivo CSV con datos procesados"),
    prediction_days: int = Form(90, description="Días para predicción futura"),
    save_features: bool = Form(True, description="Guardar dataset con características"),
    output_filename: str = Form("featured_data.csv", description="Nombre del archivo de salida")
) -> Dict[str, Any]:
    """
    Endpoint 2: Ingeniería de características.
    
    Args:
        file: Archivo CSV con datos ya procesados
        prediction_days: Días para predicción futura
        save_features: Si guardar el dataset con características
        output_filename: Nombre del archivo de salida
        
    Returns:
        Dict con resultados del feature engineering
    """
    try:
        logger.info(f"Iniciando feature engineering de archivo: {file.filename}")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="El archivo debe ser CSV")
        
        if prediction_days < 1 or prediction_days > 365:
            raise HTTPException(
                status_code=400, 
                detail="prediction_days debe estar entre 1 y 365"
            )
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_input_path = temp_file.name
    
        output_path = None
        if save_features:
            os.makedirs("data/features", exist_ok=True)
            output_path = f"data/features/{output_filename}"
        
        results = ml_service.engineer_features(
            temp_input_path, 
            prediction_days, 
            output_path
        )
        
        os.unlink(temp_input_path)
        
        if results['status'] == 'success':
            return JSONResponse(content=results, status_code=200)
        else:
            raise HTTPException(status_code=500, detail=results['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en feature engineering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/train")
async def train_model(
    file: UploadFile = File(..., description="Archivo CSV con features engineered"),
    target_column: str = Form("WillPurchase_90Days", description="Columna objetivo para predicción")
) -> Dict[str, Any]:
    """
    Endpoint 3: Entrenamiento de modelo.
    
    Args:
        file: Archivo CSV con características engineered
        target_column: Columna objetivo para predicción
        
    Returns:
        Dict con resultados del entrenamiento
    """
    try:
        logger.info(f"Iniciando entrenamiento con archivo: {file.filename}")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="El archivo debe ser CSV")
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_input_path = temp_file.name
        
        try:
            df_check = pd.read_csv(temp_input_path)
            if target_column not in df_check.columns:
                os.unlink(temp_input_path)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Columna objetivo '{target_column}' no encontrada. Columnas disponibles: {list(df_check.columns)}"
                )
        except pd.errors.EmptyDataError:
            os.unlink(temp_input_path)
            raise HTTPException(status_code=400, detail="El archivo está vacío")
        except Exception as e:
            os.unlink(temp_input_path)
            raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {str(e)}")
        
        os.makedirs("models", exist_ok=True)
        
        results = ml_service.train_model(temp_input_path, target_column)
        
        os.unlink(temp_input_path)
        
        if results['status'] == 'success':
            return JSONResponse(content=results, status_code=200)
        else:
            raise HTTPException(status_code=500, detail=results['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

def get_latest_model() -> Optional[str]:
    """
    Obtiene el modelo más reciente del directorio de modelos.
    
    Returns:
        Optional[str]: Ruta del modelo más reciente o None si no hay modelos
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        return None
    
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    
    return os.path.join(models_dir, model_files[0])

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Archivo CSV con datos de clientes"),
    model_filename: Optional[str] = Form(None, description="Nombre del modelo a usar (opcional)"),
    output_filename: str = Form("predictions.csv", description="Nombre del archivo de predicciones")
) -> Dict[str, Any]:
    """
    Endpoint 4: Predicciones.
    
    Args:
        file: Archivo CSV con datos de clientes para predicción
        model_filename: Nombre del modelo a usar (opcional, usa el más reciente si no se especifica)
        output_filename: Nombre del archivo de predicciones
        
    Returns:
        Dict con resultados de las predicciones
    """
    try:
        logger.info(f"Iniciando predicciones con archivo: {file.filename}")
        
        # Validar archivo
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="El archivo debe ser CSV")
        
        # Determinar qué modelo usar
        model_path = None
        
        if model_filename:
            # Usar modelo específico
            if not model_filename.endswith('.pkl'):
                model_filename += '.pkl'
            model_path = f"models/{model_filename}"
        else:
            # Usar modelo más reciente
            model_path = get_latest_model()
            
        if not model_path or not os.path.exists(model_path):
            # Listar modelos disponibles para ayudar al usuario
            available_models = []
            if os.path.exists("models"):
                available_models = [f for f in os.listdir("models") if f.endswith('.pkl')]
            
            error_msg = f"Modelo no encontrado: {model_path if model_path else 'No hay modelos disponibles'}"
            if available_models:
                error_msg += f". Modelos disponibles: {available_models}"
            else:
                error_msg += ". No hay modelos entrenados. Use el endpoint /train primero."
                
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_input_path = temp_file.name
        
        # Definir ruta de salida
        os.makedirs("data/predictions", exist_ok=True)
        output_path = f"data/predictions/{output_filename}"
        
        # Ejecutar predicciones
        results = ml_service.predict_customers(
            model_path, 
            temp_input_path, 
            output_path
        )
        
        # Limpiar archivo temporal
        os.unlink(temp_input_path)
        
        if results['status'] == 'success':
            # Agregar información del modelo usado
            results['model_used'] = model_path
            results['model_was_auto_selected'] = model_filename is None
            return JSONResponse(content=results, status_code=200)
        else:
            raise HTTPException(status_code=500, detail=results['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicciones: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """
    Endpoint de estado del servicio.
    
    Returns:
        Dict con estado actual del servicio
    """
    try:
        status = ml_service.get_service_status()
        
        # Agregar información de archivos disponibles
        status['available_files'] = {
            'processed_data': os.listdir('data/processed') if os.path.exists('data/processed') else [],
            'featured_data': os.listdir('data/features') if os.path.exists('data/features') else [],
            'models': [f for f in os.listdir('models') if f.endswith('.pkl')] if os.path.exists('models') else [],
            'predictions': os.listdir('data/predictions') if os.path.exists('data/predictions') else []
        }
        
        # Agregar información del modelo más reciente
        latest_model = get_latest_model()
        status['latest_model'] = {
            'path': latest_model,
            'filename': os.path.basename(latest_model) if latest_model else None
        }
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error obteniendo estado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado: {str(e)}")

@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """
    Endpoint para listar todos los modelos disponibles.
    
    Returns:
        Dict con lista de modelos disponibles
    """
    try:
        models_dir = "models"
        
        if not os.path.exists(models_dir):
            return {
                "models": [],
                "total_models": 0,
                "message": "No hay directorio de modelos"
            }
        
        model_files = []
        for filename in os.listdir(models_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(models_dir, filename)
                stat = os.stat(filepath)
                model_files.append({
                    "filename": filename,
                    "size_bytes": stat.st_size,
                    "created_timestamp": stat.st_ctime,
                    "modified_timestamp": stat.st_mtime
                })
        
        model_files.sort(key=lambda x: x['modified_timestamp'], reverse=True)
        
        return {
            "models": model_files,
            "total_models": len(model_files),
            "latest_model": model_files[0]["filename"] if model_files else None
        }
        
    except Exception as e:
        logger.error(f"Error listando modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listando modelos: {str(e)}")

@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """
    Endpoint para descargar archivos generados.
    
    Args:
        file_type: Tipo de archivo (processed, features, predictions, models)
        filename: Nombre del archivo
        
    Returns:
        Archivo para descarga
    """
    try:
        type_mapping = {
            'processed': 'data/processed',
            'features': 'data/features',
            'predictions': 'data/predictions',
            'models': 'models'
        }
        
        if file_type not in type_mapping:
            raise HTTPException(
                status_code=400, 
                detail=f"Tipo de archivo inválido. Disponibles: {list(type_mapping.keys())}"
            )
        
        file_path = f"{type_mapping[file_type]}/{filename}"
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {filename}")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error descargando archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error descargando archivo: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud de la API."""
    return {
        "status": "healthy", 
        "service": "retail-ml-pipeline-api",
        "version": "2.0.0"
    }