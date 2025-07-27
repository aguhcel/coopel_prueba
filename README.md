# Prueba Técnica - Coopel

El presente proyecto pretende ser un acercamiento sobre una solución integra para predecir si un cliente volvera a comprar un producto con ayuda de modelos de ML tradicional. Con la intención de ser expuesto mediante una API.

## Authors
- [Alejandro Lechuga (@aguhcel)](https://github.com/Irving67/)

## Estructura del Proyecto

El proyecto se encuentra segmentado de la siguiente manera:


```
coopel_prueba/
├── app/                                    # API 
│
├── coud/                                   # Propuesta de Arquitectura Cloud AWS
│
├── data/                                   # Datos empleados
│
├── models/                                 # Modelos entrenados
│
├── notebooks/                              # Notebooks empleados para el desarrollo 
│                                           # del proyecto
│
├── reports/                                # Reportes realizados
```

## Instalación Local
```bash
# 1. Clonar el repositorio
git clone git@github.com:aguhcel/coopel_prueba.git
cd coopel_prueba

# 2. Crear entorno virtual
# ALTAMENTE RECOMENDABLE CREAR UN ENTORNO POR requirements.txt
python -m venv venv
source venv/bin/activate    # En windows venv\Scripts\activate

# 3. Instalar dependecias
pip install -r 01_requirements_explore.txt
# pip install -r 02_requirements_modeling.txt
```

## Instalación con Docker (Ejecucion de la API) 
```bash
# 1. Clonar el repositorio
git clone git@github.com:aguhcel/coopel_prueba.git
cd coopel_prueba

# 2. Build de la imagen
docker build -t simple_app .

# 3. Ejecutar imagen de docker
docker run -it -p 8000:8000 -v $PWD:{PATH}/coopel_prueba simple_app
# En windows
# docker run -it -p 8000:8000 -v $cd:{PATH}/coopel_prueba simple_app
```

Deberías de ver el siguiente mensaje en pantalla
```bash
INFO:     Will watch for changes in these directories: ['{PATH}/coopel_prueba']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [1] using WatchFiles
INFO:     Started server process [8]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     192.168.65.1:47593 - "GET / HTTP/1.1" 200 OK
INFO:     192.168.65.1:47592 - "GET /docs HTTP/1.1" 200 OK
INFO:     192.168.65.1:47592 - "GET /openapi.json HTTP/1.1" 200 OK
```

## API Reference
### /docs
Swagger en donde puede probar de forma más comoda la API. 

```http
  GET http://0.0.0.0:8000/docs
```

Una vez levantado el servicio, dispondrás de varios endpoints con los que podrás interactuar con el servicio:

#### /health
Endpoint de verificación de salud de la API.

```http
  GET http://0.0.0.0:8000/health
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
|           |          | No se requieren argumentos |

#### /status
Endpoint de estado del servicio. Regresa un diccionario con el estado actual del servicio

```http
  GET http://0.0.0.0:8000/status
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
|           |          | No se requieren argumentos |

#### /models
Endpoint para listar todos los modelos disponibles. Regresa un diccionario con la lista de los modelos disponibles
```http
  GET http://0.0.0.0:8000/models
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
|           |          | No se requieren argumentos |

#### /download
Endpoint para descargar archivos generados.

```http
  GET http://0.0.0.0:8000/download/{file_type}/{filename}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `file_type`  | `string` | **Obligatorio**. Path del tipo de archivo (processed, features, predictions, models)|
| `filename`  | `string` | **Obligatorio**. Nombre del archivo |


#### /preprocess

Endpoint 1: Preprocesamiento de datos. Regresa un diccionario con los resultados del preprocesamiento.

```http
  POST http://0.0.0.0:8000/preprocess
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `file`  | `string($binary)` | **Obligatorio**. Tipo de archivo (processed, features, predictions, models)|
| `save_processed_data`  | `boolean` | Guardar datos procesados |
| `output_filename`  | `string` | Nombre del archivo de salida |

#### /feature-engineering

Endpoint 2: Ingeniería de características. Regresa un diccionario con los resultados del feature engineering.

```http
  POST http://0.0.0.0:8000/feature-engineering
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `file`  | `string($binary)` | **Obligatorio**. Archivo CSV con datos procesados|
| `prediction_days`  | `integer` | Días para predicción futura |
| `save_features`  | `boolean` | Guardar dataset con características |
| `output_filename`  | `string` | Nombre del archivo de salida |

#### /train

Endpoint 3: Entrenamiento de modelo. Regresa un diccionario con los resultados del entrenamiento.

```http
  POST http://0.0.0.0:8000/train
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `file`  | `string($binary)` | **Obligatorio**. Archivo CSV con features engineered|
| `target_column`  | `string` | Columna objetivo para predicción |

#### /predict

Endpoint 4: Predicciones. Regresa un diccionario con los resultados de las predicciones.

```http
  POST http://0.0.0.0:8000/predict
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `file`  | `string($binary)` | **Obligatorio**. Archivo CSV con datos de clientes|
| `model_filename`  | `string(string or null)` | Nombre del modelo a usar |
| `output_filename`  | `string` | Nombre del archivo de predicciones |

# Áreas de mejora.
Al momento de entregar este proyecto se encuentra en una version MVP, a continuación se detallan puntos de mejora.

* **Robustez del Pipeline de Datos**
  El pipeline asume datos perfectos. En producción, los datos cambian constantemente. 
  
  **Propuesta**: Implementar validación automática de esquemas y detección de drift.

* **Arquitectura de Modelo y Versionado**
  Solo se tiene un modelo en producción sin rollback. 
  
  **Propuesta**: Sistema de versionado semántico (v1.2.3) con metadata completo.

* **Model Registry con A/B Testing**
  No se puede comparar performance de modelos diferentes de manera controlada.

  **Propuesta**: Split traffic (90% modelo actual, 10% nuevo) con métricas en tiempo real. 

* **Features Engineering Avanzado**
   RFM básico.

   **Propuesta**: Tendencias temporales (¿está comprando más/menos frecuentemente?,Features de estacionalidad (comportamiento por mes/trimestre), Customer similarity features (comportamiento de peers).

* **Interpretabilidad y Explicabilidad**
  Predicciones sin explicación

  **Propuesta**: Integrar SHAP values para explicar cada predicción individual.
  
* **Real-time vs Batch Processing**
  Solo batch processing limita casos de uso.

  **Propuesta**: Batch, para reentrenamiento y análisis masivo. Real-time, Para scoring individual en websites/apps. Feature Store, Pre-computar features RFM para latencia <100ms.