# Arquitectura Cloud

A continuación se presenta la justificación y explicación de la arquitectura cloud propuesta para implementar sobre AWS.

La siguiente imagen muestra el diagrama de la solución sugerida para la nube.

![Propuesta Infraestructura API](/cloud/Propuesta_Cloud_AWS.png)

La arquitectura propuesta consta de 6 componentes principales organizados en dos categorías:

#### Pipeline de Datos (Secuencial):
* Almacenamiento de datos
* Procesamiento de datos / ETL
* Entrenamiento de modelos
* Registro de Modelos

#### Servicios de Aplicación (Paralelos):
* Despliegue de la API
* Monitoreo y Seguridad

Esta separación arquitectural permite desarrollo independiente del pipeline de ML y los servicios de aplicación, facilitando iteración rápida y deployment independiente de cada componente.

## 1. Almacenamiento de datos
**Servicios AWS**: Amazon S3 (Múltiples Buckets)

### Implementación:

* **S3 Raw**: Almacenamiento de archivos CSV originales con datos transaccionales.
* **S3 Clean**: Datos procesados listos para feature engineering y entrenamiento.


**Justificación:** La separación en múltiples buckets proporciona beneficios arquitecturales significativos. Permite implementar políticas diferentes ciclos de vida para cada tipo de dato, donde datos raw pueden archivarse a Glacier para cumplimiento regulatorio mientras datos procesados permanecen en acceso inmediato.

## 2. Preprocesamiento / ETL
**Servicios AWS**: AWS Lambda (Funciones Serverless)

### Implementación:

* **Lambda Preprocesamiento**: Limpieza de datos, eliminación de duplicados, manejo de valores nulos.
* **Lambda Feature Engineering**: Creación de características RFM, variables temporales, y variable objetivo.

**Justificación:** Lambda proporciona la abstracción perfecta para el codigo existente. Al ser funciones sin servidor, ayudan a un mejor seguimiento de la infraestructura, ademas de proporcionar autoescalado en caso de ser necesario. De igual forma, estas permiten la activación automatica cuando lleguen nuevos archivos a S3, dando como resultado una arquitectura impulsada por eventos (event-driven).

## 3. Entrenamiento de modelos
**Servicios AWS:** Amazon EC2 + Amazon SageMaker

### Implementación:

* EC2: Instancias para ejecutar código PyCaret existente con mínimas modificaciones.
* 
* SageMaker (Producción): Training jobs distribuidos, hyperparameter optimization, y experiment tracking.

**Justificación:**  Al tener un pipeline empleando con Pycaret, el usar EC2 nos permite mantener el mismo código mientras se obtienen los beneficios de la nube y el entrenamiento automatico por la libreria ya mencionada. Por otro lado, SageMaker nos auxilia a realizar ajustes finos ademas de realizar experimentaciones detalladas en caso de ser necesario.

## 4. Registro de modelos
**Servicios AWS:** Amazon S3 + SageMaker Model Registry

### Implementación:

* S3 PKL: Almacenamiento de archivos de modelo serializados (.pkl)
* SageMaker Registry: Metadata de modelos, métricas de performance, y lineage tracking

**Justificación:** Esta estrategía, permite el almacenamiento de los binarios .pkl ademas de tener politicas automaticas. Así mismo, SageMaker Model Registry agrega capacidades de gobierno, como lo pueden ser: el versionado de modelos, el despliegue de los mismo basados en metricas de rendimiento. 

## 5. Despliegue de la API
**Servicios AWS:** ECS Fargate / App Runner + Amazon API Gateway

### Implementación:

* ECS Fargate: Containerización de FastAPI con auto-scaling
* API Gateway: Request routing, authentication, rate limiting, y monitoring

**Justificación:** ECS Fargate proporciona un mayor control sobre el ECS. Mientras que App Runner se enfoca más a la simplicidad. Si bien, ambas divergen la finalidad es la misma hacer disponible la API. Ambas permiten desplegar un contenedor para la aplicación FastAPI existente sin modificaciones arquitecturales mayores. Por otro lado, API Gateway agrega capacidades críticas como la autenticación, limitación y transformación de solicitudes.


## 6. Monitoreo y Seguridad

**Servicios AWS:** Amazon CloudWatch + AWS IAM

### Implementación:

* CloudWatch: Tableros operacionales, alertas proactivas, y agregación de logs
* IAM: Control de acceso basado en roles con el principio del mínimo privilegio

**Justificación:** CloudWatch proporciona observabilidad completa del sistema con métricas tanto de infraestructura como de aplicación. Metricas personalizadas para ML (distribución de predicciones, model drift) permiten detection temprana de problemas. IAM garantiza una defensa de la seguridad en profundidad con permisos especificos y registros de auditoría completos.