import pandas as pd
import numpy as np
from pycaret.classification import *
import joblib
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Clase para entrenamiento y evaluación de modelos de clasificación.
    """
    
    def __init__(self):
        self.model = None
        self.model_path: Optional[str] = None
        self.evaluation_metrics: Dict[str, Any] = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga los datos del archivo CSV con manejo de encoding.
        
        Args:
            file_path (str): Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con los datos cargados
        """
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info("Datos cargados exitosamente con encoding UTF-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            logger.info("Datos cargados exitosamente con encoding ISO-8859-1")
        
        logger.info(f"Forma del dataset: {df.shape}")
        return df

    def setup_pycaret_environment(self, df: pd.DataFrame, target_column: str = 'WillPurchase_90Days', 
                                test_size: float = 0.2) -> object:
        """
        Configura el entorno de PyCaret para clasificación.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            target_column (str): Nombre de la columna objetivo
            test_size (float): Proporción del conjunto de prueba
            
        Returns:
            object: Objeto de configuración de PyCaret
        """
        logger.info("Configurando entorno de PyCaret...")
        
        clf = setup(
            data=df,
            target=target_column,
            train_size=1-test_size,
            session_id=123,
            use_gpu=False
        )
        
        logger.info(f"Dataset dividido: {int((1-test_size)*100)}% entrenamiento, {int(test_size*100)}% prueba")
        logger.info("Configuración completada exitosamente")
        
        return clf

    def compare_models_and_select_best(self, top_n: int = 10) -> List:
        """
        Compara múltiples algoritmos de machine learning y selecciona los mejores.
        
        Args:
            top_n (int): Número de mejores modelos a mostrar
            
        Returns:
            List: Lista con los mejores modelos
        """
        logger.info(f"Comparando algoritmos de machine learning (Top {top_n})...")
        
        best_models = compare_models(
            sort='Accuracy',
            n_select=top_n,
            verbose=False
        )
        
        logger.info("Comparación de modelos completada")
        return best_models

    def create_and_tune_best_model(self, model_list: List, metric: str = 'Accuracy') -> object:
        """
        Crea y optimiza el mejor modelo de la lista.
        
        Args:
            model_list (List): Lista de modelos de PyCaret
            metric (str): Métrica para seleccionar el mejor modelo
            
        Returns:
            object: Modelo optimizado
        """
        logger.info("Seleccionando y optimizando el mejor modelo...")
        
        best_model = model_list[0] if isinstance(model_list, list) else model_list
        
        logger.info(f"Modelo seleccionado: {type(best_model).__name__}")
        
        logger.info("Optimizando hiperparámetros...")
        tuned_model = tune_model(
            best_model,
            optimize=metric,
            verbose=False
        )
        
        logger.info("Optimización completada")
        return tuned_model

    def evaluate_model_performance(self, model: object) -> Tuple[object, pd.DataFrame]:
        """
        Evalúa el rendimiento del modelo.
        
        Args:
            model: Modelo entrenado de PyCaret
            
        Returns:
            Tuple[object, pd.DataFrame]: Evaluación y predicciones
        """
        logger.info("Evaluando rendimiento del modelo...")
        
        evaluation = evaluate_model(model)
        predictions = predict_model(model, verbose=False)
        
        logger.info("Evaluación completada")
        return evaluation, predictions

    def finalize_and_save_model(self, model: object, 
                              model_name: str = "best_purchase_prediction_model") -> Tuple[str, object]:
        """
        Finaliza el modelo y lo guarda en formato .pkl.
        
        Args:
            model: Modelo optimizado
            model_name (str): Nombre base para el archivo del modelo
            
        Returns:
            Tuple[str, object]: Ruta del archivo guardado y modelo finalizado
        """
        logger.info("Finalizando modelo...")
        
        final_model = finalize_model(model)
        
        models_dir = "../models"
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}"
        filepath = os.path.join(models_dir, filename)
        
        saved_model = save_model(final_model, filepath, verbose=False)
        
        pkl_path = f"{filepath}.pkl"
        logger.info(f"Modelo guardado exitosamente en: {pkl_path}")
        
        self.model = final_model
        self.model_path = pkl_path
        
        return pkl_path, final_model

    def get_feature_importance(self, model: object) -> Optional[pd.DataFrame]:
        """
        Obtiene la importancia de las características del modelo.
        
        Args:
            model: Modelo entrenado
            
        Returns:
            Optional[pd.DataFrame]: DataFrame con la importancia de características
        """
        try:
            logger.info("Calculando importancia de características...")
            
            importance_plot = plot_model(model, plot='feature', verbose=False, display_format='dataframe')
            
            logger.info("Importancia de características calculada")
            return importance_plot
        except Exception as e:
            logger.error(f"No se pudo calcular la importancia de características: {str(e)}")
            return None

    def complete_modeling_pipeline(self, df: pd.DataFrame, 
                                 target_column: str = 'WillPurchase_90Days') -> Tuple[object, str, object, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Pipeline completo de modelado con PyCaret.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            target_column (str): Columna objetivo
            
        Returns:
            Tuple: modelo_final, ruta_del_archivo, evaluación, predicciones, importancia_características
        """
        logger.info("Iniciando pipeline completo de modelado...")
        
        setup_result = self.setup_pycaret_environment(df, target_column)
        best_models = self.compare_models_and_select_best(top_n=5)
        tuned_model = self.create_and_tune_best_model(best_models)
        evaluation, predictions = self.evaluate_model_performance(tuned_model)
        feature_importance = self.get_feature_importance(tuned_model)
        model_path, final_model = self.finalize_and_save_model(tuned_model)
        
        logger.info("Pipeline de modelado completado exitosamente")
        
        return final_model, model_path, evaluation, predictions, feature_importance


class ModelPredictor:
    """
    Clase para realizar predicciones con modelos entrenados.
    """
    
    def __init__(self):
        self.loaded_model = None
        
    def load_model_and_predict(self, model_path: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Carga un modelo guardado y realiza predicciones.
        
        Args:
            model_path (str): Ruta al archivo .pkl del modelo
            new_data (pd.DataFrame): Datos para hacer predicciones
            
        Returns:
            pd.DataFrame: DataFrame con predicciones
        """
        logger.info(f"Cargando modelo desde: {model_path}")
        
        loaded_model = load_model(model_path.replace('.pkl', ''))
        predictions = predict_model(loaded_model, data=new_data, verbose=False)

        logger.info("Predicciones realizadas exitosamente")
        self.loaded_model = loaded_model
        return predictions

    def predict_new_customers(self, model_path: str, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predice si nuevos clientes comprarán en los próximos 90 días.
        
        Args:
            model_path (str): Ruta al modelo guardado
            customer_data (pd.DataFrame): Datos de los clientes
            
        Returns:
            pd.DataFrame: Predicciones con probabilidades
        """
        logger.info("Realizando predicciones para nuevos clientes...")
        
        predictions = self.load_model_and_predict(model_path, customer_data)
        
        predictions['Prediction'] = predictions['prediction_label'].map({
            1: 'Buy in 90 days',
            0: 'Will not buy in 90 days'
        })
        
        logger.info(f"Predicciones completadas para {len(predictions)} clientes")
        
        return predictions