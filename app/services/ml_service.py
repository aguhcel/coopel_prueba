import pandas as pd
from typing import Optional, Dict, Any
import logging
import os
from .preprocessing import DataProcessor
from .feature_engineering import FeatureEngineer
from .modeling import ModelTrainer, ModelPredictor

logger = logging.getLogger(__name__)

class MLService:
    """
    Servicio principal que orquesta los pipelines de ML por separado.
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.model_predictor = ModelPredictor()
        
    def preprocess_data(self, file_path: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta únicamente el pipeline de preprocesamiento.
        
        Args:
            file_path (str): Ruta al archivo de datos raw
            save_path (str, optional): Ruta para guardar datos procesados
            
        Returns:
            Dict[str, Any]: Resultados del preprocesamiento
        """
        try:
            logger.info("=== INICIANDO PREPROCESAMIENTO ===")
            
            processed_data = self.data_processor.preprocess_retail_data(file_path, save_path)
            
            results = {
                'status': 'success',
                'message': 'Preprocesamiento completado exitosamente',
                'data_shape': processed_data.shape,
                'quality_metrics': self.data_processor.quality_metrics,
                'processed_file_path': save_path if save_path else None,
                'columns': processed_data.columns.tolist()
            }
            
            logger.info("=== PREPROCESAMIENTO COMPLETADO ===")
            return results
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error en preprocesamiento: {str(e)}",
                'data_shape': None,
                'quality_metrics': {},
                'processed_file_path': None
            }
    
    def engineer_features(self, file_path: str, prediction_days: int = 90, 
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta únicamente el pipeline de feature engineering.
        
        Args:
            file_path (str): Ruta al archivo de datos procesados
            prediction_days (int): Días para predicción
            save_path (str, optional): Ruta para guardar dataset con features
            
        Returns:
            Dict[str, Any]: Resultados del feature engineering
        """
        try:
            logger.info("=== INICIANDO FEATURE ENGINEERING ===")
            
            # Cargar datos procesados
            df = pd.read_csv(file_path)
            
            # Aplicar feature engineering
            featured_data = self.feature_engineer.feature_engineering_pipeline(
                df, prediction_days, save_path
            )
            
            # Estadísticas de la variable objetivo
            target_stats = featured_data['WillPurchase_90Days'].value_counts()
            target_distribution = {
                'no_purchase': int(target_stats.get(0, 0)),
                'will_purchase': int(target_stats.get(1, 0)),
                'balance_ratio': float(target_stats.get(1, 0) / len(featured_data))
            }
            
            results = {
                'status': 'success',
                'message': 'Feature engineering completado exitosamente',
                'data_shape': featured_data.shape,
                'target_distribution': target_distribution,
                'features_created': featured_data.columns.tolist(),
                'analysis_date': str(self.feature_engineer.analysis_date) if self.feature_engineer.analysis_date else None,
                'featured_file_path': save_path if save_path else None
            }
            
            logger.info("=== FEATURE ENGINEERING COMPLETADO ===")
            return results
            
        except Exception as e:
            logger.error(f"Error en feature engineering: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error en feature engineering: {str(e)}",
                'data_shape': None,
                'target_distribution': {},
                'features_created': []
            }
    
    def train_model(self, file_path: str, target_column: str = 'WillPurchase_90Days') -> Dict[str, Any]:
        """
        Ejecuta únicamente el pipeline de entrenamiento.
        
        Args:
            file_path (str): Ruta al archivo con features
            target_column (str): Columna objetivo
            
        Returns:
            Dict[str, Any]: Resultados del entrenamiento
        """
        try:
            logger.info("=== INICIANDO ENTRENAMIENTO ===")
            
            df = pd.read_csv(file_path)
            
            final_model, model_path, evaluation, predictions, feature_importance = \
                self.model_trainer.complete_modeling_pipeline(df, target_column)
            
            accuracy = predictions['prediction_label'].eq(predictions[target_column]).mean()
            
            results = {
                'status': 'success',
                'message': 'Entrenamiento completado exitosamente',
                'model_path': model_path,
                'model_type': type(final_model).__name__,
                'data_shape': df.shape,
                'accuracy': float(accuracy),
                'feature_importance_available': feature_importance is not None,
                'evaluation_completed': evaluation is not None
            }
            
            logger.info("=== ENTRENAMIENTO COMPLETADO ===")
            return results
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error en entrenamiento: {str(e)}",
                'model_path': None,
                'model_type': None,
                'accuracy': None
            }
    
    def predict_customers(self, model_path: str, customer_data_path: str, 
                         output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            model_path (str): Ruta al modelo entrenado
            customer_data_path (str): Ruta a los datos de clientes
            output_path (str, optional): Ruta para guardar predicciones
            
        Returns:
            Dict[str, Any]: Resultados de las predicciones
        """
        try:
            logger.info("=== INICIANDO PREDICCIONES ===")
            
            customer_data = pd.read_csv(customer_data_path)
            
            predictions = self.model_predictor.predict_new_customers(model_path, customer_data)
            
            pred_stats = predictions['prediction_label'].value_counts()
            prediction_summary = {
                'total_customers': len(predictions),
                'predicted_to_buy': int(pred_stats.get(1, 0)),
                'predicted_not_to_buy': int(pred_stats.get(0, 0)),
                'buy_percentage': float(pred_stats.get(1, 0) / len(predictions) * 100)
            }
            
            # Guardar predicciones si se especifica
            if output_path:
                predictions.to_csv(output_path, index=False)
            
            results = {
                'status': 'success',
                'message': 'Predicciones completadas exitosamente',
                'prediction_summary': prediction_summary,
                'predictions_file_path': output_path if output_path else None,
                'model_used': model_path,
                'columns_in_output': predictions.columns.tolist()
            }
            
            logger.info("=== PREDICCIONES COMPLETADAS ===")
            return results
            
        except Exception as e:
            logger.error(f"Error en predicciones: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error en predicciones: {str(e)}",
                'prediction_summary': {},
                'predictions_file_path': None
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del servicio.
        
        Returns:
            Dict[str, Any]: Estado del servicio
        """
        return {
            'data_processor_ready': self.data_processor is not None,
            'feature_engineer_ready': self.feature_engineer is not None,
            'model_trainer_ready': self.model_trainer is not None,
            'model_predictor_ready': self.model_predictor is not None,
            'last_analysis_date': str(self.feature_engineer.analysis_date) if self.feature_engineer.analysis_date else None,
            'last_model_path': self.model_trainer.model_path,
            'quality_metrics': self.data_processor.quality_metrics
        }