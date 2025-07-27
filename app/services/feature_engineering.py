import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Clase para ingeniería de características RFM y temporales.
    """
    
    def __init__(self):
        self.analysis_date: Optional[pd.Timestamp] = None
        
    def create_rfm_features(self, df: pd.DataFrame, analysis_date: Optional[str] = None) -> pd.DataFrame:
        """
        Crea características RFM (Recency, Frequency, Monetary) para cada cliente.
        
        Args:
            df (pd.DataFrame): DataFrame con datos transaccionales
            analysis_date (str, optional): Fecha de análisis
            
        Returns:
            pd.DataFrame: DataFrame con métricas RFM por cliente
        """
        logger.info("=== CREANDO CARACTERÍSTICAS RFM ===")
        
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        if analysis_date is None:
            analysis_date = df['InvoiceDate'].max()
        else:
            analysis_date = pd.to_datetime(analysis_date)
        
        self.analysis_date = analysis_date
        logger.info(f"Fecha de análisis: {analysis_date}")
        
        rfm_data = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (analysis_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalValue': 'sum'
        }).reset_index()
        
        rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        rfm_data['AvgOrderValue'] = rfm_data['Monetary'] / rfm_data['Frequency']
        rfm_data['DaysActive'] = df.groupby('CustomerID')['InvoiceDate'].apply(
            lambda x: (x.max() - x.min()).days
        ).values
        
        customer_behavior = df.groupby('CustomerID').agg({
            'Quantity': ['sum', 'mean', 'std'],
            'UnitPrice': ['mean', 'std'],
            'Country': lambda x: x.mode()[0],
            'StockCode': 'nunique'
        }).reset_index()
        
        customer_behavior.columns = [
            'CustomerID', 'TotalQuantity', 'AvgQuantity', 'StdQuantity',
            'AvgUnitPrice', 'StdUnitPrice', 'MostFrequentCountry', 'ProductVariety'
        ]
        
        rfm_features = rfm_data.merge(customer_behavior, on='CustomerID', how='left')
        rfm_features['StdQuantity'] = rfm_features['StdQuantity'].fillna(0)
        rfm_features['StdUnitPrice'] = rfm_features['StdUnitPrice'].fillna(0)
        
        logger.info(f"Características RFM creadas para {len(rfm_features)} clientes")
        
        return rfm_features

    def create_rfm_scores(self, rfm_features: pd.DataFrame) -> pd.DataFrame:
        """
        Crea scores RFM usando quintiles.
        
        Args:
            rfm_features (pd.DataFrame): DataFrame con características RFM
            
        Returns:
            pd.DataFrame: DataFrame con scores RFM añadidos
        """
        logger.info("=== CREANDO SCORES RFM ===")
        
        rfm_scored = rfm_features.copy()
        
        rfm_scored['R_Score'] = pd.qcut(rfm_scored['Recency'], 5, labels=[5,4,3,2,1])
        rfm_scored['F_Score'] = pd.qcut(rfm_scored['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm_scored['M_Score'] = pd.qcut(rfm_scored['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        rfm_scored['RFM_Score'] = (
            rfm_scored['R_Score'].astype(str) + 
            rfm_scored['F_Score'].astype(str) + 
            rfm_scored['M_Score'].astype(str)
        )
        
        def assign_customer_segment(row):
            r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal_Customers'
            elif r >= 4 and f <= 2:
                return 'New_Customers'
            elif r >= 3 and f <= 2 and m >= 3:
                return 'Potential_Loyalists'
            elif r <= 2 and f >= 3 and m >= 3:
                return 'At_Risk'
            elif r <= 2 and f <= 2 and m >= 3:
                return 'Cannot_Lose'
            elif r >= 3 and f >= 3 and m <= 2:
                return 'Price_Sensitive'
            else:
                return 'Others'
        
        rfm_scored['Customer_Segment'] = rfm_scored.apply(assign_customer_segment, axis=1)
        
        segment_stats = rfm_scored['Customer_Segment'].value_counts()
        logger.info("Distribución de segmentos de clientes:")
        for segment, count in segment_stats.items():
            logger.info(f"  {segment}: {count:,} ({count/len(rfm_scored)*100:.1f}%)")
        
        return rfm_scored

    def create_target_variable(self, df: pd.DataFrame, prediction_days: int = 90, 
                             analysis_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """
        Crea variable objetivo: si el cliente volverá a comprar en los próximos X días.
        
        Args:
            df (pd.DataFrame): DataFrame con datos transaccionales
            prediction_days (int): Días hacia el futuro para la predicción
            analysis_date (str, optional): Fecha de corte para el análisis
            
        Returns:
            Tuple[pd.DataFrame, pd.Timestamp]: DataFrame con variable objetivo y fecha de análisis
        """
        logger.info(f"=== CREANDO VARIABLE OBJETIVO ({prediction_days} días) ===")
        
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        if analysis_date is None:
            analysis_date = df['InvoiceDate'].quantile(0.7)
        else:
            analysis_date = pd.to_datetime(analysis_date)
        
        logger.info(f"Fecha de corte para análisis: {analysis_date}")
        logger.info(f"Fecha límite para predicción: {analysis_date + pd.Timedelta(days=prediction_days)}")
        
        historical_data = df[df['InvoiceDate'] <= analysis_date]
        future_data = df[df['InvoiceDate'] > analysis_date]
        
        logger.info(f"Registros históricos: {len(historical_data):,}")
        logger.info(f"Registros futuros: {len(future_data):,}")
        
        historical_customers = set(historical_data['CustomerID'].unique())
        
        future_cutoff = analysis_date + pd.Timedelta(days=prediction_days)
        prediction_period_data = future_data[
            (future_data['InvoiceDate'] > analysis_date) & 
            (future_data['InvoiceDate'] <= future_cutoff)
        ]
        
        future_customers = set(prediction_period_data['CustomerID'].unique())
        
        logger.info(f"Clientes en período histórico: {len(historical_customers):,}")
        logger.info(f"Clientes que compran en período de predicción: {len(future_customers):,}")
        
        target_data = []
        
        for customer_id in historical_customers:
            will_purchase = 1 if customer_id in future_customers else 0
            target_data.append({
                'CustomerID': customer_id,
                'WillPurchase_90Days': will_purchase
            })
        
        target_df = pd.DataFrame(target_data)
        
        target_distribution = target_df['WillPurchase_90Days'].value_counts()
        logger.info("Distribución de variable objetivo:")
        logger.info(f"  No volverá a comprar (0): {target_distribution[0]:,} ({target_distribution[0]/len(target_df)*100:.1f}%)")
        logger.info(f"  Volverá a comprar (1): {target_distribution[1]:,} ({target_distribution[1]/len(target_df)*100:.1f}%)")
        
        return target_df, analysis_date

    def create_temporal_features(self, df: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
        """
        Crea características temporales adicionales.
        
        Args:
            df (pd.DataFrame): DataFrame con datos transaccionales
            analysis_date (pd.Timestamp): Fecha de corte para el análisis
            
        Returns:
            pd.DataFrame: DataFrame con características temporales por cliente
        """
        logger.info("=== CREANDO CARACTERÍSTICAS TEMPORALES ===")
        
        historical_data = df[df['InvoiceDate'] <= analysis_date].copy()
        
        historical_data['Year'] = historical_data['InvoiceDate'].dt.year
        historical_data['Month'] = historical_data['InvoiceDate'].dt.month
        historical_data['DayOfWeek'] = historical_data['InvoiceDate'].dt.dayofweek
        historical_data['Quarter'] = historical_data['InvoiceDate'].dt.quarter
        
        temporal_features = historical_data.groupby('CustomerID').agg({
            'InvoiceDate': [
                lambda x: x.nunique(),
                lambda x: (analysis_date - x.min()).days,
                lambda x: (x.max() - x.min()).days if len(x) > 1 else 0
            ],
            'Year': lambda x: x.nunique(),
            'Month': lambda x: x.nunique(),
            'DayOfWeek': lambda x: x.mode()[0] if len(x) > 0 else 0,
            'Quarter': lambda x: x.nunique()
        }).reset_index()
        
        temporal_features.columns = [
            'CustomerID', 'UniquePurchaseDays', 'CustomerAge_Days', 'ActivityPeriod_Days',
            'UniqueYears', 'UniqueMonths', 'PreferredDayOfWeek', 'UniqueQuarters'
        ]
        
        temporal_features['AvgDaysBetweenPurchases'] = (
            temporal_features['ActivityPeriod_Days'] / 
            (temporal_features['UniquePurchaseDays'] - 1)
        ).fillna(0)
        
        logger.info(f"Características temporales creadas para {len(temporal_features)} clientes")
        
        return temporal_features

    def combine_all_features(self, rfm_features: pd.DataFrame, target_df: pd.DataFrame, 
                           temporal_features: pd.DataFrame) -> pd.DataFrame:
        """
        Combina todas las características en un dataset final.
        
        Args:
            rfm_features (pd.DataFrame): Características RFM con scores
            target_df (pd.DataFrame): Variable objetivo
            temporal_features (pd.DataFrame): Características temporales
            
        Returns:
            pd.DataFrame: Dataset final con todas las características
        """
        logger.info("=== COMBINANDO TODAS LAS CARACTERÍSTICAS ===")
        
        final_dataset = rfm_features.merge(target_df, on='CustomerID', how='inner')
        final_dataset = final_dataset.merge(temporal_features, on='CustomerID', how='left')
        
        temporal_cols = ['UniquePurchaseDays', 'CustomerAge_Days', 'ActivityPeriod_Days',
                        'UniqueYears', 'UniqueMonths', 'PreferredDayOfWeek', 'UniqueQuarters',
                        'AvgDaysBetweenPurchases']
        
        for col in temporal_cols:
            if col in final_dataset.columns:
                final_dataset[col] = final_dataset[col].fillna(0)
        
        final_dataset['R_Score'] = final_dataset['R_Score'].astype(int)
        final_dataset['F_Score'] = final_dataset['F_Score'].astype(int)
        final_dataset['M_Score'] = final_dataset['M_Score'].astype(int)
        
        logger.info(f"Dataset final creado:")
        logger.info(f"  Filas: {len(final_dataset):,}")
        logger.info(f"  Columnas: {final_dataset.shape[1]}")
        logger.info(f"  Clientes únicos: {final_dataset['CustomerID'].nunique():,}")
        
        return final_dataset

    def feature_engineering_pipeline(self, df: pd.DataFrame, prediction_days: int = 90, 
                                   save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Pipeline completo de feature engineering.
        
        Args:
            df (pd.DataFrame): DataFrame procesado con datos transaccionales
            prediction_days (int): Días para predicción futura
            save_path (str, optional): Ruta para guardar el dataset final
            
        Returns:
            pd.DataFrame: Dataset final con todas las características
        """
        logger.info("=== INICIANDO PIPELINE DE FEATURE ENGINEERING ===")
        
        rfm_features = self.create_rfm_features(df)
        rfm_features = self.create_rfm_scores(rfm_features)
        target_df, analysis_date = self.create_target_variable(df, prediction_days)
        temporal_features = self.create_temporal_features(df, analysis_date)
        final_dataset = self.combine_all_features(rfm_features, target_df, temporal_features)
        
        if save_path:
            final_dataset.to_csv(save_path, index=False)
            logger.info(f"Dataset con feature engineering guardado en: {save_path}")
        
        logger.info("=== PIPELINE DE FEATURE ENGINEERING COMPLETADO ===")
        return final_dataset