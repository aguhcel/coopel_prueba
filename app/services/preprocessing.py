import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Clase principal para el preprocesamiento de datos retail.
    """
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.quality_metrics: Dict[str, Any] = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga los datos del archivo CSV con manejo de encoding.
        
        Args:
            file_path (str): Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con los datos cargados
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            Exception: Si hay errores en la carga
        """
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info("Datos cargados exitosamente con encoding UTF-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            logger.info("Datos cargados exitosamente con encoding ISO-8859-1")
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
        
        logger.info(f"Forma del dataset: {df.shape}")
        self.df = df
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina registros duplicados del dataset.
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame sin duplicados
        """
        df_work = df.copy()
        initial_shape = df_work.shape[0]
        df_clean = df_work.drop_duplicates()
        final_shape = df_clean.shape[0]
        
        removed_count = initial_shape - final_shape
        logger.info(f"Registros duplicados eliminados: {removed_count}")
        logger.info(f"Registros restantes: {final_shape}")
        
        return df_clean

    def handle_null_customer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina registros con CustomerID nulo.
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame sin CustomerID nulos
        """
        initial_shape = df.shape[0]
        null_customers = df['CustomerID'].isnull().sum()
        
        logger.info(f"Registros con CustomerID nulo: {null_customers}")
        
        if null_customers > 0:
            country_nulls = df.groupby('Country').agg({
                'CustomerID': lambda x: x.isnull().sum()
            }).query('CustomerID > 0').sort_values('CustomerID', ascending=False)
            
            logger.info("Países con CustomerID nulos:")
            for country, nulls in country_nulls['CustomerID'].items():
                total = len(df[df['Country'] == country])
                percent = (nulls / total) * 100
                logger.info(f"  {country}: {nulls} nulos de {total} transacciones ({percent:.1f}%)")
        
        df_clean = df[df['CustomerID'].notna()].copy()
        final_shape = df_clean.shape[0]
        
        logger.info(f"Registros eliminados: {initial_shape - final_shape}")
        logger.info(f"Registros restantes: {final_shape}")

        return df_clean

    def filter_cancelled_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra transacciones de cancelación (InvoiceNo que empiezan con 'C').
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame sin transacciones canceladas
        """
        initial_shape = df.shape[0]
        cancelled_mask = df['InvoiceNo'].str.startswith('C', na=False)
        cancelled_count = cancelled_mask.sum()
        
        logger.info("aqui")

        logger.info(f"Transacciones de cancelación encontradas: {cancelled_count}")
        
        if cancelled_count > 0:
            cancelled_df = df[cancelled_mask]
            logger.info("Análisis de transacciones canceladas:")
            logger.info(f"  Clientes únicos con cancelaciones: {cancelled_df['CustomerID'].nunique()}")
            logger.info(f"  Países con cancelaciones: {cancelled_df['Country'].nunique()}")
            logger.info(f"  Total quantity en cancelaciones: {cancelled_df['Quantity'].sum()}")
        
        df_clean = df[~cancelled_mask].copy()
        final_shape = df_clean.shape[0]
        
        logger.info(f"Registros eliminados: {initial_shape - final_shape}")
        logger.info(f"Registros restantes: {final_shape}")
        
        return df_clean

    def handle_inconsistent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja datos inconsistentes: Quantity negativa y UnitPrice = 0.
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame con datos consistentes
        """
        initial_shape = df.shape[0]
        
        negative_quantity = (df['Quantity'] < 0).sum()
        zero_quantity = (df['Quantity'] == 0).sum()
        zero_price = (df['UnitPrice'] == 0).sum()
        negative_price = (df['UnitPrice'] < 0).sum()
        
        logger.info("Análisis de datos inconsistentes:")
        logger.info(f"  Quantity negativa: {negative_quantity}")
        logger.info(f"  Quantity igual a 0: {zero_quantity}")
        logger.info(f"  UnitPrice igual a 0: {zero_price}")
        logger.info(f"  UnitPrice negativo: {negative_price}")
        
        consistent_mask = (
            (df['Quantity'] > 0) &
            (df['UnitPrice'] > 0)
        )
        
        df_clean = df[consistent_mask].copy()
        final_shape = df_clean.shape[0]
        
        logger.info(f"Registros eliminados: {initial_shape - final_shape}")
        logger.info(f"Registros restantes: {final_shape}")
        
        return df_clean

    def balance_geographical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balancea los datos geográficos eliminando países con pocas transacciones.
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame balanceado geográficamente
        """
        country_stats = df['Country'].value_counts()
        
        tier1 = country_stats[country_stats >= 1000].index.tolist()
        tier2 = country_stats[(country_stats >= 100) & (country_stats < 1000)].index.tolist()
        tier3 = country_stats[country_stats < 100].index.tolist()
        
        logger.info("Clasificación por tiers:")
        logger.info(f"  Tier 1 (>=1000 trans): {len(tier1)} países - {country_stats[tier1].sum():,} transacciones")
        logger.info(f"  Tier 2 (100-999 trans): {len(tier2)} países - {country_stats[tier2].sum():,} transacciones")
        logger.info(f"  Tier 3 (<100 trans): {len(tier3)} países - {country_stats[tier3].sum():,} transacciones")
        
        def assign_tier(country: str) -> str:
            if country in tier1:
                return 'Tier1_Principal'
            elif country in tier2:
                return 'Tier2_Mediano'
            else:
                return 'Tier3_Pequeno'
        
        def assign_region(country: str) -> str:
            europe = ['United Kingdom', 'Germany', 'France', 'Spain', 'Netherlands', 
                      'Belgium', 'Switzerland', 'Austria', 'Italy', 'Portugal', 'Norway',
                      'Denmark', 'Finland', 'Sweden', 'Poland', 'Cyprus']
            
            asia_pacific = ['Australia', 'Japan', 'Singapore', 'Hong Kong']
            americas = ['USA', 'Canada', 'Brazil']
            
            if country in europe:
                return 'Europa'
            elif country in asia_pacific:
                return 'Asia_Pacifico'
            elif country in americas:
                return 'Americas'
            else:
                return 'Otros'
        
        df['Country_Tier'] = df['Country'].apply(assign_tier)
        df['Region'] = df['Country'].apply(assign_region)
        
        initial_shape = df.shape[0]
        df_balanced = df[
            (df['Country_Tier'].isin(['Tier1_Principal', 'Tier2_Mediano'])) &
            (df['Region'].isin(['Europa', 'Asia_Pacifico', 'Americas']))
        ].copy()
        
        final_shape = df_balanced.shape[0]
        
        logger.info("Filtrado geográfico aplicado:")
        logger.info(f"  Registros eliminados: {initial_shape - final_shape}")
        logger.info(f"  Registros restantes: {final_shape}")
        logger.info(f"  Porcentaje mantenido: {(final_shape / initial_shape) * 100:.1f}%")
        
        return df_balanced

    def handle_skewed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja datos sesgados aplicando transformaciones.
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame con transformaciones aplicadas
        """
        logger.info("Análisis de sesgo en UnitPrice:")
        
        original_skew = stats.skew(df['UnitPrice'])
        original_kurtosis = stats.kurtosis(df['UnitPrice'])
        
        logger.info(f"  Sesgo original: {original_skew:.3f}")
        logger.info(f"  Curtosis original: {original_kurtosis:.3f}")
        
        df['UnitPrice_Log'] = np.log1p(df['UnitPrice'])
        log_skew = stats.skew(df['UnitPrice_Log'])
        
        p99 = df['UnitPrice'].quantile(0.99)
        p1 = df['UnitPrice'].quantile(0.01)
        df['UnitPrice_Winsorized'] = df['UnitPrice'].clip(lower=p1, upper=p99)
        winsor_skew = stats.skew(df['UnitPrice_Winsorized'])
        
        df['TotalValue'] = df['Quantity'] * df['UnitPrice']
        df['TotalValue_Log'] = np.log1p(df['TotalValue'].clip(lower=0))
        
        df['Price_Segment'] = pd.cut(df['UnitPrice'], 
                                    bins=[0, 1, 5, 20, np.inf], 
                                    labels=['Bajo', 'Medio', 'Alto', 'Premium'])
        
        logger.info("Transformaciones aplicadas:")
        logger.info(f"  Sesgo log-transformado: {log_skew:.3f}")
        logger.info(f"  Sesgo winsorizado: {winsor_skew:.3f}")
        logger.info(f"  Mejora en sesgo (log): {((original_skew - log_skew) / original_skew * 100):.1f}%")
        
        return df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida la calidad del dataset procesado.
        
        Args:
            df (pd.DataFrame): DataFrame procesado
            
        Returns:
            Dict[str, Any]: Métricas de calidad
        """
        logger.info("=== VALIDACIÓN DE CALIDAD DE DATOS ===")
        
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        date_range = (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days
        
        customer_stats = df.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',
            'TotalValue': 'sum'
        }).describe()
        
        quality_metrics = {
            'total_rows': int(df.shape[0]),
            'total_columns': int(df.shape[1]),
            'unique_customers': int(df['CustomerID'].nunique()),
            'date_range_days': int(date_range),
            'has_duplicates': bool(df.duplicated().sum() > 0),
            'has_null_customer_id': bool(df['CustomerID'].isnull().sum() > 0),
            'data_quality_score': 100
        }
        
        logger.info(f"Filas: {quality_metrics['total_rows']:,}")
        logger.info(f"Columnas: {quality_metrics['total_columns']}")
        logger.info(f"Clientes únicos: {quality_metrics['unique_customers']:,}")
        logger.info(f"Días de datos: {quality_metrics['date_range_days']}")
        
        self.quality_metrics = quality_metrics
        return quality_metrics

    def preprocess_retail_data(self, file_path: str, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Pipeline completo de preprocesamiento de datos retail.
        
        Args:
            file_path (str): Ruta al archivo de datos original
            save_path (str, optional): Ruta para guardar datos procesados
            
        Returns:
            pd.DataFrame: Dataset procesado
        """
        logger.info("=== INICIANDO PIPELINE DE PREPROCESAMIENTO ===")
        
        df = self.load_data(file_path)
        df = self.remove_duplicates(df)
        df = self.handle_null_customer_id(df)
        df = self.filter_cancelled_transactions(df)
        df = self.handle_inconsistent_data(df)
        df = self.balance_geographical_data(df)
        df = self.handle_skewed_data(df)
        
        self.validate_data_quality(df)
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Datos procesados guardados en: {save_path}")
        
        logger.info("=== PIPELINE COMPLETADO EXITOSAMENTE ===")
        self.df = df
        return df