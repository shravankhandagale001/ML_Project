import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('solar_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    r2_score: float
    mse: float
    rmse: float
    mae: float
    mape: float
    accuracy_percentage: float
    train_r2: float = None
    overfitting_score: float = None

class SolarEnergyPredictor:
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.target_column = 'solar_energy_output'
        self.metrics = None
        self.is_trained = False
        
        self.scaler = RobustScaler()
        
        self.expected_columns = [
            'Year', 'Month', 'Day', 'Hour', 'Minute', 'DHI', 'DNI', 'Dew Point', 
            'Temperature', 'Pressure', 'Relative Humidity', 'Snow Depth', 'Wind Speed', 
            'Solar Zenith Angle', 'Precipitable Water', 'Clearsky GHI', 'GHI', 
            'Clearsky DNI', 'Clearsky DHI'
        ]
    

    def predict_from_user_input(self, user_data: Dict) -> Dict:
        """
        Simplified prediction interface that requires only essential parameters.
        Automatically fills reasonable defaults for missing parameters.
        
        Args:
            user_data: Dictionary containing at minimum:
                - 'Hour' (0-23)
                - 'Temperature' (in °C)
                - 'Solar Zenith Angle' (degrees)
                - 'GHI' (W/m²)
                - 'Relative Humidity' (%)
                - 'Wind Speed' (m/s)
                - 'Month' (1-12)
                
        Returns:
            Prediction results dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Validate required parameters
        required_params = ['Hour', 'Temperature', 'Solar Zenith Angle', 'GHI',
                         'Relative Humidity', 'Wind Speed', 'Month']
        missing_params = [p for p in required_params if p not in user_data]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Default values for optional parameters
        defaults = {
            'Year': datetime.now().year,
            'Day': 15,  # Mid-month default
            'Minute': 0,
            'Pressure': 1013,  # hPa
            'Dew Point': user_data['Temperature'] - 5,  # Estimate from temp
            'Snow Depth': 0,
            'Precipitable Water': 10,
            'DHI': user_data['GHI'] * 0.4,  # Estimate diffuse component
            'DNI': user_data['GHI'] * 0.6,  # Estimate direct component
            'Clearsky GHI': user_data['GHI'] * 1.1,
            'Clearsky DHI': user_data['GHI'] * 0.4,
            'Clearsky DNI': user_data['GHI'] * 0.7
        }
        
        # Merge user input with defaults (user values take precedence)
        complete_data = {**defaults, **user_data}
        
        # Create a single-row DataFrame
        input_df = pd.DataFrame([complete_data])
        
        # Process the input data through the same pipeline
        processed_input = self.preprocess_data(input_df)
        processed_input = self.engineer_features(processed_input)
        
        # Make prediction
        return self.predict(processed_input)
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            
            missing_cols = set(self.expected_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing expected columns: {missing_cols}")
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            missing_stats = df.isnull().sum()
            missing_count = missing_stats[missing_stats > 0]
            if not missing_count.empty:
                logger.warning(f"Missing values found:\n{missing_count}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        self._handle_missing_values(df_processed)
        df_processed = self._create_datetime_features_fast(df_processed)
        df_processed = self._clean_solar_data(df_processed)
        df_processed = self._clean_weather_data(df_processed)
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame):
        solar_cols = ['GHI', 'DHI', 'DNI', 'Clearsky GHI', 'Clearsky DHI', 'Clearsky DNI']
        for col in solar_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        weather_numeric_cols = ['Temperature', 'Pressure', 'Relative Humidity', 
                               'Wind Speed', 'Dew Point', 'Precipitable Water']
        for col in weather_numeric_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        if 'Snow Depth' in df.columns:
            df['Snow Depth'].fillna(0, inplace=True)
        
        if 'Solar Zenith Angle' in df.columns and df['Solar Zenith Angle'].isnull().sum() > 0:
            df['Solar Zenith Angle'].fillna(df['Solar Zenith Angle'].median(), inplace=True)
    
    def _create_datetime_features_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
            
            df['day_of_year'] = df['datetime'].dt.dayofyear
            df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
            
            df['hour_angle'] = (df['Hour'] + df['Minute']/60 - 12) * 15
            df['solar_elevation_approx'] = 90 - df.get('Solar Zenith Angle', 0)
            
            df['season'] = df['Month'].map({
                12: 0, 1: 0, 2: 0,
                3: 1, 4: 1, 5: 1,
                6: 2, 7: 2, 8: 2,
                9: 3, 10: 3, 11: 3
            })
            
            # Cyclical encoding for hour and month
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            
            # Daylight indicator
            df['IsDaylight'] = ((df['Hour'] >= 6) & (df['Hour'] <= 18)).astype(int)
            
        except Exception as e:
            logger.warning(f"Could not create datetime features: {e}")
        
        return df
    
    def _clean_solar_data(self, df: pd.DataFrame) -> pd.DataFrame:
        solar_cols = ['GHI', 'DHI', 'DNI', 'Clearsky GHI', 'Clearsky DHI', 'Clearsky DNI']
        
        for col in solar_cols:
            if col in df.columns:
                negative_mask = df[col] < 0
                if negative_mask.sum() > 0:
                    df.loc[negative_mask, col] = 0
                
                if 'GHI' in col:
                    high_mask = df[col] > 1400
                    if high_mask.sum() > 0:
                        df.loc[high_mask, col] = 1400
                
                elif 'DNI' in col:
                    high_mask = df[col] > 1000
                    if high_mask.sum() > 0:
                        df.loc[high_mask, col] = 1000
        
        return df
    
    def _clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        bounds = {
            'Temperature': (-50, 60),
            'Relative Humidity': (0, 100),
            'Pressure': (900, 1100),
            'Wind Speed': (0, 50),
            'Dew Point': (-60, 50),
            'Snow Depth': (0, 1000),
            'Solar Zenith Angle': (0, 180),
            'Precipitable Water': (0, 100)
        }
        
        for col, (min_val, max_val) in bounds.items():
            if col in df.columns:
                outliers = (df[col] < min_val) | (df[col] > max_val)
                if outliers.sum() > 0:
                    df[col] = np.clip(df[col], min_val, max_val)
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, target_type: str = 'energy_output',
                             panel_specs: Dict = None) -> pd.DataFrame:
        df_with_target = df.copy()
        
        if panel_specs is None:
            panel_specs = {
                'panel_area': 20,
                'panel_efficiency': 0.20,
                'system_efficiency': 0.85,
                'temperature_coefficient': -0.004,
                'optimal_temperature': 25
            }
        
        if target_type == 'energy_output':
            solar_energy = self._calculate_energy_output(df_with_target, panel_specs)
        elif target_type == 'ghi_prediction':
            solar_energy = self._create_ghi_prediction_target(df_with_target) #this part can be removed
        elif target_type == 'efficiency_index':
            solar_energy = self._calculate_efficiency_index(df_with_target)
        else:
            raise ValueError("target_type must be 'energy_output', 'ghi_prediction', or 'efficiency_index'")
        
        df_with_target[self.target_column] = solar_energy
        
        return df_with_target
    
    def _calculate_energy_output(self, df: pd.DataFrame, panel_specs: Dict) -> np.ndarray:
        if 'GHI' not in df.columns:
            raise ValueError("GHI column required for energy output calculation")
        
        base_energy = (df['GHI'] * panel_specs['panel_area'] * 
                      panel_specs['panel_efficiency'] / 1000)
        
        if 'Temperature' in df.columns:
            temp_factor = 1 + panel_specs['temperature_coefficient'] * \
                         (df['Temperature'] - panel_specs['optimal_temperature'])
            temp_factor = np.clip(temp_factor, 0.6, 1.2)
            base_energy *= temp_factor
        
        energy_output = base_energy * panel_specs['system_efficiency']
        
        if 'Snow Depth' in df.columns:
            snow_factor = np.where(df['Snow Depth'] > 0, 
                                 np.maximum(0.1, 1 - df['Snow Depth'] / 100), 1)
            energy_output *= snow_factor
        
        if 'Wind Speed' in df.columns:
            wind_factor = 1 + np.minimum(0.05, df['Wind Speed'] / 100)
            energy_output *= wind_factor
        
        if 'Solar Zenith Angle' in df.columns:
            angle_factor = np.where(df['Solar Zenith Angle'] > 75, 
                                   np.maximum(0.1, 1 - (df['Solar Zenith Angle'] - 75) / 50), 1)
            energy_output *= angle_factor
        
        noise_factor = np.random.normal(1.0, 0.015, len(df))
        energy_output *= noise_factor
        
        energy_output = np.maximum(0, energy_output)
        
        return energy_output
    
    def _create_ghi_prediction_target(self, df: pd.DataFrame) -> np.ndarray:
        if 'GHI' not in df.columns:
            raise ValueError("GHI column required for GHI prediction target") #This can be removed
        
        ghi_target = df['GHI'].copy()
        if len(df) > 1:
            ghi_target = df['GHI'].shift(-1).fillna(df['GHI'])
        
        return ghi_target.values
    
    def _calculate_efficiency_index(self, df: pd.DataFrame) -> np.ndarray:
        required_cols = ['GHI', 'Clearsky GHI']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Columns {required_cols} required for efficiency index")
        
        efficiency_index = df['GHI'] / (df['Clearsky GHI'] + 1e-6)
        
        weather_penalty = 0
        if 'Relative Humidity' in df.columns:
            weather_penalty += (df['Relative Humidity'] / 100) * 0.1
        
        efficiency_index = efficiency_index * (1 - weather_penalty)
        efficiency_index = np.clip(efficiency_index, 0, 1.5)
        
        return efficiency_index * 100
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_engineered = df.copy()
        
        # Drop columns we won't use
        cols_to_drop = ['Month', 'Day', 'Hour', 'GHI', 'Clearsky GHI', 
                       'DHI', 'Clearsky DNI', 'DNI', 'Clearsky DHI']
        df_engineered = df_engineered.drop(columns=[col for col in cols_to_drop if col in df_engineered.columns])
        
        # Create interaction features
        if all(col in df_engineered.columns for col in ['Wind Speed', 'Temperature']):
            df_engineered['Wind_Temp'] = df_engineered['Wind Speed'] * df_engineered['Temperature']
        
        if all(col in df_engineered.columns for col in ['Temperature', 'Relative Humidity']):
            df_engineered['Temp_RelHumidity'] = df_engineered['Temperature'] * df_engineered['Relative Humidity'] / 100
        
        if all(col in df_engineered.columns for col in ['Wind Speed', 'Relative Humidity']):
            df_engineered['Wind_RelHumidity'] = df_engineered['Wind Speed'] * df_engineered['Relative Humidity']
        
        return df_engineered
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2, 
                   optimize_hyperparameters: bool = True, cv_folds: int = 3) -> Dict:
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        cols_to_drop = ['datetime', 'Year', 'Minute', self.target_column]
        feature_cols = [col for col in df.columns if col not in cols_to_drop]
        
        X = df[feature_cols].copy()
        y = df[self.target_column].copy()
        
        X = X.fillna(X.median())
        
        self.feature_names = list(X.columns)
        
        if 'datetime' in df.columns:
            sort_idx = df['datetime'].argsort()
            n_test = int(len(df) * test_size)
            
            train_idx = sort_idx[:-n_test]
            test_idx = sort_idx[-n_test:]
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        if self.model_type == 'random_forest':
            base_model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state, 
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('regressor', base_model)
        ])
        
        if optimize_hyperparameters:
            self.pipeline = self._fast_optimize_hyperparameters(
                self.pipeline, X_train, y_train, cv_folds
            )
        
        self.pipeline.fit(X_train, y_train)
        self.model = self.pipeline.named_steps['regressor']
        self.is_trained = True
        
        y_train_pred = self.pipeline.predict(X_train)
        y_test_pred = self.pipeline.predict(X_test)
        
        self.metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train, cv=cv_folds, scoring='r2', n_jobs=-1
        )
        
        self._log_model_performance(cv_scores)
        
        return {
            'train_score': self.metrics.train_r2,
            'test_score': self.metrics.r2_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'feature_importance': self._get_feature_importance()
        }
    
    def _fast_optimize_hyperparameters(self, pipeline: Pipeline, X_train: pd.DataFrame, 
                                      y_train: pd.Series, cv_folds: int) -> Pipeline:
        if self.model_type == 'random_forest':
            param_distributions = {
                'regressor__n_estimators': [50, 100, 150],
                'regressor__max_depth': [10, 15, 20, None],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
                'regressor__max_features': ['sqrt', 'log2']
            }
        
        n_iter = min(12, len(X_train) // 1000)
        n_iter = max(n_iter, 6)
        
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=0,
            random_state=self.random_state
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best hyperparameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def _calculate_metrics(self, y_train: pd.Series, y_train_pred: np.ndarray,
                          y_test: pd.Series, y_test_pred: np.ndarray) -> ModelMetrics:
        r2 = r2_score(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        
        mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-8))) * 100
        accuracy = max(0, 100 - mape)
        
        train_r2 = r2_score(y_train, y_train_pred)
        overfitting = train_r2 - r2
        
        return ModelMetrics(
            r2_score=r2,
            mse=mse,
            rmse=rmse,
            mae=mae,
            mape=mape,
            accuracy_percentage=accuracy,
            train_r2=train_r2,
            overfitting_score=overfitting
        )
    
    def _get_feature_importance(self) -> pd.DataFrame:
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _log_model_performance(self, cv_scores: np.ndarray):
        logger.info("="*70)
        logger.info("SOLAR ENERGY MODEL PERFORMANCE SUMMARY")
        logger.info("="*70)
        logger.info(f"Model Type: {self.model_type.upper()}")
        logger.info(f"Test R² Score: {self.metrics.r2_score:.4f}")
        logger.info(f"Test RMSE: {self.metrics.rmse:.2f}")
        logger.info(f"Test MAE: {self.metrics.mae:.2f}")
        logger.info(f"Test MAPE: {self.metrics.mape:.2f}%")
        logger.info(f"Model Accuracy: {self.metrics.accuracy_percentage:.2f}%")
        logger.info(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        if self.metrics.r2_score >= 0.9:
            performance = "EXCELLENT"
        elif self.metrics.r2_score >= 0.8:
            performance = "VERY GOOD" 
        elif self.metrics.r2_score >= 0.7:
            performance = "GOOD"
        elif self.metrics.r2_score >= 0.6:
            performance = "FAIR"
        else:
            performance = "NEEDS IMPROVEMENT"
        
        logger.info(f"Performance Level: {performance}")
        
        if self.metrics.overfitting_score > 0.15:
            logger.warning(f"HIGH overfitting detected (gap: {self.metrics.overfitting_score:.4f})")
        elif self.metrics.overfitting_score > 0.05:
            logger.warning(f"MILD overfitting detected (gap: {self.metrics.overfitting_score:.4f})")
        else:
            logger.info(f"Good generalization (train-test gap: {self.metrics.overfitting_score:.4f})")
        
        logger.info("="*70)
    
    def predict(self, X_new: pd.DataFrame, return_confidence: bool = True) -> Dict:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_pred = self._prepare_prediction_features(X_new)
        
        predictions = self.pipeline.predict(X_pred)
        
        results = {
            'predictions': predictions,
            'input_data': X_new,
            'features_used': X_pred
        }
        
        if return_confidence and hasattr(self.model, 'estimators_'):
            confidence_info = self._calculate_prediction_confidence(X_pred, predictions)
            results.update(confidence_info)
        
        return results
    
    def _prepare_prediction_features(self, X_new: pd.DataFrame) -> pd.DataFrame:
        missing_features = set(self.feature_names) - set(X_new.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
            for feature in missing_features:
                if any(x in feature.lower() for x in ['_ma', '_ratio', '_diff']):
                    X_new[feature] = 0
                elif 'temperature' in feature.lower():
                    X_new[feature] = 20
                elif 'humidity' in feature.lower():
                    X_new[feature] = 60
                elif 'pressure' in feature.lower():
                    X_new[feature] = 1013
                elif 'wind' in feature.lower():
                    X_new[feature] = 3
                else:
                    X_new[feature] = 0
        
        X_pred = X_new[self.feature_names].copy()
        X_pred = X_pred.fillna(X_pred.median())
        
        return X_pred
    
    def _calculate_prediction_confidence(self, X_pred: pd.DataFrame, predictions: np.ndarray) -> Dict:
        fitted_scaler = self.pipeline.named_steps['scaler']
        
        tree_predictions = np.array([
            tree.predict(fitted_scaler.transform(X_pred)) 
            for tree in self.model.estimators_
        ])
        
        prediction_std = np.std(tree_predictions, axis=0)
        prediction_var = np.var(tree_predictions, axis=0)
        
        confidence_intervals = 1.96 * prediction_std
        
        max_std = np.max(prediction_std) + 1e-8
        confidence_scores = 100 * (1 - (prediction_std / max_std))
        confidence_scores = np.clip(confidence_scores, 0, 100)
        
        lower_bounds = predictions - confidence_intervals
        upper_bounds = predictions + confidence_intervals
        
        return {
            'confidence_scores': confidence_scores,
            'prediction_std': prediction_std,
            'prediction_variance': prediction_var,
            'confidence_intervals': confidence_intervals,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds
        }
    
    def plot_results(self, training_results: Dict, figsize: Tuple[int, int] = (15, 10)):
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting results")
        
        y_test = training_results['y_test']
        y_pred = training_results['y_pred']
        feature_importance = training_results.get('feature_importance')
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Solar Energy Prediction Model Results ({self.model_type.title()})', 
                     fontsize=16, fontweight='bold')
        
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=20, c='orange')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Solar Energy Output')
        axes[0, 0].set_ylabel('Predicted Solar Energy Output')
        axes[0, 0].set_title(f'Actual vs Predicted (R² = {self.metrics.r2_score:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20, c='lightblue')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Solar Energy Output')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='lightgreen')
        axes[0, 2].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {residuals.mean():.2f}')
        axes[0, 2].set_xlabel('Prediction Errors')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Error Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        if feature_importance is not None:
            top_features = feature_importance.head(10)
            y_pos = np.arange(len(top_features))
            axes[1, 0].barh(y_pos, top_features['importance'], color='forestgreen', alpha=0.8)
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(top_features['feature'], fontsize=8)
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 10 Feature Importances')
            axes[1, 0].invert_yaxis()
            axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        cv_scores = training_results['cv_scores']
        axes[1, 1].bar(range(len(cv_scores)), cv_scores, alpha=0.8, color='orange')
        axes[1, 1].axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {cv_scores.mean():.3f}')
        axes[1, 1].set_xlabel('CV Fold')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Cross-Validation Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].axis('off')
        metrics_text = f"""
MODEL PERFORMANCE

R² Score: {self.metrics.r2_score:.4f}
RMSE: {self.metrics.rmse:.2f}
MAE: {self.metrics.mae:.2f}
MAPE: {self.metrics.mape:.2f}%
Accuracy: {self.metrics.accuracy_percentage:.2f}%

Cross-Validation:
   Mean R²: {cv_scores.mean():.4f}
   Std R²: {cv_scores.std():.4f}

Overfitting Check:
   Train R²: {self.metrics.train_r2:.4f}
   Test R²: {self.metrics.r2_score:.4f}
   Gap: {self.metrics.overfitting_score:.4f}
        """
        
        axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1, 2].set_title('Performance Summary')
        
        plt.tight_layout()
        plt.show()
    
    def load_model(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        
        model_path = filepath.with_suffix('.pkl')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.pipeline = joblib.load(model_path)
        self.model = self.pipeline.named_steps['regressor']
        self.scaler = self.pipeline.named_steps['scaler']
        
        info_path = filepath.with_suffix('.json')
        if info_path.exists():
            import json
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            
            self.model_type = model_info['model_type']
            self.feature_names = model_info['feature_names']
            self.target_column = model_info['target_column']
            
            if model_info['metrics']:
                metrics_dict = model_info['metrics']
                self.metrics = ModelMetrics(
                    r2_score=metrics_dict['r2_score'],
                    mse=0,
                    rmse=metrics_dict['rmse'],
                    mae=metrics_dict['mae'],
                    mape=metrics_dict['mape'],
                    accuracy_percentage=metrics_dict['accuracy_percentage']
                )
        
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")
    
    def generate_prediction_report(self, prediction_results: Dict, 
                                 output_file: Optional[str] = None) -> str:
        predictions = prediction_results['predictions']
        input_data = prediction_results['input_data']
        
        report_lines = [
            "="*80,
            "SOLAR ENERGY PREDICTION REPORT",
            "="*80,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model Type: {self.model_type.title()}",
            f"Target: Solar Energy Output (kWh)",
            ""
        ]
        
        if self.metrics:
            report_lines.extend([
                "MODEL PERFORMANCE:",
                "-" * 50,
                f"R² Score: {self.metrics.r2_score:.4f}",
                f"RMSE: {self.metrics.rmse:.2f} kWh",
                f"MAE: {self.metrics.mae:.2f} kWh",
                f"MAPE: {self.metrics.mape:.2f}%",
                f"Accuracy: {self.metrics.accuracy_percentage:.2f}%",
                ""
            ])
        
        report_lines.extend([
            "PREDICTION SUMMARY:",
            "-" * 50,
            f"Number of predictions: {len(predictions)}",
            f"Average predicted energy: {predictions.mean():.2f} kWh",
            f"Energy range: {predictions.min():.2f} - {predictions.max():.2f} kWh",
            f"Standard deviation: {predictions.std():.2f} kWh",
            f"Total predicted energy: {predictions.sum():.2f} kWh",
            ""
        ])
        
        n_show = min(10, len(predictions))
        report_lines.extend([
            f"SAMPLE PREDICTIONS (showing first {n_show}):",
            "-" * 50
        ])
        
        for i in range(n_show):
            pred = predictions[i]
            line = f"Sample {i+1:3d}: {pred:7.2f} kWh"
            
            if 'confidence_scores' in prediction_results:
                conf = prediction_results['confidence_scores'][i]
                line += f" (Confidence: {conf:5.1f}%)"
            
            report_lines.append(line)
        
        if len(predictions) > n_show:
            report_lines.append(f"... and {len(predictions) - n_show} more predictions")
        
        report_lines.extend([
            "",
            "PERFORMANCE INSIGHTS:",
            "-" * 50
        ])
        
        if self.metrics:
            if self.metrics.r2_score >= 0.9:
                report_lines.append("EXCELLENT model performance")
            elif self.metrics.r2_score >= 0.8:
                report_lines.append("VERY GOOD model performance")
            elif self.metrics.r2_score >= 0.7:
                report_lines.append("GOOD model performance")
            else:
                report_lines.append("FAIR model performance")
        
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report


def demo_solar_energy_workflow():
    model = SolarEnergyPredictor(model_type='random_forest', random_state=42)
    
    try:
        df = model.load_data('D:\pict_ml_project\Solar_DataSet_Techfest(1).csv')
        
        df_processed = model.preprocess_data(df)
        
        df_with_target = model.create_target_variable(
            df_processed, 
            target_type='energy_output',
            panel_specs={
                'panel_area': 25,
                'panel_efficiency': 0.21,
                'system_efficiency': 0.87,
                'temperature_coefficient': -0.0038,
                'optimal_temperature': 25
            }
        )
        
        df_final = model.engineer_features(df_with_target)
        
        training_results = model.train_model(
            df_final,
            test_size=0.2,
            optimize_hyperparameters=True,
            cv_folds=3
        )
        
        model.plot_results(training_results, figsize=(15, 10))
        
        X_test = training_results['X_test']
        sample_data = X_test.head(100)
        
        prediction_results = model.predict(sample_data, return_confidence=True)
        
        report = model.generate_prediction_report(prediction_results)
        print(report)
        
        return model, training_results, prediction_results
        
    except FileNotFoundError:
        print("DATASET FILE NOT FOUND!")
        print("Ensure 'Solar_DataSet_Techfest(1).csv' is in the current directory")
        
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        raise


def quick_train(csv_file_path: str, sample_predictions: int = 5):
    model = SolarEnergyPredictor(model_type='random_forest')
    
    df = model.load_data(csv_file_path)
    df_processed = model.preprocess_data(df)
    df_with_target = model.create_target_variable(df_processed, target_type='energy_output')
    df_final = model.engineer_features(df_with_target)
    
    results = model.train_model(df_final, optimize_hyperparameters=False, cv_folds=3)
    
    X_test = results['X_test']
    sample_data = X_test.head(100)
    predictions = model.predict(sample_data, return_confidence=False)
    
    print(f"Model R² Score: {model.metrics.r2_score:.3f}")
    print(f"Model Accuracy: {model.metrics.accuracy_percentage:.1f}%")
    print(f"Sample Predictions: {predictions['predictions'][:sample_predictions]}")
    
    return model, predictions


if __name__ == "__main__":
    demo_solar_energy_workflow()