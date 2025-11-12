"""
Utilidades para crear, entrenar y gestionar modelos de machine learning
"""
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Paths para modelos y scalers
MODEL_PATHS = {
    'logistic_regression': 'models/prediction_model_lr.pkl',
    'neural_network': 'models/prediction_model_nn.pkl'
}

SCALER_PATHS = {
    'logistic_regression': 'models/scaler_lr.pkl',
    'neural_network': 'models/scaler_nn.pkl'
}


def create_logistic_regression_model():
    """Crea un modelo de Regresión Logística con regularización adecuada para evitar sobreajuste"""
    return LogisticRegression(
        max_iter=5000,  # Más iteraciones para convergencia
        random_state=42,
        multi_class='multinomial',  # Para problemas multiclase
        solver='lbfgs',  # Algoritmo eficiente
        C=0.1,  # Mayor regularización para evitar sobreajuste y probabilidades extremas
        class_weight='balanced',  # Balancear clases
        tol=1e-4,  # Tolerancia razonable
        warm_start=False
    )


def create_neural_network_model():
    """Crea un modelo de Red Neuronal con regularización adecuada para evitar sobreajuste"""
    return MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Arquitectura más simple para evitar sobreajuste
        max_iter=2000,  # Iteraciones razonables
        random_state=42,
        learning_rate='adaptive',  # Tasa de aprendizaje adaptativa
        learning_rate_init=0.001,  # Learning rate inicial
        alpha=0.1,  # Mayor regularización para evitar sobreajuste
        early_stopping=True,  # Parada temprana
        validation_fraction=0.2,  # Más datos de validación para mejor generalización
        n_iter_no_change=20,  # Menos paciencia para evitar sobreajuste
        solver='adam',  # Optimizador Adam
        batch_size='auto',  # Batch size automático
        beta_1=0.9,  # Parámetro beta1 para Adam
        beta_2=0.999,  # Parámetro beta2 para Adam
        epsilon=1e-8,  # Epsilon para estabilidad numérica
        tol=1e-4  # Tolerancia razonable
    )


def create_model(model_type='logistic_regression'):
    """Crea un modelo según el tipo especificado"""
    if model_type == 'logistic_regression':
        return create_logistic_regression_model()
    elif model_type == 'neural_network':
        return create_neural_network_model()
    else:
        return create_logistic_regression_model()


def train_model(model_type, X_train, y_train, X_train_scaled=None, optimize_hyperparams=False):
    """
    Entrena un modelo según su tipo con optimización de hiperparámetros para máxima precisión
    
    Args:
        model_type: Tipo de modelo ('random_forest', 'logistic_regression', 'neural_network')
        X_train: Datos de entrenamiento (sin normalizar)
        y_train: Etiquetas de entrenamiento
        X_train_scaled: Datos normalizados (solo para red neuronal)
        optimize_hyperparams: Si True, optimiza hiperparámetros con GridSearchCV (desactivado por defecto para velocidad)
    
    Returns:
        model: Modelo entrenado
        scaler: Scaler usado (None si no se usa normalización)
    """
    import numpy as np
    from collections import Counter
    
    scaler = None
    
    # Mostrar distribución de clases original
    class_counts_original = Counter(y_train)
    print(f"Distribución de clases original: {dict(class_counts_original)}")
    
    # Balancear el dataset antes del entrenamiento
    print("Balanceando el dataset...")
    try:
        # Intentar usar SMOTE primero (mejor para datasets pequeños)
        if len(X_train) < 1000:
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(X_train) // 2 - 1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"Dataset balanceado con SMOTE: {len(X_train_balanced)} muestras")
        else:
            # Para datasets grandes, usar RandomOverSampler (más rápido)
            ros = RandomOverSampler(random_state=42)
            X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
            print(f"Dataset balanceado con RandomOverSampler: {len(X_train_balanced)} muestras")
    except Exception as e:
        print(f"Error al balancear con SMOTE: {str(e)}. Usando RandomOverSampler...")
        try:
            ros = RandomOverSampler(random_state=42)
            X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
            print(f"Dataset balanceado con RandomOverSampler: {len(X_train_balanced)} muestras")
        except Exception as e2:
            print(f"Error al balancear: {str(e2)}. Continuando sin balanceo...")
            X_train_balanced, y_train_balanced = X_train, y_train
    
    # Mostrar distribución de clases después del balanceo
    class_counts_balanced = Counter(y_train_balanced)
    print(f"Distribución de clases balanceada: {dict(class_counts_balanced)}")
    print(f"Entrenando {model_type} con {len(X_train_balanced)} muestras y {X_train_balanced.shape[1]} características...")
    
    if model_type == 'neural_network':
        # Crear Pipeline con StandardScaler + MLPClassifier (como en el proyecto que funciona)
        # Esto integra el scaler en el modelo, simplificando la predicción
        
        if optimize_hyperparams and len(X_train_balanced) >= 50:
            # Optimizar hiperparámetros para red neuronal con Pipeline
            print("Optimizando hiperparámetros de red neuronal con Pipeline...")
            
            # Crear Pipeline base
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.2,
                    solver='adam',
                    learning_rate='adaptive',
                    n_iter_no_change=20
                ))
            ])
            
            # Parámetros para el Pipeline (usar prefijo 'mlp__' para el modelo)
            param_grid = {
                'mlp__hidden_layer_sizes': [(100, 50), (128, 64), (150, 75), (128, 64, 32)],
                'mlp__alpha': [0.001, 0.01, 0.1],
                'mlp__learning_rate_init': [0.0005, 0.001, 0.01],
                'mlp__max_iter': [1000, 2000]
            }
            
            # Usar n_jobs=1 en Windows para evitar error _posixsubprocess
            import platform
            n_jobs_value = 1 if platform.system() == 'Windows' else -1
            grid_search = GridSearchCV(
                pipe, param_grid, cv=3, 
                scoring='accuracy', n_jobs=n_jobs_value, verbose=1
            )
            grid_search.fit(X_train_balanced, y_train_balanced)
            model = grid_search.best_estimator_  # Ya es un Pipeline
            print(f"Mejores parámetros: {grid_search.best_params_}")
            print(f"Mejor score CV: {grid_search.best_score_:.4f}")
        else:
            # Crear Pipeline con configuración simple
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', create_neural_network_model())
            ])
            model = pipe
            model.fit(X_train_balanced, y_train_balanced)
            
            # Calibrar probabilidades de la red neuronal también
            from sklearn.calibration import CalibratedClassifierCV
            print("Calibrando probabilidades de la red neuronal para mayor confiabilidad...")
            try:
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated_model.fit(X_train_balanced, y_train_balanced)
                model = calibrated_model
            except Exception as calib_error:
                print(f"Advertencia: No se pudo calibrar la red neuronal: {str(calib_error)}. Usando modelo sin calibrar.")
        
        # Verificar si convergió (solo para modelos no calibrados)
        if hasattr(model, 'base_estimator'):
            # Si es CalibratedClassifierCV, verificar el modelo base
            base = model.base_estimator
            if hasattr(base, 'named_steps') and 'mlp' in base.named_steps:
                mlp = base.named_steps['mlp']
                if hasattr(mlp, 'n_iter_'):
                    print(f"Convergencia: {mlp.n_iter_} iteraciones")
            elif hasattr(base, 'n_iter_'):
                print(f"Convergencia: {base.n_iter_} iteraciones")
        elif hasattr(model, 'named_steps') and 'mlp' in model.named_steps:
            mlp = model.named_steps['mlp']
            if hasattr(mlp, 'n_iter_'):
                print(f"Convergencia: {mlp.n_iter_} iteraciones")
        elif hasattr(model, 'n_iter_'):
            print(f"Convergencia: {model.n_iter_} iteraciones")
        
        # El scaler ya está integrado en el Pipeline, retornar None
        return model, None
    
    else:  # logistic_regression
        # Crear Pipeline con StandardScaler + LogisticRegression (como en el proyecto que funciona)
        # Esto integra el scaler en el modelo, simplificando la predicción
        
        # Calcular pesos personalizados para clases desbalanceadas
        class_counts = Counter(y_train_balanced)
        total_samples = len(y_train_balanced)
        n_classes = len(class_counts)
        
        # Calcular pesos usando sklearn (más preciso)
        unique_classes = np.unique(y_train_balanced)
        sklearn_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_balanced)
        class_weights_dict_sklearn = dict(zip(unique_classes, sklearn_weights))
        
        print(f"Distribución de clases: {dict(class_counts)}")
        print(f"Pesos sklearn (balanced): {class_weights_dict_sklearn}")
        
        if optimize_hyperparams and len(X_train_balanced) >= 50:
            # Optimización con Pipeline integrado
            print("Optimizando hiperparámetros de Regresión Logística con Pipeline...")
            
            # Crear Pipeline base
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('logistic', LogisticRegression(random_state=42))
            ])
            
            # Parámetros para el Pipeline (usar prefijo 'logistic__' para el modelo)
            param_distributions = {
                'logistic__C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                'logistic__solver': ['lbfgs', 'saga', 'newton-cg'],
                'logistic__class_weight': ['balanced', class_weights_dict_sklearn],
                'logistic__max_iter': [5000, 10000],
                'logistic__tol': [1e-4, 1e-5],
                'logistic__multi_class': ['multinomial', 'ovr'],
                'logistic__penalty': ['l2']
            }
            
            # Usar n_jobs=1 en Windows para evitar error _posixsubprocess
            import platform
            n_jobs_value = 1 if platform.system() == 'Windows' else -1
            print("Buscando mejores parámetros (búsqueda aleatoria optimizada, cv=3)...")
            random_search = RandomizedSearchCV(
                pipe, param_distributions, cv=3,
                scoring='balanced_accuracy', n_jobs=n_jobs_value, verbose=0,
                n_iter=10, random_state=42
            )
            random_search.fit(X_train_balanced, y_train_balanced)
            
            model = random_search.best_estimator_  # Ya es un Pipeline
            print(f"Mejor modelo encontrado:")
            print(f"Mejores parámetros: {random_search.best_params_}")
            print(f"Mejor score CV (balanced_accuracy): {random_search.best_score_:.4f}")
            
            # Calibrar probabilidades del Pipeline
            from sklearn.calibration import CalibratedClassifierCV
            print("Calibrando probabilidades del modelo para mayor confiabilidad...")
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated_model.fit(X_train_balanced, y_train_balanced)
            model = calibrated_model
            
            # Si el score no es suficientemente alto, probar también L1
            if random_search.best_score_ < 0.75:
                print("Score < 75%, probando también penalización L1...")
                pipe_l1 = Pipeline([
                    ('scaler', StandardScaler()),
                    ('logistic', LogisticRegression(random_state=42))
                ])
                
                param_distributions_l1 = {
                    'logistic__C': [0.01, 0.05, 0.1, 0.5, 1.0],
                    'logistic__solver': ['saga'],
                    'logistic__class_weight': ['balanced', class_weights_dict_sklearn],
                    'logistic__max_iter': [5000, 10000],
                    'logistic__tol': [1e-4, 1e-5],
                    'logistic__multi_class': ['multinomial', 'ovr'],
                    'logistic__penalty': ['l1']
                }
                
                random_search_l1 = RandomizedSearchCV(
                    pipe_l1, param_distributions_l1, cv=3,
                    scoring='balanced_accuracy', n_jobs=n_jobs_value, verbose=0,
                    n_iter=5, random_state=42
                )
                random_search_l1.fit(X_train_balanced, y_train_balanced)
                
                # Elegir el mejor entre L2 y L1
                if random_search_l1.best_score_ > random_search.best_score_:
                    temp_model = random_search_l1.best_estimator_
                    # Calibrar también el modelo L1
                    calibrated_l1 = CalibratedClassifierCV(temp_model, method='isotonic', cv=3)
                    calibrated_l1.fit(X_train_balanced, y_train_balanced)
                    model = calibrated_l1
                    print(f"L1 mejor que L2: {random_search_l1.best_score_:.4f} vs {random_search.best_score_:.4f}")
                    print(f"Mejores parámetros L1: {random_search_l1.best_params_}")
                else:
                    print(f"L2 mejor que L1: {random_search.best_score_:.4f} vs {random_search_l1.best_score_:.4f}")
        else:
            # Para datasets pequeños, usar Pipeline con configuración simple
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('logistic', LogisticRegression(
                    max_iter=5000,
                    random_state=42,
                    multi_class='multinomial',
                    solver='lbfgs',
                    C=0.1,
                    class_weight='balanced',
                    tol=1e-4
                ))
            ])
            model = pipe
            model.fit(X_train_balanced, y_train_balanced)
            
            # Calibrar probabilidades usando validación cruzada
            from sklearn.calibration import CalibratedClassifierCV
            print("Calibrando probabilidades del modelo para mayor confiabilidad...")
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated_model.fit(X_train_balanced, y_train_balanced)
            model = calibrated_model
        
        # El scaler ya está integrado en el Pipeline, retornar None
        return model, None


def force_retrain_model(model_type='logistic_regression'):
    """
    Fuerza el reentrenamiento eliminando el modelo guardado.
    Útil cuando se han mejorado los hiperparámetros y se quiere entrenar un modelo nuevo.
    """
    model_path = MODEL_PATHS.get(model_type)
    scaler_path = SCALER_PATHS.get(model_type)
    
    if model_path and os.path.exists(model_path):
        os.remove(model_path)
        print(f"Modelo {model_type} eliminado: {model_path}")
    
    if scaler_path and os.path.exists(scaler_path):
        os.remove(scaler_path)
        print(f"Scaler {model_type} eliminado: {scaler_path}")
    
    print(f"Modelo {model_type} listo para reentrenamiento con nuevas configuraciones.")


def save_model(model, model_type, scaler=None):
    """
    Guarda un modelo. Si el modelo es un Pipeline, el scaler ya está integrado.
    Si scaler es None pero el modelo no es Pipeline, se guarda el scaler por separado (compatibilidad).
    """
    model_path = MODEL_PATHS[model_type]
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modelo {model_type} guardado en {model_path}")
    
    # Si el modelo es un Pipeline, el scaler ya está integrado
    if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
        print(f"Scaler integrado en Pipeline para {model_type}")
    elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'named_steps'):
        # Si es CalibratedClassifierCV con Pipeline base
        if 'scaler' in model.base_estimator.named_steps:
            print(f"Scaler integrado en Pipeline (dentro de CalibratedClassifierCV) para {model_type}")
    elif scaler is not None and model_type in SCALER_PATHS:
        # Guardar scaler por separado solo si no está en Pipeline (compatibilidad con modelos antiguos)
        scaler_path = SCALER_PATHS[model_type]
        joblib.dump(scaler, scaler_path)
        print(f"Scaler {model_type} guardado por separado en {scaler_path}")


def load_model(model_type):
    """Carga un modelo y su scaler si existe"""
    model_path = MODEL_PATHS[model_type]
    
    if not os.path.exists(model_path):
        return None, None
    
    model = joblib.load(model_path)
    print(f"Modelo {model_type} cargado desde {model_path}")
    
    scaler = None
    if model_type in SCALER_PATHS:
        scaler_path = SCALER_PATHS[model_type]
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Scaler {model_type} cargado desde {scaler_path}")
    
    return model, scaler


def prepare_features_for_prediction(X, model_type, scaler=None, model=None):
    """
    Prepara las características para predicción (normaliza si es necesario)
    
    Args:
        X: Datos sin normalizar
        model_type: Tipo de modelo
        scaler: Scaler para normalización (si existe, para compatibilidad con modelos antiguos)
        model: El modelo mismo (para verificar si es Pipeline)
    
    Returns:
        X_ready: Datos listos para predicción
    """
    # Si el modelo es un Pipeline, el scaler ya está integrado, no hacer nada
    if model is not None:
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            # El modelo es un Pipeline con scaler integrado, retornar X sin cambios
            # El Pipeline aplicará el scaler automáticamente en predict_proba()
            return X
        elif hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'named_steps'):
            # Si es CalibratedClassifierCV con Pipeline base
            if 'scaler' in model.base_estimator.named_steps:
                return X
    
    # Si no es Pipeline, usar scaler manual (compatibilidad con modelos antiguos)
    if scaler is not None and model_type in ['neural_network', 'logistic_regression']:
        return scaler.transform(X)
    return X

