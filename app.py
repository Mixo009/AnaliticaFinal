from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, balanced_accuracy_score
import joblib
import os
import io
import json
from model_utils import (
    create_model, train_model as train_ml_model, save_model, load_model,
    prepare_features_for_prediction, MODEL_PATHS, SCALER_PATHS,
    force_retrain_model
)

app = Flask(__name__)
CORS(app)

# Mapeo de clases global (usado tanto en individual como batch)
DIAGNOSIS_LABELS = {1: 'Dengue', 2: 'Malaria', 3: 'Leptospirosis'}

# Crear directorio para modelos y archivos subidos
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Modelos globales (se cargarán o entrenarán al iniciar)
models = {}
scalers = {}  # Para almacenar los scalers de normalización (especialmente para red neuronal)
model_columns = {}  # Para almacenar las columnas usadas por cada modelo
current_model_type = 'logistic_regression'  # Modelo por defecto

def get_model(model_type='logistic_regression'):
    """Obtiene el modelo según el tipo especificado"""
    global models, scalers, model_columns
    
    if model_type not in models:
        model, scaler = load_model(model_type)
        # Solo guardar en el diccionario si el modelo existe (no guardar None)
        if model is not None:
            models[model_type] = model
            if scaler is not None:
                scalers[model_type] = scaler
            
            # Cargar columnas del modelo si existen
            columns_path = f'models/{model_type}_columns.json'
            if os.path.exists(columns_path):
                try:
                    with open(columns_path, 'r') as f:
                        model_columns[model_type] = json.load(f)
                except Exception as e:
                    print(f"Error al cargar columnas del modelo {model_type}: {str(e)}")
    
    return models.get(model_type)  # Retornar None si no existe

def save_model_with_columns(model, model_type, scaler, columns=None):
    """Guarda el modelo, scaler y columnas"""
    global model_columns
    save_model(model, model_type, scaler)
    if columns is not None:
        model_columns[model_type] = columns
        columns_path = f'models/{model_type}_columns.json'
        with open(columns_path, 'w') as f:
            json.dump(columns, f)

def normalize_binary(v):
    """Normaliza valores binarios a 0 o 1"""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('1', 'true', 'si', 'sí', 'on', 'y', 'yes'): return 1
        if s in ('0', 'false', 'no', 'off', 'n'): return 0
        try:
            return float(s)
        except Exception:
            return 0
    return int(v) if isinstance(v, (bool, int)) else (float(v) if isinstance(v, float) else 0)

def feature_columns():
    """
    Retorna las columnas en el orden EXACTO del proyecto que funciona.
    Este es el orden correcto según FEATURE_SECTIONS del proyecto que funciona.
    """
    return [
        # Demografía (orden exacto del proyecto que funciona)
        'age', 'male', 'female', 'urban_origin', 'rural_origin',
        # Ocupación
        'homemaker', 'student', 'professional', 'merchant', 'agriculture_livestock', 'various_jobs', 'unemployed',
        # Clínicos
        'hospitalization_days', 'body_temperature',
        # Síntomas
        'fever', 'headache', 'dizziness', 'loss_of_appetite', 'weakness',
        'myalgias', 'arthralgias', 'eye_pain', 'hemorrhages', 'vomiting',
        'abdominal_pain', 'chills', 'hemoptysis', 'edema', 'jaundice',
        'bruises', 'petechiae', 'rash', 'diarrhea', 'respiratory_difficulty', 'itching',
        # Laboratorio (nombres exactos del proyecto que funciona: AST, ALT, ALP sin paréntesis)
        'hematocrit', 'hemoglobin', 'red_blood_cells', 'white_blood_cells',
        'neutrophils', 'eosinophils', 'basophils', 'monocytes', 'lymphocytes', 'platelets',
        'AST', 'ALT', 'ALP',  # Sin paréntesis como en el proyecto que funciona
        'total_bilirubin', 'direct_bilirubin', 'indirect_bilirubin',
        'total_proteins', 'albumin', 'creatinine', 'urea'
    ]

def prepare_df(df, expected_columns=None):
    """
    Prepara el DataFrame para predicción, exactamente como en el proyecto que funciona.
    Si expected_columns es None, usa feature_columns() para mantener compatibilidad.
    """
    # Usar feature_columns() si no se proporcionan columnas esperadas
    if expected_columns is None:
        cols = feature_columns()
    else:
        cols = expected_columns
    
    # Asegurar que todas las columnas estén presentes
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    
    # Normalizar campos binarios
    binary_cols = ['male', 'female', 'urban_origin', 'rural_origin', 
                   'homemaker', 'student', 'professional', 'merchant', 
                   'agriculture_livestock', 'various_jobs', 'unemployed',
                   'fever', 'headache', 'dizziness', 'loss_of_appetite', 'weakness',
                   'myalgias', 'arthralgias', 'eye_pain', 'hemorrhages', 'vomiting',
                   'abdominal_pain', 'chills', 'hemoptysis', 'edema', 'jaundice',
                   'bruises', 'petechiae', 'rash', 'diarrhea', 'respiratory_difficulty', 'itching']
    
    for c in binary_cols:
        if c in df.columns:
            df[c] = df[c].apply(normalize_binary)
    
    # Normalizar campos numéricos (usar nombres exactos del proyecto que funciona)
    numeric_cols = ['age', 'body_temperature', 'hospitalization_days',
                    'hematocrit', 'hemoglobin', 'red_blood_cells', 'white_blood_cells',
                    'neutrophils', 'eosinophils', 'basophils', 'monocytes', 'lymphocytes', 'platelets',
                    'AST', 'ALT', 'ALP',  # Sin paréntesis como en el proyecto que funciona
                    'total_bilirubin', 'direct_bilirubin', 'indirect_bilirubin',
                    'total_proteins', 'albumin', 'creatinine', 'urea']
    
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # Retornar solo las columnas en el orden correcto
    # IMPORTANTE: Usar el orden de expected_columns si se proporciona, sino usar feature_columns()
    return df[cols]

def train_model_if_needed():
    """Entrena modelos básicos si no existen"""
    global models
    
    for model_type in MODEL_PATHS.keys():
        if model_type not in models:
            get_model(model_type)

def init_models():
    """Inicializa los modelos automáticamente al iniciar la aplicación - igual que Proyecto_Final"""
    global models, scalers, model_columns
    
    # Intentar cargar modelos desde disco primero
    for model_type in MODEL_PATHS.keys():
        model = get_model(model_type)
        if model is None:
            # Si no existe, intentar entrenar con datos por defecto usando el mismo método que Proyecto_Final
            default_data_file = 'DEMALE-HSJM_2025_data.xlsx'
            if os.path.exists(default_data_file):
                try:
                    print(f'[ML] Entrenando modelo {model_type} con datos por defecto (metodo Proyecto_Final)...')
                    df = pd.read_excel(default_data_file, engine='openpyxl')
                    df.columns = df.columns.str.strip()
                    
                    # Buscar columna de etiquetas
                    label_variations = ['diagnosis', 'Diagnosis', 'DIAGNOSIS', 'Etiqueta', 'Target', 'Clase', 'Y', 'Label']
                    label_col = None
                    for var in label_variations:
                        if var in df.columns:
                            label_col = var
                            break
                    
                    if label_col:
                        y = df[label_col].astype(int)
                        X = prepare_df(df.drop(columns=[label_col]))
                        
                        # Entrenar modelo EXACTAMENTE como Proyecto_Final (sin calibración para regresión logística)
                        if model_type == 'logistic_regression':
                            # Usar el mismo método que Proyecto_Final
                            from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
                            from sklearn.pipeline import Pipeline
                            from sklearn.preprocessing import StandardScaler
                            from sklearn.linear_model import LogisticRegression
                            from imblearn.over_sampling import SMOTE
                            
                            # Separar en train y test
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            
                            # Aplicar SMOTE solo en entrenamiento
                            smote = SMOTE(random_state=42)
                            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                            
                            # Pipeline + GridSearchCV igual que Proyecto_Final
                            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                            pipe_log = Pipeline([
                                ('scaler', StandardScaler()),
                                ('logistic', LogisticRegression(max_iter=1000, multi_class='ovr')),
                            ])
                            grid_log = {
                                'logistic__C': [0.01, 0.1, 1, 10, 100],
                                'logistic__penalty': ['l2'],
                                'logistic__solver': ['liblinear'],
                                'logistic__class_weight': ['balanced'],
                            }
                            gs_log = GridSearchCV(pipe_log, grid_log, cv=cv, scoring='balanced_accuracy', n_jobs=1)
                            gs_log.fit(X_train_balanced, y_train_balanced)
                            
                            # Entrenar modelo final con todos los datos balanceados (igual que Proyecto_Final)
                            final_model_log = gs_log.best_estimator_
                            X_all_balanced, y_all_balanced = smote.fit_resample(X, y)
                            final_model_log.fit(X_all_balanced, y_all_balanced)
                            
                            model = final_model_log
                            scaler = None  # Ya está en el Pipeline
                            
                            if model is not None:
                                models[model_type] = model
                                save_model_with_columns(model, model_type, scaler, list(X.columns))
                                print(f'[ML] Modelo {model_type} entrenado exitosamente (metodo Proyecto_Final)')
                        elif model_type == 'neural_network':
                            # Usar el mismo método que Proyecto_Final para red neuronal
                            from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
                            from sklearn.pipeline import Pipeline
                            from sklearn.preprocessing import StandardScaler
                            from sklearn.neural_network import MLPClassifier
                            from imblearn.over_sampling import SMOTE
                            
                            # Separar en train y test
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            
                            # Aplicar SMOTE solo en entrenamiento
                            smote = SMOTE(random_state=42)
                            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                            
                            # Pipeline + GridSearchCV igual que Proyecto_Final
                            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                            pipe_mlp = Pipeline([
                                ('scaler', StandardScaler()),
                                ('mlp', MLPClassifier(
                                    max_iter=1500,
                                    batch_size=16,
                                    learning_rate_init=0.001,
                                    solver='adam',
                                    random_state=42,
                                    early_stopping=True,
                                    validation_fraction=0.15,
                                    n_iter_no_change=20,
                                    warm_start=False
                                )),
                            ])
                            grid_mlp = {
                                'mlp__hidden_layer_sizes': [(32,), (32, 16), (24, 12), (40, 20)],
                                'mlp__alpha': [0.01, 0.05, 0.1, 0.2],
                                'mlp__learning_rate': ['adaptive'],
                                'mlp__activation': ['tanh', 'logistic'],
                                'mlp__learning_rate_init': [0.0005, 0.001, 0.002],
                                'mlp__tol': [1e-5],
                            }
                            gs_mlp = GridSearchCV(pipe_mlp, grid_mlp, cv=cv, scoring='balanced_accuracy', n_jobs=1)
                            gs_mlp.fit(X_train_balanced, y_train_balanced)
                            
                            # Entrenar modelo final con todos los datos balanceados (igual que Proyecto_Final)
                            final_model_mlp = gs_mlp.best_estimator_
                            X_all_balanced_mlp, y_all_balanced_mlp = smote.fit_resample(X, y)
                            final_model_mlp.fit(X_all_balanced_mlp, y_all_balanced_mlp)
                            
                            model = final_model_mlp
                            scaler = None  # Ya está en el Pipeline
                            
                            if model is not None:
                                models[model_type] = model
                                save_model_with_columns(model, model_type, scaler, list(X.columns))
                                print(f'[ML] Modelo {model_type} entrenado exitosamente (metodo Proyecto_Final)')
                except Exception as e:
                    print(f'[ML] Error al entrenar modelo {model_type}: {str(e)}')
                    import traceback
                    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_individual():
    """Endpoint para predicción individual - igual que Proyecto_Final"""
    try:
        # Extraer datos del request
        if request.is_json:
            data = request.json
            model_type = data.get('model_type', 'logistic_regression')
            # Convertir model_type a formato Proyecto_Final
            if model_type == 'logistic_regression':
                model_type = 'logistic'
            elif model_type == 'neural_network':
                model_type = 'nn'
        else:
            data = request.form.to_dict()
            model_type = data.get('model_type', 'logistic')
        
        # Validar y limpiar
        model_type = model_type if model_type in ('logistic', 'nn', 'logistic_regression', 'neural_network') else 'logistic'
        if model_type == 'logistic_regression':
            model_type = 'logistic'
        elif model_type == 'neural_network':
            model_type = 'nn'
        
        # Obtener modelo
        if model_type == 'logistic':
            model_key = 'logistic_regression'
        else:
            model_key = 'neural_network'
        
        model = get_model(model_key)
        
        # Si el modelo no existe, intentar inicializarlo
        if model is None:
            print(f'[DEBUG] Modelo {model_type} no encontrado, intentando inicializar...')
            init_models()
            model = get_model(model_key)
        
        if model is None:
            return jsonify(success=False, error=f'Modelo {model_type} no inicializado. Por favor, entrene el modelo primero usando el endpoint /train_model'), 500
        
        # Mapear datos del formulario a nombres del modelo (como Proyecto_Final espera)
        payload = {}
        
        # Campos básicos
        if 'edad' in data:
            payload['age'] = float(data['edad'])
        if 'temperatura' in data:
            payload['body_temperature'] = float(data['temperatura'])
        if 'dias_hospitalizacion' in data and data['dias_hospitalizacion']:
            payload['hospitalization_days'] = float(data['dias_hospitalizacion'])
        else:
            payload['hospitalization_days'] = 0.0
        
        # Género
        genero = data.get('genero', '')
        payload['male'] = 1 if genero == 'masculino' else 0
        payload['female'] = 1 if genero == 'femenino' else 0
        
        # Origen
        origen = data.get('origen', '')
        payload['urban_origin'] = 1 if origen == 'urbano' else 0
        payload['rural_origin'] = 1 if origen == 'rural' else 0
        
        # Síntomas
        sintomas = data.get('sintomas', [])
        if not isinstance(sintomas, list):
            sintomas = []
        symptom_columns = [
            'fever', 'headache', 'dizziness', 'loss_of_appetite', 'weakness',
            'myalgias', 'arthralgias', 'eye_pain', 'hemorrhages', 'vomiting',
            'abdominal_pain', 'chills', 'hemoptysis', 'edema', 'jaundice',
            'bruises', 'petechiae', 'rash', 'diarrhea', 'respiratory_difficulty', 'itching'
        ]
        for symptom in symptom_columns:
            payload[symptom] = 1 if symptom in sintomas else 0
        
        # Laboratorio
        laboratorio = data.get('laboratorio', {})
        lab_mapping = {
            'hematocrito': 'hematocrit',
            'hemoglobina': 'hemoglobin',
            'globulos_rojos': 'red_blood_cells',
            'globulos_blancos': 'white_blood_cells',
            'neutrofilos': 'neutrophils',
            'eosinofilos': 'eosinophils',
            'basofilos': 'basophils',
            'monocitos': 'monocytes',
            'linfocitos': 'lymphocytes',
            'plaquetas': 'platelets',
            'ast': 'AST',
            'alt': 'ALT',
            'fosfatasa_alcalina': 'ALP',
            'bilirrubina_total': 'total_bilirubin',
            'bilirrubina_directa': 'direct_bilirubin',
            'bilirrubina_indirecta': 'indirect_bilirubin',
            'proteinas_totales': 'total_proteins',
            'albumina': 'albumin',
            'creatinina': 'creatinine',
            'urea': 'urea',
        }
        for form_key, model_key in lab_mapping.items():
            if form_key in laboratorio and laboratorio[form_key]:
                payload[model_key] = float(laboratorio[form_key])
            else:
                payload[model_key] = 0.0
        
        # Ocupaciones - mapear si vienen del formulario, sino poner en 0
        occupation_columns = ['homemaker', 'student', 'professional', 'merchant', 
                             'agriculture_livestock', 'various_jobs', 'unemployed']
        ocupaciones_data = data.get('ocupaciones', {})
        for occ in occupation_columns:
            # Si viene del formulario, usar ese valor, sino 0
            if isinstance(ocupaciones_data, dict) and occ in ocupaciones_data:
                payload[occ] = 1 if ocupaciones_data[occ] else 0
            elif occ in data:
                # Si viene directamente en data
                payload[occ] = 1 if data[occ] else 0
            else:
                payload[occ] = 0
        
        # Preparar datos exactamente como Proyecto_Final
        df = pd.DataFrame([payload])
        X = prepare_df(df)
        
        # Debug: verificar que los datos estén correctos
        print(f'[DEBUG] Shape de X: {X.shape}')
        print(f'[DEBUG] Primeros valores: {X.iloc[0, :10].tolist()}')
        print(f'[DEBUG] Valores no cero: {(X.iloc[0] != 0).sum()}/{len(X.columns)}')
        
        # Predicción exactamente como Proyecto_Final
        probs = model.predict_proba(X)[0]
        pred = int(np.argmax(probs) + 1)
        
        # Debug: mostrar probabilidades
        print(f'[DEBUG] Probabilidades raw: {probs}')
        print(f'[DEBUG] Predicción: {pred}')
        
        # Respuesta en formato compatible (mantener ambos formatos)
        return jsonify({
            'success': True,
            'prediction': pred,
            'prediction_label': DIAGNOSIS_LABELS.get(pred, f'Clase {pred}'),
            'diagnosis': pred,
            'diagnosis_label': DIAGNOSIS_LABELS.get(pred, f'Clase {pred}'),
            'probability': {
                'Dengue': float(probs[0]),
                'Malaria': float(probs[1]),
                'Leptospirosis': float(probs[2])
            },
            'probabilities': {
                1: float(probs[0]),
                2: float(probs[1]),
                3: float(probs[2])
            },
            'model_used': model_type
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Endpoint para predicción por lotes - SEPARADO de individual, igual que Proyecto_Final"""
    try:
        # Obtener tipo de modelo (convertir formato)
        model_type = request.form.get('model_type', 'logistic_regression')
        if model_type == 'logistic_regression':
            model_type = 'logistic'
        elif model_type == 'neural_network':
            model_type = 'nn'
        else:
            model_type = 'logistic'
        
        # Obtener modelo desde memoria (igual que Proyecto_Final)
        if model_type == 'logistic':
            model_key = 'logistic_regression'
        else:
            model_key = 'neural_network'
        
        model = get_model(model_key)
        
        # Si el modelo no existe, intentar inicializarlo
        if model is None:
            print(f'[DEBUG] Modelo {model_type} no encontrado en batch, intentando inicializar...')
            init_models()
            model = get_model(model_key)
        
        if model is None:
            return jsonify(success=False, error=f'Modelo {model_type} no inicializado'), 500
        
        print(f'[DEBUG] Usando modelo: {model_type}')
        
        # Verificar que se envió un archivo
        if 'file' not in request.files:
            return jsonify(success=False, error='No se adjuntó archivo'), 400
        
        file = request.files['file']
        
        # Verificar que el archivo tiene nombre
        if file.filename == '' or not file.filename:
            return jsonify(success=False, error='No se seleccionó ningún archivo'), 400
        
        print(f'[DEBUG] Archivo recibido: {file.filename}')
        
        try:
            # Leer el archivo según su extensión (igual que Proyecto_Final)
            filename_lower = file.filename.lower()
            if filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls'):
                df = pd.read_excel(file)
                print(f'[DEBUG] Archivo Excel leído: {len(df)} filas, {len(df.columns)} columnas')
            elif filename_lower.endswith('.csv'):
                # Intentar diferentes encodings para CSV
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                for encoding in encodings:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, encoding=encoding)
                        print(f'[DEBUG] Archivo CSV leído con encoding {encoding}: {len(df)} filas, {len(df.columns)} columnas')
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    return jsonify(success=False, error='Error al leer el archivo CSV. Verifique el encoding del archivo.'), 400
            else:
                return jsonify(success=False, error='Formato no soportado. Use .xlsx, .xls o .csv'), 400
            
            # Verificar que el DataFrame no está vacío
            if df.empty:
                return jsonify(success=False, error='El archivo está vacío'), 400
            
        except pd.errors.EmptyDataError:
            return jsonify(success=False, error='El archivo está vacío o no contiene datos'), 400
        except pd.errors.ParserError as e:
            print(f'[ERROR] Error de parsing: {str(e)}')
            return jsonify(success=False, error=f'Error al parsear el archivo: {str(e)}'), 400
        except Exception as e:
            print(f'[ERROR] Error al leer archivo: {str(e)}')
            return jsonify(success=False, error=f'Error al leer el archivo: {str(e)}'), 400
        
        # Preparar datos exactamente como Proyecto_Final
        has_true = 'diagnosis' in df.columns
        
        # Si hay valores reales, aplicar SMOTE para balancear el dataset
        smote_applied = False
        original_distribution = None
        balanced_distribution = None
        X_balanced = None
        y_balanced = None
        
        if has_true:
            # Preparar datos originales
            X_original = prepare_df(df.drop(columns=['diagnosis']))
            y_original = df['diagnosis'].astype(int).tolist()
            
            # Calcular distribución original
            from collections import Counter
            original_distribution = dict(Counter(y_original))
            
            # Aplicar SMOTE para balancear el dataset
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_original, y_original)
            
            # Calcular distribución balanceada
            balanced_distribution = dict(Counter(y_balanced))
            smote_applied = True
            
            print(f'[DEBUG] Dataset original: {len(X_original)} muestras, distribución: {original_distribution}')
            print(f'[DEBUG] Dataset balanceado con SMOTE: {len(X_balanced)} muestras, distribución: {balanced_distribution}')
            
            # Usar datos balanceados para predicción y evaluación
            X = X_balanced
            y_true = y_balanced
        else:
            # Si no hay valores reales, usar datos originales
            X = prepare_df(df)
            y_true = None
        
        # Predicción en datos balanceados (si aplicó SMOTE) o originales
        probs = model.predict_proba(X)
        preds = [int(np.argmax(p) + 1) for p in probs]
        
        # Crear resultados - mostrar solo los originales si se aplicó SMOTE
        results = []
        if smote_applied:
            # Si se aplicó SMOTE, solo mostrar resultados de las muestras originales
            # (las primeras len(df) muestras corresponden a las originales)
            original_count = len(df)
            for i, p in enumerate(probs[:original_count], start=1):
                result = {
                    'row': i,
                    'diagnosis': int(np.argmax(p) + 1),
                    'diagnosis_label': DIAGNOSIS_LABELS.get(int(np.argmax(p) + 1), f'Clase {int(np.argmax(p) + 1)}'),
                    'probabilities': {1: float(p[0]), 2: float(p[1]), 3: float(p[2])}
                }
                true_val = int(df['diagnosis'].iloc[i-1])
                result['true_diagnosis'] = true_val
                result['correct'] = result['diagnosis'] == true_val
                results.append(result)
        else:
            # Si no se aplicó SMOTE, mostrar todos los resultados
            for i, p in enumerate(probs, start=1):
                result = {
                    'row': i,
                    'diagnosis': int(np.argmax(p) + 1),
                    'diagnosis_label': DIAGNOSIS_LABELS.get(int(np.argmax(p) + 1), f'Clase {int(np.argmax(p) + 1)}'),
                    'probabilities': {1: float(p[0]), 2: float(p[1]), 3: float(p[2])}
                }
                results.append(result)
        
        # Calcular resumen por clase (para el frontend) - basado en resultados mostrados
        summary = {
            'Dengue': 0,
            'Malaria': 0,
            'Leptospirosis': 0
        }
        for result in results:
            label = result['diagnosis_label']
            if label in summary:
                summary[label] += 1
        
        # Calcular evaluación en datos balanceados (si se aplicó SMOTE) o originales
        evaluation = None
        metrics = None
        confusion_matrix_data = None
        confusion_matrix_classes = None
        if has_true and y_true is not None:
            # Calcular métricas en datos balanceados
            acc = accuracy_score(y_true, preds)
            bacc = balanced_accuracy_score(y_true, preds)
            cm = confusion_matrix(y_true, preds, labels=[1,2,3]).tolist()
            report = classification_report(y_true, preds, output_dict=True)
            evaluation = {
                'accuracy': float(acc),
                'balanced_accuracy': float(bacc),
                'confusion_matrix': cm,
                'classification_report': report
            }
            
            # También crear objeto metrics con formato esperado por frontend
            metrics = {
                'accuracy': float(acc),
                'precision': float(report.get('weighted avg', {}).get('precision', 0)),
                'recall': float(report.get('weighted avg', {}).get('recall', 0)),
                'f1_score': float(report.get('weighted avg', {}).get('f1-score', 0))
            }
            confusion_matrix_data = cm
            confusion_matrix_classes = [DIAGNOSIS_LABELS.get(i, f'Clase {i}') for i in [1, 2, 3]]
        
        # Información de SMOTE para mostrar en el frontend
        smote_info = None
        if smote_applied:
            smote_info = {
                'applied': True,
                'original_distribution': {int(k): int(v) for k, v in original_distribution.items()},
                'balanced_distribution': {int(k): int(v) for k, v in balanced_distribution.items()},
                'original_total': len(df),
                'balanced_total': len(X_balanced),
                'message': 'Dataset balanceado con SMOTE'
            }
        
        # Respuesta con formato compatible con el frontend
        return jsonify(
            success=True, 
            total_predictions=len(results),
            total_registros=len(results),  # Para compatibilidad con frontend
            results=results, 
            summary=summary,  # Para mostrar resumen de predicciones
            evaluation=evaluation,  # Para métricas (formato Proyecto_Final)
            metrics=metrics,  # Para métricas (formato frontend)
            confusion_matrix=confusion_matrix_data,  # Para matriz de confusión (en datos balanceados)
            confusion_matrix_classes=confusion_matrix_classes,  # Para clases de la matriz
            smote_info=smote_info,  # Información sobre SMOTE aplicado
            model_used=model_type
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500

@app.route('/force_retrain', methods=['POST'])
def force_retrain():
    """Endpoint para forzar reentrenamiento eliminando modelos antiguos"""
    try:
        model_type = request.form.get('model_type', 'logistic_regression')
        if model_type not in MODEL_PATHS:
            return jsonify({'error': 'Tipo de modelo no válido'}), 400
        
        force_retrain_model(model_type)
        
        return jsonify({
            'success': True,
            'message': f'Modelo {model_type} eliminado. Listo para reentrenamiento con nuevas configuraciones.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """Endpoint para entrenar modelo con nuevo dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        # Leer archivo
        filename = file.filename.lower()
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file, encoding='utf-8')
            elif filename.endswith(('.xlsx', '.xls')):
                # Intentar leer el Excel con diferentes opciones
                try:
                    df = pd.read_excel(file, engine='openpyxl')
                except Exception as e1:
                    try:
                        df = pd.read_excel(file, engine='openpyxl', header=0)
                    except Exception as e2:
                        return jsonify({
                            'error': f'Error al leer archivo Excel: {str(e1)}',
                            'detalles': f'Error adicional: {str(e2)}'
                        }), 400
            else:
                return jsonify({'error': 'Formato de archivo no soportado. Use .csv o .xlsx'}), 400
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'El archivo está vacío'}), 400
        except Exception as e:
            return jsonify({'error': f'Error al leer el archivo: {str(e)}'}), 400
        
        # Verificar que el DataFrame no esté vacío
        if df.empty:
            return jsonify({'error': 'El archivo no contiene datos'}), 400
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Buscar columna de etiquetas
        label_columns = ['Etiqueta', 'Target', 'Clase', 'Y', 'Label']
        label_col = None
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            return jsonify({'error': 'No se encontró columna de etiquetas'}), 400
        
        # Separar características y etiquetas
        feature_cols = [col for col in df.columns if col != label_col]
        X = df[feature_cols].values
        y = df[label_col].values
        
        # Entrenar modelo (usar Regresión Logística por defecto)
        global models
        model_type = request.form.get('model_type', 'logistic_regression')
        if model_type not in MODEL_PATHS:
            model_type = 'logistic_regression'
        
        # Forzar reentrenamiento con nuevas configuraciones mejoradas
        if model_type == 'logistic_regression':
            print("Forzando reentrenamiento de regresión logística con mejoras optimizadas...")
            force_retrain_model(model_type)
        
        # Entrenar modelo usando model_utils
        model, scaler = train_ml_model(model_type, X, y)
        models[model_type] = model
        if scaler is not None:
            scalers[model_type] = scaler
        save_model_with_columns(model, model_type, scaler, feature_cols)
        
        return jsonify({
            'success': True,
            'message': 'Modelo entrenado exitosamente con configuraciones optimizadas',
            'features': feature_cols,
            'samples': len(X)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Eliminar modelos antiguos para forzar reentrenamiento con método Proyecto_Final
    if os.path.exists(MODEL_PATHS['logistic_regression']):
        print('[ML] Eliminando modelo antiguo de regresion logistica para reentrenar con metodo Proyecto_Final...')
        force_retrain_model('logistic_regression')
    if os.path.exists(MODEL_PATHS['neural_network']):
        print('[ML] Eliminando modelo antiguo de red neuronal para reentrenar con metodo Proyecto_Final...')
        force_retrain_model('neural_network')
    
    # Inicializar modelos al iniciar (igual que Proyecto_Final)
    print('[ML] Inicializando modelos...')
    init_models()
    print('[ML] Modelos inicializados')
    app.run(debug=True, port=5000)
