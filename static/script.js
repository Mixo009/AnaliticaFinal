// Funcionalidad de pestañas
function showTab(tabName) {
    // Ocultar todas las pestañas
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remover clase active de todos los botones
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Mostrar la pestaña seleccionada
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Activar el botón correspondiente
    event.target.classList.add('active');
    
    // Limpiar resultados
    clearResults();
}

function clearResults() {
    document.getElementById('individual-result').style.display = 'none';
    document.getElementById('batch-result').style.display = 'none';
}

function showLoading() {
    document.getElementById('loading').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.classList.add('show');
    
    setTimeout(() => {
        errorDiv.style.display = 'none';
        errorDiv.classList.remove('show');
    }, 5000);
}

// Tema: persistencia y toggle
(function initTheme() {
    try {
        const saved = localStorage.getItem('theme');
        const isDark = saved === 'dark';
        if (isDark) document.body.classList.add('theme-dark');
        const toggle = document.getElementById('theme-toggle');
        if (toggle) {
            toggle.checked = isDark;
            toggle.addEventListener('change', () => {
                const dark = toggle.checked;
                document.body.classList.toggle('theme-dark', dark);
                localStorage.setItem('theme', dark ? 'dark' : 'light');
            });
        }
    } catch (e) {
        // Silenciar fallos de localStorage
    }
})();

// Predicción Individual
document.getElementById('individual-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Obtener valores de campos obligatorios
    const edad = parseFloat(document.getElementById('edad').value);
    const generoRadio = document.querySelector('input[name="genero"]:checked');
    const genero = generoRadio ? generoRadio.value : null;
    const origenRadio = document.querySelector('input[name="origen"]:checked');
    const origen = origenRadio ? origenRadio.value : null;
    const temperatura = parseFloat(document.getElementById('temperatura').value);
    const diasHospitalizacion = document.getElementById('dias_hospitalizacion').value ? 
        parseFloat(document.getElementById('dias_hospitalizacion').value) : null;
    
    // Obtener síntomas seleccionados
    const sintomasCheckboxes = document.querySelectorAll('input[name="sintomas"]:checked');
    const sintomas = Array.from(sintomasCheckboxes).map(cb => cb.value);
    
    // Obtener valores de laboratorio (opcionales, default 0)
    const getLabValue = (id) => {
        const value = document.getElementById(id).value;
        return value ? parseFloat(value) : 0.0;
    };
    
    const laboratorio = {
        hematocrito: getLabValue('hematocrito'),
        hemoglobina: getLabValue('hemoglobina'),
        globulos_rojos: getLabValue('globulos_rojos'),
        globulos_blancos: getLabValue('globulos_blancos'),
        neutrofilos: getLabValue('neutrofilos'),
        eosinofilos: getLabValue('eosinofilos'),
        basofilos: getLabValue('basofilos'),
        monocitos: getLabValue('monocitos'),
        linfocitos: getLabValue('linfocitos'),
        plaquetas: getLabValue('plaquetas'),
        ast: getLabValue('ast'),
        alt: getLabValue('alt'),
        fosfatasa_alcalina: getLabValue('fosfatasa_alcalina'),
        bilirrubina_total: getLabValue('bilirrubina_total'),
        bilirrubina_directa: getLabValue('bilirrubina_directa'),
        bilirrubina_indirecta: getLabValue('bilirrubina_indirecta'),
        proteinas_totales: getLabValue('proteinas_totales'),
        albumina: getLabValue('albumina'),
        creatinina: getLabValue('creatinina'),
        urea: getLabValue('urea')
    };
    
    // Validar campos obligatorios
    if (!edad || isNaN(edad) || edad < 0 || edad > 100) {
        showError('Por favor ingrese una edad válida (0-100)');
        return;
    }
    
    if (!genero) {
        showError('Por favor seleccione un género');
        return;
    }
    
    if (!origen) {
        showError('Por favor seleccione un origen');
        return;
    }
    
    if (!temperatura || isNaN(temperatura) || temperatura < 35 || temperatura > 42) {
        showError('Por favor ingrese una temperatura válida (35-42°C)');
        return;
    }
    
    // Validar días de hospitalización si se proporciona
    if (diasHospitalizacion !== null && (isNaN(diasHospitalizacion) || diasHospitalizacion < 1 || diasHospitalizacion > 30)) {
        showError('Por favor ingrese días de hospitalización válidos (1-30)');
        return;
    }
    
    // Obtener ocupaciones (si existen en el formulario)
    const ocupaciones = {};
    const occupationFields = ['homemaker', 'student', 'professional', 'merchant', 
                             'agriculture_livestock', 'various_jobs', 'unemployed'];
    occupationFields.forEach(occ => {
        const checkbox = document.querySelector(`input[name="${occ}"]`);
        if (checkbox) {
            ocupaciones[occ] = checkbox.checked ? 1 : 0;
        }
    });
    
    // Construir objeto de datos
    const formData = {
        edad: edad,
        genero: genero,
        origen: origen,
        temperatura: temperatura,
        dias_hospitalizacion: diasHospitalizacion,
        sintomas: sintomas,
        laboratorio: laboratorio,
        ocupaciones: ocupaciones,  // Agregar ocupaciones
        model_type: document.getElementById('model-selector').value
    };
    
    showLoading();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Error en la predicción');
        }
        
        // Verificar que la respuesta tenga los datos necesarios
        if (!data.prediction_label && !data.diagnosis_label) {
            throw new Error('Respuesta del servidor incompleta: falta la etiqueta de predicción');
        }
        if (!data.probability && !data.probabilities) {
            throw new Error('Respuesta del servidor incompleta: faltan las probabilidades');
        }
        
        // Mostrar resultado con diseño mejorado
        const predictionLabel = data.prediction_label || data.diagnosis_label || 'Desconocido';
        // Usar probability o probabilities (formato Proyecto_Final)
        const probabilityData = data.probability || (data.probabilities ? {
            'Dengue': data.probabilities[1] || 0,
            'Malaria': data.probabilities[2] || 0,
            'Leptospirosis': data.probabilities[3] || 0
        } : {});
        
        // Verificar que haya probabilidades
        const probValues = Object.values(probabilityData);
        if (probValues.length === 0) {
            throw new Error('No se recibieron probabilidades del servidor');
        }
        
        const maxProb = Math.max(...probValues);
        const maxProbPercent = isNaN(maxProb) ? '0.0' : (maxProb * 100).toFixed(1);
        
        // Iconos para cada enfermedad
        const diseaseIcons = {
            'Dengue': '🦟',
            'Malaria': '🩸',
            'Leptospirosis': '🐀'
        };
        
        // Descripciones de enfermedades
        const diseaseDescriptions = {
            'Dengue': 'Enfermedad viral transmitida por mosquitos Aedes',
            'Malaria': 'Enfermedad parasitaria transmitida por mosquitos Anopheles',
            'Leptospirosis': 'Enfermedad bacteriana transmitida por animales'
        };
        
        // Colores para cada enfermedad
        const diseaseColors = {
            'Dengue': { bg: '#fee2e2', text: '#991b1b', bar: '#dc2626' },
            'Malaria': { bg: '#dbeafe', text: '#1e40af', bar: '#3b82f6' },
            'Leptospirosis': { bg: '#dcfce7', text: '#166534', bar: '#22c55e' }
        };
        
        // Crear HTML mejorado para el resultado
        const resultBox = document.querySelector('#individual-result .result-box');
        if (!resultBox) {
            showError('Error: No se encontró el contenedor de resultados');
            return;
        }
        
        const icon = diseaseIcons[predictionLabel] || '🏥';
        const description = diseaseDescriptions[predictionLabel] || 'Predicción de enfermedad';
        const colors = diseaseColors[predictionLabel] || { bg: '#f3f4f6', text: '#374151', bar: '#6b7280' };
        
        resultBox.innerHTML = `
            <div class="prediction-main" style="background: linear-gradient(135deg, ${colors.bg} 0%, ${colors.bar} 100%); padding: 2rem; border-radius: 1rem; margin-bottom: 1.5rem; text-align: center; color: white;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">${icon}</div>
                <h2 style="font-size: 2rem; margin: 0.5rem 0; color: white;">${predictionLabel}</h2>
                <p style="font-size: 1rem; margin: 0.5rem 0; opacity: 0.9;">${description}</p>
                <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 1.5rem;">
                    <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: bold;">Confianza: ${maxProbPercent}%</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: bold;">${document.getElementById('model-selector').options[document.getElementById('model-selector').selectedIndex].text}</span>
                </div>
            </div>
            <div class="probabilities-section">
                <h3 style="margin-bottom: 1rem; color: #374151;">Probabilidades por Enfermedad</h3>
                <div class="probabilities-list">
                    ${Object.entries(probabilityData)
                        .sort((a, b) => b[1] - a[1])
                        .map(([disease, prob]) => {
                            const percent = (prob * 100).toFixed(1);
                            const dIcon = diseaseIcons[disease] || '🏥';
                            const dColors = diseaseColors[disease] || { bar: '#6b7280' };
                            return `
                                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; padding: 0.75rem; background: #f9fafb; border-radius: 0.5rem;">
                                    <span style="font-size: 1.5rem;">${dIcon}</span>
                                    <div style="flex: 1;">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                            <span style="font-weight: 600; color: #374151;">${disease}</span>
                                            <span style="font-weight: 600; color: #6b7280;">${percent}%</span>
                                        </div>
                                        <div style="background: #e5e7eb; height: 0.5rem; border-radius: 0.25rem; overflow: hidden;">
                                            <div style="background: ${dColors.bar}; height: 100%; width: ${percent}%; transition: width 0.3s ease;"></div>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                </div>
            </div>
        `;
        
        document.getElementById('individual-result').style.display = 'block';
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
});

// Predicción por Lotes
document.getElementById('batch-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Por favor seleccione un archivo');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Agregar tipo de modelo seleccionado
    const modelType = document.getElementById('model-selector').value;
    formData.append('model_type', modelType);
    
    showLoading();
    
    try {
        const response = await fetch('/predict_batch', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            // Mostrar error más detallado si está disponible
            let errorMsg = data.error || 'Error en la predicción por lotes';
            if (data.columnas_encontradas) {
                errorMsg += `\n\nColumnas encontradas en el archivo: ${data.columnas_encontradas.join(', ')}`;
            }
            if (data.sugerencia) {
                errorMsg += `\n\nSugerencia: ${data.sugerencia}`;
            }
            throw new Error(errorMsg);
        }
        
        // Mostrar resultados
        displayBatchResults(data);
        document.getElementById('batch-result').style.display = 'block';
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
});

// Guardar datos para descarga CSV
let currentBatchData = null;

function displayBatchResults(data) {
    currentBatchData = data;
    
    // Mostrar mensaje de SMOTE si se aplicó
    if (data.smote_info && data.smote_info.applied) {
        const batchResult = document.getElementById('batch-result');
        if (batchResult) {
            // Crear o actualizar mensaje de SMOTE
            let smoteMessage = batchResult.querySelector('.smote-message');
            if (!smoteMessage) {
                smoteMessage = document.createElement('div');
                smoteMessage.className = 'smote-message';
                smoteMessage.style.cssText = 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem; text-align: center; font-weight: 600; box-shadow: 0 4px 6px rgba(0,0,0,0.1);';
                batchResult.insertBefore(smoteMessage, batchResult.firstChild);
            }
            const origDist = data.smote_info.original_distribution;
            const balDist = data.smote_info.balanced_distribution;
            const origTotal = data.smote_info.original_total;
            const balTotal = data.smote_info.balanced_total;
            smoteMessage.innerHTML = `
                <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">📊 ${data.smote_info.message}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">
                    Original: ${origTotal} muestras → Balanceado: ${balTotal} muestras (${balTotal - origTotal} casos sintéticos generados)
                </div>
            `;
        }
    } else {
        // Ocultar mensaje si no se aplicó SMOTE
        const batchResult = document.getElementById('batch-result');
        if (batchResult) {
            const smoteMessage = batchResult.querySelector('.smote-message');
            if (smoteMessage) {
                smoteMessage.remove();
            }
        }
    }
    
    // Mostrar ventana de información
    showPredictionInfoModal(data);
    
    // Mostrar resumen con tarjetas
    displaySummaryCards(data);
    
    // Mostrar métricas si están disponibles
    const metricsSection = document.getElementById('metrics-section');
    const metricsGrid = document.getElementById('metrics-grid');
    
    if (data.metrics) {
        metricsSection.style.display = 'block';
        metricsGrid.innerHTML = '';
        
        // Métricas globales
        const metrics = [
            { label: 'Exactitud Global', value: data.metrics.accuracy, key: 'accuracy', color: 'blue' },
            { label: 'Precisión (Precision)', value: data.metrics.precision, key: 'precision', color: 'green' },
            { label: 'Sensibilidad (Recall)', value: data.metrics.recall, key: 'recall', color: 'purple' },
            { label: 'F1-Score', value: data.metrics.f1_score, key: 'f1_score', color: 'orange' }
        ];
        
        metrics.forEach(metric => {
            const card = document.createElement('div');
            card.className = 'metric-card';
            card.innerHTML = `
                <h5>${metric.label}</h5>
                <div class="value">${(metric.value * 100).toFixed(1)}%</div>
            `;
            metricsGrid.appendChild(card);
        });
        
        // Mostrar métricas por clase
        if (data.metrics_by_class) {
            displayMetricsByClass(data.metrics_by_class, data.metrics);
        }
    } else {
        metricsSection.style.display = 'none';
    }
    
    // Mostrar matriz de confusión si está disponible
    const confusionMatrixSection = document.getElementById('confusion-matrix-section');
    const confusionMatrixDiv = document.getElementById('confusion-matrix');
    
    if (data.confusion_matrix && data.confusion_matrix_classes) {
        confusionMatrixSection.style.display = 'block';
        
        const cm = data.confusion_matrix;
        const cmNormalized = data.confusion_matrix_normalized || null;
        const classes = data.confusion_matrix_classes;
        
        // Calcular totales por fila para porcentajes
        const rowTotals = cm.map(row => row.reduce((a, b) => a + b, 0));
        const maxValue = Math.max(...cm.flat());
        
        let html = '<table class="confusion-table">';
        html += '<thead><tr><th>Valores Reales</th>';
        classes.forEach(cls => {
            html += `<th>${cls}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        for (let i = 0; i < cm.length; i++) {
            const rowLabel = classes[i] || `Clase ${i}`;
            html += `<tr><th>${rowLabel}</th>`;
            for (let j = 0; j < cm[i].length; j++) {
                const value = cm[i][j];
                // Usar matriz normalizada si está disponible, sino calcular porcentaje
                let percentage;
                if (cmNormalized && cmNormalized[i] && cmNormalized[i][j] !== undefined) {
                    percentage = (cmNormalized[i][j] * 100).toFixed(1);
                } else {
                    percentage = rowTotals[i] > 0 ? (value / rowTotals[i] * 100).toFixed(1) : 0;
                }
                const intensity = maxValue > 0 ? (value / maxValue) * 100 : 0;
                html += `<td class="confusion-cell" style="background-color: rgba(59, 130, 246, ${0.3 + intensity / 200})">
                    <div class="confusion-value">${value}</div>
                    <div class="confusion-percentage">${percentage}%</div>
                </td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        
        // Agregar leyenda
        html += '<div class="confusion-legend">';
        html += '<span>0</span><span>Bajo</span><span>Medio</span><span>Alto</span><span>Máximo</span>';
        html += '</div>';
        
        confusionMatrixDiv.innerHTML = html;
    } else {
        confusionMatrixSection.style.display = 'none';
    }
    
    // Mostrar tabla de resultados
    displayResultsTable(data);
}

function displaySummaryCards(data) {
    const summaryCards = document.getElementById('summary-cards');
    summaryCards.innerHTML = '';
    
    if (!data.summary) {
        return;
    }
    
    const classColors = {
        'Dengue': { bg: '#dbeafe', text: '#1e40af', label: 'Dengue' },
        'Malaria': { bg: '#fef3c7', text: '#92400e', label: 'Malaria' },
        'Leptospirosis': { bg: '#fee2e2', text: '#991b1b', label: 'Leptospirosis' }
    };
    
    // Tarjeta de total
    const totalCard = document.createElement('div');
    totalCard.className = 'summary-card total-card';
    totalCard.innerHTML = `
        <div class="summary-label">Total de registros</div>
        <div class="summary-value">${data.total_registros}</div>
    `;
    summaryCards.appendChild(totalCard);
    
    // Tarjetas por clase
    Object.keys(classColors).forEach(className => {
        const count = data.summary[className] || 0;
        const colors = classColors[className];
        
        const card = document.createElement('div');
        card.className = 'summary-card';
        card.style.backgroundColor = colors.bg;
        card.innerHTML = `
            <div class="summary-label" style="color: ${colors.text}">${colors.label}</div>
            <div class="summary-value" style="color: ${colors.text}">${count}</div>
        `;
        summaryCards.appendChild(card);
    });
}

function displayMetricsByClass(metricsByClass, globalMetrics) {
    const metricsByClassSection = document.getElementById('metrics-by-class-section');
    metricsByClassSection.innerHTML = '';
    
    if (!metricsByClass || Object.keys(metricsByClass).length === 0) {
        return;
    }
    
    let html = '<h5>Métricas por Clase</h5>';
    html += '<table class="metrics-by-class-table">';
    html += '<thead><tr><th>Clase</th><th>Precisión</th><th>Recall</th><th>F1-Score</th><th>Exactitud</th><th>Soporte</th></tr></thead><tbody>';
    
    Object.keys(metricsByClass).forEach(className => {
        const m = metricsByClass[className];
        html += `<tr>
            <td><strong>${className}</strong></td>
            <td>${(m.precision * 100).toFixed(1)}%</td>
            <td>${(m.recall * 100).toFixed(1)}%</td>
            <td>${(m.f1_score * 100).toFixed(1)}%</td>
            <td>${(m.precision * 100).toFixed(1)}%</td>
            <td>${m.support}</td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    
    // Promedios
    if (globalMetrics.macro_avg && globalMetrics.weighted_avg) {
        html += '<div class="metrics-averages">';
        html += '<div class="avg-box macro-avg">';
        html += '<h6>Promedio Macro</h6>';
        html += `<p>Precisión: ${(globalMetrics.macro_avg.precision * 100).toFixed(1)}%</p>`;
        html += `<p>Recall: ${(globalMetrics.macro_avg.recall * 100).toFixed(1)}%</p>`;
        html += `<p>F1-Score: ${(globalMetrics.macro_avg.f1_score * 100).toFixed(1)}%</p>`;
        html += '</div>';
        html += '<div class="avg-box weighted-avg">';
        html += '<h6>Promedio Ponderado</h6>';
        html += `<p>Precisión: ${(globalMetrics.weighted_avg.precision * 100).toFixed(1)}%</p>`;
        html += `<p>Recall: ${(globalMetrics.weighted_avg.recall * 100).toFixed(1)}%</p>`;
        html += `<p>F1-Score: ${(globalMetrics.weighted_avg.f1_score * 100).toFixed(1)}%</p>`;
        html += '</div>';
        html += '</div>';
    }
    
    metricsByClassSection.innerHTML = html;
}

function displayResultsTable(data) {
    const table = document.getElementById('results-table');
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');
    
    // Usar results o predicciones (compatibilidad con ambos formatos)
    const results = data.results || data.predicciones || [];
    
    if (!results || results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 2rem; color: var(--text-secondary);">No hay predicciones para mostrar</td></tr>';
        return;
    }
    
    // Crear encabezados - verificar si hay evaluación para mostrar columnas adicionales
    let headerHTML = '<tr><th>Fila</th><th>Predicción</th><th>Probabilidades</th>';
    if (data.evaluation && results[0] && results[0].true_diagnosis !== undefined) {
        headerHTML += '<th>Valor Real</th><th>Correcto</th>';
    }
    headerHTML += '</tr>';
    thead.innerHTML = headerHTML;
    
    // Crear filas con mejor formato
    tbody.innerHTML = '';
    results.forEach((row, index) => {
        const tr = document.createElement('tr');
        
        // Fila
        const tdFila = document.createElement('td');
        tdFila.textContent = index + 1;
        tr.appendChild(tdFila);
        
        // Predicción con tag de color
        const tdPred = document.createElement('td');
        // Usar diagnosis_label (formato backend) o Prediccion_Label (formato antiguo)
        const predictionLabel = row.diagnosis_label || row['Prediccion_Label'] || 'Desconocido';
        const tagClass = getTagClass(predictionLabel);
        tdPred.innerHTML = `<span class="prediction-tag ${tagClass}">${predictionLabel}</span>`;
        tr.appendChild(tdPred);
        
        // Probabilidades - usar el formato del backend (probabilities)
        const tdProb = document.createElement('td');
        let probHTML = '';
        
        if (row.probabilities) {
            // Formato del backend: {1: float, 2: float, 3: float}
            const probs = row.probabilities;
            const diseases = ['Dengue', 'Malaria', 'Leptospirosis'];
            const probValues = [
                { name: diseases[0], value: probs[1] || 0 },
                { name: diseases[1], value: probs[2] || 0 },
                { name: diseases[2], value: probs[3] || 0 }
            ];
            
            // Ordenar por probabilidad descendente
            probValues.sort((a, b) => b.value - a.value);
            
            // Mostrar todas las probabilidades
            probHTML = '<div style="display: flex; flex-direction: column; gap: 0.25rem;">';
            probValues.forEach(prob => {
                const percent = (prob.value * 100).toFixed(1);
                probHTML += `<div style="font-size: 0.85rem;">
                    <span style="font-weight: 600;">${prob.name}:</span> 
                    <span style="color: var(--text-secondary);">${percent}%</span>
                </div>`;
            });
            probHTML += '</div>';
        } else {
            // Formato antiguo: buscar Probabilidad_*
            let maxProb = 0;
            Object.keys(row).forEach(key => {
                if (key.startsWith('Probabilidad_')) {
                    const prob = parseFloat(row[key]);
                    if (prob > maxProb) {
                        maxProb = prob;
                    }
                }
            });
            
            if (maxProb > 0) {
                const percentage = (maxProb * 100).toFixed(2);
                probHTML = `
                    <div class="probability-container">
                        <div class="probability-bar" style="width: ${percentage}%"></div>
                        <span class="probability-text">${percentage}%</span>
                    </div>
                `;
            } else {
                probHTML = 'N/A';
            }
        }
        
        tdProb.innerHTML = probHTML;
        tr.appendChild(tdProb);
        
        // Agregar columnas de evaluación si existen
        if (data.evaluation && row.true_diagnosis !== undefined) {
            // Valor real
            const tdTrue = document.createElement('td');
            const trueLabel = DIAGNOSIS_LABELS[row.true_diagnosis] || `Clase ${row.true_diagnosis}`;
            tdTrue.textContent = trueLabel;
            tr.appendChild(tdTrue);
            
            // Correcto
            const tdCorrect = document.createElement('td');
            if (row.correct) {
                tdCorrect.innerHTML = '<span style="color: #10b981; font-weight: 600;">✓</span>';
            } else {
                tdCorrect.innerHTML = '<span style="color: #ef4444; font-weight: 600;">✗</span>';
            }
            tr.appendChild(tdCorrect);
        }
        
        tbody.appendChild(tr);
    });
}

function getTagClass(predictionLabel) {
    const label = predictionLabel.toLowerCase();
    if (label.includes('dengue')) return 'tag-dengue';
    if (label.includes('malaria')) return 'tag-malaria';
    if (label.includes('leptospirosis')) return 'tag-leptospirosis';
    return 'tag-default';
}

// Función para descargar CSV
// Funciones para el modal de información
function showPredictionInfoModal(data) {
    const modal = document.getElementById('prediction-info-modal');
    const content = document.getElementById('prediction-info-content');
    
    if (!modal || !content) return;
    
    // Construir contenido del modal
    let html = '';
    
    // Información general
    html += '<div class="info-section">';
    html += '<div class="info-section-title">📋 Información General</div>';
    html += `<div class="info-item">
        <span class="info-label">📊 Total de Datos Leídos</span>
        <span class="info-value">${data.total_registros || data.total_predictions || 0} registros</span>
    </div>`;
    html += `<div class="info-item">
        <span class="info-label">👥 Pacientes Procesados</span>
        <span class="info-value">${data.total_registros || data.total_predictions || 0} pacientes</span>
    </div>`;
    html += `<div class="info-item">
        <span class="info-label">🤖 Modelo Utilizado</span>
        <span class="info-value">${data.model_used === 'nn' ? 'Red Neuronal' : data.model_used === 'logistic' ? 'Regresión Logística' : data.model_used || 'N/A'}</span>
    </div>`;
    html += '</div>';
    
    // Información de SMOTE si se aplicó
    if (data.smote_info && data.smote_info.applied) {
        html += '<div class="info-section">';
        html += '<div class="info-section-title">⚖️ Balanceo del Dataset con SMOTE</div>';
        html += `<div class="info-item">
            <span class="info-label">✅ Estado</span>
            <span class="info-value" style="color: #10b981; font-weight: 600;">Dataset Balanceado</span>
        </div>`;
        html += `<div class="info-item">
            <span class="info-label">📈 Muestras Originales</span>
            <span class="info-value">${data.smote_info.original_total} muestras</span>
        </div>`;
        html += `<div class="info-item">
            <span class="info-label">📊 Muestras Balanceadas</span>
            <span class="info-value">${data.smote_info.balanced_total} muestras</span>
        </div>`;
        html += `<div class="info-item">
            <span class="info-label">➕ Casos Sintéticos Generados</span>
            <span class="info-value" style="color: #667eea; font-weight: 600;">${data.smote_info.balanced_total - data.smote_info.original_total} casos</span>
        </div>`;
        
        // Distribución original
        if (data.smote_info.original_distribution) {
            html += '<div style="margin-top: 1rem; padding: 0.75rem; background: #f3f4f6; border-radius: 8px;">';
            html += '<div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--text-primary);">Distribución Original:</div>';
            const origDist = data.smote_info.original_distribution;
            Object.keys(origDist).sort().forEach(key => {
                const label = DIAGNOSIS_LABELS[parseInt(key)] || `Clase ${key}`;
                html += `<div class="distribution-item">
                    <span class="distribution-label">${label}</span>
                    <span class="distribution-value">${origDist[key]} casos</span>
                </div>`;
            });
            html += '</div>';
        }
        
        // Distribución balanceada
        if (data.smote_info.balanced_distribution) {
            html += '<div style="margin-top: 1rem; padding: 0.75rem; background: #eef2ff; border-radius: 8px;">';
            html += '<div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--text-primary);">Distribución Balanceada:</div>';
            const balDist = data.smote_info.balanced_distribution;
            Object.keys(balDist).sort().forEach(key => {
                const label = DIAGNOSIS_LABELS[parseInt(key)] || `Clase ${key}`;
                html += `<div class="distribution-item">
                    <span class="distribution-label">${label}</span>
                    <span class="distribution-value">${balDist[key]} casos</span>
                </div>`;
            });
            html += '</div>';
        }
        
        html += '</div>';
    } else {
        html += '<div class="info-section">';
        html += '<div class="info-section-title">⚖️ Balanceo del Dataset</div>';
        html += `<div class="info-item">
            <span class="info-label">ℹ️ Estado</span>
            <span class="info-value">No se aplicó balanceo (archivo sin columna de diagnóstico)</span>
        </div>`;
        html += '</div>';
    }
    
    // Resumen de predicciones
    if (data.summary) {
        html += '<div class="info-section">';
        html += '<div class="info-section-title">🎯 Resumen de Predicciones</div>';
        Object.keys(data.summary).forEach(disease => {
            const count = data.summary[disease] || 0;
            html += `<div class="info-item">
                <span class="info-label">${disease}</span>
                <span class="info-value">${count} casos</span>
            </div>`;
        });
        html += '</div>';
    }
    
    content.innerHTML = html;
    modal.style.display = 'flex';
}

function closePredictionInfoModal() {
    const modal = document.getElementById('prediction-info-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Cerrar modal con ESC
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closePredictionInfoModal();
    }
});

// Constante para etiquetas de diagnóstico (usada en el modal)
const DIAGNOSIS_LABELS = {1: 'Dengue', 2: 'Malaria', 3: 'Leptospirosis'};

document.getElementById('download-csv-btn')?.addEventListener('click', function() {
    if (!currentBatchData || !currentBatchData.predicciones) {
        showError('No hay datos para descargar');
        return;
    }
    
    // Convertir a CSV
    const headers = ['Fila', 'Predicción', 'Probabilidad'];
    const rows = currentBatchData.predicciones.map((row, index) => {
        let maxProb = 0;
        Object.keys(row).forEach(key => {
            if (key.startsWith('Probabilidad_')) {
                const prob = parseFloat(row[key]);
                if (prob > maxProb) maxProb = prob;
            }
        });
        
        return [
            index + 1,
            row['Prediccion_Label'] || 'Desconocido',
            (maxProb * 100).toFixed(2) + '%'
        ];
    });
    
    const csvContent = [
        headers.join(','),
        ...rows.map(row => row.join(','))
    ].join('\n');
    
    // Descargar
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `predicciones_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
});

// Validación de archivo
document.getElementById('file-upload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const fileName = file.name.toLowerCase();
        if (!fileName.endsWith('.csv') && !fileName.endsWith('.xlsx') && !fileName.endsWith('.xls')) {
            showError('Por favor seleccione un archivo .csv o .xlsx');
            e.target.value = '';
        }
    }
});

