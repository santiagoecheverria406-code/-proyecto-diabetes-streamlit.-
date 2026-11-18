# 🩺 Predicción de diabetes — Streamlit (Tercer corte)

**Universidad de la Costa · Minería de Datos**
**Integrantes:** Shadia Romero, Valentina Cabrera, Santiago Echeverria y Tatiana Piedrahita.

> ⚠️ **Prototipo académico.** Esta aplicación no es un dispositivo médico y no sustituye el diagnóstico ni la decisión de un profesional de la salud.

🔗 **Aplicación pública:** _(pegar aquí la URL de Streamlit Community Cloud)_

## Descripción
Clasificación binaria de diabetes (`Outcome`: 0 = sin diabetes, 1 = con diabetes) a partir de 8 indicadores clínicos y demográficos: embarazos, glucosa, presión arterial, grosor de piel, insulina, IMC, función de antecedentes familiares y edad.

**Datos:** Pima Indians Diabetes Database, 768 registros y 9 columnas (referencia: https://www.kaggle.com/datasets/hasibur013/diabetes-dataset).

## Modelo final
**Random Forest optimizado** (`n_estimators=300`, `max_depth=5`, `min_samples_leaf=10`, `class_weight="balanced_subsample"`), guardado con `joblib` como un **pipeline completo**:
`ceros inválidos → NaN → imputación por mediana → estandarización → clasificador`.
La app **carga** `modelo_diabetes_pipeline.joblib`; no reentrena.

### Resultados (conjunto de prueba, 154 registros)
| Métrica | Modelo base (Reg. Logística) | **Modelo final (Random Forest)** |
|---|---|---|
| Accuracy | 0.708 | **0.740** |
| Precision | 0.600 | **0.600** |
| Recall (diabetes) | 0.500 | **0.778** |
| F1-score | 0.545 | **0.677** |
| AUC-ROC | 0.813 | **0.820** |
| Falsos negativos | 27 | **12** |

Matriz de confusión del modelo final (filas = real, columnas = predicción): `[[72, 28], [12, 42]]`.

**Validación cruzada (5 folds estratificados, solo entrenamiento):** accuracy 0.780 ± 0.017, recall 0.762 ± 0.032, F1 0.707 ± 0.018, AUC 0.839 ± 0.019.

### Qué se cambió respecto al segundo corte y por qué
1. **Fuga de información corregida.** La imputación por mediana se hacía sobre todo el dataset antes de dividir; ahora los ceros inválidos → NaN, la imputación y el escalado ocurren **dentro del pipeline** y se ajustan solo con entrenamiento.
2. **Selección por recall, no solo por accuracy.** El modelo base tenía buen accuracy pero detectaba solo el 50% de los casos con diabetes. Se compararon 8 modelos con validación cruzada y se optimizaron 3 con `GridSearchCV` usando F2 (pondera el recall).
3. **Resultado:** los falsos negativos bajaron de 27 a 12 en prueba, con accuracy > 0,70.

## Estructura del repositorio
| Archivo | Contenido |
|---|---|
| `app.py` | Aplicación Streamlit |
| `pipeline_utils.py` | Transformador propio (`ZeroToNaN`), necesario para cargar el `.joblib` |
| `modelo_diabetes_pipeline.joblib` | Pipeline completo serializado |
| `diabetes.csv` | Datos (EDA de la app) |
| `metrics.json`, `verificacion_registro.json` | Métricas y registro de prueba para verificar app vs. notebook |
| `Proyecto_Diabetes_Tercer_Corte.ipynb` | Notebook actualizado (comparación, optimización, antes/después, guardado) |
| `requirements.txt` | Dependencias con versiones fijas (las mismas con las que se guardó el modelo) |

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud
1. Subir esta carpeta a un repositorio **público** de GitHub.
2. Entrar a https://share.streamlit.io → *Create app* → elegir el repositorio, rama `main` y archivo principal `app.py`.
3. En *Advanced settings* seleccionar **Python 3.12**.
4. *Deploy* y verificar que la URL pública abra y prediga.

## Pruebas realizadas
| Prueba | Resultado |
|---|---|
| Caso normal (riesgo bajo) | Predice clase 0 con probabilidad baja |
| Caso de riesgo alto | Predice clase 1 con probabilidad alta |
| Entrada inválida (glucosa = 0, IMC = 250) | La app rechaza la entrada y muestra mensajes de rango |
| Datos opcionales omitidos (insulina, piel) | El pipeline los imputa y se informa al usuario |
| Registro conocido del conjunto de prueba | La probabilidad de la app coincide con la del notebook (`verificacion_registro.json`) |

## Limitaciones
Datos acotados (768 registros, mujeres ≥ 21 años de una población específica), ceros no medidos en insulina y grosor de piel (imputados), recall ≈ 78 % (≈ 1 de cada 5 casos no se detecta), conjunto de prueba pequeño (154), clases desbalanceadas y asociación ≠ causalidad. Un uso real requeriría datos autorizados, validación clínica y supervisión profesional.

## Uso de inteligencia artificial
> **Revisar y completar antes de entregar:** todos los integrantes deben leer esta sección, comprobar que es fiel a lo que hicieron y poder explicar el código.

- **Herramienta:** Claude (Anthropic) _(el primer informe usó ChatGPT)_.
- **Propósito:** apoyar la revisión del modelo, el diseño del pipeline, la optimización de hiperparámetros, la construcción de `app.py` y la redacción del README.
- **Prompt principal:** _«Necesito que me ayudes a realizar esta actividad (Integración y despliegue del modelo con Streamlit)»_, adjuntando el enunciado, el notebook y el informe del primer corte, y pidiendo hacer el modelado desde cero. _(Reemplazar o ampliar con el prompt exacto si lo modificaron.)_
- **Cómo verificamos la respuesta:** se ejecutó el notebook completo; se comprobó que el `.joblib` recargado reproduce las predicciones; se probó la app con un caso normal, uno inválido y un registro del conjunto de prueba (coincide con el notebook); se revisó la fuga de información y se compararon resultados antes y después.
- **Problemas encontrados:** (1) un primer archivo recibido era una plantilla con datos sintéticos de otro tema, no nuestro proyecto, y se descartó; (2) el notebook del primer corte imputaba antes de dividir los datos (fuga de información); (3) la IA no pudo descargar el dataset desde Kaggle y usó una copia pública del Pima en GitHub; (4) un error de pandas 3 (arreglo de solo lectura) al calcular correlaciones.
- **Ajustes realizados:** limpieza e imputación dentro del pipeline, selección por recall y F2, versiones fijas en `requirements.txt`, validación explícita de rangos en la app y corrección del error de pandas. _(Verificar que `diabetes.csv` sea idéntico al descargado de Kaggle; si no, volver a ejecutar el notebook con el suyo.)_
