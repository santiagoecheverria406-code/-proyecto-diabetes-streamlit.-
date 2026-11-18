"""Aplicación Streamlit — Predicción de diabetes (prototipo académico).

Carga el pipeline completo (preprocesamiento + modelo) guardado con joblib.
La aplicación NO reentrena el modelo.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline_utils import FEATURES, ZERO_INVALID_COLS  # noqa: F401  (necesario para cargar el .joblib)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "modelo_diabetes_pipeline.joblib"
DATA_PATH = BASE / "diabetes.csv"
METRICS_PATH = BASE / "metrics.json"
VERIF_PATH = BASE / "verificacion_registro.json"

INTEGRANTES = ["Shadia Romero", "Valentina Cabrera", "Santiago Echeverria", "Tatiana Piedrahita"]
C_NEG, C_POS = "#3B82F6", "#EF4444"

st.set_page_config(page_title="Predicción de diabetes", page_icon="🩺", layout="wide")


# ----------------------------------------------------------------------------
# Carga de recursos (en caché: no se reentrena nada)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
data = load_data()
metrics = load_json(METRICS_PATH)
verif = load_json(VERIF_PATH)

# ----------------------------------------------------------------------------
# Variables del modelo: etiquetas, unidades y rangos válidos
#   lo = mínimo válido, hi = máximo válido (se valida en el código)
#   optional = el dato puede omitirse (se trata como faltante y el pipeline lo imputa)
# ----------------------------------------------------------------------------
VARS = {
    "Pregnancies": dict(label="Número de embarazos", unit="", lo=0, hi=20, default=1, step=1, kind="int", optional=False),
    "Glucose": dict(label="Glucosa plasmática", unit="mg/dL", lo=40, hi=400, default=110, step=1, kind="int", optional=False),
    "BloodPressure": dict(label="Presión arterial diastólica", unit="mm Hg", lo=30, hi=160, default=70, step=1, kind="int", optional=False),
    "SkinThickness": dict(label="Grosor del pliegue cutáneo (tríceps)", unit="mm", lo=5, hi=100, default=25, step=1, kind="int", optional=True),
    "Insulin": dict(label="Insulina sérica a las 2 horas", unit="µU/mL", lo=10, hi=900, default=100, step=1, kind="int", optional=True),
    "BMI": dict(label="Índice de masa corporal (IMC)", unit="kg/m²", lo=10.0, hi=70.0, default=28.0, step=0.1, kind="float", optional=False),
    "DiabetesPedigreeFunction": dict(label="Función de antecedentes familiares de diabetes", unit="índice", lo=0.05, hi=2.5, default=0.4, step=0.001, kind="float", optional=False),
    "Age": dict(label="Edad", unit="años", lo=18, hi=100, default=33, step=1, kind="int", optional=False),
}

EXAMPLES = {
    "— Valores por defecto —": None,
    "Caso normal (riesgo bajo)": dict(Pregnancies=1, Glucose=92, BloodPressure=70, SkinThickness=22, Insulin=85, BMI=24.5, DiabetesPedigreeFunction=0.25, Age=27),
    "Caso de riesgo alto": dict(Pregnancies=6, Glucose=168, BloodPressure=82, SkinThickness=38, Insulin=240, BMI=36.2, DiabetesPedigreeFunction=0.85, Age=52),
    "Entrada inválida (para probar validación)": dict(Pregnancies=2, Glucose=0, BloodPressure=70, SkinThickness=25, Insulin=100, BMI=250.0, DiabetesPedigreeFunction=0.4, Age=30),
}


def _widget_key(name):
    return f"in_{name}"


def apply_example():
    """Callback: rellena el formulario con un ejemplo."""
    ex = EXAMPLES.get(st.session_state["example_choice"])
    for name, spec in VARS.items():
        val = spec["default"] if ex is None else ex[name]
        st.session_state[_widget_key(name)] = int(val) if spec["kind"] == "int" else float(val)
        if spec["optional"]:
            st.session_state[_widget_key(name) + "_na"] = False


def validate(values):
    """Valida tipos y rangos. Devuelve lista de mensajes de error."""
    errors = []
    for name, spec in VARS.items():
        v = values[name]
        if v is None:  # dato omitido (solo permitido en variables opcionales)
            if not spec["optional"]:
                errors.append(f"**{spec['label']}** es obligatoria.")
            continue
        if not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
            errors.append(f"**{spec['label']}** debe ser un número válido.")
        elif not (spec["lo"] <= v <= spec["hi"]):
            unit = f" {spec['unit']}" if spec["unit"] else ""
            errors.append(f"**{spec['label']}**: {v:g}{unit} está fuera del rango válido "
                          f"({spec['lo']:g} – {spec['hi']:g}{unit}).")
    return errors


def predict(values):
    row = pd.DataFrame([{k: (np.nan if values[k] is None else values[k]) for k in FEATURES}])
    proba = float(model.predict_proba(row)[0, 1])
    pred = int(model.predict(row)[0])
    return pred, proba


# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.title("🩺 Predicción de diabetes con minería de datos")
st.caption("Universidad de la Costa · Minería de Datos · Proyecto de clase — Tercer corte")
st.warning(
    "**Prototipo académico.** Esta aplicación fue desarrollada con fines educativos. "
    "No es un dispositivo médico ni sustituye el diagnóstico, la valoración ni la decisión de un profesional de la salud."
)

tab_proy, tab_eda, tab_pred, tab_eval, tab_lim = st.tabs(
    ["📌 Proyecto", "📊 Exploración (EDA)", "🔮 Predicción", "✅ Evaluación del modelo", "⚠️ Limitaciones"]
)

# ----------------------------------------------------------------------------
# 1. Proyecto
# ----------------------------------------------------------------------------
with tab_proy:
    st.header("Presentación del proyecto")
    st.markdown(
        "**Objetivo.** Clasificar si una persona presenta diabetes (*Outcome = 1*) o no (*Outcome = 0*) "
        "a partir de ocho indicadores clínicos y demográficos, y estimar la probabilidad asociada a esa clasificación."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Integrantes")
        for n in INTEGRANTES:
            st.markdown(f"- {n}")
        st.subheader("Fuente de los datos")
        st.markdown(
            "*Diabetes Dataset*, basado en la **Pima Indians Diabetes Database** (Kaggle: "
            "[hasibur013/diabetes-dataset](https://www.kaggle.com/datasets/hasibur013/diabetes-dataset)). "
            "Contiene **768 registros** y **9 columnas**."
        )
    with c2:
        st.subheader("Resumen")
        m1, m2, m3 = st.columns(3)
        m1.metric("Registros", f"{len(data):,}")
        m2.metric("Variables de entrada", len(FEATURES))
        m3.metric("% con diabetes", f"{data['Outcome'].mean():.1%}")
        st.markdown(
            f"**Modelo final:** {metrics['modelo_final']}, guardado como un pipeline completo "
            "(ceros inválidos → faltantes → imputación por mediana → estandarización → clasificador). "
            "La aplicación **carga** ese archivo; no entrena de nuevo."
        )

    st.subheader("Variables utilizadas por el modelo")
    desc = pd.DataFrame(
        [{"Variable": k, "Descripción": v["label"], "Unidad": v["unit"] or "—",
          "Rango válido en la app": f"{v['lo']:g} – {v['hi']:g}"} for k, v in VARS.items()]
    )
    st.dataframe(desc, hide_index=True)

# ----------------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------------
with tab_eda:
    st.header("Exploración de datos")
    st.caption("En estos gráficos los ceros no válidos de glucosa e IMC se omiten (se tratan como valores faltantes).")
    eda = data.copy()
    for c in ZERO_INVALID_COLS:
        eda[c] = eda[c].replace(0, np.nan)
    lab = {0: "Sin diabetes", 1: "Con diabetes"}

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("1. Distribución de la variable objetivo")
        counts = eda["Outcome"].map(lab).value_counts().reindex(["Sin diabetes", "Con diabetes"])
        fig, ax = plt.subplots(figsize=(5, 3.6))
        bars = ax.bar(counts.index, counts.values, color=[C_NEG, C_POS])
        ax.bar_label(bars, padding=2)
        ax.set_ylabel("Número de pacientes")
        ax.set_title("Clases desbalanceadas (≈ 65 % / 35 %)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        st.subheader("2. Glucosa según la presencia de diabetes")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for v, col in [(0, C_NEG), (1, C_POS)]:
            ax.hist(eda.loc[eda["Outcome"] == v, "Glucose"].dropna(), bins=20, alpha=0.65, color=col, label=lab[v])
        ax.set_xlabel("Glucosa (mg/dL)")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    g3, g4 = st.columns(2)
    with g3:
        st.subheader("3. IMC vs. glucosa")
        fig, ax = plt.subplots(figsize=(5, 3.8))
        for v, col in [(0, C_NEG), (1, C_POS)]:
            s = eda[eda["Outcome"] == v]
            ax.scatter(s["BMI"], s["Glucose"], alpha=0.45, s=18, color=col, label=lab[v])
        ax.set_xlabel("IMC (kg/m²)")
        ax.set_ylabel("Glucosa (mg/dL)")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
    with g4:
        st.subheader("Variables más influyentes en el modelo")
        imp = pd.Series(metrics["importancias"]).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.barh(imp.index, imp.values, color="#6366F1")
        ax.set_xlabel("Importancia (Random Forest)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
    st.info("La glucosa es la variable que más separa ambas clases, seguida del IMC y la edad. "
            "Esto describe asociaciones en los datos; no demuestra causalidad.")

# ----------------------------------------------------------------------------
# 3. Predicción
# ----------------------------------------------------------------------------
with tab_pred:
    st.header("Predicción individual")
    st.caption("Ingrese los datos del paciente. Los rangos se validan antes de calcular la predicción.")

    # Valores iniciales
    for name, spec in VARS.items():
        st.session_state.setdefault(_widget_key(name), int(spec["default"]) if spec["kind"] == "int" else float(spec["default"]))
        if spec["optional"]:
            st.session_state.setdefault(_widget_key(name) + "_na", False)

    st.selectbox("Cargar un ejemplo (opcional)", list(EXAMPLES.keys()), key="example_choice", on_change=apply_example)

    with st.form("form_prediccion"):
        cols = st.columns(3)
        values = {}
        for i, (name, spec) in enumerate(VARS.items()):
            with cols[i % 3]:
                unit = f" ({spec['unit']})" if spec["unit"] else ""
                label = f"{spec['label']}{unit}"
                # El widget acepta un rango amplio; el rango clínico válido se comprueba en validate()
                is_int = spec["kind"] == "int"
                val = st.number_input(
                    label, key=_widget_key(name),
                    min_value=0 if is_int else 0.0,
                    max_value=100000 if is_int else 100000.0,
                    step=spec["step"],
                    help=f"Rango válido: {spec['lo']:g} – {spec['hi']:g}",
                    format="%d" if is_int else ("%.3f" if name == "DiabetesPedigreeFunction" else "%.1f"),
                )
                if spec["optional"]:
                    na = st.checkbox("No dispongo de este dato", key=_widget_key(name) + "_na")
                    values[name] = None if na else val
                else:
                    values[name] = val
        submitted = st.form_submit_button("Calcular predicción", type="primary")

    if submitted:
        errors = validate(values)
        if errors:
            st.error("No se puede calcular la predicción. Corrija los siguientes datos:")
            for e in errors:
                st.markdown(f"- {e}")
        else:
            pred, proba = predict(values)
            st.subheader("Resultado")
            r1, r2 = st.columns([1, 2])
            r1.metric("Probabilidad estimada de diabetes", f"{proba:.1%}")
            with r2:
                if pred == 1:
                    st.error("**El modelo clasifica este caso como: posible diabetes (clase 1).**")
                else:
                    st.success("**El modelo clasifica este caso como: sin diabetes (clase 0).**")
                st.progress(min(max(proba, 0.0), 1.0), text="Probabilidad de la clase *diabetes* (umbral de decisión: 50 %)")
            omitted = [VARS[k]["label"] for k, v in values.items() if v is None]
            if omitted:
                st.info("Datos omitidos: " + ", ".join(omitted) +
                        ". El pipeline los reemplazó por la mediana del conjunto de entrenamiento, "
                        "por lo que la estimación es menos específica.")
            outside = [VARS[k]["label"] for k in FEATURES
                       if values[k] is not None and k not in ("Pregnancies",)
                       and not (data.loc[data[k] > 0, k].min() <= values[k] <= data[k].max())]
            if outside:
                st.warning("Valores fuera del rango observado en los datos de entrenamiento (extrapolación poco fiable): "
                           + ", ".join(outside) + ".")
            st.caption("Es una estimación estadística, no un diagnóstico. Consulte siempre a un profesional de la salud.")

    # Verificación con un registro conocido del conjunto de prueba
    with st.expander("🔎 Verificación con un registro conocido del conjunto de prueba"):
        st.write("Se aplica el pipeline a un registro del conjunto de prueba y se compara con el resultado guardado por el notebook.")
        # En el notebook el pipeline recibió el valor 0 original; aquí lo pasamos tal cual para reproducirlo exactamente
        row = pd.DataFrame([verif["entrada"]])[FEATURES]
        p_app = float(model.predict_proba(row)[0, 1])
        c_app = int(model.predict(row)[0])
        st.dataframe(row, hide_index=True)
        v1, v2, v3 = st.columns(3)
        v1.metric("Probabilidad (app)", f"{p_app:.4f}")
        v2.metric("Probabilidad (notebook)", f"{verif['probabilidad_diabetes']:.4f}")
        v3.metric("Valor real del registro", "Diabetes" if verif["valor_real"] == 1 else "Sin diabetes")
        st.caption("Clasificación del modelo para este registro: " + ("**posible diabetes**" if c_app == 1 else "**sin diabetes**")
                   + " (la verificación comprueba que app y notebook den el mismo resultado, no que acierten).")
        if abs(p_app - verif["probabilidad_diabetes"]) < 1e-9 and c_app == verif["prediccion"]:
            st.success("✅ La predicción de la app coincide con la del notebook.")
        else:
            st.error("❌ La predicción de la app NO coincide con la del notebook.")
        st.caption("Nota: los ceros de Insulina y Grosor de piel en el registro original son valores no medidos; "
                   "el pipeline los convierte en faltantes y los imputa igual que en el notebook.")

# ----------------------------------------------------------------------------
# 4. Evaluación
# ----------------------------------------------------------------------------
with tab_eval:
    st.header("Evaluación del modelo")
    t = metrics["prueba"]
    st.caption(f"Conjunto de prueba: {metrics['test_size']} registros (20 %, estratificado). "
               f"Entrenamiento: {metrics['train_size']} registros.")
    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{t['accuracy']:.3f}")
    b.metric("Precision", f"{t['precision']:.3f}")
    c.metric("Recall (diabetes)", f"{t['recall']:.3f}")
    d.metric("F1-score", f"{t['f1']:.3f}")
    e.metric("AUC-ROC", f"{t['roc_auc']:.3f}")

    l, r = st.columns(2)
    with l:
        st.subheader("Matriz de confusión")
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Sin diabetes", "Con diabetes"])
        ax.set_yticks([0, 1], ["Sin diabetes", "Con diabetes"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"- **Falsos negativos:** {fn} pacientes con diabetes no detectados.\n"
                    f"- **Falsos positivos:** {fp} pacientes sin diabetes señalados como positivos.")
    with r:
        st.subheader("Validación cruzada (5 folds, entrenamiento)")
        cv = pd.DataFrame(metrics["validacion_cruzada"]).T
        cv["Resultado"] = cv.apply(lambda x: f"{x['media']:.3f} ± {x['std']:.3f}", axis=1)
        cv.index = ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]
        st.dataframe(cv[["Resultado"]])
        st.caption("Media ± desviación estándar entre folds: resultados estables.")

    st.subheader("Antes y después de la optimización (conjunto de prueba)")
    base, fin = metrics["base_prueba"], metrics["prueba"]
    fn_base = metrics["base_confusion_matrix"][1][0]
    comp = pd.DataFrame({
        "Regresión Logística (base)": [base["accuracy"], base["precision"], base["recall"], base["f1"], base["roc_auc"], fn_base],
        "Random Forest optimizado (final)": [fin["accuracy"], fin["precision"], fin["recall"], fin["f1"], fin["roc_auc"], fn],
    }, index=["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC", "Falsos negativos"])
    st.dataframe(comp.style.format("{:.3f}", subset=pd.IndexSlice[comp.index[:-1], :]).format("{:.0f}", subset=pd.IndexSlice[["Falsos negativos"], :]))

    with st.expander("Comparación de modelos candidatos (validación cruzada)"):
        mc = pd.DataFrame(metrics["comparacion_modelos"]).set_index("modelo")
        st.dataframe(mc[["accuracy", "precision", "recall", "f1", "roc_auc"]].sort_values("recall", ascending=False).round(3))

    st.info("**Por qué este modelo.** En un contexto de salud los falsos negativos (no detectar a un paciente con diabetes) "
            "son más costosos que los falsos positivos; por eso se priorizó el **recall** de la clase diabetes y no solo el accuracy.")

# ----------------------------------------------------------------------------
# 5. Limitaciones
# ----------------------------------------------------------------------------
with tab_lim:
    st.header("Limitaciones del modelo")
    st.markdown("""
- **Datos acotados:** solo 768 registros del conjunto Pima, que corresponden a mujeres de al menos 21 años de una población específica. El modelo puede no generalizar a hombres, menores ni otras poblaciones.
- **Calidad de los datos:** muchas mediciones de insulina (≈ 49 %) y grosor de piel (≈ 30 %) estaban registradas como cero, es decir, no medidas. Se imputaron con la mediana, lo que introduce incertidumbre.
- **Rendimiento moderado:** el modelo aún se equivoca. Con ≈ 78 % de recall, alrededor de 1 de cada 5 pacientes con diabetes no es detectado, y varios pacientes sanos son señalados como positivos.
- **Conjunto de prueba pequeño (154 registros):** una sola predicción cambia el recall en casi 2 puntos porcentuales; las métricas tienen margen de error.
- **Desbalance de clases:** ≈ 65 % sin diabetes y ≈ 35 % con diabetes; se mitigó con ponderación de clases, pero no se eliminó.
- **Asociación no es causalidad:** la importancia de las variables describe patrones estadísticos.
- **Uso responsable:** es un prototipo académico. Un uso real exigiría datos autorizados, validación clínica, revisión de sesgos y supervisión de profesionales de la salud.
""")
    st.caption("Proyecto desarrollado por: " + ", ".join(INTEGRANTES) + ".")
