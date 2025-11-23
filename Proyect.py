# Proyect.py
"""
Proyecto: Clasificador Iris - Panel interactivo con Streamlit
Autor: [NOMBRES_DEL_EQUIPO]  <- Edita aquí para agregar los nombres de los miembros del equipo
Descripción:
 - Carga el dataset Iris
 - Diseña un flujo de trabajo (exploración, preprocesamiento, modelado, evaluación)
 - Entrena un RandomForest y muestra métricas (Accuracy, Precision, Recall, F1)
 - Permite ingresar una nueva muestra (sepalo/petalo) y predecir la especie
 - Visualiza la nueva muestra en un scatter 3D junto al dataset
 - Incluye visualizaciones: histogramas, scatter matrix, matriz de confusión, importancias
Instrucciones:
 - Instalar dependencias (requirements.txt)
 - Ejecutar: streamlit run Proyect.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import StringIO

# -------------------------
# Configuración de la página
# -------------------------
st.set_page_config(page_title="Iris - Clasificador (RandomForest)", layout="wide", initial_sidebar_state="expanded")
st.title("🌸 Clasificación de Iris — Proyecto")

# -------------------------
# Sidebar: metadata y parámetros
# -------------------------
st.sidebar.header("Información del proyecto / Equipo")
team_names = st.sidebar.text_area("Nombres del equipo, value=ERNESTO DIAZ, NICOLLE ALGARIN", help="Escribe los nombres de los miembros del equipo tal como deben aparecer en el repo y la presentación.")
video_link = st.sidebar.text_input("Enlace a la presentación en video (opcional)", value="", help="Pega el enlace de tu video (YouTube/Drive).")

st.sidebar.markdown("---")
st.sidebar.header("Hiperparámetros del modelo")
n_estimators = st.sidebar.slider("n_estimators (nº de árboles)", 10, 500, 100, step=10)
max_depth_option = st.sidebar.checkbox("Fijar max_depth", value=False)
max_depth = st.sidebar.slider("max_depth", 1, 30, 5, step=1) if max_depth_option else None
test_size = st.sidebar.slider("Tamaño del test (fracción)", 0.1, 0.5, 0.2, step=0.05)
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))
do_crossval = st.sidebar.checkbox("Hacer validación cruzada (5 folds)", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Instrucciones: cambia hiperparámetros para reentrenar el modelo en esta sesión (cacheado).")

# -------------------------
# Cargar datos Iris
# -------------------------
@st.cache_data
def load_iris():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # normalizar nombres de columnas
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    df["target"] = iris.target
    df["species"] = pd.Categorical([iris.target_names[i] for i in iris.target])
    return df, iris

df, iris_bunch = load_iris()
FEATURES = list(df.columns[:4])
TARGET_NAMES = list(iris_bunch.target_names)

# -------------------------
# Mostrar flujo de trabajo (metodología)
# -------------------------
with st.expander("📋 Flujo de trabajo (metodología)", expanded=True):
    st.markdown("""
    **Flujo de trabajo propuesto**
    1. *Comprensión de datos:* inspección de variables, estadísticas descriptivas, visualizaciones (histogramas, scatter matrix).
    2. *Preprocesamiento:* mantener variables numéricas; en este dataset no se requieren imputaciones ni codificaciones complejas.
    3. *Modelado:* RandomForest (robusto, poco sensible a escalado).
    4. *Validación:* uso de separación Train/Test y validación cruzada estratificada para estimar generalización.
    5. *Evaluación:* métricas: Accuracy, Precision (weighted), Recall (weighted), F1 (weighted), matriz de confusión.
    6. *Interpretación:* importancia de variables y visualizaciones 2D/3D para entender la separación entre clases.
    7. *Entrega:* panel interactivo en Streamlit, documentación (README) y video explicativo.
    """)

# -------------------------
# Entrenamiento del modelo (cacheado)
# -------------------------
@st.cache_resource
def train_rf(n_estimators, max_depth, test_size, random_state, do_crossval):
    X = df[FEATURES].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # métricas
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, target_names=TARGET_NAMES, zero_division=0, output_dict=False),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "feature_importances": model.feature_importances_
    }

    # validación cruzada (si se solicita)
    cv_scores = None
    if do_crossval:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        metrics["cv_scores"] = cv_scores.tolist()

    return model, (X_train, X_test, y_train, y_test, y_pred), metrics

model, split_info, metrics = train_rf(n_estimators=n_estimators, max_depth=max_depth, test_size=test_size, random_state=random_state, do_crossval=do_crossval)
X_train, X_test, y_train, y_test, y_pred = split_info

# -------------------------
# Panel principal: métricas
# -------------------------
st.subheader("📈 Métricas del modelo (conjunto de prueba)")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Exactitud (Accuracy)", f"{metrics['accuracy']:.4f}")
col_b.metric("Precisión (weighted)", f"{metrics['precision']:.4f}")
col_c.metric("Recall (weighted)", f"{metrics['recall']:.4f}")
col_d.metric("F1-score (weighted)", f"{metrics['f1']:.4f}")

with st.expander("Ver reporte de clasificación y matriz de confusión"):
    st.text("Reporte de clasificación (texto):")
    st.text(metrics["classification_report"])
    st.write("Matriz de confusión (filas = verdad, columnas = predicción)")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES)
    st.dataframe(cm_df.style.background_gradient(cmap="Blues"))

    if "cv_scores" in metrics:
        st.markdown("**Validación cruzada (accuracy) — 5 folds**")
        st.write(metrics["cv_scores"])
        st.write(f"Media: {np.mean(metrics['cv_scores']):.4f}  —  Desviación estándar: {np.std(metrics['cv_scores']):.4f}")

# -------------------------
# Visualizaciones del dataset
# -------------------------
st.subheader("🔍 Visualizaciones del dataset")

# estadísticas descriptivas
with st.expander("Estadísticas descriptivas"):
    st.dataframe(df.describe().T)

# Histogramas
st.markdown("### Histogramas por característica")
fig_hist, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()
for i, feat in enumerate(FEATURES):
    sns.histplot(data=df, x=feat, hue="species", ax=axes[i], multiple="stack", edgecolor=None)
    axes[i].set_title(feat.replace("_", " ").title())
plt.tight_layout()
st.pyplot(fig_hist)

# Scatter matrix (plotly)
st.markdown("### Matriz de dispersión interactiva (Scatter matrix)")
fig_sm = px.scatter_matrix(df, dimensions=FEATURES, color="species",
                           labels={c: c.replace("_", " ") for c in FEATURES},
                           title="Scatter matrix de las 4 características")
fig_sm.update_layout(height=700)
st.plotly_chart(fig_sm, use_container_width=True)

# Importancia de características
st.markdown("### Importancia de características (RandomForest)")
fi = metrics["feature_importances"]
fi_df = pd.DataFrame({"feature": FEATURES, "importance": fi}).sort_values("importance", ascending=False)
st.bar_chart(fi_df.set_index("feature"))

# -------------------------
# Panel interactivo de predicción
# -------------------------
st.subheader("🧪 Predicción interactiva — Ingresa una nueva muestra")

with st.form("form_predict", clear_on_submit=False):
    c1, c2 = st.columns(2)
    sepal_length = c1.number_input("Longitud del sépalo (cm)", min_value=0.0, value=float(df[FEATURES[0]].mean()), step=0.1)
    sepal_width  = c2.number_input("Anchura del sépalo (cm)", min_value=0.0, value=float(df[FEATURES[1]].mean()), step=0.1)
    petal_length = c1.number_input("Longitud del pétalo (cm)", min_value=0.0, value=float(df[FEATURES[2]].mean()), step=0.1)
    petal_width  = c2.number_input("Anchura del pétalo (cm)", min_value=0.0, value=float(df[FEATURES[3]].mean()), step=0.1)
    submitted = st.form_submit_button("Predecir")

if submitted:
    x_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(x_new)[0]
    proba = model.predict_proba(x_new)[0]
    st.success(f"🌿 Especie predicha: **{iris_bunch.target_names[pred]}**")
    st.write("Probabilidades (por clase):")
    probs = {iris_bunch.target_names[i]: float(f"{p:.4f}") for i,p in enumerate(proba)}
    st.json(probs)

    # Visualización 3D interactiva
    st.markdown("### Visualización 3D — compara la nueva muestra con el dataset")
    axis_x = st.selectbox("Eje X", options=FEATURES, index=0, key="axis_x")
    axis_y = st.selectbox("Eje Y", options=FEATURES, index=1, key="axis_y")
    axis_z = st.selectbox("Eje Z", options=FEATURES, index=2, key="axis_z")

    fig3d = px.scatter_3d(df, x=axis_x, y=axis_y, z=axis_z, color="species", symbol="species",
                         labels={c: c.replace("_", " ") for c in FEATURES},
                         title="Posición de la nueva muestra (X) respecto al dataset")
    # Añadir nueva muestra con marcador distinto
    coords = {
        FEATURES[0]: sepal_length,
        FEATURES[1]: sepal_width,
        FEATURES[2]: petal_length,
        FEATURES[3]: petal_width
    }
    fig3d.add_trace(go.Scatter3d(
        x=[coords[axis_x]],
        y=[coords[axis_y]],
        z=[coords[axis_z]],
        mode='markers+text',
        marker=dict(size=8, color='black', symbol='x'),
        text=[f"Nueva ({iris_bunch.target_names[pred]})"],
        name='Nueva muestra'
    ))
    fig3d.update_layout(height=700)
    st.plotly_chart(fig3d, use_container_width=True)
else:
    st.markdown("Si quieres predecir, completa las medidas arriba y presiona **Predecir**. También puedes explorar el dataset en 3D abajo.")
    # mostrar dataset en 3D por defecto
    fig3d_all = px.scatter_3d(df, x=FEATURES[0], y=FEATURES[1], z=FEATURES[2], color="species", symbol="species",
                              labels={c: c.replace("_"," ") for c in FEATURES},
                              title="Dataset Iris (3D) — ejes por defecto")
    fig3d_all.update_layout(height=600)
    st.plotly_chart(fig3d_all, use_container_width=True)

# -------------------------
# Exportar / descargar modelo y resumen
# -------------------------
st.markdown("---")
st.subheader("Exportar resultados")

col1, col2 = st.columns(2)
with col1:
    if st.button("Descargar CSV (dataset)"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Descargar dataset (CSV)", data=csv, file_name="iris_dataset.csv", mime="text/csv")
with col2:
    if st.button("Guardar modelo (pickle)"):
        model_filename = "rf_iris_model.joblib"
        joblib.dump(model, model_filename)
        with open(model_filename, "rb") as f:
            st.download_button(label="Descargar modelo (joblib)", data=f, file_name=model_filename, mime="application/octet-stream")

# -------------------------
# Recomendaciones y cierre
# -------------------------
st.markdown("---")
st.subheader("Conclusiones y recomendaciones")
st.markdown("""
- Se siguió un flujo de trabajo clásico: exploración → modelado → evaluación → interpretación.
- RandomForest funciona muy bien con este dataset; las métricas suelen ser altas.
- Recomendaciones de mejora:
  - Buscar hiperparámetros óptimos con GridSearchCV o RandomizedSearchCV.
  - Añadir explicabilidad (SHAP) para interpretar predicciones individuales.
  - Ampliar la interfaz con más análisis (curva ROC, decisiones por clase).
""")

# Mostrar equipo y enlace de video si se proporcionó
if team_names:
    st.markdown("### Equipo")
    st.write([n.strip() for n in team_names.split(",") if n.strip()])

if video_link:
    st.markdown("### Enlace a la presentación en video")
    st.write(video_link)

# Footer
st.caption("Proyecto Iris - Implementación con Streamlit. Edita el archivo para añadir los nombres definitivos del equipo y actualizar el README / requirements.")

