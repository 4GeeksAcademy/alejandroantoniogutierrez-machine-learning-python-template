# 🏖️ ClubPredict — Predicción de Membresías Vacacionales

Aplicación web desarrollada con **Streamlit** que predice si un prospecto comprará una membresía en un club vacacional de hoteles, usando un modelo de **Random Forest** entrenado sobre un dataset sintético de 2,000 registros.

🔗 **App en producción:** `https://clubpredict.onrender.com` *(reemplaza con tu URL)*

---

## 📁 Estructura del Proyecto

```
clubpredict/
├── app.py              # Aplicación Streamlit
├── train_model.py      # Script para generar dataset y entrenar el modelo
├── model.pkl           # Modelo entrenado (Random Forest)
├── encoders.pkl        # Encoders de variables categóricas
├── model_meta.json     # Métricas y metadatos del modelo
├── club_vacacional.csv # Dataset sintético generado
├── requirements.txt    # Dependencias Python
├── render.yaml         # Configuración de despliegue en Render
└── README.md
```

---

## 🤖 Modelo

| Parámetro | Valor |
|-----------|-------|
| Algoritmo | Random Forest Classifier |
| Registros de entrenamiento | 2,000 |
| Variables de entrada | 10 |
| Accuracy | ~68% |
| AUC-ROC | ~0.67 |
| Balanceo de clases | `class_weight='balanced'` |

### Variables del modelo

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `edad` | Numérica | Edad del prospecto (18–75) |
| `ingreso_anual` | Numérica | Ingreso anual estimado en USD |
| `nacionalidad` | Categórica | País de origen (10 nacionalidades) |
| `estado_civil` | Categórica | Soltero / Casado / Divorciado / Viudo |
| `tiene_hijos` | Binaria | Si tiene hijos (0/1) |
| `visitas_previas` | Numérica | Número de visitas anteriores al resort |
| `dias_estancia` | Numérica | Días de estancia actual |
| `canal_contacto` | Categórica | Cómo llegó el prospecto (Presencial/Online/Telefónico/Referido) |
| `calificacion_presentacion` | Ordinal | Calificación de la presentación de ventas (1–5) |
| `oferta_especial` | Binaria | Si se le hizo una oferta especial (0/1) |

---

## 🚀 Despliegue en Render

### Paso 1 — Preparar el repositorio

```bash
git init
git add .
git commit -m "Initial commit - ClubPredict"
git remote add origin https://github.com/TU_USUARIO/clubpredict.git
git push -u origin main
```

### Paso 2 — Crear servicio en Render

1. Ve a [render.com](https://render.com) e inicia sesión
2. Haz clic en **New → Web Service**
3. Conecta tu repositorio de GitHub
4. Configura el servicio:
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Plan:** Free
5. Haz clic en **Create Web Service**

> ⚠️ El servicio gratuito de Render puede tardar ~30 segundos en arrancar tras inactividad.

---

## 💻 Ejecución Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# (Opcional) Regenerar dataset y modelo
python train_model.py

# Ejecutar la app
streamlit run app.py
```

---

## 📊 Dataset Sintético

El dataset fue generado con lógica de negocio realista para un club vacacional:
- **2,000 prospectos** con 10 variables cada uno
- Las probabilidades de compra varían según nacionalidad, ingreso, edad, canal de contacto y otros factores
- Balanceo ~66% compra / ~34% no compra

---

## 🛠️ Recursos Externos

- [Streamlit](https://streamlit.io) — Framework de la interfaz web
- [scikit-learn](https://scikit-learn.org) — Entrenamiento del modelo
- [Render](https://render.com) — Plataforma de despliegue
- [Google Fonts](https://fonts.google.com) — Tipografías (Playfair Display + Inter)

---

## 👤 Autor

Desarrollado como proyecto del módulo de despliegue con Streamlit.
