"""
train_model.py
Genera el dataset sintético y entrena el modelo de predicción de membresías.
Ejecutar una vez antes de desplegar: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pickle
import json

np.random.seed(42)
N = 2000

# ── Dataset sintético ─────────────────────────────────────────────────────────
nacionalidades = {
    'Mexicano': 0.42, 'Estadounidense': 0.55, 'Canadiense': 0.50,
    'Colombiano': 0.32, 'Argentino': 0.30, 'Español': 0.37,
    'Brasileño': 0.35, 'Alemán': 0.40, 'Francés': 0.38, 'Británico': 0.44,
}
nac_lista = list(nacionalidades.keys())

nacionalidad = np.random.choice(nac_lista, size=N,
                                p=[0.25,0.18,0.10,0.10,0.08,0.07,0.07,0.05,0.05,0.05])
edad = np.random.normal(42, 12, N).clip(18, 75).astype(int)
ingreso_anual = np.random.normal(55000, 20000, N).clip(15000, 150000).astype(int)
visitas_previas = np.random.poisson(1.2, N).clip(0, 8)
dias_estancia = np.random.choice([3,4,5,6,7,10,14], size=N,
                                  p=[0.15,0.20,0.25,0.15,0.12,0.08,0.05])
tiene_hijos = np.random.choice([0,1], size=N, p=[0.40,0.60])
estado_civil = np.random.choice(['Soltero','Casado','Divorciado','Viudo'], size=N,
                                 p=[0.25,0.55,0.15,0.05])
canal_contacto = np.random.choice(['Presencial','Online','Telefónico','Referido'], size=N,
                                   p=[0.35,0.30,0.20,0.15])
calificacion_presentacion = np.random.choice([1,2,3,4,5], size=N,
                                              p=[0.10,0.15,0.25,0.30,0.20])
oferta_especial = np.random.choice([0,1], size=N, p=[0.50,0.50])

# Probabilidad de compra basada en reglas de negocio
prob_compra = np.zeros(N)
for i in range(N):
    p = nacionalidades[nacionalidad[i]]
    if 35 <= edad[i] <= 55:       p += 0.08
    elif edad[i] < 25:             p -= 0.15
    if ingreso_anual[i] > 70000:   p += 0.10
    elif ingreso_anual[i] < 25000: p -= 0.18
    p += visitas_previas[i] * 0.04
    if dias_estancia[i] >= 7:      p += 0.06
    if tiene_hijos[i]:             p += 0.05
    if estado_civil[i] == 'Casado': p += 0.06
    if canal_contacto[i] == 'Referido': p += 0.14
    elif canal_contacto[i] == 'Online': p -= 0.06
    p += (calificacion_presentacion[i] - 3) * 0.07
    if oferta_especial[i]:         p += 0.08
    prob_compra[i] = np.clip(p, 0.02, 0.95)

compro = (np.random.rand(N) < prob_compra).astype(int)

df = pd.DataFrame({
    'nacionalidad': nacionalidad, 'edad': edad, 'ingreso_anual': ingreso_anual,
    'visitas_previas': visitas_previas, 'dias_estancia': dias_estancia,
    'tiene_hijos': tiene_hijos, 'estado_civil': estado_civil,
    'canal_contacto': canal_contacto,
    'calificacion_presentacion': calificacion_presentacion,
    'oferta_especial': oferta_especial, 'compro_membresia': compro
})
df.to_csv('club_vacacional.csv', index=False)
print(f"Dataset: {len(df)} registros | Tasa de compra: {compro.mean():.1%}")

# ── Encoders ──────────────────────────────────────────────────────────────────
le_nac   = LabelEncoder()
le_civil = LabelEncoder()
le_canal = LabelEncoder()

df['nac_enc']   = le_nac.fit_transform(df['nacionalidad'])
df['civil_enc'] = le_civil.fit_transform(df['estado_civil'])
df['canal_enc'] = le_canal.fit_transform(df['canal_contacto'])

features = ['edad','ingreso_anual','visitas_previas','dias_estancia',
            'tiene_hijos','calificacion_presentacion','oferta_especial',
            'nac_enc','civil_enc','canal_enc']

X = df[features]
y = df['compro_membresia']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

print(f"Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")
print(classification_report(y_test, y_pred))

# ── Guardar artefactos ────────────────────────────────────────────────────────
scaler = StandardScaler().fit(X_train)

with open('model.pkl', 'wb') as f:   pickle.dump(model, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump({'nac': le_nac, 'civil': le_civil,
                 'canal': le_canal, 'scaler': scaler}, f)

fi = dict(zip(features, model.feature_importances_))
meta = {
    'modelo': 'Random Forest (class_weight=balanced)',
    'accuracy': float(acc), 'auc': float(auc),
    'nacionalidades': nac_lista,
    'estados_civiles': list(le_civil.classes_),
    'canales': list(le_canal.classes_),
    'features': features,
    'feature_importance': fi
}
with open('model_meta.json', 'w') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n✅ Archivos generados: model.pkl | encoders.pkl | model_meta.json | club_vacacional.csv")
