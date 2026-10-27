# 🎼 Generación de Corales de Bach con Modelos de Secuencia

Laboratorio de Procesamiento de Lenguaje Natural enfocado en modelos secuenciales para generación musical.

El objetivo del proyecto es aprender patrones presentes en corales de Johann Sebastian Bach y generar continuaciones musicales de cuatro voces mediante arquitecturas recurrentes.

## Flujo general

**Coral → 32 pasos de contexto → Modelo secuencial → 16 pasos futuros → Generación libre → MIDI**

Cada paso musical contiene cuatro voces:

- Soprano
- Alto
- Tenor
- Bajo

Cada nota se representa mediante un valor MIDI entre `0` y `127`. El valor `0` representa silencio.

---

## Dataset

Se utilizó el dataset `jsb_chorales`.

Particiones utilizadas:

- Train: 229 corales
- Validation: 76 corales
- Test: 77 corales

Cada archivo contiene cuatro columnas:

```text
note0
note1
note2
note3
```

Para entrenamiento se utilizaron ventanas de 48 pasos:

- primeros 32 pasos → entrada `X`
- siguientes 16 pasos → objetivo `Y`

Formas:

```text
X: (batch, 32, 4)
Y: (batch, 16, 4)
```

Cantidad de ventanas obtenidas:

```text
Train: 5681
Validation: 982
Test: 1011
```

---

## Arquitecturas evaluadas

Se entrenaron cuatro experimentos.

### E1 — Encoder-Decoder LSTM

Configuración:

- LSTM
- 1 capa recurrente
- hidden size: 128
- embedding por nota: 16
- Adam
- learning rate: 0.001
- batch size: 64
- máximo 20 épocas
- seed: 42

Resultados:

```text
Val Loss: 0.760435
Val Accuracy por nota: 0.788235
Val Accuracy por paso: 0.597632
Parámetros entrenables: 281088
Tiempo: 156.98 s
```

---

### E2 — Encoder-Decoder GRU

Se mantuvo el mismo protocolo de E1 cambiando únicamente la celda recurrente de LSTM a GRU.

Resultados:

```text
Val Loss: 0.660780
Val Accuracy por nota: 0.812977
Val Accuracy por paso: 0.634865
Parámetros entrenables: 231424
Tiempo: 112.71 s
```

La GRU obtuvo mejores resultados que la LSTM y utilizó menos parámetros.

---

### E3 — Encoder-Decoder GRU con atención aditiva

Se agregó un mecanismo de atención aditiva sobre los 32 estados producidos por el encoder.

Resultados:

```text
Val Loss: 0.708219
Val Accuracy por nota: 0.793772
Val Accuracy por paso: 0.609789
Parámetros entrenables: 313472
Tiempo: 197.64 s
```

Aunque el mecanismo de atención permitió al decoder consultar diferentes posiciones del contexto, en este experimento no superó el desempeño de E2.

---

### E4 — GRU con hidden size 256

Como E2 obtuvo la menor pérdida de validación entre E1, E2 y E3, se tomó esa arquitectura como base y se modificó un único hiperparámetro:

```text
hidden size: 128 → 256
```

El resto de la configuración permaneció igual.

Resultados:

```text
Mejor época: 13
Val Loss: 0.611728
Val Accuracy por nota: 0.818658
Val Accuracy por paso: 0.641675
Parámetros entrenables: 642560
Tiempo: 141.59 s
```

E4 obtuvo la menor pérdida de validación y fue seleccionado como modelo final.

---

## Comparación de experimentos

| Experimento | Arquitectura | Val Loss | Acc. nota | Acc. paso | Parámetros |
|---|---|---:|---:|---:|---:|
| E1 | LSTM H128 | 0.760435 | 0.788235 | 0.597632 | 281088 |
| E2 | GRU H128 | 0.660780 | 0.812977 | 0.634865 | 231424 |
| E3 | GRU H128 + atención | 0.708219 | 0.793772 | 0.609789 | 313472 |
| E4 | GRU H256 | **0.611728** | **0.818658** | **0.641675** | 642560 |

El modelo seleccionado fue **E4 — Encoder-Decoder GRU con hidden size 256**.

---

## Evaluación final en Test

Solo el modelo seleccionado fue evaluado sobre el conjunto de test.

Resultados:

```text
Test Loss: 0.631339
Test Accuracy por nota: 0.813798
Test Accuracy por paso: 0.626175
```

---

## Mecanismo de atención

Para E3 se generó un mapa de atención de tamaño:

```text
16 × 32
```

Las filas corresponden a los 16 pasos generados y las columnas a los 32 pasos del contexto.

Las posiciones con mayor atención promedio fueron:

```text
Paso 15: 0.0725
Paso 30: 0.0718
Paso 32: 0.0701
Paso 31: 0.0694
Paso 16: 0.0670
```

El modelo no distribuye uniformemente la atención y asigna mayor peso a determinadas posiciones del contexto. Estos pesos se interpretan únicamente como información sobre dónde concentró su atención el modelo y no como una explicación causal de sus predicciones.

---

## Generación musical

El modelo ganador fue utilizado en modo autoregresivo.

Se utilizaron los primeros 32 pasos de dos corales distintos de validación y se generaron 32 pasos nuevos para cada uno.

La generación se realizó con:

```text
Temperature = 1.0
```

Cuando se requieren más de 16 pasos, la generación se realiza por bloques y se utilizan nuevamente los últimos 32 pasos disponibles como contexto.

Se generaron dos archivos MIDI:

```text
midis/bach_generado_1.mid
midis/bach_generado_2.mid
```

Tempo utilizado:

```text
120 BPM
```

Las notas consecutivas iguales dentro de una misma voz se fusionan al crear el MIDI.

---

## MLflow

Los experimentos E1, E2, E3 y E4 fueron registrados en MLflow.

Se registraron:

- parámetros de entrenamiento
- train loss
- validation loss
- accuracy por nota
- accuracy por paso completo
- número de parámetros
- tiempo de entrenamiento
- curvas de pérdida
- checkpoints

El modelo ganador E4 fue registrado en MLflow Model Registry con:

```text
Modelo: bach-chorales-mariana
Versión: 1
Alias: champion
```

La aplicación carga el modelo mediante el alias `champion`.

---

## FastAPI

El modelo ganador se expone mediante una API desarrollada con FastAPI.

Endpoints:

### GET `/health`

Verifica que la API y el modelo estén disponibles.

Ejemplo:

```json
{
  "status": "ok",
  "model": "bach-chorales-mariana",
  "version": "1"
}
```

### POST `/generate`

Recibe:

- `seed`: exactamente 32 pasos × 4 voces
- `steps`: entre 1 y 64
- `temperature`: entre 0.5 y 1.5

Ejemplo conceptual:

```json
{
  "seed": [
    [72, 67, 60, 48]
  ],
  "steps": 32,
  "temperature": 1.0
}
```

El `seed` real debe contener exactamente 32 filas.

La respuesta devuelve únicamente los pasos nuevos generados junto con el nombre y versión del modelo.

---

## Validación de la API

Se probaron los siguientes casos:

- `GET /health` → código 200
- generación válida de 32 pasos → código 200
- generación de 33 pasos → código 200
- seed inválido con menos de 32 pasos → código 422

Esto verifica tanto la generación por bloques como las validaciones de entrada.

---

## Demo pública

Mientras la instancia de AWS SageMaker, FastAPI y el túnel ngrok estén activos:

```text
https://sandbank-motivator-tile.ngrok-free.dev/docs
```

El dominio también es utilizado para otro laboratorio, por lo que solo uno de los proyectos se expone públicamente a la vez.

---

## Estructura del repositorio

```text
.
├── 01_preparacion_datos.ipynb
├── app.py
├── README.md
├── requirements.txt
├── checkpoints/
│   ├── E1_best_lstm.pt
│   ├── E2_best_gru.pt
│   ├── E3_best_gru_attention.pt
│   └── E4_best_gru_h256.pt
├── evidencias/
│   ├── E1_curva_loss.png
│   ├── E2_curva_loss.png
│   ├── E3_curva_loss.png
│   ├── E4_curva_loss.png
│   └── E3_mapa_atencion.png
└── midis/
    ├── bach_generado_1.mid
    └── bach_generado_2.mid
```

---

## Instalación

Instalar dependencias:

```bash
pip install -r requirements.txt
```

El dataset `jsb_chorales` debe descargarse y extraerse localmente en las carpetas:

```text
train/
valid/
test/
```

---

## Ejecutar la API

Desde la carpeta del proyecto:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Documentación Swagger local:

```text
http://localhost:8000/docs
```

---

## Tecnologías utilizadas

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- MLflow
- FastAPI
- Uvicorn
- PrettyMIDI
- AWS SageMaker
- ngrok

---

## Resultado final

El proyecto implementa y compara diferentes arquitecturas recurrentes para continuación automática de corales de Bach.

El mejor resultado fue obtenido por una **GRU Encoder-Decoder con hidden size 256**, que posteriormente fue registrada como modelo `champion`, evaluada sobre test, utilizada para generar archivos MIDI y expuesta mediante una API REST.