# Pepe AI 🌿

**Pepe AI** es una solución de visión artificial avanzada diseñada para la detección y conteo automático de yemas en viñedos. Utilizando el modelo **YOLO26s** potenciado por la técnica de **SAHI (Slicing Aided Hyper Inference)**, el sistema es capaz de identificar yemas de tamaño pequeño en imágenes con una alta precisión.

---

## 📋 Requisitos Previos

*   **Python**: Versión 3.10 (Obligatorio).
*   **Hardware**: Se recomienda una GPU NVIDIA con soporte para **CUDA 12.1** para un rendimiento óptimo (Solo entrenamiento).

---

## 🚀 Instalación y Configuración

Sigue estos pasos en orden para configurar tu entorno de trabajo:

### 1. Crear el Entorno Virtual
Abre una terminal en la carpeta de tu proyecto y ejecuta los siguientes comandos:
```bash
# Crear el entorno virtual con Python 3.10
python -m venv .venv

# Activar el entorno virtual (Windows)
.venv\Scripts\activate
```

### 2. Actualizar PIP e Instalar Dependencias
Es fundamental actualizar el gestor de paquetes antes de instalar las librerías:

```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar todas las dependencias (incluyendo soporte GPU para Torch)
pip install -r requirements.txt
```

### 3. Descarga de Datos y Datasets
Para que el proyecto funcione correctamente, debes descargar los archivos necesarios, **descomprimirlos** y ubicarlos en la raíz del proyecto.

1.  Descarga los archivos `.zip` de los siguientes enlaces: **(SOLO ENTRENAMIENTO)**
    *   **dataset.zip**: [[Descargar](https://drive.google.com/file/d/1FwQ556rmGgQcJcQZ5DbFGxkW7GTPbsQ-/view?usp=sharing)]
    *   **sliced_dataset_high_res.zip**: [[Descargar](https://drive.google.com/file/d/1CRA3DJ7NWFNbPOQoTwq-Y0TGr9TqD4pE/view?usp=sharing)]
    *   **visualizacion_dataset_completo.zip**: [[Descargar](https://drive.google.com/file/d/1xHniUjR7b9Ycoa4OokAOJ09Zg_6GkOjq/view?usp=sharing)]
2.  **Descomprime** cada archivo de modo que las carpetas aparezcan directamente en el directorio principal.
3.  Asegúrate de que la estructura de archivos final sea la siguiente:
```text
Pepe_Yemas/
├── .venv/
├── dataset/
├── runs/
├── scripts/
├── sliced_dataset_high_res/
├── visualizacion_dataset_completo/
├── app.py
├── data.yaml
├── readme.md
└── requirements.txt
```

---

## 🖥️ Uso de la Aplicación

**Pepe AI** ofrece una interfaz gráfica intuitiva desarrollada en **Streamlit** para facilitar la visualización y comparativa de resultados.

### Ejecución
Con el entorno virtual activado, lanza la aplicación con el siguiente comando:
```bash
streamlit run app.py
```

### Funcionalidades principales:
*   **Selector de Confianza**: Slider dinámico para ajustar el umbral de detección (Recomendado: **0.25**).
*   **Comparativa Visual**: El sistema busca la imagen en el dataset original para comparar las etiquetas manuales con lo detectado por la IA.
*   **Guardado bajo demanda**: Las imágenes solo se guardarán en la carpeta `outputs/` si presionas el botón **"Guardar Resultado"**.

---

## 🛠️ Tecnologías Utilizadas

*   **YOLOv261s (Ultralytics)**: Motor principal de detección de objetos.
*   **SAHI**: Procesamiento por secciones (slicing) de 800x800.
*   **Streamlit**: Interfaz de usuario moderna.
*   **PyTorch**: Framework configurado para aceleración por hardware (CUDA 12.1).

---


## ⚙️ Información de Hardware adicional
Para el entrenamiento del último modelo de detección, se ha empleado el siguiente hardware:

*   **NVIDIA RTX 3060 Ti**: GPU
*   **16 GB RAM DDR4**: Memoria RAM
*   **Intel Core i5-11400F**: CPU

Con estas características, el modelo YOLO26s ha tenido un tiempo de entrenamiento aproximado de **4 horas**

---

## 📊 Métricas obtenidas ##
Tras el entrenamiento del modelo YOLO26s, se ha realizado una evaluación con imágenes no empleadas durante el entrenamiento, obteniendo los siguientes resultados:
* **76%** de precisión
* **63%** de recall
* **71%** de mAP50
* **31%** de mAP50-95 

Evaluando a Pepe mediante el conjunto de pruebas, se estima que tiene una tasa de error de **+/- 12%**

---

**Pepe AI** - *By AGM Global*