# Proyecto: Recomendación de Áreas en Desarrollo de Software

## 1. Problema a resolver

Actualmente, las personas que inician o desean avanzar en el desarrollo de software se enfrentan a una gran variedad de áreas y tecnologías, como desarrollo web, arquitectura, infraestructura, ciencia de datos, inteligencia artificial y ciberseguridad.

Debido a esta variedad, surge una dificultad para decidir qué camino seguir, especialmente cuando no se tiene experiencia o claridad sobre en qué área enfocarse.

### ¿Qué se quiere predecir?

El modelo busca predecir el área o tipo de tecnología que una persona debería aprender según su perfil.

### ¿Por qué usar IA?

Un modelo de IA permite analizar múltiples características del usuario y generar recomendaciones personalizadas, facilitando la toma de decisiones de aprendizaje.

---

## 2. Definición de las features

### Nivel del usuario
Representa la experiencia (principiante, intermedio, avanzado).  
Permite definir la complejidad de las tecnologías recomendadas.

### Interés principal
Indica el enfoque del usuario (diseño, lógica, datos, seguridad, IA, móvil, juegos, DevOps, etc.).  
Permite orientar la recomendación hacia áreas específicas.

### Tiempo disponible
Indica cuánto tiempo puede dedicar al aprendizaje (bajo, medio, alto).  
Es importante porque algunas áreas requieren mayor dedicación.

---

## 3. Dataset inicial

Cada fila representa una persona con sus características.

Cada columna representa:
- un feature (nivel, interés, tiempo)
- o la variable objetivo (área recomendada)

```python
data = {
    'nivel': ['principiante', 'principiante', 'intermedio', 'avanzado', 'intermedio'],
    'interes': ['datos', 'diseño', 'logica', 'seguridad', 'datos'],
    'tiempo': ['alto', 'medio', 'alto', 'alto', 'bajo'],
    'area_recomendada': ['ciencia de datos', 'desarrollo web', 'backend', 'ciberseguridad', 'analisis de datos']
} 
```
## 4. Variable objetivo

**area_recomendada**

Representa el área sugerida para aprender. Es una variable categórica, donde cada número representa un área específica.
Ejemplo:
Desarrollo web = 0
Ciencia de datos = 1
Backend = 2
Ciberseguridad = 3
Análisis de datos = 4

---

## 5. Entrenamiento del modelo

Se utilizan las siguientes librerías:

- **pandas**: manejo de datos
- **LabelEncoder**: convertir texto a números
- **train_test_split**: dividir datos
- **DecisionTreeClassifier**: modelo de decisión

Se realizan los siguientes pasos:

- Codificación de datos  
- Separación de variables (X, y)  
- División en entrenamiento y prueba  
- Entrenamiento del modelo  

---

## 6. Predicción

Se ingresan los datos de una nueva persona:

- nivel  
- interés  
- tiempo  

El modelo devuelve un área recomendada basada en los patrones aprendidos.

---

## Plan de trabajo

**Semana 1**
- Ampliar dataset
- Agregar nuevas áreas
- Validar coherencia
- Mejorar features
- Documentar proceso

**Semana 2**
- Probar otros modelos
- Comparar resultados
- Ajustar parámetros

**Semana 3**
- Realizar pruebas
- Validar resultados
- Mejorar modelo

---

## Entrega Semana 1 

### Generación del dataset
Se generaron combinaciones automáticas de las variables **nivel**, **interés** y **tiempo** utilizando la librería `itertools`, lo que permitió crear un dataset más amplio.

### Reglas de recomendación
Se definieron reglas mediante un diccionario para asignar el área recomendada según el perfil del usuario, permitiendo asi generar datos coherentes para el entrenamiento del modelo.

### Validación de datos
Se implementó un filtro para detectar incoherencias en el dataset, asegurando que perfiles de nivel principiante no sean asociados con áreas avanzadas.

### Transformación de datos
- Se utilizó `map()` para convertir la variable **nivel** respetando su orden lógico.
- Se utilizó `LabelEncoder` para las variables categóricas sin necesidad de un orden (**interés** y **tiempo**).

### Modelo
Se utilizó un modelo de tipo **Decision Tree**, limitando su profundidad (`max_depth=4`).
