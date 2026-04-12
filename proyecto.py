import pandas as pd
import itertools

#se importan las herramientas necesarias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# valores posibles para cada variable
niveles = ['principiante', 'intermedio', 'avanzado']
intereses = ['diseño', 'logica', 'datos', 'seguridad', 'ia', 'movil', 'juegos', 'devops']
tiempos = ['bajo', 'medio', 'alto']

# generar combinaciones
combinaciones = list(itertools.product(niveles, intereses, tiempos))

df = pd.DataFrame(combinaciones, columns=['nivel', 'interes', 'tiempo'])
df.head()

reglas = {
    'diseño': 'desarrollo web',
    'logica': {
        'principiante': 'backend',
        'default': 'arquitectura de software'
    },
    'datos': {
        'principiante': 'analisis de datos',
        'default': 'ciencia de datos'
    },
    'seguridad': {
        'avanzado': 'ciberseguridad',
        'default': 'seguridad basica'
    },
    'ia': {
        'principiante': 'fundamentos de ia',
        'default': 'machine learning'
    },
    'movil': {
        'principiante': 'apps moviles basicas',
        'default': 'desarrollo movil'
    },
    'juegos': {
        'principiante': 'programacion de videojuegos',
        'default': 'desarrollador de videojuegos'
    },
    'devops': {
        'principiante': 'automatizacion devops',
        'default': 'infraestructura cloud'
    }
}

def recomendar_area(nivel, interes):
    regla = reglas[interes]
    if isinstance(regla, dict):
        return regla.get(nivel, regla['default'])
    else:
        return regla
    
df['area_recomendada'] = df.apply(
    lambda row: recomendar_area(row['nivel'], row['interes']),
    axis=1
)

# validar si hay incoherencias
incoherencias = df[(df['nivel'] == 'principiante') & (df['area_recomendada'] == 'arquitectura de software')]
print("Incoherencias encontradas:")
print(incoherencias)

# se asignan los valores numericos a cada nivel
map_nivel = {
    'principiante': 0,
    'intermedio': 1,
    'avanzado': 2
}
df['nivel'] = df['nivel'].map(map_nivel)
le_interes = LabelEncoder()
le_tiempo = LabelEncoder()
le_area = LabelEncoder()

#se convierten los valores
df['interes'] = le_interes.fit_transform(df['interes'])
df['tiempo'] = le_tiempo.fit_transform(df['tiempo'])
df['area_recomendada'] = le_area.fit_transform(df['area_recomendada'])

# se separan las variables
X = df[['nivel', 'interes', 'tiempo']] # x contiene las caracteristicas
y = df['area_recomendada'] # y contiene la variable objetivo

# se dividen los datos, 30% prueba y 70% entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# se crea el modelo con maximo 4 niveles antes de decidir
modelo = DecisionTreeClassifier(max_depth=4)

# se entrena el modelo
modelo.fit(X_train, y_train)

# se crea un ejemplo con los datos de una persona
ejemplo = [[
    1,  # 0 = principiante, 1 = intermedio, 2 = avanzado
    le_interes.transform(['diseño'])[0],
    le_tiempo.transform(['alto'])[0]
]]
# se utiliza ese nuevo ejemplo para predecir
ejemplo_df = pd.DataFrame(ejemplo, columns=['nivel', 'interes', 'tiempo'])
prediccion = modelo.predict(ejemplo_df)

# se decodifica el resultado a texto
resultado = le_area.inverse_transform(prediccion)

print("Área recomendada:", resultado[0])