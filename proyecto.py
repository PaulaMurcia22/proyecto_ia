import pandas as pd

#se importan las herramientas necesarias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data = {
    'nivel': ['principiante', 'principiante', 'intermedio', 'avanzado', 'intermedio'],
    'interes': ['datos', 'diseño', 'logica', 'seguridad', 'datos'],
    'tiempo': ['alto', 'medio', 'alto', 'alto', 'bajo'],
    'area_recomendada': ['ciencia de datos', 'desarrollo web', 'backend', 'ciberseguridad', 'analisis de datos']
}

df = pd.DataFrame(data)
df

# se codifican los datos de texto a numeros
le_nivel = LabelEncoder()
le_interes = LabelEncoder()
le_tiempo = LabelEncoder()
le_area = LabelEncoder()

#se convierten los valores
df['nivel'] = le_nivel.fit_transform(df['nivel'])
df['interes'] = le_interes.fit_transform(df['interes'])
df['tiempo'] = le_tiempo.fit_transform(df['tiempo'])
df['area_recomendada'] = le_area.fit_transform(df['area_recomendada'])

# se separan las variables
X = df[['nivel', 'interes', 'tiempo']] # x contiene las caracteristicas
y = df['area_recomendada'] # y contiene la variable objetivo

# se dividen los datos, 30% prueba y 70% entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# se crea el modelo
modelo = DecisionTreeClassifier()

# se entrena el modelo
modelo.fit(X_train, y_train)

# se crea un ejemplo con los datos de una persona
ejemplo = [[
    le_nivel.transform(['principiante'])[0], # se convierten a numero
    le_interes.transform(['diseño'])[0],
    le_tiempo.transform(['alto'])[0]
]]
# se utiliza ese nuevo ejemplo para predecir
ejemplo_df = pd.DataFrame(ejemplo, columns=['nivel', 'interes', 'tiempo'])
prediccion = modelo.predict(ejemplo_df)

# se decodifica el resultado a texto
resultado = le_area.inverse_transform(prediccion)

print("Área recomendada:", resultado[0])


# se crea un ejemplo con los datos de una persona
ejemplo2 = [[
    le_nivel.transform(['intermedio'])[0], # se convierten a numero
    le_interes.transform(['logica'])[0],
    le_tiempo.transform(['alto'])[0]
]]
# se utiliza ese nuevo ejemplo para predecir
ejemplo_df2 = pd.DataFrame(ejemplo2, columns=['nivel', 'interes', 'tiempo'])
prediccion2 = modelo.predict(ejemplo_df2)

# se decodifica el resultado a texto
resultado2 = le_area.inverse_transform(prediccion2)

print("Área recomendada:", resultado2[0])