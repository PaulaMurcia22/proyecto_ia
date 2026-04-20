import pathlib
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


RAIZ_PROYECTO = pathlib.Path(__file__).parent
ARCHIVO_DATOS_PREDETERMINADO = RAIZ_PROYECTO / "data" / "datos_prueba.csv"
REGENERAR_CSV = False

# Opciones de intereses disponibles.
OPCIONES_INTERES = {
    "1": "diseño",
    "2": "frontend",
    "3": "backend",
    "4": "datos",
    "5": "inteligencia artificial",
    "6": "seguridad",
    "7": "devops",
    "8": "móvil",
}

# Opciones de nivel disponibles.
OPCIONES_NIVEL = {
    "1": "principiante",
    "2": "intermedio",
    "3": "avanzado",
}

# Cada nivel tiene un rango esperado de años de experiencia:
# - principiante: entre 0 y 2 años
# - intermedio: entre 3 y 5 años
# - avanzado: 5 años o mas
RANGOS_EXPERIENCIA_POR_NIVEL = {
    "principiante": (0, 2),
    "intermedio": (3, 5),
    "avanzado": (5, None),
}

ORDEN_NIVELES = ["principiante", "intermedio", "avanzado"]

# Catalogo de rutas posibles segun el interes
RUTAS_POR_INTERES = {
    "diseño": [
        {"area": "Fundamentos de diseño", "nivel": "principiante", "min": 0, "max": 1},
        {"area": "Diseño UI", "nivel": "principiante", "min": 1, "max": 2},
        {"area": "Diseño UX", "nivel": "intermedio", "min": 3, "max": 4},
        {"area": "Sistemas de diseño", "nivel": "intermedio", "min": 4, "max": 5},
        {"area": "Dirección de diseño", "nivel": "avanzado", "min": 5, "max": None},
    ],
    "frontend": [
        {"area": "Fundamentos de HTML y CSS", "nivel": "principiante", "min": 0, "max": 1},
        {"area": "JavaScript y componentes UI", "nivel": "principiante", "min": 1, "max": 2},
        {"area": "Integración con APIs", "nivel": "intermedio", "min": 3, "max": 4},
        {"area": "Arquitectura frontend", "nivel": "intermedio", "min": 4, "max": 5},
        {"area": "Microfrontends", "nivel": "avanzado", "min": 5, "max": None},
    ],
    "backend": [
        {"area": "Fundamentos de backend", "nivel": "principiante", "min": 0, "max": 1},
        {"area": "APIs REST", "nivel": "principiante", "min": 1, "max": 2},
        {"area": "Integración con bases de datos", "nivel": "intermedio", "min": 3, "max": 4},
        {"area": "Arquitectura de servicios", "nivel": "intermedio", "min": 4, "max": 5},
        {"area": "Sistemas distribuidos", "nivel": "avanzado", "min": 5, "max": None},
    ],
    "datos": [
        {"area": "Fundamentos de datos", "nivel": "principiante", "min": 0, "max": 1},
        {"area": "Análisis de datos", "nivel": "principiante", "min": 1, "max": 2},
        {"area": "Ciencia de datos", "nivel": "intermedio", "min": 3, "max": 4},
        {"area": "Ingeniería de datos", "nivel": "intermedio", "min": 4, "max": 5},
        {"area": "Liderazgo analítico", "nivel": "avanzado", "min": 5, "max": None},
    ],
    "inteligencia artificial": [
        {"area": "Fundamentos de IA", "nivel": "principiante", "min": 0, "max": 1},
        {"area": "Modelos clásicos de machine learning", "nivel": "principiante", "min": 1, "max": 2},
        {"area": "Deep Learning aplicado", "nivel": "intermedio", "min": 3, "max": 4},
        {"area": "MLOps", "nivel": "intermedio", "min": 4, "max": 5},
        {"area": "Arquitectura de soluciones de IA", "nivel": "avanzado", "min": 5, "max": None},
    ],
    "seguridad": [
        {"area": "Seguridad básica", "nivel": "principiante", "min": 0, "max": 1},
        {"area": "Hardening de aplicaciones", "nivel": "principiante", "min": 1, "max": 2},
        {"area": "Pentesting web", "nivel": "intermedio", "min": 3, "max": 4},
        {"area": "Automatización de seguridad", "nivel": "intermedio", "min": 4, "max": 5},
        {"area": "Arquitectura de seguridad", "nivel": "avanzado", "min": 5, "max": None},
    ],
    "devops": [
        {"area": "Automatización inicial", "nivel": "principiante", "min": 0, "max": 1},
        {"area": "Integración continua", "nivel": "principiante", "min": 1, "max": 2},
        {"area": "Entrega continua", "nivel": "intermedio", "min": 3, "max": 4},
        {"area": "Infraestructura como código", "nivel": "intermedio", "min": 4, "max": 5},
        {"area": "Plataforma cloud", "nivel": "avanzado", "min": 5, "max": None},
    ],
    "móvil": [
        {"area": "Fundamentos móviles", "nivel": "principiante", "min": 0, "max": 1},
        {"area": "Interfaces móviles", "nivel": "principiante", "min": 1, "max": 2},
        {"area": "Consumo de APIs móviles", "nivel": "intermedio", "min": 3, "max": 4},
        {"area": "Arquitectura móvil", "nivel": "intermedio", "min": 4, "max": 5},
        {"area": "Publicación y escalado móvil", "nivel": "avanzado", "min": 5, "max": None},
    ],
}

def generar_datos_prueba(total_filas: int = 400, semilla: int = 42) -> pd.DataFrame:
    rng = random.Random(semilla)

    rutas = [
        (interes, ruta)
        for interes, rutas_interes in RUTAS_POR_INTERES.items()
        for ruta in rutas_interes
    ]

    filas_por_ruta = max(1, total_filas // len(rutas))
    filas = []

    for interes, ruta in rutas:
        nivel = ruta["nivel"]
        minimo = ruta["min"]
        maximo = ruta["max"]

        rango_minimo, rango_maximo = RANGOS_EXPERIENCIA_POR_NIVEL[nivel]
        minimo = max(minimo, rango_minimo)

        if maximo is None:
            maximo = minimo + 3

        if rango_maximo is not None:
            maximo = min(maximo, rango_maximo)

        if maximo < minimo:
            maximo = minimo

        for _ in range(filas_por_ruta):
            experiencia = round(rng.uniform(minimo, maximo), 1)
            filas.append(
                {
                    "nivel": nivel,
                    "interes": interes,
                    "experiencia": experiencia,
                    "area_recomendada": ruta["area"],
                }
            )

    datos = pd.DataFrame(filas)
    datos = datos.sample(frac=1, random_state=semilla).reset_index(drop=True)
    return datos


# mostrar el menu y pedir una opcion valida
def pedir_opcion(titulo: str, opciones: dict[str, str]) -> str:
    print(f"\n{titulo}")

    # Se imprimen todas las opciones disponibles.
    for numero, texto in opciones.items():
        print(f"{numero}. {texto}")

    # Se repite hasta que el usuario escriba una opcion valida.
    while True:
        seleccion = input("Ingresa el número de tu opción: ").strip()

        # Si la opcion existe, devolvemos el valor asociado.
        if seleccion in opciones:
            return opciones[seleccion]

        # Si no existe, se vuelve a pedir.
        print("Opción inválida. Intenta de nuevo con uno de los números mostrados.")


# Se genera un texto del rango esperado segun el nivel ingresado
def describir_rango(nivel: str) -> str:
    # Se recupera el rango segun el nivel
    minimo, maximo = RANGOS_EXPERIENCIA_POR_NIVEL[nivel]

    # Si maximo es None, se ajusta el texto
    if maximo is None:
        return f"{minimo} o más años"

    # Si existe maximo, se devuelve el rango completo
    return f"{minimo} a {maximo} años"


# Se valida que la experiencia ingresada cumpla con el rango definido para ese nivel
def pedir_experiencia_para_nivel(nivel: str) -> float:
    # Se obtiene el rango permitido para el nivel seleccionado
    minimo, maximo = RANGOS_EXPERIENCIA_POR_NIVEL[nivel]

    # Se muestra un mensaje indicando el rango que se espera
    mensaje = f"Experiencia en esa área ({describir_rango(nivel)}): "

    while True:
        try:
            experiencia = float(input(mensaje).strip())
        except ValueError:
            # se solicita que sea un numero
            print("Debes ingresar un número válido.")
            continue

        # Si la experiencia esta por debajo o por arriba del minimo /maximo, se muestra un mensaje
        if experiencia < minimo:
            print(f"Para nivel {nivel}, la experiencia mínima es {minimo}.")
            continue

        if maximo is not None and experiencia > maximo:
            print(f"Para nivel {nivel}, la experiencia máxima es {maximo}.")
            continue

        # si cumple con todo, se devuelve el valor
        return experiencia

def main() -> None:
    # se utiliza la ruta definida
    ruta_csv = ARCHIVO_DATOS_PREDETERMINADO
    print(f"Cargando datos desde: {ruta_csv}")

    if REGENERAR_CSV or not ruta_csv.exists():
        total_rutas = sum(len(rutas) for rutas in RUTAS_POR_INTERES.values())
        total_filas = total_rutas * 10
        datos_generados = generar_datos_prueba(total_filas=total_filas, semilla=42)
        datos_generados.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
        print(f"CSV generado automaticamente con {len(datos_generados)} filas.")
    # Si el archivo no existe que se termina la ejecucion
    if not ruta_csv.exists():
        print(f"No se encontró el archivo CSV: {ruta_csv}")
        raise SystemExit(1)

    # Se lee el archivo CSV
    datos = pd.read_csv(ruta_csv, encoding="utf-8-sig")

    # Se definen las columnas necesarias para el modelo
    columnas_esperadas = {"nivel", "interes", "experiencia", "area_recomendada"}

    # Se identifica si falta alguna columna
    columnas_faltantes = columnas_esperadas.difference(datos.columns)
    if columnas_faltantes:
        print("El CSV no tiene las columnas necesarias:",", ".join(sorted(columnas_faltantes)),)
        raise SystemExit(1)

    datos = datos.copy()

    # se normalizan las columnas de texto
    datos["nivel"] = datos["nivel"].astype(str).str.strip().str.lower()
    datos["interes"] = datos["interes"].astype(str).str.strip().str.lower()
    datos["area_recomendada"] = datos["area_recomendada"].astype(str).str.strip()

    # se valida que experiencia sea numerica
    datos["experiencia"] = pd.to_numeric(datos["experiencia"], errors="coerce")
    if datos["experiencia"].isna().any():
        print("La columna 'experiencia' debe contener únicamente números.")
        raise SystemExit(1)

    #se validan posibles incoherencias en los datos antes de entrenar el modelo
    incoherencias = []
    for indice, fila in datos.iterrows():
        nivel = fila["nivel"]
        interes = fila["interes"]
        experiencia = float(fila["experiencia"])
        area = fila["area_recomendada"]

        # se valida que el nivel del CSV debe existir en los niveles permitidos.
        if nivel not in ORDEN_NIVELES:
            incoherencias.append(
                {"fila": indice + 2, "motivo": f"nivel no válido: {nivel}"}
            )
            continue

        # se valida que el interes del CSV debe existir en los intereses permitidos.
        if interes not in RUTAS_POR_INTERES:
            incoherencias.append(
                {"fila": indice + 2, "motivo": f"interés no válido: {interes}"}
            )
            continue

        # Se obtiene el rango valido segun el nivel de la fila.
        minimo, maximo = RANGOS_EXPERIENCIA_POR_NIVEL[nivel]

        # se valida que la experiencia no quede por debajo del minimo o por encima del maximo
        if experiencia < minimo:
            incoherencias.append(
                {
                    "fila": indice + 2,
                    "motivo": f"experiencia {experiencia} por debajo del rango para {nivel}",
                }
            )
        if maximo is not None and experiencia > maximo:
            incoherencias.append(
                {
                    "fila": indice + 2,
                    "motivo": f"experiencia {experiencia} por encima del rango para {nivel}",
                }
            )

        # Se calculan las areas validas para el interes de esa fila
        areas_validas = {ruta["area"] for ruta in RUTAS_POR_INTERES[interes]}

        # El area recomendada del CSV debe pertenecer al interes seleccionado
        if area not in areas_validas:
            incoherencias.append(
                {
                    "fila": indice + 2,
                    "motivo": f"área '{area}' no corresponde al interés '{interes}'",
                }
            )

    incoherencias = pd.DataFrame(incoherencias)
    # Si no hay incoherencias, se muestra un mensaje
    if incoherencias.empty:
        print("El CSV es coherente con las reglas definidas.")
    else:
        # Si se encontraron incoherencias, se muestran
        print("\nSe encontraron incoherencias en el CSV:")
        print(incoherencias.to_string(index=False))

    # -----------------------------------------------------------------------------
    # PREPARACION DEL MODELO
    # -----------------------------------------------------------------------------

    # se transformas las columnas de texto a numero
    # OrdinalEncoder se usa en nivel para mantener el orden.
    le_nivel = OrdinalEncoder(categories=[ORDEN_NIVELES])
    # LabelEncoder se usa en interes y area
    le_interes = LabelEncoder()
    le_area = LabelEncoder()

    # se convierten los valores
    datos["nivel_codificado"] = le_nivel.fit_transform(datos[["nivel"]])
    datos["interes_codificado"] = le_interes.fit_transform(datos["interes"])
    datos["area_codificada"] = le_area.fit_transform(datos["area_recomendada"])

    X = datos[["interes_codificado", "nivel_codificado", "experiencia"]]
    y = datos["area_codificada"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
    modelo.fit(X_train, y_train)

    # -----------------------------------------------------------------------------
    # ENTRADA
    # -----------------------------------------------------------------------------
    print("\n--- Recomendador áreas de aprendizaje ---")

    interes = pedir_opcion("Selecciona tu área de interés:", OPCIONES_INTERES)
    nivel = pedir_opcion("Selecciona tu nivel en esa área:", OPCIONES_NIVEL)
    experiencia = pedir_experiencia_para_nivel(nivel)

    top_recomendaciones = []

    # Posicion del nivel del usuario dentro del orden general.
    indice_nivel_usuario = ORDEN_NIVELES.index(nivel)

    # Se recorren todas las rutas posibles del interes seleccionado
    for ruta in RUTAS_POR_INTERES[interes]:
        rango = (
            f"{ruta['min']} o más años"
            if ruta["max"] is None
            else f"{ruta['min']} a {ruta['max']} años"
        )
        distancia = 0.0
        if experiencia < ruta["min"]:
            distancia = ruta["min"] - experiencia
        elif ruta["max"] is not None and experiencia > ruta["max"]:
            distancia = experiencia - ruta["max"]

        indice_nivel_ruta = ORDEN_NIVELES.index(ruta["nivel"])
        diferencia_nivel = abs(indice_nivel_usuario - indice_nivel_ruta)
        puntaje = max(0, 100 - (distancia * 20) - (diferencia_nivel * 20))

        top_recomendaciones.append(
            {
                "area": ruta["area"],
                "nivel": ruta["nivel"],
                "rango": rango,
                "puntaje": puntaje,
            }
        )

    # Se ordenan las rutas de mayor a menor puntaje y se mantienen las 3 mejores
    top_recomendaciones.sort(key=lambda item: item["puntaje"], reverse=True)
    top_recomendaciones = top_recomendaciones[:3]

    # -----------------------------------------------------------------------------
    # SALIDA
    # -----------------------------------------------------------------------------
    print("\nResumen de tu perfil:")
    print(f"Interés: {interes}")
    print(f"Nivel: {nivel}")
    print(f"Experiencia en esa área: {experiencia} años")

    # se imprime el top 3 con su porcentaje 
    print("\nTop 3 rutas sugeridas:")
    for posicion, recomendacion in enumerate(top_recomendaciones, start=1):
        ajuste = min(recomendacion["puntaje"], 100)
        print(
            f"{posicion}. {recomendacion['area']} "
            f"(nivel sugerido: {recomendacion['nivel']}, rango: {recomendacion['rango']}, "
            f"ajuste estimado: {ajuste:.1f}%)"
        )

    # se muestra la recomendacion final.
    print("\nRuta final recomendada:")
    print(top_recomendaciones[0]["area"])

if __name__ == "__main__":
    main()
