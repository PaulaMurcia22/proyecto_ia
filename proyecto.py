import pathlib
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

PUNTOS_EXPERIENCIA_DATOS = {
    "principiante": [0.0, 0.5, 1.0, 1.5, 2.0],
    "intermedio": [3.0, 3.5, 4.0, 4.5, 5.0],
    "avanzado": [5.0, 6.0, 7.0, 8.0, 10.0],
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

def elegir_area_recomendada(interes: str, nivel: str, experiencia: float) -> str:
    rutas = [ruta for ruta in RUTAS_POR_INTERES[interes] if ruta["nivel"] == nivel]

    candidatas = [
        ruta
        for ruta in rutas
        if experiencia >= ruta["min"] and (ruta["max"] is None or experiencia <= ruta["max"])
    ]
    if candidatas:
        return max(candidatas, key=lambda ruta: ruta["min"])["area"]

    def distancia_a_rango(ruta: dict[str, object]) -> float:
        minimo = float(ruta["min"])
        maximo = ruta["max"]
        if experiencia < minimo:
            return minimo - experiencia
        if maximo is not None and experiencia > float(maximo):
            return experiencia - float(maximo)
        return 0.0

    return min(rutas, key=distancia_a_rango)["area"]


def generar_datos_prueba(total_filas: int = 400, semilla: int = 42) -> pd.DataFrame:
    filas = []

    combinaciones = [
        (interes, nivel, experiencia)
        for interes in RUTAS_POR_INTERES
        for nivel in ORDEN_NIVELES
        for experiencia in PUNTOS_EXPERIENCIA_DATOS[nivel]
    ]

    if not combinaciones:
        return pd.DataFrame(columns=["nivel", "interes", "experiencia", "area_recomendada"])

    repeticiones_base = max(1, total_filas // len(combinaciones))
    resto = max(0, total_filas - (repeticiones_base * len(combinaciones)))

    for indice, (interes, nivel, experiencia) in enumerate(combinaciones):
        repeticiones = repeticiones_base + (1 if indice < resto else 0)
        area_recomendada = elegir_area_recomendada(interes, nivel, experiencia)

        for _ in range(repeticiones):
            filas.append(
                {
                    "nivel": nivel,
                    "interes": interes,
                    "experiencia": experiencia,
                    "area_recomendada": area_recomendada,
                }
            )

    datos = pd.DataFrame(filas)
    datos = datos.sample(frac=1, random_state=semilla).reset_index(drop=True)
    return datos


def preparar_datos(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder, OrdinalEncoder, list[str]]:
    datos = datos.copy()
    datos["nivel"] = datos["nivel"].astype(str).str.strip().str.lower()
    datos["interes"] = datos["interes"].astype(str).str.strip().str.lower()
    datos["area_recomendada"] = datos["area_recomendada"].astype(str).str.strip()

    datos["experiencia"] = pd.to_numeric(datos["experiencia"], errors="coerce")
    if datos["experiencia"].isna().any():
        raise ValueError("La columna 'experiencia' debe contener solo números.")

    le_nivel = OrdinalEncoder(categories=[ORDEN_NIVELES])
    le_area = LabelEncoder()

    datos["nivel_codificado"] = le_nivel.fit_transform(datos[["nivel"]])
    datos["area_codificada"] = le_area.fit_transform(datos["area_recomendada"])

    interes_codificado = pd.get_dummies(datos["interes"], prefix="interes")
    X = pd.concat(
        [interes_codificado, datos[["nivel_codificado", "experiencia"]]],
        axis=1,
    )
    y = datos["area_codificada"]

    return X, y, le_area, le_nivel, list(X.columns)


def preparar_fila_usuario(interes: str, nivel: str, experiencia: float, columnas_caracteristicas: list[str]) -> pd.DataFrame:
    fila = pd.DataFrame([{columna: 0.0 for columna in columnas_caracteristicas}])

    # Nombre de la columna one-hot para el interés (ej: 'interes_frontend')
    columna_interes = f"interes_{interes}"
    if columna_interes not in fila.columns:
        raise ValueError(f"El interés '{interes}' no está disponible en el modelo.")

    # Se cambia la columna del interés seleccionado
    fila.at[0, columna_interes] = 1.0

    fila.at[0, "nivel_codificado"] = float(ORDEN_NIVELES.index(nivel))
    fila.at[0, "experiencia"] = float(experiencia)

    return fila[columnas_caracteristicas]


def calcular_ajuste_perfil(ruta: dict[str, object], nivel_usuario: str, experiencia_usuario: float) -> float:
    cumple_nivel = ruta["nivel"] == nivel_usuario
    cumple_experiencia = experiencia_usuario >= float(ruta["min"]) and (
        ruta["max"] is None or experiencia_usuario <= float(ruta["max"])
    )

    coincidencias = int(cumple_nivel) + int(cumple_experiencia)
    return (coincidencias / 2) * 100.0


def obtener_top_recomendaciones(
    modelo: DecisionTreeClassifier,
    fila_usuario: pd.DataFrame,
    le_area: LabelEncoder,
    interes_usuario: str,
    nivel_usuario: str,
    experiencia_usuario: float,
    cantidad: int = 3,
) -> list[dict[str, float]]:
    probabilidades = modelo.predict_proba(fila_usuario)[0]

    rutas_interes = RUTAS_POR_INTERES.get(interes_usuario, [])
    areas_permitidas = {ruta["area"] for ruta in rutas_interes}
    indice_nivel_usuario = ORDEN_NIVELES.index(nivel_usuario)

    def encontrar_ruta(area_name: str):
        for ruta in rutas_interes:
            if ruta["area"] == area_name:
                return ruta
        return None

    def puntuar_ruta(area_name: str) -> tuple[int, float, str]:
        ruta = encontrar_ruta(area_name)
        if not ruta:
            return (0, 0.0, area_name)

        cumple_nivel = ORDEN_NIVELES.index(ruta["nivel"]) == indice_nivel_usuario
        cumple_experiencia = experiencia_usuario >= ruta["min"] and (
            ruta["max"] is None or experiencia_usuario <= ruta["max"]
        )

        if cumple_nivel and cumple_experiencia:
            cumplimiento = 2
        elif cumple_nivel or cumple_experiencia:
            cumplimiento = 1
        else:
            cumplimiento = 0

        confianza = float(probabilidades[modelo.classes_ == le_area.transform([area_name])[0]][0]) if area_name in le_area.classes_ else 0.0
        return (cumplimiento, confianza, area_name)

    candidatos = [area for area in le_area.classes_ if area in areas_permitidas]
    candidatos_ordenados = sorted(candidatos, key=puntuar_ruta, reverse=True)[:cantidad]

    recomendaciones = []
    for area in candidatos_ordenados:
        area_codificada = le_area.transform([area])[0]
        indice = list(modelo.classes_).index(area_codificada)
        ruta = encontrar_ruta(area)
        recomendaciones.append(
            {
                "area": area,
                "confianza": float(probabilidades[indice]),
                "ajuste": calcular_ajuste_perfil(ruta, nivel_usuario, experiencia_usuario) if ruta else 0.0,
            }
        )

    return recomendaciones


def explicar_decision(
    modelo: DecisionTreeClassifier,
    fila_usuario: pd.DataFrame,
    interes_usuario: str,
    nivel_usuario: str,
    experiencia_usuario: float,
    top_recomendaciones: list[dict[str, float]],
) -> list[str]:

    mensajes: list[str] = []

    rutas = RUTAS_POR_INTERES.get(interes_usuario, [])
    total_rutas = len(rutas)

    indice_nivel_usuario = ORDEN_NIVELES.index(nivel_usuario)

    descartadas_por_nivel = 0
    descartadas_por_experiencia = 0
    cumplian = 0

    # Para cada ruta en el interés, evaluamos si cumple nivel y experiencia
    for ruta in rutas:
        nivel_ruta = ruta["nivel"]
        minimo = ruta["min"]
        maximo = ruta["max"]

        indice_nivel_ruta = ORDEN_NIVELES.index(nivel_ruta)

        cumple_nivel = indice_nivel_usuario == indice_nivel_ruta
        cumple_experiencia = True
        if experiencia_usuario < minimo:
            cumple_experiencia = False
        if maximo is not None and experiencia_usuario > maximo:
            cumple_experiencia = False

        if cumple_nivel and cumple_experiencia:
            cumplian += 1
        else:
            if not cumple_nivel:
                descartadas_por_nivel += 1
            if not cumple_experiencia:
                descartadas_por_experiencia += 1

    mensajes.append(f"Según tu interés ingresado: {interes_usuario}, se evaluaron {total_rutas} rutas.")
    mensajes.append(
        f"{cumplian} ruta(s) cumplían tanto nivel como experiencia; "
        f"se descartaron {descartadas_por_nivel} por nivel y {descartadas_por_experiencia} por experiencia."
    )

    # Comparación entre la primera y la segunda recomendación 
    if top_recomendaciones:
        mejor = top_recomendaciones[0]
        mensajes.append(
            f"Resultado final: {mejor['area']} (ajuste al perfil: {mejor['ajuste']:.1f}%)."
        )

        if len(top_recomendaciones) > 1:
            segunda = top_recomendaciones[1]

            def encontrar_ruta(area_name: str):
                for rutas in RUTAS_POR_INTERES.values():
                    for r in rutas:
                        if r["area"] == area_name:
                            return r
                return None

            ruta_mejor = encontrar_ruta(mejor["area"])
            ruta_segunda = encontrar_ruta(segunda["area"])

            razones = []
            if ruta_mejor and ruta_segunda:
                if ruta_mejor["nivel"] == nivel_usuario and ruta_segunda["nivel"] != nivel_usuario:
                    razones.append("cumplió el requisito de nivel mientras la otra no")

                minimo_mejor = ruta_mejor["min"]
                max_mejor = ruta_mejor["max"]
                minimo_seg = ruta_segunda["min"]
                max_seg = ruta_segunda["max"]

                cumple_mejor_exp = (experiencia_usuario >= minimo_mejor) and (
                    max_mejor is None or experiencia_usuario <= max_mejor
                )
                cumple_seg_exp = (experiencia_usuario >= minimo_seg) and (
                    max_seg is None or experiencia_usuario <= max_seg
                )

                if cumple_mejor_exp and not cumple_seg_exp:
                    razones.append("cumplió el requisito de experiencia mientras la otra no")

            if razones:
                mensajes.append(
                    f"En comparación, {segunda['area']} no fue elegida porque " + "; ".join(razones) + "."
                )
            else:
                mensajes.append(
                    f"En comparación, {mejor['area']} obtuvo mayor probabilidad para tu perfil que {segunda['area']}."
                )

    return mensajes


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

    X, y, le_area, _le_nivel, columnas_caracteristicas = preparar_datos(datos)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = DecisionTreeClassifier(max_depth=6, random_state=42)
    modelo.fit(X_train, y_train)

    # -----------------------------------------------------------------------------
    # ENTRADA
    # -----------------------------------------------------------------------------
    print("\n--- Recomendador áreas de aprendizaje ---")
    print("Este modelo de inteligencia artificial te sugiere una ruta de aprendizaje a partir de tu interés, tu nivel y tu experiencia.")
    print("Responde con el número de la opción que mejor describa tu perfil.")

    interes = pedir_opcion("Selecciona tu área de interés principal:", OPCIONES_INTERES)
    nivel = pedir_opcion("Selecciona tu nivel actual en esa área:", OPCIONES_NIVEL)
    experiencia = pedir_experiencia_para_nivel(nivel)

    fila_usuario = preparar_fila_usuario(
        interes=interes,
        nivel=nivel,
        experiencia=experiencia,
        columnas_caracteristicas=columnas_caracteristicas,
    )

    top_recomendaciones = obtener_top_recomendaciones(
        modelo=modelo,
        fila_usuario=fila_usuario,
        le_area=le_area,
        interes_usuario=interes,
        nivel_usuario=nivel,
        experiencia_usuario=experiencia,
        cantidad=3,
    )

    ruta_final = top_recomendaciones[0]["area"]
    explicacion = explicar_decision(
        modelo,
        fila_usuario,
        interes_usuario=interes,
        nivel_usuario=nivel,
        experiencia_usuario=experiencia,
        top_recomendaciones=top_recomendaciones,
    )

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
        print(
            f"{posicion}. {recomendacion['area']} "
            f"(ajuste al perfil: {recomendacion['ajuste']:.1f}%)"
        )

    # se muestra la recomendacion final.
    print("\nRuta final recomendada:")
    print(f"{ruta_final} (ajuste al perfil: {top_recomendaciones[0]['ajuste']:.1f}%)")

    print("\nPor qué el modelo dio este resultado:")
    if explicacion:
        for item in explicacion:
            print(f"- {item}")
    else:
        print("- El árbol llegó directamente a esta recomendación con tu combinación de datos.")

    if len(top_recomendaciones) > 1:
        segunda_opcion = top_recomendaciones[1]
        print(
            f"- '{ruta_final}' quedó por encima de '{segunda_opcion['area']}' "
            f"porque el árbol la colocó mejor dentro de las rutas de tu interés."
        )

    # -------------------------------------------------------------------------
    # Tabla con las rutas del interés seleccionado y si cumplen requisitos
    # -------------------------------------------------------------------------
    rutas = RUTAS_POR_INTERES.get(interes, [])
    if rutas:
        tabla = pd.DataFrame(rutas)

        def formato_rango(row):
            if pd.isna(row["max"]):
                return f"{row['min']} o más"
            return f"{row['min']} a {row['max']} años"

        tabla["rango"] = tabla.apply(formato_rango, axis=1)

        idx_nivel_usuario = ORDEN_NIVELES.index(nivel)

        def cumple_nivel(row):
            return ORDEN_NIVELES.index(row["nivel"]) == idx_nivel_usuario

        def cumple_experiencia(row):
            minimo = row["min"]
            maximo = row["max"]
            if experiencia < minimo:
                return False
            if maximo is not None and experiencia > maximo:
                return False
            return True

        tabla["cumple_nivel"] = tabla.apply(lambda r: "Sí" if cumple_nivel(r) else "No", axis=1)
        tabla["cumple_experiencia"] = tabla.apply(lambda r: "Sí" if cumple_experiencia(r) else "No", axis=1)
        tabla["cumple_ambos"] = tabla.apply(
            lambda r: "Sí" if (cumple_nivel(r) and cumple_experiencia(r)) else "No", axis=1
        )

        print("\nDetalle de rutas para tu interés:")
        print(tabla[["area", "nivel", "rango", "cumple_nivel", "cumple_experiencia", "cumple_ambos"]].to_string(index=False))

if __name__ == "__main__":
    main()
