import pathlib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from proyecto import ORDEN_NIVELES, RANGOS_EXPERIENCIA_POR_NIVEL, RUTAS_POR_INTERES


RAIZ_PROYECTO = pathlib.Path(__file__).parent
RUTA_DATOS = RAIZ_PROYECTO / "data" / "datos_prueba.csv"


def validar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    # Valida coherencia de niveles, intereses, experiencia y area recomendada.
    incoherencias = []

    for indice, fila in datos.iterrows():
        nivel = str(fila["nivel"]).strip().lower()
        interes = str(fila["interes"]).strip().lower()
        area = str(fila["area_recomendada"]).strip()

        try:
            experiencia = float(fila["experiencia"])
        except (TypeError, ValueError):
            incoherencias.append({"fila": indice + 2, "motivo": "experiencia no numerica"})
            continue

        if nivel not in ORDEN_NIVELES:
            incoherencias.append(
                {"fila": indice + 2, "motivo": f"nivel no valido: {nivel}"}
            )
            continue

        if interes not in RUTAS_POR_INTERES:
            incoherencias.append(
                {"fila": indice + 2, "motivo": f"interes no valido: {interes}"}
            )
            continue

        minimo, maximo = RANGOS_EXPERIENCIA_POR_NIVEL[nivel]
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

        areas_validas = {ruta["area"] for ruta in RUTAS_POR_INTERES[interes]}
        if area not in areas_validas:
            incoherencias.append(
                {
                    "fila": indice + 2,
                    "motivo": f"area '{area}' no corresponde al interes '{interes}'",
                }
            )

    return pd.DataFrame(incoherencias)


def preparar_datos(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Normaliza el texto y lo convierte de texto a numero
    datos = datos.copy()
    datos["nivel"] = datos["nivel"].astype(str).str.strip().str.lower()
    datos["interes"] = datos["interes"].astype(str).str.strip().str.lower()
    datos["area_recomendada"] = datos["area_recomendada"].astype(str).str.strip()

    datos["experiencia"] = pd.to_numeric(datos["experiencia"], errors="coerce")
    if datos["experiencia"].isna().any():
        raise ValueError("La columna 'experiencia' debe contener solo numeros.")

    le_nivel = OrdinalEncoder(categories=[ORDEN_NIVELES])
    le_interes = LabelEncoder()
    le_area = LabelEncoder()

    datos["nivel_codificado"] = le_nivel.fit_transform(datos[["nivel"]])
    datos["interes_codificado"] = le_interes.fit_transform(datos["interes"])
    datos["area_codificada"] = le_area.fit_transform(datos["area_recomendada"])

    X = datos[["interes_codificado", "nivel_codificado", "experiencia"]]
    y = datos["area_codificada"]

    return X, y


def evaluar_modelos(X: pd.DataFrame, y: pd.Series) -> list[dict[str, float]]:
    modelos = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    resultados = []
    min_por_clase = int(y.value_counts().min())
    n_splits = max(2, min(5, min_por_clase))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    for nombre, modelo in modelos.items():
        cv_accuracy = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(modelo, X, y, cv=cv, scoring="f1_macro")

        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)

        resultados.append(
            {
                "modelo": nombre,
                "cv_accuracy": float(cv_accuracy.mean()),
                "cv_f1": float(cv_f1.mean()),
                "test_accuracy": float(accuracy_score(y_test, predicciones)),
                "test_f1": float(f1_score(y_test, predicciones, average="macro")),
            }
        )
    return resultados


def imprimir_resultados(
    datos: pd.DataFrame,
    incoherencias: pd.DataFrame,
    resultados: list[dict[str, float]],
    total_rutas: int,
    filas_por_ruta: int,
    semilla: int,
) -> None:
    mejor = max(resultados, key=lambda item: (item["cv_f1"], item["cv_accuracy"]))

    print("\n# Resultados")
    print(f"- Total filas dataset: {len(datos)}")
    print(f"- Total rutas: {total_rutas}")
    print(f"- Filas por ruta: {filas_por_ruta}")
    print(f"- Incoherencias: {len(incoherencias)}")

    print("\n## Comparacion de modelos")
    print("| Modelo | CV Accuracy | CV F1 macro | Test Accuracy | Test F1 macro |")
    print("| ------ | ----------- | ----------- | ------------- | ------------- |")

    for resultado in resultados:
        print(
            "| {modelo} | {cv_accuracy:.3f} | {cv_f1:.3f} | {test_accuracy:.3f} | {test_f1:.3f} |".format(
                **resultado
            )
        )

    print("\n## Modelo recomendado")
    print(f"- Mejor modelo por CV F1 macro: {mejor['modelo']}")
    print(f"- CV F1 macro: {mejor['cv_f1']:.3f}")
    print(f"- CV Accuracy: {mejor['cv_accuracy']:.3f}")


def main() -> None:
    # Carga el CSV generado desde proyecto.py.
    if not RUTA_DATOS.exists():
        raise SystemExit(
            "No se encontro el CSV. Ejecuta proyecto.py para generarlo automaticamente."
        )

    datos = pd.read_csv(RUTA_DATOS, encoding="utf-8-sig")
    incoherencias = validar_datos(datos)

    X, y = preparar_datos(datos)
    resultados = evaluar_modelos(X, y)

    total_rutas = sum(len(rutas) for rutas in RUTAS_POR_INTERES.values())
    filas_por_ruta = max(1, len(datos) // total_rutas)

    imprimir_resultados(
        datos,
        incoherencias,
        resultados,
        total_rutas,
        filas_por_ruta,
        42,
    )


if __name__ == "__main__":
    main()
