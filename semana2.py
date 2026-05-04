import pathlib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from proyecto import ORDEN_NIVELES


RAIZ_PROYECTO = pathlib.Path(__file__).parent
RUTA_DATOS = RAIZ_PROYECTO / "data" / "datos_prueba.csv"


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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)

        resultados.append(
            {
                "modelo": nombre,
                "test_accuracy": float(accuracy_score(y_test, predicciones)),
                "test_f1": float(f1_score(y_test, predicciones, average="macro")),
            }
        )
    return resultados


def imprimir_resultados(
    datos: pd.DataFrame,
    resultados: list[dict[str, float]],
) -> None:
    mejor = max(resultados, key=lambda item: item["test_f1"])

    print("\n# Resultados")
    print(f"- Total filas dataset: {len(datos)}")

    print("\n## Comparacion de modelos")
    print("| Modelo | Test Accuracy | Test F1 macro |")
    print("| ------ | ------------- | ------------- |")

    for resultado in resultados:
        print(
            "| {modelo} | {test_accuracy:.3f} | {test_f1:.3f} |".format(
                **resultado
            )
        )

    print("\n## Modelo recomendado")
    print(f"- Mejor modelo por Test F1 macro: {mejor['modelo']}")
    print(f"- Test F1 macro: {mejor['test_f1']:.3f}")
    print(f"- Test Accuracy: {mejor['test_accuracy']:.3f}")


def main() -> None:
    # Carga el CSV generado desde proyecto.py.
    if not RUTA_DATOS.exists():
        raise SystemExit(
            "No se encontro el CSV. Ejecuta proyecto.py para generarlo automaticamente."
        )

    datos = pd.read_csv(RUTA_DATOS, encoding="utf-8-sig")
    X, y = preparar_datos(datos)
    resultados = evaluar_modelos(X, y)

    imprimir_resultados(
        datos,
        resultados,
    )


if __name__ == "__main__":
    main()
