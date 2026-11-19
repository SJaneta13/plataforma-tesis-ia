from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TOTAL_ESPERADO = 727

ARCHIVOS_PREDETERMINADOS = [
    Path("outputs/text_analysis/case1_es/news_sentiment_scored.csv"),
    Path("outputs/text_analysis/case2_en/news_sentiment_scored.csv"),
    Path("outputs/text_analysis/case3_ec_media/news_sentiment_scored.csv"),
]

COLUMNAS_ETIQUETA_CANDIDATAS = [
    "sentiment_label",
    "sentiment_raw_label",
    "sentimiento",
    "sentiment",
    "tone_label",
    "label",
]

COLUMNAS_SCORE_CANDIDATAS = [
    "sentiment_score",
    "sentiment_compound",
    "compound",
    "tone",
    "score",
]

ORDEN = ["Negativo", "Positivo", "Neutral"]


def quitar_acentos(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)

    return "".join(
        caracter
        for caracter in normalizado
        if not unicodedata.combining(caracter)
    )


def normalizar_etiqueta(valor: object) -> str | None:
    """Normaliza etiquetas existentes sin crear clasificaciones nuevas."""
    if pd.isna(valor):
        return None

    texto = quitar_acentos(str(valor).strip().lower())

    equivalencias = {
        "negative": "Negativo",
        "negativo": "Negativo",
        "negativa": "Negativo",
        "neg": "Negativo",

        "positive": "Positivo",
        "positivo": "Positivo",
        "positiva": "Positivo",
        "pos": "Positivo",

        "neutral": "Neutral",
        "neutro": "Neutral",
        "neutra": "Neutral",
        "neu": "Neutral",
    }

    return equivalencias.get(texto)


def cargar_csv(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    errores: list[str] = []

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                ruta,
                encoding=encoding,
                sep=None,
                engine="python",
            )
        except Exception as exc:
            errores.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"No se pudo leer {ruta}.\n" + "\n".join(errores)
    )


def buscar_columna(
    df: pd.DataFrame,
    candidatas: list[str],
) -> str | None:
    columnas = {
        str(col).strip().lower(): str(col)
        for col in df.columns
    }

    for candidata in candidatas:
        if candidata.lower() in columnas:
            return columnas[candidata.lower()]

    return None


def auditar_archivo(
    ruta: Path,
    nombre_caso: str,
) -> pd.DataFrame:
    df = cargar_csv(ruta)

    columna_etiqueta = buscar_columna(
        df,
        COLUMNAS_ETIQUETA_CANDIDATAS,
    )

    if columna_etiqueta is None:
        posibles = [
            str(col)
            for col in df.columns
            if any(
                palabra in str(col).lower()
                for palabra in (
                    "sentiment",
                    "sentimiento",
                    "tone",
                    "label",
                )
            )
        ]

        raise KeyError(
            f"No se encontró una columna de etiqueta en {ruta}.\n"
            f"Columnas relacionadas encontradas: {posibles}"
        )

    resultado = df.copy()

    resultado["caso_origen"] = nombre_caso
    resultado["sentimiento_normalizado"] = resultado[
        columna_etiqueta
    ].apply(normalizar_etiqueta)

    validas = resultado["sentimiento_normalizado"].notna().sum()
    vacias = resultado[columna_etiqueta].isna().sum()

    no_reconocidas = resultado.loc[
        resultado[columna_etiqueta].notna()
        & resultado["sentimiento_normalizado"].isna(),
        columna_etiqueta,
    ]

    print("\n" + "=" * 70)
    print(f"CASO: {nombre_caso}")
    print(f"Archivo: {ruta}")
    print(f"Filas: {len(resultado)}")
    print(f"Columna utilizada: {columna_etiqueta}")
    print(f"Etiquetas válidas: {validas}")
    print(f"Etiquetas vacías: {vacias}")
    print(f"Etiquetas no reconocidas: {len(no_reconocidas)}")

    print("\nDistribución normalizada:")
    print(
        resultado["sentimiento_normalizado"]
        .value_counts(dropna=False)
        .to_string()
    )

    if not no_reconocidas.empty:
        print("\nValores no reconocidos:")
        print(
            no_reconocidas
            .astype(str)
            .value_counts()
            .to_string()
        )

    return resultado


def comprobar_duplicados(df: pd.DataFrame) -> None:
    columnas_url = [
        "url",
        "URL",
        "article_url",
        "source_url",
        "link",
    ]

    columna_url = next(
        (
            columna
            for columna in columnas_url
            if columna in df.columns
        ),
        None,
    )

    if columna_url is None:
        print(
            "\nADVERTENCIA: no se encontró una columna URL "
            "para revisar duplicados."
        )
        return

    urls = (
        df[columna_url]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    duplicados = urls.dropna().duplicated(keep=False).sum()

    print(f"\nRegistros con URL duplicada: {duplicados}")

    if duplicados > 0:
        print(
            "ADVERTENCIA: revisa si los duplicados ya fueron "
            "controlados durante la preparación del corpus."
        )


def construir_resumen(df: pd.DataFrame) -> pd.DataFrame:
    conteos = (
        df["sentimiento_normalizado"]
        .value_counts()
        .reindex(ORDEN, fill_value=0)
    )

    total_validas = int(conteos.sum())

    return pd.DataFrame(
        {
            "sentimiento": conteos.index,
            "frecuencia": conteos.values.astype(int),
            "porcentaje": conteos.values / total_validas * 100,
        }
    )


def guardar_figura(
    resumen: pd.DataFrame,
    ruta_salida: Path,
) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    figura, eje = plt.subplots(figsize=(9, 5.2))

    barras = eje.bar(
        resumen["sentimiento"],
        resumen["frecuencia"],
        width=0.62,
        color="#1F6685",
    )

    eje.set_title(
        "Distribución de sentimiento en noticias digitales",
        fontsize=16,
        pad=16,
    )
    eje.set_xlabel(
        "Clasificación de sentimiento",
        fontsize=11,
    )
    eje.set_ylabel(
        "Número de noticias",
        fontsize=11,
    )

    eje.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )
    eje.set_axisbelow(True)

    eje.spines["top"].set_visible(False)
    eje.spines["right"].set_visible(False)

    maximo = int(resumen["frecuencia"].max())
    limite = maximo * 1.18 if maximo > 0 else 1

    eje.set_ylim(0, limite)

    for barra, frecuencia, porcentaje in zip(
        barras,
        resumen["frecuencia"],
        resumen["porcentaje"],
    ):
        eje.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + limite * 0.018,
            f"{int(frecuencia)}\n({porcentaje:.1f} %)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    total = int(resumen["frecuencia"].sum())

    figura.text(
        0.5,
        0.015,
        f"Noticias con etiqueta válida: n = {total}",
        ha="center",
        fontsize=9,
    )

    figura.tight_layout(rect=(0, 0.05, 1, 1))

    figura.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figura)

    print(f"\nFigura guardada en: {ruta_salida}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combina los tres casos de noticias, audita las "
            "etiquetas existentes y genera la figura de sentimiento."
        )
    )

    parser.add_argument(
        "--case1",
        default=str(ARCHIVOS_PREDETERMINADOS[0]),
    )
    parser.add_argument(
        "--case2",
        default=str(ARCHIVOS_PREDETERMINADOS[1]),
    )
    parser.add_argument(
        "--case3",
        default=str(ARCHIVOS_PREDETERMINADOS[2]),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/visualization/news/"
            "distribucion_sentimiento_noticias.png"
        ),
    )
    parser.add_argument(
        "--summary",
        default=(
            "outputs/visualization/news/"
            "distribucion_sentimiento_noticias.csv"
        ),
    )
    parser.add_argument(
        "--audit-output",
        default=(
            "outputs/visualization/news/"
            "noticias_sentimiento_consolidado.csv"
        ),
    )

    args = parser.parse_args()

    try:
        casos = [
            (
                Path(args.case1),
                "Noticias en español",
            ),
            (
                Path(args.case2),
                "Noticias en inglés",
            ),
            (
                Path(args.case3),
                "Medios ecuatorianos",
            ),
        ]

        dataframes = [
            auditar_archivo(ruta, nombre)
            for ruta, nombre in casos
        ]

        consolidado = pd.concat(
            dataframes,
            ignore_index=True,
            sort=False,
        )

        print("\n" + "=" * 70)
        print("AUDITORÍA CONSOLIDADA")
        print(f"Total de filas: {len(consolidado)}")
        print(f"Total esperado: {TOTAL_ESPERADO}")

        if len(consolidado) != TOTAL_ESPERADO:
            raise ValueError(
                f"Los tres archivos suman {len(consolidado)} "
                f"registros y se esperaban {TOTAL_ESPERADO}."
            )

        comprobar_duplicados(consolidado)

        validas = consolidado[
            "sentimiento_normalizado"
        ].notna().sum()

        sin_clasificar = len(consolidado) - validas

        print(f"Etiquetas válidas: {validas}")
        print(f"Sin etiqueta válida: {sin_clasificar}")

        # Se permite generar la figura si quedan registros no
        # clasificados, pero se informa explícitamente la cobertura.
        if validas == 0:
            raise ValueError(
                "No existe ninguna etiqueta de sentimiento válida."
            )

        resumen = construir_resumen(consolidado)

        print("\nRESUMEN FINAL")
        print(
            resumen.to_string(
                index=False,
                formatters={
                    "porcentaje": lambda valor: f"{valor:.1f} %"
                },
            )
        )

        ruta_resumen = Path(args.summary)
        ruta_resumen.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        resumen.to_csv(
            ruta_resumen,
            index=False,
            encoding="utf-8-sig",
        )

        ruta_auditoria = Path(args.audit_output)
        ruta_auditoria.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        columnas_auditoria = [
            columna
            for columna in [
                "caso_origen",
                "sentimiento_normalizado",
                "url",
                "URL",
                "title",
                "titulo",
                "language",
                "idioma",
            ]
            if columna in consolidado.columns
        ]

        consolidado[columnas_auditoria].to_csv(
            ruta_auditoria,
            index=False,
            encoding="utf-8-sig",
        )

        guardar_figura(
            resumen,
            Path(args.output),
        )

        print(f"Resumen guardado en: {ruta_resumen}")
        print(
            f"Archivo de auditoría guardado en: "
            f"{ruta_auditoria}"
        )

        if sin_clasificar > 0:
            print(
                "\nADVERTENCIA METODOLÓGICA:"
                f"\nLa figura representa {validas} de "
                f"{len(consolidado)} noticias."
                "\nDebes informar esta cobertura en la tesis."
            )
        else:
            print(
                "\nVALIDACIÓN COMPLETA:"
                "\nLas 727 noticias cuentan con una etiqueta "
                "normalizada de sentimiento."
            )

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()