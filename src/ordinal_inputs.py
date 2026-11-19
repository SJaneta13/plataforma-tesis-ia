"""Selección histórica de predictores del modelo ordinal."""

CONCEPTUAL_OVERLAP_COLS = [
    (
        "Percepción sobre el uso de IA en campañas políticas "
        "[La presencia de bots o cuentas falsas en redes sociales "
        "reduce mi confianza en la información política digital.]"
    ),
    (
        "Percepción sobre el uso de IA en campañas políticas "
        "[La presencia de bots o cuentas falsas en redes sociales "
        "reduce mi confianza en la información política digital.]_num"
    ),
    (
        "Percepción sobre el uso de IA en campañas políticas "
        "[La inteligencia artificial puede mejorar la transparencia "
        "del proceso electoral.]"
    ),
    (
        "Percepción sobre el uso de IA en campañas políticas "
        "[La inteligencia artificial puede mejorar la transparencia "
        "del proceso electoral.]_num"
    ),
    (
        "¿Ha cambiado su confianza en el proceso electoral al saber que "
        "existen bots, deepfakes o manipulación mediante IA?"
    ),
]

# =========================
# Feature building
# =========================
def select_ordinal_inputs(df, *, sensitivity):
    DROP_COLS = [
        "confianza_idx",
        "confianza_idx_round",
        "limpieza_num",
        "fraude_num",
        "fraude_rev",
        "confianza_limpieza",
        "confianza_fraude",
        "Marca temporal",
        "confianza_3",
    ]

    if sensitivity:
        overlap_present = [
            col for col in CONCEPTUAL_OVERLAP_COLS if col in df.columns
        ]

        DROP_COLS.extend(overlap_present)

        print(
            "\nRegresión ordinal: análisis de sensibilidad activado."
        )

        for col in overlap_present:
            print(f" - Excluida: {col}")

    TEXT_COL = (
        "¿Qué recomendaciones haría para garantizar un uso responsable y "
        "transparente de la inteligencia artificial en campañas políticas "
        "digitales en Ecuador, considerando la experiencia de la campaña "
        "entre Luisa González y Daniel Noboa?"
    )
    if TEXT_COL in df.columns:
        DROP_COLS.append(TEXT_COL)

    X = df.drop(
        columns=[c for c in DROP_COLS if c in df.columns], errors="ignore"
    ).copy()

    # Variables independientes según tesis:
    # Conocimiento sobre IA, frecuencia de exposición, edad, género, facultad, rol
    base_keep = [
        "edad",
        "genero",
        "rol_uce",
        "conoce_ia",
        "percibe_automatizacion",
        "identifica_ia",
        "¿Ha cambiado su confianza en el proceso electoral al saber que "
        "existen bots, deepfakes o manipulación mediante IA?",
        "¿Considera que el uso de IA en campañas digitales influyó en su "
        "decisión de voto en las elecciones presidenciales de 2025?",
        "¿Realiza alguna verificación (chequea fuentes, busca noticias, "
        "consulta otras personas) de la información política digital antes "
        "de compartirla?",
        "¿Conoce alguna medida, ley o regulación sobre el uso de IA en "
        "campañas políticas en Ecuador?",
        "¿Cree que debería regularse el uso de IA en campañas políticas "
        "digitales en el país?",
    ]

    if sensitivity:
        base_keep = [
            col for col in base_keep
            if col not in CONCEPTUAL_OVERLAP_COLS
        ]

    likert_num_cols = [
        c
        for c in X.columns
        if isinstance(c, str)
        and c.endswith("_num")
        and c not in CONCEPTUAL_OVERLAP_COLS
    ]

    keep_cols = [c for c in base_keep if c in X.columns] + likert_num_cols

    # Comprobar que no se conserven simultáneamente
    duplicate_pairs = [
        col
        for col in keep_cols
        if isinstance(col, str)
        and col.endswith("_num")
        and col[:-4] in keep_cols
    ]

    if duplicate_pairs:
        raise RuntimeError(
            "El modelo ordinal conserva representaciones duplicadas:\n- "
            + "\n- ".join(
                f"{col[:-4]} / {col}"
                for col in duplicate_pairs
            )
        )

    X = X[keep_cols].copy()
    return X
