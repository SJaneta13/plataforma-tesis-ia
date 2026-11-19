# dashboard/app.py
import base64
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(
    page_title="UCE · IA y Política",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.styles import inject_global_css
from dashboard.filters import render_global_filters
from dashboard.data_loader import (
    load_survey,
    load_news_cases,
    load_model_metrics_summary,
)


from dashboard.sections.resumen_general import render_resumen_general
from dashboard.sections.perfil_muestra import render_perfil_muestra
from dashboard.sections.conocimiento_ia import render_conocimiento_ia
from dashboard.sections.percepcion_ciudadana import render_percepcion_ciudadana
from dashboard.sections.confianza_electoral import render_confianza_electoral
from dashboard.sections.modelos_predictivos import render_modelos_predictivos
from dashboard.sections.evidencia_gdelt import render_evidencia_digital
from dashboard.sections.datos_artefactos import render_datos_artefactos


def render_sidebar_header():
    seal = Path(__file__).with_name("assets") / "sello-uce.png"
    encoded = base64.b64encode(seal.read_bytes()).decode("ascii")
    st.sidebar.markdown(
        f"""
        <img class="institution-seal" src="data:image/png;base64,{encoded}"
             alt="Sello de la Universidad Central del Ecuador">
        <div class="sidebar-title">IA y confianza electoral</div>
        <div class="sidebar-subtitle">
            Universidad Central del Ecuador · Quito, elecciones 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation():
    return st.sidebar.radio(
        "Navegación",
        [
            "Resumen General",
            "Perfil de la muestra",
            "Conocimiento de IA",
            "Percepción ciudadana",
            "Confianza electoral",
            "Modelos predictivos",
            "Evidencia digital y triangulación",
            "Datos y artefactos",
        ],
        label_visibility="collapsed",
    )


def render_sidebar_footer():
    st.sidebar.divider()
    st.sidebar.caption("Investigadores:")
    st.sidebar.caption("Silvia Janeta · Cristian Toca")
    st.sidebar.caption("Tutor: Ing. Paulo Llaguno")


def main():
    inject_global_css()

    survey_df = load_survey()
    news_df = load_news_cases()
    metrics_df = load_model_metrics_summary()

    render_sidebar_header()
    section = render_sidebar_navigation()

    if survey_df.empty:
        st.error("La base procesada de encuestas está ausente, vacía o no se pudo leer.")
        st.stop()

    survey_df_filtered, _ = render_global_filters(survey_df)

    if len(survey_df_filtered) != len(survey_df) and section in ("Resumen General", "Confianza electoral", "Modelos predictivos"):
        st.caption(
            "Los filtros describen el segmento seleccionado. Los contrastes publicados "
            "y las métricas de entrenamiento conservan su muestra original."
        )

    if section in ("Resumen General", "Evidencia digital y triangulación"):
        counts = news_df["corpus_type"].value_counts().to_dict() if "corpus_type" in news_df else {}
        if counts != {"news_gdelt": 727, "social_media": 233}:
            st.warning(
                "El corpus cargado difiere del reportado en la tesis "
                "(727 noticias y 233 publicaciones de TikTok). "
                "Los conteos visibles corresponden a los registros disponibles."
            )

    render_sidebar_footer()

    if section == "Resumen General":
        render_resumen_general(
            survey_df=survey_df_filtered,
            news_df=news_df,
            metrics_df=metrics_df,
        )

    elif section == "Perfil de la muestra":
        render_perfil_muestra(survey_df_filtered)

    elif section == "Conocimiento de IA":
        render_conocimiento_ia(survey_df_filtered)

    elif section == "Percepción ciudadana":
        render_percepcion_ciudadana(survey_df_filtered)

    elif section == "Confianza electoral":
        render_confianza_electoral(survey_df_filtered, survey_df)

    elif section == "Modelos predictivos":
        render_modelos_predictivos(survey_df)

    elif section == "Evidencia digital y triangulación":
        render_evidencia_digital(survey_df_filtered, news_df)

    elif section == "Datos y artefactos":
        render_datos_artefactos(
            survey_df=survey_df,
            survey_df_filtered=survey_df_filtered,
        )

if __name__ == "__main__":
    main()
