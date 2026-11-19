# Percepción de IA y confianza electoral en la comunidad universitaria UCE

Plataforma analítica del trabajo de titulación de Silvia Janeta y Cristian Toca, de la carrera de Sistemas de Información de la Universidad Central del Ecuador. Presenta resultados sobre conocimiento, percepción y confianza electoral frente al uso de inteligencia artificial en campañas políticas digitales de las elecciones presidenciales de Ecuador de 2025.

La base del estudio comprende 575 encuestas válidas de participantes de 18 a 64 años residentes en Quito, seleccionados por conveniencia. La evidencia contextual reúne 727 noticias y 233 publicaciones de TikTok del período del 1 de febrero al 30 de abril de 2025.

## Funcionalidades

| Módulo | Contenido |
| --- | --- |
| Resumen General | Muestra, corpus, resultados de las hipótesis e indicadores principales |
| Perfil de la muestra | Edad, género, rol institucional y unidad académica |
| Conocimiento de IA | Conocimiento declarado y reconocimiento percibido de automatización y contenido sintético |
| Percepción ciudadana | Comunicación política, manipulación, riesgo y regulación |
| Confianza electoral | Distribuciones descriptivas y contrastes de H1 y H2 |
| Modelos predictivos | Comparación de métricas, sensibilidad, importancia por permutación, matriz de confusión, estabilidad y simulador |
| Evidencia digital y triangulación | Fuentes, sentimiento categórico, evolución temporal y contexto interpretativo |
| Datos y artefactos | Calidad de datos, diccionario de variables, operacionalización y descargas técnicas |

Los filtros permiten explorar segmentos de la encuesta. Los contrastes publicados de H1 y H2 utilizan la muestra completa. Las métricas de entrenamiento corresponden a las corridas registradas.

## Arquitectura

La interfaz consume los artefactos producidos por el proceso analítico. La carga de datos se concentra en `dashboard/data_loader.py`, la predicción en `dashboard/model_backend.py` y la presentación en los módulos de `dashboard/sections/`. Los estilos y componentes comunes se comparten entre módulos.

```text
tesis_ia_campanas/
├── dashboard/
│   ├── app.py                     # Entrada de Streamlit y navegación
│   ├── data_loader.py             # Lectura de datos y artefactos
│   ├── model_backend.py           # Inferencia del modelo del simulador
│   ├── filters.py                 # Segmentación descriptiva
│   ├── common.py                  # Gráficos y elementos compartidos
│   ├── components.py              # Componentes de presentación
│   ├── styles.py                  # Aplicación del tema
│   ├── assets/                    # CSS y sello institucional
│   └── sections/                  # Ocho módulos de consulta
├── src/
│   ├── 01_prepare_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_modeling.py
│   ├── 03a_ordinal_logit.py
│   ├── ordinal_inputs.py
│   ├── evaluation/
│   ├── 04_scraping/
│   ├── 05_text_analysis/
│   ├── 06_visualization/
│   └── 07_dashboard_artifacts/
├── data/processed/encuestas/       # Bases procesadas de consulta y modelado
├── outputs/
│   ├── latest/tables/             # Tablas consumidas por el dashboard
│   ├── latest/models_dashboard/   # Modelo y metadatos del simulador
│   ├── runs/                     # Cuatro corridas de referencia
│   ├── evaluation/common_matrix/  # Comparación complementaria
│   ├── text_analysis/             # Corpus digital y resultados de sentimiento
│   └── run_registry.json          # Identificadores de corridas
├── backend/public/surveys/tables/ # Tablas descriptivas de encuestas
├── .streamlit/config.toml
├── requirements.txt
├── requirements-training.txt
├── run_model_pipeline.py
├── run_news_pipeline.py
└── README.md
```

## Organización del análisis con CRISP-DM

| Fase | Aplicación en el proyecto |
| --- | --- |
| Comprensión del problema | Delimitación de la población UCE, período electoral, constructos e hipótesis |
| Comprensión de los datos | Inspección de encuestas y corpus contextual, diccionario y reporte de calidad |
| Preparación | Depuración y codificación en `src/01_prepare_data.py` y `src/02_feature_engineering.py` |
| Modelado | Random Forest, SVM-RBF y modelo ordinal para tres y cinco niveles |
| Evaluación | Particiones estratificadas 70/30, cinco semillas, métricas, sensibilidad y comparación complementaria |
| Despliegue | Consulta de resultados y simulación exploratoria mediante Streamlit |

## Requisitos

- Python 3.11 de 64 bits.
- Acceso a PyPI para instalar las dependencias.
- Git para versionar y publicar el proyecto.
- Cuenta de GitHub y cuenta de Streamlit Community Cloud para el alojamiento propuesto.

`requirements.txt` fija las dependencias de ejecución, incluida la versión 1.8.0 de scikit-learn con la que se guardaron los modelos. `requirements-training.txt` añade Matplotlib y Statsmodels para los scripts de modelado y evaluación.

## Instalación y ejecución

Abre una terminal en la carpeta donde se encuentran este README y `requirements.txt`.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-local\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

### macOS y Linux

```bash
python3.11 -m venv .venv-local
./.venv-local/bin/python -m pip install -r requirements.txt
./.venv-local/bin/python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Abre [la aplicación local](http://127.0.0.1:8501). Pulsa Ctrl+C en la terminal para detenerla. En los siguientes arranques basta con el último comando del bloque correspondiente. Si el puerto está ocupado, añade `--server.port 8502`.

## Resultados y trazabilidad

`outputs/run_registry.json` identifica las cuatro corridas completas y de sensibilidad del 21 de julio de 2026, para tres y cinco niveles. `outputs/latest/tables/model_dashboard_manifest.json` registra su relación con las tablas finales. El simulador utiliza Random Forest de cinco niveles reajustado con la muestra disponible.

Las métricas publicadas provienen de cinco particiones estratificadas con semillas 0, 7, 13, 21 y 42. Random Forest y SVM-RBF usaron 26 predictores originales en la configuración completa y 23 en sensibilidad. El ordinal usó 15 y 14, respectivamente. Sus 33 y 30 columnas reportadas corresponden a la codificación de categorías. En la corrida histórica del ordinal, la codificación e imputación se calcularon antes de separar entrenamiento y prueba.

La comprobación complementaria en `outputs/evaluation/common_matrix/` aplica una matriz común por partición, con preprocesamiento ajustado al entrenamiento, hiperparámetros históricos y enlaces ordinales históricos. Permite examinar la sensibilidad a la homologación de entradas. Esta comprobación conserva la búsqueda histórica de parámetros y debe interpretarse con ese alcance.

Para reproducirla desde la raíz en Windows:

```powershell
.\.venv-local\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv-local\Scripts\python.exe -m src.evaluation.compare_common_matrix
```

El resultado incluye `comparison.csv`, `metrics_by_seed.csv` y `predictors.json`.

El flujo completo `python run_model_pipeline.py` requiere las fuentes originales de encuesta en las rutas que esperan los scripts de preparación. El ZIP de distribución incluye los datos procesados y los modelos necesarios para ejecutar la plataforma y la comparación complementaria. Una nueva corrida de entrenamiento actualiza el registro y los artefactos de presentación, por lo que debe ejecutarse en una copia de trabajo.

## Publicación inicial en GitHub

El procedimiento siguiente utiliza una carpeta nueva extraída del ZIP y un repositorio nuevo. El archivo ZIP distribuye los archivos del proyecto y permite iniciar su propio historial de Git.

1. En GitHub, crea un repositorio llamado `plataforma-tesis-ia` en tu cuenta. Selecciona **Private** y deja sin seleccionar las opciones para añadir README, licencia y gitignore. Esos archivos se gestionan desde la carpeta local.
2. Extrae el ZIP en Descargas. En PowerShell, ejecuta lo siguiente. Si ya existe la carpeta de destino, elige otro nombre.

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\tesis_ia_campanas.zip" -DestinationPath "$env:USERPROFILE\Downloads\publicacion_tesis"
Set-Location -LiteralPath "$env:USERPROFILE\Downloads\publicacion_tesis\tesis_ia_campanas"
git --version
git init -b main
```

3. Configura la identidad para los commits de este repositorio. Sustituye el nombre y el correo por tu identidad de GitHub. Puedes usar el correo privado proporcionado por GitHub en Settings → Emails.

```powershell
git config user.name "TU_NOMBRE"
git config user.email "TU_CORREO_DE_GITHUB"
```

4. Prepara los archivos y revisa lo que se va a publicar.

```powershell
git add .
git status --short
git diff --cached --stat
```

Confirma que aparecen `dashboard/`, los CSV procesados, las tablas y los modelos finales. Los entornos virtuales, cachés, archivos de secretos y fuentes crudas se excluyen mediante `.gitignore`.

5. Crea el commit y conecta el repositorio. Sustituye `TU_USUARIO` por el propietario real del repositorio.

```powershell
git commit -m "Publica plataforma analitica de tesis"
git remote add origin https://github.com/TU_USUARIO/plataforma-tesis-ia.git
git remote -v
git push -u origin main
```

Completa la autenticación de GitHub cuando Git la solicite. En GitHub, abre el repositorio y comprueba que el README se renderiza y que `dashboard/app.py` aparece en esa ruta. Si el remoto contiene commits previos, utiliza un repositorio nuevo vacío o integra esos cambios antes de subir.

[Guía oficial para publicar código local](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

## Actualizaciones y ramas

Usa `main` para una versión de consulta validada. Para cambios de interfaz o análisis, crea una rama de trabajo, por ejemplo:

```powershell
git switch -c mejora/presentacion
git add dashboard README.md
git diff --cached --stat
git commit -m "Mejora presentacion de resultados"
git push -u origin mejora/presentacion
```

Abre un Pull Request en GitHub hacia `main`. Antes de integrarlo, revisa las ocho páginas, los filtros, las métricas y el simulador. Documenta en el Pull Request cualquier cambio en fuentes, transformaciones, población o corridas. Los datos y resultados requieren una revisión distinta de un ajuste visual.

## Alojamiento en Streamlit Community Cloud

1. Entra en [Streamlit Community Cloud](https://share.streamlit.io/) y conecta tu cuenta de GitHub.
2. Pulsa **Create app** y selecciona el repositorio que acabas de publicar.
3. Configura estos valores.

| Campo | Valor |
| --- | --- |
| Repository | `TU_USUARIO/plataforma-tesis-ia` |
| Branch | `main` |
| Main file path | `dashboard/app.py` |
| Python en Advanced settings | `3.11` |

4. Pulsa **Deploy** y revisa el registro de instalación. El servicio instala `requirements.txt` desde la raíz.
5. En la aplicación publicada comprueba los 575 casos de encuesta y los 960 registros digitales. Revisa las ocho páginas, las opciones de 5, 8 y 11 variables, los filtros y la predicción del simulador.
6. Configura y verifica quién puede acceder a la aplicación antes de compartir su enlace. La privacidad del repositorio y el acceso a la aplicación deben revisarse por separado.

El CSV procesado contiene respuestas individuales sobre el voto y marcas de tiempo. El uso de un repositorio privado limita quién accede al código y los datos. La exposición pública de la aplicación requiere autorización para los datos y el nivel de detalle mostrado. Las credenciales, si se incorporan posteriormente, se administran mediante Secrets de Streamlit.

[Documentación oficial de despliegue](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Verificación y resolución de problemas

| Situación | Comprobación |
| --- | --- |
| Falla la instalación | Comprueba Python 3.11 y ejecuta pip con el intérprete de `.venv-local` |
| Falta un módulo Python | Reinstala `requirements.txt` en el mismo entorno que ejecuta Streamlit |
| Falla la lectura de modelos | Comprueba scikit-learn 1.8.0 y la presencia de `outputs/latest/models_dashboard/` |
| Faltan tablas o casos digitales en la nube | Comprueba los archivos del repositorio y las reglas de `.gitignore` |
| Los conteos difieren de la tesis | Revisa el corpus, su período y los archivos cargados antes de interpretar resultados |
| Cambios de CSS no aparecen | Recarga la página y ejecuta Rerun en Streamlit |
| Git informa que falta identidad | Configura `user.name` y `user.email` antes del commit |
| Git informa que origin ya existe | Revisa `git remote -v` y confirma el repositorio de destino |
| Git rechaza el push por historial previo | Integra el historial remoto o usa un repositorio nuevo vacío |

## Alcance y limitaciones

La muestra por conveniencia describe a los participantes encuestados. Los resultados muestran asociaciones exploratorias y no permiten inferir causalidad ni representatividad de la población de Quito. H1 obtuvo respaldo parcial con asociación negativa débil. H2 careció de confirmación estadística para la interacción reportada.

La evidencia digital aporta contexto y su distribución depende de las fuentes consultadas. La clasificación de sentimiento procede de los artefactos analíticos disponibles. El simulador presenta escenarios exploratorios del modelo y requiere interpretación junto con su procedencia y sus métricas de evaluación.

## Autores

Silvia Janeta y Cristian Toca. Carrera de Sistemas de Información, Universidad Central del Ecuador.

Tutor académico, Ing. Paulo Llaguno.
