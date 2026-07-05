import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = Path("frutas_mobilenet.h5")
CLASSES_PATH = Path("class_names.json")
TOP_K = 5

st.set_page_config(
    page_title="Clasificador de Frutas 🍎",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilos CSS Avanzados (Diseño Moderno y Limpio)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    /* Ajustes globales de fuente */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Título con degradado moderno */
    .titulo-app {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #ff4b4b 0%, #ff8533 50%, #1e90ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitulo-app {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Tarjeta de resultado principal */
    .tarjeta-resultado {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }

    .tarjeta-resultado:hover {
        transform: translateY(-2px);
    }

    .clase-predicha {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.5rem 0;
        letter-spacing: -0.025em;
    }

    .confianza-badge {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    .confianza-badge-baja {
        display: inline-block;
        background-color: #fef3c7;
        color: #b45309;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }

    /* Footer */
    .footer-app {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 4rem;
        border-top: 1px solid #e2e8f0;
        padding-top: 1.5rem;
    }

    /* Ajuste para las etiquetas de las barras de progreso */
    .progreso-label {
        font-weight: 600;
        color: #334155;
        font-size: 0.95rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .progreso-porcentaje {
        text-align: right;
        font-weight: 700;
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Carga de modelo y clases
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de inteligencia artificial...")
def cargar_modelo():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource(show_spinner=False)
def cargar_clases():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Traducción dinámica de nombres de clase (inglés -> español)
#
# Fruits-360 nombra sus clases en inglés (ej. "Apple Golden 1", "Cherry Wax
# Black"). En vez de traducir a mano las ~260 clases, se traduce PALABRA POR
# PALABRA usando este diccionario, así funciona automáticamente sin importar
# cuántas ni cuáles clases traiga class_names.json.
#
# Las variedades/marcas (Braeburn, Granny Smith, Kaiser, etc.) se dejan tal
# cual, ya que en español también se usan esos nombres comercialmente.
# ---------------------------------------------------------------------------
PALABRAS_ES = {
    "apple": "Manzana", "apricot": "Albaricoque", "avocado": "Aguacate",
    "banana": "Plátano", "beans": "Frijoles", "beetroot": "Remolacha",
    "blackberry": "Mora", "blueberry": "Arándano", "cabbage": "Col",
    "cactus": "Cactus", "fruit": "Fruta", "cantaloupe": "Melón",
    "carambula": "Carambola", "carrot": "Zanahoria", "cauliflower": "Coliflor",
    "celery": "Apio", "cherimoya": "Chirimoya", "cherry": "Cereza",
    "chestnut": "Castaña", "clementine": "Clementina", "cocos": "Coco",
    "corn": "Maíz", "husk": "Cáscara", "cucumber": "Pepino", "ripe": "Maduro",
    "dates": "Dátiles", "eggplant": "Berenjena", "fig": "Higo",
    "ginger": "Jengibre", "root": "Raíz", "gooseberry": "Grosella espinosa",
    "granadilla": "Granadilla", "grape": "Uva", "grapefruit": "Pomelo",
    "guava": "Guayaba", "hazelnut": "Avellana", "huckleberry": "Arándano silvestre",
    "kaki": "Caqui", "kiwi": "Kiwi", "kohlrabi": "Colirrábano",
    "kumquats": "Kumquat", "lemon": "Limón", "limes": "Lima", "lychee": "Lichi",
    "mandarine": "Mandarina", "mango": "Mango", "mangostan": "Mangostán",
    "maracuja": "Maracuyá", "melon": "Melón", "piel": "Piel", "de": "de",
    "sapo": "Sapo", "mulberry": "Mora", "nectarine": "Nectarina",
    "flat": "Plana", "nut": "Nuez", "forest": "Bosque", "pecan": "Pecana",
    "onion": "Cebolla", "peeled": "Pelada", "orange": "Naranja",
    "papaya": "Papaya", "passion": "Pasión", "peach": "Durazno",
    "pear": "Pera", "pepino": "Pepino dulce", "pepper": "Pimiento",
    "physalis": "Physalis", "with": "con", "pineapple": "Piña",
    "mini": "Mini", "pitahaya": "Pitahaya", "plum": "Ciruela",
    "pomegranate": "Granada", "pomelo": "Pomelo", "sweetie": "Dulce",
    "potato": "Papa", "washed": "Lavada", "sweet": "Dulce", "quince": "Membrillo",
    "rambutan": "Rambután", "raspberry": "Frambuesa", "redcurrant": "Grosella roja",
    "salak": "Salak", "strawberry": "Fresa", "wedge": "Gajo",
    "tamarillo": "Tamarillo", "tangelo": "Tangelo", "tomato": "Tomate",
    "not": "no", "ripened": "madurado", "walnut": "Nuez de nogal",
    "watermelon": "Sandía", "zucchini": "Calabacín", "dark": "Oscuro",
    "almond": "Almendra", "red": "Roja", "yellow": "Amarilla", "green": "Verde",
    "black": "Negra", "white": "Blanca", "pink": "Rosa", "blue": "Azul",
    "wax": "Cera", "rainier": "Rainier", "lady": "Lady", "finger": "Finger",
    "crimson": "Crimson", "snow": "Snow", "golden": "Golden",
    "granny": "Granny", "smith": "Smith", "delicious": "Delicious",
    "braeburn": "Braeburn", "meyer": "Meyer", "abate": "Abate",
    "forelle": "Forelle", "kaiser": "Kaiser", "monster": "Monster",
    "williams": "Williams", "stone": "Piedra", "heart": "Corazón",
    "maroon": "Granate",
}


def traducir_nombre_clase(nombre_clase: str) -> str:
    """Traduce dinámicamente cada palabra del nombre de la clase al español."""
    palabras = nombre_clase.replace("_", " ").strip().split()
    traducidas = []
    for palabra in palabras:
        clave = palabra.lower()
        if clave.isdigit():
            traducidas.append(palabra)
        elif clave in PALABRAS_ES:
            traducidas.append(PALABRAS_ES[clave])
        else:
            traducidas.append(palabra.capitalize())
    return " ".join(traducidas)


# Alias para no romper el resto del código
formatear_nombre = traducir_nombre_clase


def predecir(modelo, class_names, img: Image.Image):
    img_resized = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img_resized, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = modelo.predict(arr, verbose=0)[0]

    k = min(TOP_K, len(class_names))
    top_idx = np.argsort(preds)[-k:][::-1]
    resultados = [(class_names[i], float(preds[i])) for i in top_idx]
    return resultados


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🍓 Acerca del Proyecto")
    st.caption("Sistema inteligente de reconocimiento visual optimizado para dispositivos móviles y web.")

    with st.container(border=True):
        st.markdown("**Dataset:** Fruits-360")
        st.markdown("**Arquitectura:** MobileNetV2")
        st.markdown("**Desarrollado por:** Erik Guillen Reyes")
        st.markdown("**Curso:** IS-701 Inteligencia Artificial")

    st.markdown("---")

    modelo_existe = MODEL_PATH.exists() and CLASSES_PATH.exists()
    if modelo_existe:
        clases_preview = cargar_clases()
        st.success(f"● Modelo en línea ({len(clases_preview)} clases)", icon="🟢")
        with st.expander("🔍 Explorar clases soportadas"):
            st.write(", ".join(formatear_nombre(c) for c in clases_preview))
    else:
        st.error("No se encontró el modelo entrenado", icon="🔴")

    st.markdown("---")
    st.markdown("### ⚙️ Configuración")
    umbral_confianza = st.slider(
        "Umbral mínimo de confianza",
        min_value=0, max_value=100, value=60, step=5,
        help="Nivel mínimo de certeza para considerar una predicción como 'segura'."
    )

# ---------------------------------------------------------------------------
# Encabezado principal
# ---------------------------------------------------------------------------
st.markdown('<p class="titulo-app">Clasificador de Frutas y Verduras</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitulo-app">Sube una imagen o usa la cámara para identificarla en tiempo real mediante Deep Learning</p>',
    unsafe_allow_html=True,
)

if not modelo_existe:
    st.error(
        f"Faltan archivos esenciales: No se encontró el modelo en '{MODEL_PATH}' o las clases en '{CLASSES_PATH}'. "
        "Por favor, asegúrate de subir la carpeta completa al repositorio.", icon="❌"
    )
    st.stop()

modelo = cargar_modelo()
class_names = cargar_clases()

# ---------------------------------------------------------------------------
# Cuerpo Principal (Layout de 2 Columnas Asimétricas para Mejor Balance)
# ---------------------------------------------------------------------------
col_izq, col_der = st.columns([11, 12], gap="large")

with col_izq:
    st.markdown("#### 📤 Captura de Imagen")

    with st.container(border=True):
        tab_subir, tab_camara = st.tabs(["📁 Subir Archivo", "📷 Usar Cámara"])
        imagen = None

        with tab_subir:
            archivo = st.file_uploader(
                "Elige una imagen",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            if archivo is not None:
                imagen = Image.open(archivo)

        with tab_camara:
            foto = st.camera_input("Toma una foto", label_visibility="collapsed")
            if foto is not None:
                imagen = Image.open(foto)

    if imagen is not None:
        st.markdown("")
        with st.container(border=True):
            st.image(imagen, caption="Vista previa de la muestra", use_container_width=True)

with col_der:
    st.markdown("#### 🔍 Análisis y Resultados")

    if imagen is None:
        st.info("El sistema está listo. Sube un archivo o toma una fotografía en el panel izquierdo para comenzar el análisis.", icon="💡")
    else:
        with st.spinner("Analizando patrones visuales..."):
            resultados = predecir(modelo, class_names, imagen)

        top_clase, top_prob = resultados[0]
        top_clase_es = formatear_nombre(top_clase)
        porcentaje_exacto = top_prob * 100
        es_confiable = porcentaje_exacto >= umbral_confianza

        # Renderizado de Tarjeta Principal
        clase_badge = "confianza-badge" if es_confiable else "confianza-badge-baja"
        texto_estado = "PREDICCIÓN SEGURA ✅" if es_confiable else "REVISIÓN RECOMENDADA ⚠️"

        st.markdown(
            f"""
            <div class="tarjeta-resultado">
                <div style="font-size: 0.85rem; font-weight: 700; color: #64748b; letter-spacing: 0.05em;">{texto_estado}</div>
                <div class="clase-predicha">{top_clase_es}</div>
                <div class="{clase_badge}">Confianza: {porcentaje_exacto:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not es_confiable:
            st.warning(
                f"La certeza del modelo ({porcentaje_exacto:.1f}%) es menor al umbral del {umbral_confianza}%. "
                "Esto puede deberse a mala iluminación, ángulos extraños o que el objeto no pertenece claramente al dataset.",
                icon="⚠️"
            )

        # Sección de distribución de probabilidades (Top de predicciones ordenado)
        st.markdown("##### 📊 Distribución de Probabilidad (Top K)")

        with st.container(border=True):
            st.markdown("<div style='padding: 0.5rem 0;'>", unsafe_allow_html=True)
            for clase, prob in resultados:
                nombre = formatear_nombre(clase)
                p_cien = prob * 100

                # Columnas internas para una cuadrícula perfecta
                c_lbl, c_bar, c_pct = st.columns([6, 12, 3])

                with c_lbl:
                    st.markdown(f"<p class='progreso-label'>{nombre}</p>", unsafe_allow_html=True)
                with c_bar:
                    # Forzar límites limpios en la barra de progreso
                    st.progress(min(max(prob, 0.0), 1.0))
                with c_pct:
                    st.markdown(f"<p class='progreso-porcentaje'>{p_cien:.1f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<p class="footer-app">Infraestructura Core: MobileNetV2 · Dataset de Referencia: Fruits-360 · Interfaz: Streamlit Engine</p>',
    unsafe_allow_html=True,
)
