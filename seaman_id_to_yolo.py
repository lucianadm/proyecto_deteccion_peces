# -*- coding: utf-8 -*-
"""
Procesa archivos Seaman_Fish_ID_Training_Data_*.ID para YOLO y genera JSON limpio.

Usa estas tres partes separadas:
1) Maneouver: lector binario tipo lee_manouver.ipynb.
2) ID: lector robusto tipo lee_encabezado_deID.ipynb para encontrar encabezados y matrices.
3) Procesamiento visual: fondo/alineado/paleta/por canal tipo extrae_info_p_yolo.ipynb.

Salida por cada Training_Data:
- JPG completa por canal: base_CH1.jpg, base_CH2.jpg, ...
- JPG sin fondo por canal: base_CH1_SIN_FONDO.jpg, base_CH2_SIN_FONDO.jpg, ... (arranca en el fondo; sin franja negra)
- NPY de Data1, Data2, Data3 y Data4 (si existe) alineados usando EXACTAMENTE la misma sección que la JPG _SIN_FONDO.
- JSON por canal con solo metadata útil para identificación acústica/especie.
- No genera canal virtual artificial. Si el archivo ya trae un canal virtual real, lo procesa como un canal más.
"""

from __future__ import annotations

import os
import re
import glob
import json
import struct
import gc
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


# ============================================================
# CONFIGURACIÓN DEL ALGORITMO
# ============================================================
# Estos valores reproducen el procesamiento usado para generar el dataset.
# Las rutas de entrada/salida NO se configuran aquí: se pasan por línea de
# comandos; ver --help y README.md.

# Copia opcional de todos los archivos generados a una única carpeta.
# Se activa con --global-output-dir.
guardar_copia_global = False
carpeta_todas_imagenes_yolo = None
prefijo_salida_global = "dataset"

# None = procesa todos los pings/bloques. Se puede limitar con --max-blocks.
max_bloques_por_archivo = None

# Fuente utilizada para construir la imagen de ecosonda.
# Data3 corresponde a S5 = DB DATA WITHOUT TVG.
usar_fuente_para_imagen = "Data3"

# Detección/alineación del fondo.
metodo_fondo = "A"
usar_depth_in_samples_para_fondo = True
min_fraccion_depth_valido = 0.50

min_range_ratio = 0.55
max_range_ratio = 0.90
suavizado_perfil = 9
max_salto_fondo = 50
ventana_suavizado = 21

# Altura máxima usada por algunas vistas de referencia. La imagen final sin
# fondo conserva toda la columna de agua disponible por encima del fondo.
hmax_sobre_fondo_m = 60.0
margen_quitar_fondo_samples = 0

dpi_salida = 200

peso_ch1 = 0.85
peso_ch2 = 0.15

contraste_ch1 = {"suavizar": False}
contraste_ch2 = {"suavizar": True}


# ============================================================
# FUNCIONES BÁSICAS BINARIAS
# ============================================================

HEADER_SIZE = 24


def read_i32(f):
    return struct.unpack("<i", f.read(4))[0]


def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_f32(f):
    return struct.unpack("<f", f.read(4))[0]


def read_text(f, n):
    raw = f.read(n)
    txt = raw.decode("latin1", errors="ignore")
    txt = txt.replace("\x00", "")
    txt = "".join(c for c in txt if 32 <= ord(c) <= 126)
    return txt.strip()


def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode(errors="ignore")
    return obj

def copiar_a_carpeta_global(archivo_origen):
    """
    Copia cualquier archivo generado a una carpeta global,
    agregando un prefijo al nombre.
    """
    if not guardar_copia_global:
        return

    archivo_origen = Path(archivo_origen)

    if not archivo_origen.exists():
        return

    carpeta_global = Path(carpeta_todas_imagenes_yolo)
    carpeta_global.mkdir(parents=True, exist_ok=True)

    # Evita duplicar el prefijo si el archivo ya fue guardado como
    # 01_22_1_nombre.ext, 02_20_nombre.ext, etc.
    if archivo_origen.name.startswith(f"{prefijo_salida_global}_"):
        nombre_destino = archivo_origen.name
    else:
        nombre_destino = f"{prefijo_salida_global}_{archivo_origen.name}"

    archivo_destino = carpeta_global / nombre_destino

    import shutil
    shutil.copy2(archivo_origen, archivo_destino)

    print("Copia global guardada:", archivo_destino)
# ============================================================
# LECTOR MANEOUVER - basado en lee_manouver.ipynb
# ============================================================
def nombre_con_prefijo(nombre):
    return f"{prefijo_salida_global}_{nombre}"

def limpiar_texto_fijo(b):
    txt = b.decode("latin1", errors="ignore").replace("\x00", "")
    txt = "".join(c for c in txt if 32 <= ord(c) <= 126)
    return txt.strip()


def limpiar_version(b):
    txt = b.decode("latin1", errors="ignore").replace("\x00", "")
    m = re.search(r"\d+\.\d+\.\d+\.\d+", txt)
    return m.group(0) if m else limpiar_texto_fijo(b)


def descripcion_sounder(sounder_type):
    if sounder_type == 1:
        return "ECHOSOUNDER_SIB_BEAM - Single beam"
    elif sounder_type == 2:
        return "ECHOSOUNDER_SPB_BEAM - Split beam"
    elif sounder_type == 6:
        return "ECHOSOUNDER_SI_BEAM - Single beam Size Indicator"
    else:
        return "Tipo desconocido"


def buscar_archivo_maneouver(input_dir):
    input_dir = Path(input_dir)
    candidatos = sorted(input_dir.glob("Seaman_Fish_ID_File_Features_Maneouver_*.ID"))
    if not candidatos:
        candidatos = sorted(input_dir.glob("*File_Features*Maneouver*.ID"))
    return candidatos[0] if candidatos else None


def leer_maneouver_id(path):
    path = Path(path)
    datos = {}

    with open(path, "rb") as f:
        total_size = path.stat().st_size

        datos["Version"] = limpiar_version(f.read(24))

        datos["Start_Date"] = read_i32(f)
        datos["Start_Time"] = read_i32(f)
        datos["End_Date"] = read_i32(f)
        datos["End_Time"] = read_i32(f)

        datos["Pings_Total_Amount"] = read_u32(f)
        datos["Files_Total_Amount"] = read_u32(f)

        datos["Sounder_Type"] = read_u32(f)
        datos["Sounder_Type_Description"] = descripcion_sounder(datos["Sounder_Type"])
        datos["Channel_Amount"] = read_u32(f)

        datos["Serial_Number_FMF"] = read_u32(f)
        datos["Fishing_Method"] = read_text(f, 24)

        datos["Data_Type_1"] = read_text(f, 8)
        datos["Data_Name_1"] = read_text(f, 24)
        datos["Data_Type_2"] = read_text(f, 8)
        datos["Data_Name_2"] = read_text(f, 24)
        datos["Data_Type_3"] = read_text(f, 8)
        datos["Data_Name_3"] = read_text(f, 24)

        pos_actual = f.tell()
        bytes_restantes = total_size - pos_actual
        raw_obs = f.read(bytes_restantes)
        try:
            observaciones = raw_obs.decode("utf-16-le", errors="ignore").replace("\x00", "").strip()
        except Exception:
            observaciones = ""
        datos["Observations"] = observaciones

    return datos


def extraer_maneouver_util(m, training_files_found=None):
    """
    Solo contexto que puede aportar a identificación de especies.
    Se excluyen: versión, offsets, nombres de archivo, número de serie,
    cantidad administrativa de archivos/pings, etc.
    """
    return {
        "fishing_method": m.get("Fishing_Method"),
        "sounder_type": m.get("Sounder_Type"),
        "sounder_type_description": m.get("Sounder_Type_Description"),
        "channel_amount_configured": m.get("Channel_Amount"),
        "maneuver_start_date": m.get("Start_Date"),
        "maneuver_start_time": m.get("Start_Time"),
        "maneuver_end_date": m.get("End_Date"),
        "maneuver_end_time": m.get("End_Time"),
        "acoustic_data_products": {
            "data1": m.get("Data_Name_1"),
            "data2": m.get("Data_Name_2"),
            "data3": m.get("Data_Name_3"),
            "data4": "DB DATA / Graphic Filtered Data (S7), si está presente en el Training_Data",
        },
    }


# ============================================================
# LECTOR ID - basado en lee_encabezado_deID.ipynb
# ============================================================


def leer_cabecera_training_data_desde_pos(f, path):
    """
    Intenta leer la cabecera general de Training_Data v5.0.

    En algunos archivos exportados la cabecera real no queda perfectamente alineada
    con esta lectura; por eso este programa usa esta cabecera solo como referencia
    y, si algo no cierra, busca el primer HEADER de sondaje.
    """
    header = {}
    file_size = Path(path).stat().st_size
    header["_file_size_bytes"] = file_size

    start0 = f.tell()

    # Variante v5 esperada: Version_Data_Fish_ID ocupa 24 bytes antes de fechas.
    f.seek(0)
    header["Version_Data_Fish_ID"] = limpiar_version(f.read(24))
    try:
        header["Start_Date"] = read_i32(f)
        header["Start_Time"] = read_i32(f)
        header["End_Date"] = read_i32(f)
        header["End_Time"] = read_i32(f)
        header["Pings_Amount"] = read_u32(f)
        header["Sounder_Type"] = read_u32(f)
        header["Sounder_Type_Description"] = descripcion_sounder(header["Sounder_Type"])
        header["Channel_Amount"] = read_u32(f)
        header["DB_By_Color_CH1"] = read_u32(f)
        header["DB_By_Color_CH2"] = read_u32(f)
        header["Max_Samples_On_File_CH1"] = read_u32(f)
        header["Max_Samples_On_File_CH2"] = read_u32(f)
        header["_offset_despues_cabecera"] = f.tell()
        header["_header_variant"] = "v5_with_version"
        return header
    except Exception:
        pass

    # Fallback viejo: fechas desde el byte cero.
    f.seek(start0)
    header = {"_file_size_bytes": file_size}
    header["Start_Date"] = read_i32(f)
    header["Start_Time"] = read_i32(f)
    header["End_Date"] = read_i32(f)
    header["End_Time"] = read_i32(f)
    header["Pings_Amount"] = read_u32(f)
    header["Channel_Amount"] = read_u32(f)
    header["Max_Samples_On_File_CH1"] = read_u32(f)
    header["Max_Samples_On_File_CH2"] = read_u32(f)
    header["_offset_despues_cabecera"] = f.tell()
    header["_header_variant"] = "legacy_without_version"
    return header


def header_general_valido(header):
    if not (1 <= header.get("Channel_Amount", -1) <= 8):
        return False
    if not (1 <= header.get("Pings_Amount", -1) <= 300000):
        return False
    if not (1 <= header.get("Max_Samples_On_File_CH1", -1) <= 200000):
        return False
    if not (1 <= header.get("Max_Samples_On_File_CH2", -1) <= 200000):
        return False
    return True


def buscar_header(f, start=0, max_bytes=None):
    f.seek(start)
    data = f.read() if max_bytes is None else f.read(max_bytes)
    idx = data.find(b"HEADER")
    if idx == -1:
        return None
    pos = start + idx
    f.seek(pos)
    return pos


def _leer_array_por_keyword(f, keyword, n, leer_datos=True):
    """
    Lee el array asociado a un keyword de Fish ID.
    En v5:
      - FILTERED SAMPLES DATA -> int32
      - CORRELATOR DATA / DETECTOR DATA -> int32
      - DB DATA WITHOUT TVG -> int16
      - DB DATA -> int16
    """
    kw = (keyword or "").upper()
    if ("FILTERED" in kw) or ("CORRELATOR" in kw) or ("DETECTOR" in kw):
        dtype = "<i4"
        nbytes = 4 * n
    else:
        dtype = "<i2"
        nbytes = 2 * n

    if leer_datos:
        arr = np.fromfile(f, dtype=dtype, count=n)
        if arr.size != n:
            raise ValueError(f"No se pudieron leer {n} muestras para {keyword!r}. Leídas: {arr.size}")
        return arr
    else:
        f.seek(nbytes, 1)
        return None


def _peek_keyword(f):
    pos = f.tell()
    raw = f.read(HEADER_SIZE)
    f.seek(pos)
    if len(raw) < HEADER_SIZE:
        return ""
    return _texto_bytes_limpio(raw)


def _keyword_es_header(txt):
    return "HEADER" in (txt or "").upper()


def leer_columna_v5(f, leer_datos=True):
    """
    Lee un bloque de sondaje Fish ID v5.

    Lee siempre Data1, Data2 y Data3. Además, si antes del próximo HEADER aparece
    otro bloque de datos, lo guarda como Data4. Esto es importante porque en v5
    Single Beam puede incluir Data4 = DB DATA / S7.
    """
    p = {}
    p["_offset_inicio"] = f.tell()
    p["KeyWord_Header"] = read_text(f, HEADER_SIZE)

    if "HEADER" not in p["KeyWord_Header"]:
        raise ValueError(f"No se encontró HEADER válido en offset {p['_offset_inicio']}")

    p["Maneouver_Ping"] = read_u32(f)
    p["Channel"] = read_i32(f)
    p["Channel_Name"] = f"CH{p['Channel']}"

    p["Transmission_Mode"] = read_u32(f)

# En estos ID viene el nombre del transductor antes de los parámetros
    p["Transducer_Name"] = read_text(f, 100)

# Estos dos floats aparecen antes de la frecuencia
    p["Beam_1"] = read_f32(f)
    p["Beam_2"] = read_f32(f)

    p["TX_Center_Frequency"] = read_i32(f)



    p["TX_Band_Width"] = read_i32(f)
    p["Sampling_Frequency"] = read_i32(f)
    p["Center_Freq_After_FPGA"] = read_i32(f)
    p["Power"] = read_u32(f)
    p["Frequency_Modulation"] = read_u32(f)
    p["Window_RX_Name"] = read_text(f, 8)
    p["Pulse_Width"] = read_u32(f)
    p["SamplesXMeter"] = read_f32(f)
    p["Sounder_Scale"] = read_u32(f)
    p["SubScale_1"] = read_u32(f)
    p["SubScale_2"] = read_u32(f)
    p["Total_Scale_Samples"] = read_u32(f)
    p["Date"] = read_i32(f)
    p["Time"] = read_i32(f)
    p["Latitude"] = read_f32(f)
    p["Card_N_S"] = read_text(f, 4)
    p["Longitude"] = read_f32(f)
    p["Card_E_W"] = read_text(f, 4)
    p["Vessel_Speed"] = read_f32(f)
    p["Vessel_Course"] = read_f32(f)
    p["Vessel_Heading"] = read_f32(f)
    p["Transducer_Temperature"] = read_f32(f)
    p["Net_Temperature"] = read_f32(f)
    p["Ping_Rate"] = read_f32(f)
    p["Depth_In_Samples"] = read_u32(f)
    p["Depth"] = read_f32(f)
    p["SV"] = read_f32(f)
    p["Roughness"] = read_i32(f)
    p["Hardness"] = read_i32(f)
    p["_offset_fin_header"] = f.tell()

    n = int(p["Total_Scale_Samples"])
    if not (1 <= n <= 200000):
        raise ValueError(f"Total_Scale_Samples inválido: {n}")

    p["KeyWord_Data1"] = read_text(f, HEADER_SIZE)
    Data1 = _leer_array_por_keyword(f, p["KeyWord_Data1"], n, leer_datos=leer_datos)

    p["KeyWord_Data2"] = read_text(f, HEADER_SIZE)
    Data2 = _leer_array_por_keyword(f, p["KeyWord_Data2"], n, leer_datos=leer_datos)

    p["KeyWord_Data3"] = read_text(f, HEADER_SIZE)
    Data3 = _leer_array_por_keyword(f, p["KeyWord_Data3"], n, leer_datos=leer_datos)

    # Data4 opcional: en SIB suele ser DB DATA / S7. Lo leemos solo si el
    # siguiente keyword no es HEADER y parece un bloque de datos real.
    Data4 = None
    p["KeyWord_Data4"] = None
    pos_antes_extra = f.tell()
    kw_extra = _peek_keyword(f)
    if kw_extra and (not _keyword_es_header(kw_extra)):
        # Evitar avanzar si lo que sigue no parece un bloque de datos conocido.
        kwu = kw_extra.upper()
        parece_dato = any(tok in kwu for tok in ["DATA", "SAMPLES", "CORRELATOR", "DETECTOR", "DB"])
        if parece_dato:
            p["KeyWord_Data4"] = read_text(f, HEADER_SIZE)
            Data4 = _leer_array_por_keyword(f, p["KeyWord_Data4"], n, leer_datos=leer_datos)
        else:
            f.seek(pos_antes_extra)

    p["_offset_fin_bloque"] = f.tell()
    return Data1, Data2, Data3, Data4, p


# Alias para compatibilidad con nombres previos
leer_columna_v4 = leer_columna_v5



def _texto_bytes_limpio(raw: bytes) -> str:
    txt = raw.decode("latin1", errors="ignore").replace("\x00", "")
    txt = "".join(c for c in txt if 32 <= ord(c) <= 126)
    return txt.strip()


def _parece_keyword_virtual(txt: str) -> bool:
    """
    Detecta si un bloque intermedio corresponde a un canal virtual REAL
    ya guardado en el archivo. No genera ni fusiona nada.
    """
    t = (txt or "").upper()
    palabras_virtuales = ["VIRTUAL", "VIRT", "VIRTU", "VCH", "VC"]
    return any(pal in t for pal in palabras_virtuales)


def intentar_leer_virtual_real_entre_headers(path, pos_ini, pos_fin, n_samples, p_ref):
    """
    Algunos archivos guardan canales virtuales reales entre dos bloques HEADER,
    sin un HEADER propio. Este lector intenta capturarlos sin inventar/fusionar
    canales: solo lee la virtual si ya está físicamente en el archivo.

    Formato esperado del bloque virtual:
        keyword_1 (24 bytes) + int16[n]
        keyword_2 (24 bytes) + int16[n]

    Si los keywords no parecen virtuales, no devuelve nada.
    """
    if pos_fin is None or pos_fin <= pos_ini:
        return []

    n = int(n_samples)
    if n <= 0:
        return []

    bytes_necesarios = HEADER_SIZE + 2 * n + HEADER_SIZE + 2 * n
    if (pos_fin - pos_ini) < bytes_necesarios:
        return []

    virtuales = []

    try:
        with open(path, "rb") as fv:
            fv.seek(pos_ini)
            raw_kw1 = fv.read(HEADER_SIZE)
            kw1 = _texto_bytes_limpio(raw_kw1)
            arr1 = np.fromfile(fv, dtype="<i2", count=n)

            raw_kw2 = fv.read(HEADER_SIZE)
            kw2 = _texto_bytes_limpio(raw_kw2)
            arr2 = np.fromfile(fv, dtype="<i2", count=n)

        if len(arr1) != n or len(arr2) != n:
            return []

        # Solo se acepta como virtual si al menos uno de los keywords lo indica.
        # Esto evita confundir Data4/correlator u otros productos con una virtual.
        es_virtual = _parece_keyword_virtual(kw1) or _parece_keyword_virtual(kw2)
        if not es_virtual:
            return []

        p1 = dict(p_ref)
        p1["Channel"] = "VIRT_S5V"
        p1["Channel_Name"] = "VIRT_S5V"
        p1["IsVirtual"] = True
        p1["Virtual_Source"] = kw1
        p1["KeyWord_Data3"] = kw1

        p2 = dict(p_ref)
        p2["Channel"] = "VIRT_S7V"
        p2["Channel_Name"] = "VIRT_S7V"
        p2["IsVirtual"] = True
        p2["Virtual_Source"] = kw2
        p2["KeyWord_Data3"] = kw2

        zeros_i32 = np.zeros(n, dtype=np.int32)
        virtuales.append(("VIRT_S5V", zeros_i32, zeros_i32, arr1.astype(np.int16, copy=False), p1))
        virtuales.append(("VIRT_S7V", zeros_i32.copy(), zeros_i32.copy(), arr2.astype(np.int16, copy=False), p2))

    except Exception:
        return []

    return virtuales


def leer_id_completo_robusto(path, max_bloques=None, verbose=True):
    """
    Lee todos los bloques HEADER del ID. Si la cabecera general no es válida,
    busca el primer HEADER, como en las pruebas que hicimos.

    Devuelve:
        fp_header, channels, Z, Data1_stack, Data2_stack, Data4_stack, bloques

    Donde:
        Z = Data3 / DB DATA WITHOUT TVG, shape por canal: (samples, pings)
        Data1_stack = Data1, shape por canal: (samples, pings)
        Data2_stack = Data2, shape por canal: (samples, pings)
        Data4_stack = Data4/S7 si existe, shape por canal: (samples, pings)
    """
    path = Path(path)

    with open(path, "rb") as f:
        f.seek(0)
        header = leer_cabecera_training_data_desde_pos(f, path)

        if header_general_valido(header):
            modo = "CABECERA_GENERAL_OK"
            f.seek(header["_offset_despues_cabecera"])
        else:
            modo = "SIN_CABECERA_GENERAL_VALIDA_BUSCANDO_HEADER"
            header["_warning"] = "Cabecera general inválida; se buscó el primer HEADER."
            pos_header = buscar_header(f, start=0)
            if pos_header is None:
                raise ValueError("No se encontró HEADER en el archivo.")
            header["_offset_despues_cabecera"] = None
            header["_offset_primer_header"] = pos_header
            f.seek(pos_header)

        bloques = []
        datos_por_canal: Dict[Any, Dict[str, Any]] = {}

        while True:
            if max_bloques is not None and len(bloques) >= max_bloques:
                break

            pos = f.tell()
            if pos >= header["_file_size_bytes"]:
                break

            probe = f.read(6)
            f.seek(pos)
            if probe != b"HEADER":
                pos_header = buscar_header(f, start=pos, max_bytes=500000)
                if pos_header is None:
                    break
                f.seek(pos_header)

            try:
                d1, d2, d3, d4, p = leer_columna_v5(f, leer_datos=True)
            except Exception:
                break

            ch = int(p["Channel"])
            if ch not in datos_por_canal:
                datos_por_canal[ch] = {
                    "Data1_list": [],
                    "Data2_list": [],
                    "Data3_list": [],
                    "Data4_list": [],
                    "Parameters": [],
                }

            datos_por_canal[ch]["Data1_list"].append(d1)
            datos_por_canal[ch]["Data2_list"].append(d2)
            datos_por_canal[ch]["Data3_list"].append(d3)
            datos_por_canal[ch]["Data4_list"].append(d4)
            datos_por_canal[ch]["Parameters"].append(p)
            bloques.append(p)

            # ------------------------------------------------------------
            # Canal virtual REAL ya presente en el archivo
            # ------------------------------------------------------------
            # No se genera ninguna virtual por fusión. Solo se captura si
            # entre este bloque y el próximo HEADER hay datos con keyword
            # que indique virtual.
            pos_fin_bloque = f.tell()
            pos_header_siguiente = buscar_header(f, start=pos_fin_bloque, max_bytes=500000)

            virtuales_reales = intentar_leer_virtual_real_entre_headers(
                path=path,
                pos_ini=pos_fin_bloque,
                pos_fin=pos_header_siguiente,
                n_samples=int(p.get("Total_Scale_Samples", 0)),
                p_ref=p,
            )

            for chv, vd1, vd2, vd3, pv in virtuales_reales:
                if chv not in datos_por_canal:
                    datos_por_canal[chv] = {
                        "Data1_list": [],
                        "Data2_list": [],
                        "Data3_list": [],
                        "Data4_list": [],
                        "Parameters": [],
                    }
                datos_por_canal[chv]["Data1_list"].append(vd1)
                datos_por_canal[chv]["Data2_list"].append(vd2)
                datos_por_canal[chv]["Data3_list"].append(vd3)
                datos_por_canal[chv]["Data4_list"].append(None)
                datos_por_canal[chv]["Parameters"].append(pv)

            if pos_header_siguiente is None:
                break
            f.seek(pos_header_siguiente)

        if not bloques:
            raise ValueError("No se pudo leer ningún bloque de datos.")

    # Construir stacks rectangulares por canal: (samples, pings)
    canales_ordenados = sorted(datos_por_canal.keys(), key=lambda x: (isinstance(x, str), str(x)))
    channels = []
    data1_stacks = []
    data2_stacks = []
    data3_stacks = []
    data4_stacks = []

    for ch in canales_ordenados:
        info = datos_por_canal[ch]
        max_samples = max(len(x) for x in info["Data3_list"])
        n_pings = len(info["Data3_list"])

        A1 = np.zeros((max_samples, n_pings), dtype=np.int32)
        A2 = np.zeros((max_samples, n_pings), dtype=np.int32)
        A3 = np.zeros((max_samples, n_pings), dtype=np.int16)
        tiene_data4 = any(x is not None for x in info.get("Data4_list", []))
        A4 = np.zeros((max_samples, n_pings), dtype=np.int16) if tiene_data4 else None

        for i in range(n_pings):
            d1 = info["Data1_list"][i]
            d2 = info["Data2_list"][i]
            d3 = info["Data3_list"][i]
            d4 = info.get("Data4_list", [None] * n_pings)[i]
            A1[:len(d1), i] = d1
            A2[:len(d2), i] = d2
            A3[:len(d3), i] = d3
            if A4 is not None and d4 is not None:
                A4[:len(d4), i] = d4

        # Diagnóstico al levantar S3 / Correlator desde el .ID
        imprimir_diagnostico_s3(f"{ch}_Data2_S3_CORRELATOR_DETECTOR", A2, etapa="LEVANTADO DEL ID")

        channel_name = ch if isinstance(ch, str) else f"CH{ch}"
        channels.append({
            "Channel": ch,
            "Channel_Name": channel_name,
            "IsVirtual": isinstance(ch, str) and str(ch).upper().startswith("VIRT"),
            "Parameters": info["Parameters"],
        })
        data1_stacks.append(A1)
        data2_stacks.append(A2)
        data3_stacks.append(A3)
        data4_stacks.append(A4)

    # No apilar canales entre sí: pueden tener distinta cantidad de muestras.
    # Cada entrada de estas listas es una matriz (samples, pings) de un canal.
    Data1_stack = data1_stacks
    Data2_stack = data2_stacks
    Z = data3_stacks
    Data4_stack = data4_stacks

    header["_modo_lectura"] = modo
    header["_bloques_leidos"] = len(bloques)
    header["_canales_detectados"] = canales_ordenados

    if verbose:
        print(f"Modo lectura: {modo}")
        print(f"Bloques leídos: {len(bloques)}")
        print(f"Canales detectados: {canales_ordenados}")
        
        for i, Zch in enumerate(Z):
            print(f"{channels[i]['Channel_Name']} Data3 shape: {Zch.shape} (samples, pings)")

    return header, channels, Z, Data1_stack, Data2_stack, Data4_stack, bloques


# ============================================================
# METADATA ÚTIL PARA JSON
# ============================================================


def stat_vals(vals):
    vals_limpios = []
    for v in vals:
        try:
            vf = float(v)
        except Exception:
            continue
        if np.isfinite(vf):
            vals_limpios.append(vf)

    if not vals_limpios:
        return None

    return {
        "median": float(np.nanmedian(vals_limpios)),
        "min": float(np.nanmin(vals_limpios)),
        "max": float(np.nanmax(vals_limpios)),
    }


def stat_param(params, key):
    return stat_vals([p.get(key) for p in params if key in p])


def median_param(params, key):
    st = stat_param(params, key)
    return None if st is None else st["median"]


def obtener_parametro_representativo(channels, ch_idx, key):
    vals = []
    for p in channels[ch_idx]["Parameters"]:
        try:
            v = float(p.get(key, np.nan))
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if len(vals) == 0:
        return np.nan
    return float(np.nanmedian(vals))


def modo_acustico(transmission_mode, frequency_modulation, bandwidth_hz):
    tm = 0 if transmission_mode is None else transmission_mode
    fm = 0 if frequency_modulation is None else frequency_modulation
    bw = 0 if bandwidth_hz is None else bandwidth_hz
    return "CHIRP" if (tm != 0 or fm != 0 or bw > 1000) else "CW"


def resumir_metadata_canal(channels, ch_idx):
    """
    Metadata útil para clasificación acústica/especie.
    No incluye rutas, offsets, nombres de imágenes, nombres de npy ni datos administrativos.
    """
    params = channels[ch_idx]["Parameters"]
    p0 = params[0] if params else {}

    freq_hz = median_param(params, "TX_Center_Frequency")
    bw_hz = median_param(params, "TX_Band_Width")
    fs_hz = median_param(params, "Sampling_Frequency")
    center_fpga_hz = median_param(params, "Center_Freq_After_FPGA")
    transmission_mode = median_param(params, "Transmission_Mode")
    frequency_modulation = median_param(params, "Frequency_Modulation")

    return {
        "channel": channels[ch_idx]["Channel"],
        "channel_name": channels[ch_idx]["Channel_Name"],

        "acoustic_setup": {
            "tx_center_frequency_hz": freq_hz,
            "tx_center_frequency_khz": None if freq_hz is None else freq_hz / 1000.0,
            "tx_band_width_hz": bw_hz,
            "tx_band_width_khz": None if bw_hz is None else bw_hz / 1000.0,
            "sampling_frequency_hz": fs_hz,
            "center_freq_after_fpga_hz": center_fpga_hz,
            "transmission_mode": transmission_mode,
            "frequency_modulation": frequency_modulation,
            "mode": modo_acustico(transmission_mode, frequency_modulation, bw_hz),
            "Transducer_Beam_Angle_3dB_Port_Starboard": (
                stat_param(params, "Transducer_Beam_Angle_3dB_Port_Starboard")
                or stat_param(params, "Beam_1")
            ),
            "Transducer_Beam_Angle_3dB_Fore_Aft": (
                stat_param(params, "Transducer_Beam_Angle_3dB_Fore_Aft")
                or stat_param(params, "Beam_2")
            ),
            "transducer_index": median_param(params, "Transducer_Index"),
            "power": median_param(params, "Power"),
            "pulse_width_us": median_param(params, "Pulse_Width"),
            "window_rx_name": p0.get("Window_RX_Name"),
            "transducer_name": p0.get("Transducer_Name"),
        },

        "data_products_present": {
            "data1": p0.get("KeyWord_Data1"),
            "data2": p0.get("KeyWord_Data2"),
            "data3": p0.get("KeyWord_Data3"),
            "data4": p0.get("KeyWord_Data4"),
            "image_source_used": usar_fuente_para_imagen,
        },

        "range_geometry": {
            "samples_x_meter": stat_param(params, "SamplesXMeter"),
            "total_scale_samples": stat_param(params, "Total_Scale_Samples"),
            "sounder_scale_m": stat_param(params, "Sounder_Scale"),
            "subscale_1": stat_param(params, "SubScale_1"),
            "subscale_2": stat_param(params, "SubScale_2"),
            "depth_m": stat_param(params, "Depth"),
            "depth_in_samples": stat_param(params, "Depth_In_Samples"),
        },

        "navigation_environment": {
            "date": stat_param(params, "Date"),
            "time": stat_param(params, "Time"),
            "latitude_raw": stat_param(params, "Latitude"),
            "longitude_raw": stat_param(params, "Longitude"),
            "vessel_speed_knots": stat_param(params, "Vessel_Speed"),
            "vessel_course_deg": stat_param(params, "Vessel_Course"),
            "ping_rate_hz": stat_param(params, "Ping_Rate"),
            "bottom_roughness": stat_param(params, "Roughness"),
            "bottom_hardness": stat_param(params, "Hardness"),
            "sv": stat_param(params, "SV"),
            "vessel_heading_deg": stat_param(params, "Vessel_Heading"),
            "transducer_temperature": stat_param(params, "Transducer_Temperature"),
            "net_temperature": stat_param(params, "Net_Temperature"),
            "latitude_cardinal": p0.get("Card_N_S"),
            "longitude_cardinal": p0.get("Card_E_W"),
        },
    }


def metadata_training_util(path, header, channels):
    channel_summaries = [resumir_metadata_canal(channels, i) for i in range(len(channels))]

    freqs = []
    modes = []
    for c in channel_summaries:
        fkhz = c.get("acoustic_setup", {}).get("tx_center_frequency_khz")
        if fkhz is not None:
            freqs.append(fkhz)
        modes.append(c.get("acoustic_setup", {}).get("mode"))

    freqs = sorted(freqs)

    return {
        "acoustic_summary": {
            "frequencies_khz": freqs,
            "is_multifrequency": len(set(freqs)) > 1,
            "has_chirp_or_fm": any(m == "CHIRP" for m in modes),
            "channels_detected": [c["channel"] for c in channel_summaries],
        },
        "channels_summary": channel_summaries,
    }



def obtener_tamano_jpg(path_jpg):
    """
    Devuelve el tamaño real del JPG guardado como (ancho_px, alto_px).
    """
    try:
        from PIL import Image
        with Image.open(path_jpg) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None, None


def construir_geometria_vertical_bottom_aligned(
    path_jpg,
    altura_m,
    profundidad_fondo_m=None,
):
    """
    Geometría vertical del flujo general.

    La imagen fue alineada desde el fondo:
    - fila superior: máxima altura sobre el fondo;
    - fila inferior: 0 m sobre el fondo.
    """
    ancho_px, alto_px = obtener_tamano_jpg(path_jpg)

    altura_m = np.asarray(altura_m, dtype=float)
    altura_max_m = (
        float(altura_m[-1])
        if altura_m.size and np.isfinite(altura_m[-1])
        else None
    )

    metros_por_pixel = None
    if altura_max_m is not None and alto_px is not None and alto_px > 1:
        metros_por_pixel = altura_max_m / float(alto_px - 1)

    profundidad_superior_m = None
    if (
        profundidad_fondo_m is not None
        and altura_max_m is not None
        and np.isfinite(profundidad_fondo_m)
    ):
        profundidad_superior_m = (
            float(profundidad_fondo_m) - altura_max_m
        )

    return {
        "saved_image_width_px": ancho_px,
        "saved_image_height_px": alto_px,
        "vertical_reference": "bottom_aligned",
        "top_height_above_bottom_m": altura_max_m,
        "bottom_height_above_bottom_m": 0.0,
        "vertical_extent_m": altura_max_m,
        "meters_per_vertical_pixel": metros_por_pixel,
        "bottom_depth_m_used": profundidad_fondo_m,
        "top_depth_m_approx": profundidad_superior_m,
        "bottom_depth_m_approx": profundidad_fondo_m,
        "pixel_y_direction": "top_to_bottom_increasing_depth",
        "height_above_bottom_formula": (
            "height_m = top_height_above_bottom_m "
            "- y_px * meters_per_vertical_pixel"
        ),
        "depth_formula": (
            "depth_m = bottom_depth_m_used - height_m"
        ),
    }


def calcular_metros_por_sample_desde_params(params):
    """
    Para Split Beam usa primero Depth / Depth_In_Samples.
    Si no está disponible, usa 1 / SamplesXMeter.
    """
    valores = []

    for p in (params or []):
        try:
            depth = float(p.get("Depth", np.nan))
            depth_samples = float(
                p.get("Depth_In_Samples", np.nan)
            )
        except Exception:
            continue

        if (
            np.isfinite(depth)
            and np.isfinite(depth_samples)
            and depth >= 0
            and depth_samples > 0
        ):
            valores.append(depth / depth_samples)

    if valores:
        return float(np.nanmedian(valores))

    sxm = median_param(params or [], "SamplesXMeter")
    try:
        sxm = float(sxm)
        if np.isfinite(sxm) and sxm > 0:
            return 1.0 / sxm
    except Exception:
        pass

    return None


def construir_geometria_vertical_surface_view(
    path_jpg,
    cut_samples,
    params,
):
    """
    Geometría vertical de la rama Split Beam dedicada.

    La imagen conserva desde la superficie hasta el corte anterior al fondo.
    """
    ancho_px, alto_px = obtener_tamano_jpg(path_jpg)
    metros_por_sample = calcular_metros_por_sample_desde_params(
        params
    )

    profundidad_inferior_m = None
    if metros_por_sample is not None:
        profundidad_inferior_m = (
            float(cut_samples) * metros_por_sample
        )

    metros_por_pixel = None
    if (
        profundidad_inferior_m is not None
        and alto_px is not None
        and alto_px > 1
    ):
        metros_por_pixel = (
            profundidad_inferior_m / float(alto_px - 1)
        )

    return {
        "saved_image_width_px": ancho_px,
        "saved_image_height_px": alto_px,
        "vertical_reference": "surface_to_bottom_cut",
        "first_saved_sample": 0,
        "last_saved_sample_exclusive": int(cut_samples),
        "saved_samples_count": int(cut_samples),
        "meters_per_original_sample": metros_por_sample,
        "top_depth_m_approx": 0.0,
        "bottom_depth_m_approx": profundidad_inferior_m,
        "vertical_extent_m": profundidad_inferior_m,
        "meters_per_vertical_pixel": metros_por_pixel,
        "pixel_y_direction": "top_to_bottom_increasing_depth",
        "depth_formula": (
            "depth_m = y_px * meters_per_vertical_pixel"
        ),
    }


def metadata_imagen_canal(base_metadata, channel_summary, extra=None):
    """
    JSON mínimo por canal: solo información potencialmente útil para especie.
    No guarda nombres de imagen, npy, offsets, modo de lectura ni archivos fuente.
    """
    extra = extra or {}
    return {
        "maneouver_context": base_metadata["maneouver_metadata"],
        "file_acoustic_summary": base_metadata["training_data_metadata"]["acoustic_summary"],
        "channel_metadata": channel_summary,
        "processed_view": {
            "aligned_from_bottom": True,
            "height_above_bottom_m": extra.get("height_above_bottom_m"),
            "bottom_alignment_source": extra.get("bottom_alignment_source"),
            "image_geometry": extra.get("image_geometry"),
        },
    }


# ============================================================
# PALETA Y PROCESAMIENTO VISUAL - tomado de extrae_info_p_yolo.ipynb
# ============================================================


def make_ecosonda_cmap_oscura():
    colores = [
        "#000000", "#03152b", "#083b73", "#0f6fc6", "#14b7d6",
        "#7ccf3a", "#d7d700", "#ff8c00", "#d40000", "#5a0000",
    ]
    cmap = LinearSegmentedColormap.from_list("ecosonda_oscura", colores, N=256)
    cmap.set_bad("black")
    return cmap


fish_cmap_oscura = make_ecosonda_cmap_oscura()


def media_movil_1d(x, ventana=9):
    x = np.asarray(x, dtype=np.float32)
    if ventana < 3:
        return x
    if ventana % 2 == 0:
        ventana += 1
    pad = ventana // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(ventana, dtype=np.float32) / ventana
    return np.convolve(xpad, kernel, mode="valid")


def suavizar_fondo(fondo, ventana=21):
    fondo = np.asarray(fondo, dtype=np.float32)
    if ventana < 3:
        return fondo
    return media_movil_1d(fondo, ventana=ventana)


def interpolar_nans_1d(y):
    y = np.asarray(y, dtype=np.float32)
    x = np.arange(y.size)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return None
    return np.interp(x, x[mask], y[mask]).astype(np.float32)


def obtener_ventana_valida(perfil, min_range_ratio=0.55, max_range_ratio=0.90):
    perfil = np.asarray(perfil, dtype=np.float32)
    n = perfil.size
    r0 = int(min_range_ratio * n)
    r1 = int(max_range_ratio * n)
    r0 = max(0, min(r0, n - 2))
    r1 = max(r0 + 2, min(r1, n))
    tramo = perfil[r0:r1]
    tramo_interp = interpolar_nans_1d(tramo)
    return r0, r1, tramo_interp


def detectar_fondo_metodo_a(Sv, min_range_ratio=0.55, max_range_ratio=0.90, suavizado_perfil=9, max_salto=50):
    Sv = np.asarray(Sv, dtype=np.float32)
    n_ping, n_range = Sv.shape
    fondo = np.zeros(n_ping, dtype=np.int32)
    fondo_prev = None

    for i in range(n_ping):
        perfil = Sv[i, :]
        r0, r1, tramo_interp = obtener_ventana_valida(perfil, min_range_ratio, max_range_ratio)
        if tramo_interp is None:
            fondo_i = fondo_prev if fondo_prev is not None else int(0.75 * n_range)
            fondo[i] = fondo_i
            fondo_prev = fondo_i
            continue
        tramo_suave = media_movil_1d(tramo_interp, ventana=suavizado_perfil)
        grad = np.diff(tramo_suave)
        if grad.size < 3:
            fondo_i = fondo_prev if fondo_prev is not None else int(0.75 * n_range)
        else:
            j = int(np.argmax(grad)) + 1
            fondo_i = r0 + j
        if fondo_prev is not None and abs(fondo_i - fondo_prev) > max_salto:
            lo = max(r0, fondo_prev - max_salto)
            hi = min(r1, fondo_prev + max_salto)
            i0 = max(0, lo - r0)
            i1 = max(0, hi - r0)
            grad_local = grad[i0:i1] if grad.size else np.array([])
            fondo_i = r0 + i0 + int(np.argmax(grad_local)) + 1 if grad_local.size > 0 else fondo_prev
        fondo[i] = fondo_i
        fondo_prev = fondo_i
    return fondo


def detectar_fondo_metodo_b(Sv, min_range_ratio=0.55, max_range_ratio=0.90, suavizado_perfil=9, max_salto=50):
    Sv = np.asarray(Sv, dtype=np.float32)
    n_ping, n_range = Sv.shape
    fondo = np.zeros(n_ping, dtype=np.int32)
    fondo_prev = None

    for i in range(n_ping):
        perfil = Sv[i, :]
        r0, r1, tramo_interp = obtener_ventana_valida(perfil, min_range_ratio, max_range_ratio)
        if tramo_interp is None:
            fondo_i = fondo_prev if fondo_prev is not None else int(0.75 * n_range)
            fondo[i] = fondo_i
            fondo_prev = fondo_i
            continue
        tramo_suave = media_movil_1d(tramo_interp, ventana=suavizado_perfil)
        grad = np.diff(tramo_suave)
        if grad.size < 3:
            fondo_i = fondo_prev if fondo_prev is not None else int(0.75 * n_range)
        else:
            gmax = np.max(grad)
            if not np.isfinite(gmax) or gmax <= 0:
                j = int(np.argmax(tramo_suave))
            else:
                candidatos = np.where(grad >= 0.5 * gmax)[0]
                j = int(candidatos[0]) + 1 if candidatos.size else int(np.argmax(grad)) + 1
            fondo_i = r0 + j
        if fondo_prev is not None and abs(fondo_i - fondo_prev) > max_salto:
            lo = max(r0, fondo_prev - max_salto)
            hi = min(r1, fondo_prev + max_salto)
            i0 = max(0, lo - r0)
            i1 = max(0, hi - r0)
            grad_local = grad[i0:i1] if grad.size else np.array([])
            fondo_i = r0 + i0 + int(np.argmax(grad_local)) + 1 if grad_local.size > 0 else fondo_prev
        fondo[i] = fondo_i
        fondo_prev = fondo_i
    return fondo


def calcular_metros_por_sample(channels, ch_idx):
    vals = []
    for p in channels[ch_idx]["Parameters"]:
        try:
            depth = float(p.get("Depth", np.nan))
            depth_samples = float(p.get("Depth_In_Samples", np.nan))
        except Exception:
            continue
        if np.isfinite(depth) and np.isfinite(depth_samples) and depth_samples > 0:
            vals.append(depth / depth_samples)
    if len(vals) == 0:
        # fallback: usar SamplesXMeter si está
        sxm = obtener_parametro_representativo(channels, ch_idx, "SamplesXMeter")
        if np.isfinite(sxm) and sxm > 0:
            return 1.0 / sxm
        raise ValueError(f"No pude calcular metros/sample para CH{ch_idx+1}")
    return float(np.nanmedian(vals))


def obtener_matriz_desde_stack(ch_idx, STACK, channels=None):
    if STACK is None or ch_idx >= len(STACK):
        return None
    if STACK[ch_idx] is None:
        return None
    # Preservar complejos, especialmente S3 / Correlator en modo CHIRP.
    arr = np.asarray(STACK[ch_idx])
    if np.iscomplexobj(arr):
        M = arr.astype(np.complex64, copy=False)
    else:
        M = arr.astype(np.float32, copy=False)
    if M.shape[0] > M.shape[1]:
        M = M.T
    if channels is not None:
        n_real = obtener_parametro_representativo(channels, ch_idx, "Total_Scale_Samples")
        if np.isfinite(n_real):
            n_real = int(n_real)
            if n_real > 0 and n_real < M.shape[1]:
                M = M[:, :n_real]
    return M


def normalizar_percentil(M, p_low=2, p_high=98):
    M = np.asarray(M, dtype=np.float32)
    vals = M[np.isfinite(M)]
    if vals.size == 0:
        return np.zeros_like(M, dtype=np.float32)
    vmin = np.nanpercentile(vals, p_low)
    vmax = np.nanpercentile(vals, p_high)
    if vmax <= vmin:
        return np.zeros_like(M, dtype=np.float32)
    return np.clip((M - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


def preparar_imagen_auto(M, suavizar=False, quitar_luvia=True):
    M = np.asarray(M, dtype=np.float32).copy()

    if suavizar and gaussian_filter is not None:
        M = gaussian_filter(M, sigma=(0.25, 0.40))

    if quitar_luvia:
        col_med = np.nanmedian(M, axis=1, keepdims=True)
        M = M - 1.1 * col_med

    vals = M[np.isfinite(M)]
    if vals.size == 0:
        return np.zeros_like(M, dtype=np.float32)

    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med)) + 1e-6
    sigma = 1.4826 * mad
    vmin = med - 0.3 * sigma
    vmax = med + 3.2 * sigma

    if vmax <= vmin:
        return np.zeros_like(M, dtype=np.float32)

    M_norm = np.clip((M - vmin) / (vmax - vmin), 0, 1)
    M_vis = np.log1p(6.0 * M_norm) / np.log1p(6.0)
    M_vis = np.power(M_vis, 2.3)
    return M_vis.astype(np.float32)


def preparar_imagen_auto_usando_referencia(M, M_referencia, suavizar=False, quitar_luvia=True):
    """
    Aplica a M el MISMO criterio visual de preparar_imagen_auto,
    pero calcula vmin/vmax usando M_referencia.

    Uso previsto:
      - M_referencia = imagen recortada/alineada que ya se ve bien.
      - M = imagen FULL sin recorte.

    Así la JPG FULL usa los mismos colores/contraste que la JPG actual.
    """
    M = np.asarray(M, dtype=np.float32).copy()
    M_ref = np.asarray(M_referencia, dtype=np.float32).copy()

    if suavizar and gaussian_filter is not None:
        M = gaussian_filter(M, sigma=(0.25, 0.40))
        M_ref = gaussian_filter(M_ref, sigma=(0.25, 0.40))

    if quitar_luvia:
        col_med = np.nanmedian(M, axis=1, keepdims=True)
        M = M - 1.1 * col_med

        col_med_ref = np.nanmedian(M_ref, axis=1, keepdims=True)
        M_ref = M_ref - 1.1 * col_med_ref

    vals = M_ref[np.isfinite(M_ref)]
    if vals.size == 0:
        return np.zeros_like(M, dtype=np.float32)

    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med)) + 1e-6
    sigma = 1.4826 * mad
    vmin = med - 0.3 * sigma
    vmax = med + 3.2 * sigma

    if vmax <= vmin:
        return np.zeros_like(M, dtype=np.float32)

    M_norm = np.clip((M - vmin) / (vmax - vmin), 0, 1)
    M_vis = np.log1p(6.0 * M_norm) / np.log1p(6.0)
    M_vis = np.power(M_vis, 2.3)
    return M_vis.astype(np.float32)


def guardar_full_con_contraste_del_recorte(Sv_full, M_recortada_bottom_up, archivo_salida, contraste):
    """
    Guarda una JPG FULL sin recorte ni alineación, usando los colores/contraste
    calculados desde la misma imagen recortada de referencia.

    IMPORTANTE: no elimina fondo, no alinea y no recorta.
    """
    if Sv_full is None or Sv_full.size == 0:
        print("  matriz FULL vacía")
        return
    if M_recortada_bottom_up is None or M_recortada_bottom_up.size == 0:
        print("  matriz de referencia recortada vacía")
        return

    M_ref_plot = M_recortada_bottom_up[:, ::-1]
    M_full_plot = np.asarray(Sv_full, dtype=np.float32).copy()

    M_vis_full = preparar_imagen_auto_usando_referencia(
        M_full_plot,
        M_ref_plot,
        suavizar=contraste.get("suavizar", False),
        quitar_luvia=True,
    )

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(
        M_vis_full.T,
        aspect="auto",
        origin="upper",
        cmap=fish_cmap_oscura,
        vmin=0,
        vmax=1,
    )
    ax.axis("off")
    plt.savefig(archivo_salida, dpi=dpi_salida, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def quitar_fondo_en_full(Sv_full, fondo, margen_samples=0):
    """
    Devuelve una copia de la imagen FULL, pero con el fondo eliminado.

    Sv_full tiene forma (pings, samples/rango). Para cada ping se toma
    fondo[i] como posición del fondo y se ponen en NaN todas las muestras
    desde ese índice hacia abajo.

    margen_samples > 0 elimina también algunas muestras por arriba del fondo,
    útil si queda una línea brillante residual del eco de fondo.
    """
    M = np.asarray(Sv_full, dtype=np.float32).copy()
    fondo = np.asarray(fondo, dtype=np.float32)

    n_ping, n_range = M.shape
    n = min(n_ping, fondo.size)
    margen_samples = int(max(0, margen_samples))

    for i in range(n):
        if not np.isfinite(fondo[i]):
            continue
        fi = int(round(float(fondo[i])))
        fi = max(0, min(fi, n_range - 1))
        inicio = max(0, fi - margen_samples)
        M[i, inicio:] = np.nan

    return M


def guardar_full_sin_fondo_con_contraste_del_recorte(
    M_sin_fondo_bottom_up,
    altura_m,
    M_recortada_bottom_up,
    archivo_salida,
    contraste,
):
    """
    Guarda la JPG SIN_FONDO recortada desde el fondo hacia arriba.

    A diferencia de la versión anterior, NO deja una franja negra donde estaba
    el fondo. La imagen arranca en el fondo detectado/alineado y llega hacia
    arriba hasta la superficie disponible.

    Usa el mismo criterio de colores/contraste que la JPG actual, tomando como
    referencia la imagen recortada M_recortada_bottom_up.
    """
    if M_sin_fondo_bottom_up is None or M_sin_fondo_bottom_up.size == 0:
        print("  matriz SIN_FONDO vacía")
        return
    if M_recortada_bottom_up is None or M_recortada_bottom_up.size == 0:
        print("  matriz de referencia recortada vacía")
        return

    M_ref_plot = M_recortada_bottom_up[:, ::-1]
    M_sin_plot = M_sin_fondo_bottom_up[:, ::-1]

    M_vis = preparar_imagen_auto_usando_referencia(
        M_sin_plot,
        M_ref_plot,
        suavizar=contraste.get("suavizar", False),
        quitar_luvia=True,
    )

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(
        M_vis.T,
        aspect="auto",
        origin="upper",
        cmap=fish_cmap_oscura,
        vmin=0,
        vmax=1,
        extent=[0, M_sin_plot.shape[0], altura_m[-1], 0],
    )
    ax.axis("off")
    plt.savefig(archivo_salida, dpi=dpi_salida, bbox_inches="tight", pad_inches=0)
    plt.close(fig)




def ajustar_matriz_a_pixeles_de_jpg(M, jpg_path):
    """
    Ajusta una matriz (pings, samples) al tamaño real en pixeles
    de la JPG ya generada, SIN tocar la JPG.

    Esto sirve para que el .npy asociado tenga correspondencia directa
    con la imagen _SIN_FONDO.jpg:
        - eje 0 del npy = ancho de la JPG (pings/pixeles horizontales)
        - eje 1 del npy = alto de la JPG (samples/pixeles verticales)

    No cambia la sección física: toma la misma matriz ya alineada/recortada
    que se usó para graficar la JPG _SIN_FONDO y la remuestrea por vecino
    más cercano al tamaño final con el que matplotlib guardó la imagen.
    """
    M_arr = np.asarray(M)
    if np.iscomplexobj(M_arr):
        M = M_arr.astype(np.complex64, copy=False)
    else:
        M = M_arr.astype(np.float32, copy=False)
    jpg_path = Path(jpg_path)

    if M.ndim != 2:
        raise ValueError(f"La matriz a guardar debe ser 2D, pero tiene shape {M.shape}")

    try:
        from PIL import Image
        with Image.open(jpg_path) as img:
            ancho_jpg, alto_jpg = img.size  # PIL devuelve (width, height)
    except Exception as e:
        print(f"ADVERTENCIA: no pude leer tamaño de JPG {jpg_path}. Se guarda matriz sin remuestrear. Motivo: {e}")
        return M

    n_ping, n_samples = M.shape

    if n_ping <= 0 or n_samples <= 0 or ancho_jpg <= 0 or alto_jpg <= 0:
        return M

    # Si ya coincide, no hacemos nada.
    if n_ping == ancho_jpg and n_samples == alto_jpg:
        return M

    # Vecino más cercano: conserva valores de la matriz cruda/alineada,
    # sin interpolar amplitudes entre muestras.
    idx_ping = np.rint(np.linspace(0, n_ping - 1, ancho_jpg)).astype(int)
    idx_samp = np.rint(np.linspace(0, n_samples - 1, alto_jpg)).astype(int)

    idx_ping = np.clip(idx_ping, 0, n_ping - 1)
    idx_samp = np.clip(idx_samp, 0, n_samples - 1)

    M_pix = M[np.ix_(idx_ping, idx_samp)]
    if np.iscomplexobj(M_pix):
        M_pix = M_pix.astype(np.complex64, copy=False)
    else:
        M_pix = M_pix.astype(np.float32, copy=False)
    return M_pix


def alinear_desde_fondo_en_metros(Sv, fondo, metros_por_sample, dz=None, hmax=None):
    # Preserva matrices complejas. Para S3 complejo, interpola real e imaginario
    # por separado y devuelve una matriz complex64.
    Sv_arr = np.asarray(Sv)
    es_complejo = np.iscomplexobj(Sv_arr)
    Sv = Sv_arr.astype(np.complex64 if es_complejo else np.float32, copy=False)

    fondo = np.asarray(fondo, dtype=np.float32)
    n_ping, n_range = Sv.shape
    if dz is None:
        dz = metros_por_sample
    if hmax is None:
        hmax = float(np.nanpercentile(fondo * metros_por_sample, 5))
    if not np.isfinite(hmax) or hmax <= 0:
        raise ValueError("hmax inválido. Revisar fondo detectado.")
    altura_m = np.arange(0, hmax, dz, dtype=np.float32)

    if es_complejo:
        M = np.full((n_ping, len(altura_m)), np.nan + 1j * np.nan, dtype=np.complex64)
    else:
        M = np.full((n_ping, len(altura_m)), np.nan, dtype=np.float32)

    for i in range(n_ping):
        fi = int(np.round(fondo[i]))
        if fi <= 3 or fi >= n_range:
            continue
        perfil = Sv[i, :fi + 1]
        idx = np.arange(fi + 1)
        altura_original = (fi - idx) * metros_por_sample
        x = altura_original[::-1]
        y = perfil[::-1]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue

        if es_complejo:
            yr = np.real(y)
            yi = np.imag(y)
            mr = mask & np.isfinite(yr)
            mi = mask & np.isfinite(yi)
            if mr.sum() >= 3 and mi.sum() >= 3:
                real_interp = np.interp(altura_m, x[mr], yr[mr], left=np.nan, right=np.nan)
                imag_interp = np.interp(altura_m, x[mi], yi[mi], left=np.nan, right=np.nan)
                M[i, :] = real_interp + 1j * imag_interp
        else:
            M[i, :] = np.interp(altura_m, x[mask], y[mask], left=np.nan, right=np.nan)

    return M, altura_m


def generar_virtual_alineado(M1, M2, modo="ponderado"):
    A = normalizar_percentil(M1, 2, 98)
    B = normalizar_percentil(M2, 5, 99)
    if gaussian_filter is not None:
        B = gaussian_filter(B, sigma=(0.6, 0.8))

    if modo == "fusion_inteligente":
        umbral_ch1 = np.nanpercentile(A[np.isfinite(A)], 90)
        A_fuertes = np.where(A > umbral_ch1, A, 0)
        umbral_ch2 = np.nanpercentile(B[np.isfinite(B)], 85)
        B_util = np.where(B > umbral_ch2, B, 0)
        V = A.copy()
        mask_util = B_util > 0
        V[mask_util] += 0.12 * B_util[mask_util]
        V = np.maximum(V, A_fuertes)
    elif modo == "fusion_selectiva":
        umbral_ch2 = np.nanpercentile(B[np.isfinite(B)], 92)
        B_fuerte = np.where(B > umbral_ch2, B, 0)
        V = np.maximum(A, 0.75 * B_fuerte)
    elif modo == "ponderado":
        V = peso_ch1 * A + peso_ch2 * B
    elif modo == "promedio":
        V = 0.5 * A + 0.5 * B
    elif modo == "max":
        V = np.maximum(A, B)
    else:
        raise ValueError(f"modo_virtual no soportado: {modo}")
    return np.clip(V, 0, 1).astype(np.float32)


def guardar_alineado_con_contraste(M_bottom_up, archivo_salida, altura_m, contraste):
    if M_bottom_up is None or M_bottom_up.size == 0:
        print("  matriz vacía")
        return
    M_plot = M_bottom_up[:, ::-1]
    M_vis = preparar_imagen_auto(M_plot, suavizar=contraste.get("suavizar", False), quitar_luvia=True)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(
        M_vis.T,
        aspect="auto",
        origin="upper",
        cmap=fish_cmap_oscura,
        vmin=0,
        vmax=1,
        extent=[0, M_plot.shape[0], altura_m[-1], 0],
    )
    ax.axis("off")
    plt.savefig(archivo_salida, dpi=dpi_salida, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def guardar_virtual_corregido(M_bottom_up, archivo_salida, altura_m):
    M_plot = M_bottom_up[:, ::-1]
    M_vis = np.clip(M_plot, 0, 1)
    M_vis = np.power(M_vis, 1.6)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(
        M_vis.T,
        aspect="auto",
        origin="upper",
        cmap=fish_cmap_oscura,
        vmin=0,
        vmax=1,
        extent=[0, M_plot.shape[0], altura_m[-1], 0],
    )
    ax.axis("off")
    plt.savefig(archivo_salida, dpi=dpi_salida, bbox_inches="tight", pad_inches=0)
    plt.close(fig)




def fondo_desde_depth_in_samples(channels, ch_idx, n_ping, n_range, min_fraccion_valida=0.50):
    """
    Genera el vector de fondo usando Depth_In_Samples de la cabecera de cada ping.

    Esto evita que la detección automática confunda capas/peces con el fondo y genere
    zigzags artificiales. Si faltan algunos valores, se interpolan. Si no hay suficientes
    valores válidos, devuelve None para usar detección automática como fallback.
    """
    params = channels[ch_idx].get("Parameters", [])
    vals = np.full(n_ping, np.nan, dtype=np.float32)

    n = min(n_ping, len(params))
    for i in range(n):
        try:
            ds = float(params[i].get("Depth_In_Samples", np.nan))
        except Exception:
            ds = np.nan

        if np.isfinite(ds) and 0 < ds < n_range:
            vals[i] = ds

    valid = np.isfinite(vals)
    frac_valid = float(valid.sum()) / float(n_ping) if n_ping > 0 else 0.0

    if valid.sum() < 2 or frac_valid < min_fraccion_valida:
        return None

    x = np.arange(n_ping, dtype=np.float32)
    vals_interp = np.interp(x, x[valid], vals[valid]).astype(np.float32)

    # Suavizado leve por si hay saltos aislados en el header.
    vals_interp = suavizar_fondo(vals_interp, ventana=ventana_suavizado)
    vals_interp = np.clip(vals_interp, 1, n_range - 1)

    return vals_interp


def obtener_fondo_para_canal(Sv, channels, ch_idx, detector_fondo):
    """
    Primero intenta usar Depth_In_Samples de la cabecera.
    Si no es válido, recién ahí usa el detector automático A/B.
    """
    n_ping, n_range = Sv.shape

    if usar_depth_in_samples_para_fondo:
        fondo_header = fondo_desde_depth_in_samples(
            channels,
            ch_idx,
            n_ping=n_ping,
            n_range=n_range,
            min_fraccion_valida=min_fraccion_depth_valido,
        )
        if fondo_header is not None:
            return fondo_header, "Depth_In_Samples_header"

    fondo_auto = detector_fondo(
        Sv,
        min_range_ratio=min_range_ratio,
        max_range_ratio=max_range_ratio,
        suavizado_perfil=suavizado_perfil,
        max_salto=max_salto_fondo,
    )
    fondo_auto = suavizar_fondo(fondo_auto, ventana=ventana_suavizado)

    return fondo_auto, f"detector_{metodo_fondo.upper()}"



def imprimir_diagnostico_s3(nombre_fuente, matriz, etapa=""):
    """Imprime diagnóstico de S3/Correlator al cargar/guardar."""
    if "S3" not in str(nombre_fuente).upper() and "CORRELATOR" not in str(nombre_fuente).upper():
        return

    arr = np.asarray(matriz)
    flat = arr.ravel()
    primeros = flat[:10]

    print("\n" + "=" * 70)
    print(f"DIAGNÓSTICO S3 / CORRELATOR {etapa}")
    print("=" * 70)
    print("Nombre:", nombre_fuente)
    print("dtype:", arr.dtype)
    print("shape:", arr.shape)
    print("ndim:", arr.ndim)
    print("size:", arr.size)
    print("¿Es complejo?:", bool(np.iscomplexobj(arr)))
    print("Primeros 10 valores:")
    print(primeros)

    if np.iscomplexobj(arr):
        print("Primeros 10 valores - parte real:")
        print(np.real(primeros))
        print("Primeros 10 valores - parte imaginaria:")
        print(np.imag(primeros))
        print("Primeros 10 valores - magnitud:")
        print(np.abs(primeros))
    print("=" * 70 + "\n")

# ============================================================
# PROCESAMIENTO DE UN ID + JSON
# ============================================================




# ============================================================
# LECTOR JPG ORIGINAL (COPIADO DEL PROGRAMA VIEJO, NO TOCAR)
# Se usa SOLO para generar las JPG exactamente como antes.
# El backend/JSON/NPY sigue usando el lector v5 completo de arriba.
# ============================================================

def leer_cabecera_training_data_desde_pos_JPG(f, path):
    header = {}
    header["Start_Date"] = read_i32(f)
    header["Start_Time"] = read_i32(f)
    header["End_Date"] = read_i32(f)
    header["End_Time"] = read_i32(f)
    header["Pings_Amount"] = read_u32(f)
    header["Channel_Amount"] = read_u32(f)
    header["Max_Samples_On_File_CH1"] = read_u32(f)
    header["Max_Samples_On_File_CH2"] = read_u32(f)
    header["_offset_despues_cabecera"] = f.tell()
    header["_file_size_bytes"] = Path(path).stat().st_size
    return header



def leer_columna_v4_JPG(f, leer_datos=True):
    p = {}
    p["_offset_inicio"] = f.tell()
    p["KeyWord_Header"] = read_text(f, HEADER_SIZE)

    if "HEADER" not in p["KeyWord_Header"]:
        raise ValueError(f"No se encontró HEADER válido en offset {p['_offset_inicio']}")

    p["Maneouver_Ping"] = read_u32(f)
    p["Channel"] = read_i32(f)
    p["Channel_Name"] = f"CH{p['Channel']}"

    p["Transmission_Mode"] = read_u32(f)
    p["Transducer_Index"] = read_u32(f)
    p["TX_Center_Frequency"] = read_i32(f)
    p["TX_Band_Width"] = read_i32(f)
    p["Sampling_Frequency"] = read_i32(f)
    p["Center_Freq_After_FPGA"] = read_i32(f)
    p["Power"] = read_u32(f)
    p["Frequency_Modulation"] = read_u32(f)
    p["Window_RX_Name"] = read_text(f, 8)
    p["Pulse_Width"] = read_u32(f)
    p["SamplesXMeter"] = read_f32(f)
    p["Sounder_Scale"] = read_u32(f)
    p["SubScale_1"] = read_u32(f)
    p["SubScale_2"] = read_u32(f)
    p["Total_Scale_Samples"] = read_u32(f)
    p["Date"] = read_i32(f)
    p["Time"] = read_i32(f)
    p["Latitude"] = read_f32(f)
    p["Card_N_S"] = read_text(f, 4)
    p["Longitude"] = read_f32(f)
    p["Card_E_W"] = read_text(f, 4)
    p["Vessel_Speed"] = read_f32(f)
    p["Vessel_Course"] = read_f32(f)
    p["Vessel_Heading"] = read_f32(f)
    p["Transducer_Temperature"] = read_f32(f)
    p["Net_Temperature"] = read_f32(f)
    p["Ping_Rate"] = read_f32(f)
    p["Depth_In_Samples"] = read_u32(f)
    p["Depth"] = read_f32(f)
    p["SV"] = read_f32(f)
    p["Roughness"] = read_i32(f)
    p["Hardness"] = read_i32(f)
    p["_offset_fin_header"] = f.tell()

    n = int(p["Total_Scale_Samples"])
    if not (1 <= n <= 200000):
        raise ValueError(f"Total_Scale_Samples inválido: {n}")

    p["KeyWord_Data1"] = read_text(f, HEADER_SIZE)
    Data1 = np.fromfile(f, dtype="<i4", count=n) if leer_datos else None
    if not leer_datos:
        f.seek(4 * n, 1)

    p["KeyWord_Data2"] = read_text(f, HEADER_SIZE)
    Data2 = np.fromfile(f, dtype="<i4", count=n) if leer_datos else None
    if not leer_datos:
        f.seek(4 * n, 1)

    p["KeyWord_Data3"] = read_text(f, HEADER_SIZE)
    Data3 = np.fromfile(f, dtype="<i2", count=n) if leer_datos else None
    if not leer_datos:
        f.seek(2 * n, 1)

    p["_offset_fin_bloque"] = f.tell()
    return Data1, Data2, Data3, p




def leer_id_completo_robusto_JPG(path, max_bloques=None, verbose=True):
    """
    Lee todos los bloques HEADER del ID. Si la cabecera general no es válida,
    busca el primer HEADER, como en las pruebas que hicimos.

    Devuelve:
        fp_header, channels, Z, Data1_stack, Data2_stack, bloques

    Donde:
        Z = Data3 / DB DATA WITHOUT TVG, shape por canal: (samples, pings)
        Data1_stack = Data1, shape por canal: (samples, pings)
        Data2_stack = Data2, shape por canal: (samples, pings)
    """
    path = Path(path)

    with open(path, "rb") as f:
        f.seek(0)
        header = leer_cabecera_training_data_desde_pos_JPG(f, path)

        if header_general_valido(header):
            modo = "CABECERA_GENERAL_OK"
            f.seek(header["_offset_despues_cabecera"])
        else:
            modo = "SIN_CABECERA_GENERAL_VALIDA_BUSCANDO_HEADER"
            header["_warning"] = "Cabecera general inválida; se buscó el primer HEADER."
            pos_header = buscar_header(f, start=0)
            if pos_header is None:
                raise ValueError("No se encontró HEADER en el archivo.")
            header["_offset_despues_cabecera"] = None
            header["_offset_primer_header"] = pos_header
            f.seek(pos_header)

        bloques = []
        datos_por_canal: Dict[Any, Dict[str, Any]] = {}

        while True:
            if max_bloques is not None and len(bloques) >= max_bloques:
                break

            pos = f.tell()
            if pos >= header["_file_size_bytes"]:
                break

            probe = f.read(6)
            f.seek(pos)
            if probe != b"HEADER":
                pos_header = buscar_header(f, start=pos, max_bytes=500000)
                if pos_header is None:
                    break
                f.seek(pos_header)

            try:
                d1, d2, d3, p = leer_columna_v4_JPG(f, leer_datos=True)
            except Exception as e:
                print("Error leyendo bloque en offset", f.tell(), ":", e)
                break

            ch = int(p["Channel"])
            if ch not in datos_por_canal:
                datos_por_canal[ch] = {
                    "Data1_list": [],
                    "Data2_list": [],
                    "Data3_list": [],
                    "Parameters": [],
                }

            datos_por_canal[ch]["Data1_list"].append(d1)
            datos_por_canal[ch]["Data2_list"].append(d2)
            datos_por_canal[ch]["Data3_list"].append(d3)
            datos_por_canal[ch]["Parameters"].append(p)
            bloques.append(p)

            # ------------------------------------------------------------
            # Canal virtual REAL ya presente en el archivo
            # ------------------------------------------------------------
            # No se genera ninguna virtual por fusión. Solo se captura si
            # entre este bloque y el próximo HEADER hay datos con keyword
            # que indique virtual.
            pos_fin_bloque = f.tell()
            pos_header_siguiente = buscar_header(f, start=pos_fin_bloque, max_bytes=500000)

            virtuales_reales = intentar_leer_virtual_real_entre_headers(
                path=path,
                pos_ini=pos_fin_bloque,
                pos_fin=pos_header_siguiente,
                n_samples=int(p.get("Total_Scale_Samples", 0)),
                p_ref=p,
            )

            for chv, vd1, vd2, vd3, pv in virtuales_reales:
                if chv not in datos_por_canal:
                    datos_por_canal[chv] = {
                        "Data1_list": [],
                        "Data2_list": [],
                        "Data3_list": [],
                        "Parameters": [],
                    }
                datos_por_canal[chv]["Data1_list"].append(vd1)
                datos_por_canal[chv]["Data2_list"].append(vd2)
                datos_por_canal[chv]["Data3_list"].append(vd3)
                datos_por_canal[chv]["Parameters"].append(pv)

            if pos_header_siguiente is None:
                break
            f.seek(pos_header_siguiente)

        if not bloques:
            raise ValueError("No se pudo leer ningún bloque de datos.")

    # Construir stacks rectangulares por canal: (samples, pings)
    canales_ordenados = sorted(datos_por_canal.keys(), key=lambda x: (isinstance(x, str), str(x)))
    channels = []
    data1_stacks = []
    data2_stacks = []
    data3_stacks = []

    for ch in canales_ordenados:
        info = datos_por_canal[ch]
        max_samples = max(len(x) for x in info["Data3_list"])
        n_pings = len(info["Data3_list"])

        A1 = np.zeros((max_samples, n_pings), dtype=np.int32)
        A2 = np.zeros((max_samples, n_pings), dtype=np.int32)
        A3 = np.zeros((max_samples, n_pings), dtype=np.int16)

        for i in range(n_pings):
            d1 = info["Data1_list"][i]
            d2 = info["Data2_list"][i]
            d3 = info["Data3_list"][i]
            A1[:len(d1), i] = d1
            A2[:len(d2), i] = d2
            A3[:len(d3), i] = d3

        # Diagnóstico al levantar S3 / Correlator desde el .ID
        imprimir_diagnostico_s3(f"{ch}_Data2_S3_CORRELATOR_DETECTOR", A2, etapa="LEVANTADO DEL ID")

        channel_name = ch if isinstance(ch, str) else f"CH{ch}"
        channels.append({
            "Channel": ch,
            "Channel_Name": channel_name,
            "IsVirtual": isinstance(ch, str) and str(ch).upper().startswith("VIRT"),
            "Parameters": info["Parameters"],
        })
        data1_stacks.append(A1)
        data2_stacks.append(A2)
        data3_stacks.append(A3)

    # No apilar canales entre sí: pueden tener distinta cantidad de muestras.
    # Cada entrada de estas listas es una matriz (samples, pings) de un canal.
    Data1_stack = data1_stacks
    Data2_stack = data2_stacks
    Z = data3_stacks

    header["_modo_lectura"] = modo
    header["_bloques_leidos"] = len(bloques)
    header["_canales_detectados"] = canales_ordenados

    if verbose:
        print(f"Modo lectura: {modo}")
        print(f"Bloques leídos: {len(bloques)}")
        print(f"Canales detectados: {canales_ordenados}")
        
        for i, Zch in enumerate(Z):
            print(f"{channels[i]['Channel_Name']} Data3 shape: {Zch.shape} (samples, pings)")

    return header, channels, Z, Data1_stack, Data2_stack, bloques




# ============================================================
# LECTOR ID v6 CORREGIDO SEGÚN MATLAB ORIGINAL
# Reemplaza los lectores viejos manteniendo la misma interfaz:
#   header, channels, Z, Data1_stack, Data2_stack, Data4_stack, bloques
# Matrices en listas por canal con shape (samples, pings), como esperaba el script.
# ============================================================

ECHOSOUNDER_SIB = 1
ECHOSOUNDER_SPB = 2
TRANSMISSION_CHIRP = 1


def _read_exact_v6(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"Se esperaban {n} bytes y se leyeron {len(b)} en offset {f.tell()}.")
    return b


def _read_text_v6(f, n):
    raw = _read_exact_v6(f, n)
    txt = raw.decode("latin1", errors="ignore").replace("\x00", "")
    txt = "".join(c for c in txt if 32 <= ord(c) <= 126)
    return txt.strip()


def _read_i32_v6(f):
    return struct.unpack("<i", _read_exact_v6(f, 4))[0]


def _read_u32_v6(f):
    return struct.unpack("<I", _read_exact_v6(f, 4))[0]


def _read_f32_v6(f):
    return struct.unpack("<f", _read_exact_v6(f, 4))[0]


def leer_cabecera_training_data_v6(f, path):
    """
    Cabecera global v6 según Read_Fish_ID_File_v6.m.
    Tamaño: 48 bytes.
    """
    header = {}
    header["_file_size_bytes"] = Path(path).stat().st_size
    header["Version_Data_Fish_ID"] = _read_f32_v6(f)

    if abs(float(header["Version_Data_Fish_ID"]) - 6.0) > 1e-5:
        raise ValueError(
            f"Versión inválida: se esperaba 6.0 y se leyó {header['Version_Data_Fish_ID']}"
        )

    header["Start_Date"] = _read_i32_v6(f)
    header["Start_Time"] = _read_i32_v6(f)
    header["End_Date"] = _read_i32_v6(f)
    header["End_Time"] = _read_i32_v6(f)
    header["Pings_Amount"] = _read_u32_v6(f)
    header["Sounder_Type"] = _read_u32_v6(f)
    header["Sounder_Type_Description"] = descripcion_sounder(header["Sounder_Type"])
    header["Channel_Amount"] = _read_u32_v6(f)
    header["DB_By_Color_CH1"] = _read_f32_v6(f)
    header["DB_By_Color_CH2"] = _read_f32_v6(f)
    header["Max_Samples_On_File_CH1"] = _read_u32_v6(f)
    header["Max_Samples_On_File_CH2"] = _read_u32_v6(f)
    header["_offset_despues_cabecera"] = f.tell()
    header["_header_variant"] = "v6_matlab_original"

    if header["Sounder_Type"] not in (ECHOSOUNDER_SIB, ECHOSOUNDER_SPB):
        raise ValueError(f"Sounder_Type no soportado: {header['Sounder_Type']}")
    if not (1 <= header["Channel_Amount"] <= 16):
        raise ValueError(f"Channel_Amount inválido: {header['Channel_Amount']}")
    if not (1 <= header["Pings_Amount"] <= 1_000_000):
        raise ValueError(f"Pings_Amount inválido: {header['Pings_Amount']}")

    return header


def leer_data_header_v6(f):
    """Header por columna/ping según Read_Data_Header_From_File_v6.m."""
    p = {}
    p["_offset_inicio"] = f.tell()
    p["KeyWord_Header"] = _read_text_v6(f, HEADER_SIZE)

    if "HEADER" not in p["KeyWord_Header"]:
        raise ValueError(
            f"No se encontró HEADER válido en offset {p['_offset_inicio']}. "
            f"Leído: {p['KeyWord_Header']!r}"
        )

    p["Maneouver_Ping"] = _read_u32_v6(f)
    p["Channel"] = _read_i32_v6(f)
    p["Channel_Name"] = f"CH{p['Channel']}"
    p["Transmission_Mode"] = _read_u32_v6(f)
    p["Transducer_Name"] = _read_text_v6(f, 100)
    p["Transducer_Beam_Angle_3dB_Port_Starboard"] = _read_f32_v6(f)
    p["Transducer_Beam_Angle_3dB_Fore_Aft"] = _read_f32_v6(f)
    p["Beam_1"] = p["Transducer_Beam_Angle_3dB_Port_Starboard"]
    p["Beam_2"] = p["Transducer_Beam_Angle_3dB_Fore_Aft"]
    p["TX_Center_Frequency"] = _read_i32_v6(f)
    p["TX_Band_Width"] = _read_i32_v6(f)
    p["Sampling_Frequency"] = _read_i32_v6(f)
    p["Center_Freq_After_FPGA"] = _read_i32_v6(f)
    p["Power"] = _read_u32_v6(f)
    p["Frequency_Modulation"] = _read_u32_v6(f)
    p["Window_RX_Name"] = _read_text_v6(f, 8)
    p["Pulse_Width"] = _read_u32_v6(f)
    p["SamplesXMeter"] = _read_f32_v6(f)
    p["Sounder_Scale"] = _read_u32_v6(f)
    p["SubScale_1"] = _read_u32_v6(f)
    p["SubScale_2"] = _read_u32_v6(f)
    p["Total_Scale_Samples"] = _read_u32_v6(f)
    p["Date"] = _read_i32_v6(f)
    p["Time"] = _read_i32_v6(f)
    p["Latitude"] = _read_f32_v6(f)
    p["Card_N_S"] = _read_text_v6(f, 4)
    p["Longitude"] = _read_f32_v6(f)
    p["Card_E_W"] = _read_text_v6(f, 4)
    p["Vessel_Speed"] = _read_f32_v6(f)
    p["Vessel_Course"] = _read_f32_v6(f)
    p["Vessel_Heading"] = _read_f32_v6(f)
    p["Transducer_Temperature"] = _read_f32_v6(f)
    p["Net_Temperature"] = _read_f32_v6(f)
    p["Ping_Rate"] = _read_f32_v6(f)
    p["Depth_In_Samples"] = _read_u32_v6(f)
    p["Depth"] = _read_f32_v6(f)
    p["SV"] = _read_f32_v6(f)
    p["Roughness"] = _read_i32_v6(f)
    p["Hardness"] = _read_i32_v6(f)
    p["_offset_fin_header"] = f.tell()

    n = int(p["Total_Scale_Samples"])
    if not (1 <= n <= 1_000_000):
        raise ValueError(f"Total_Scale_Samples inválido: {n} en offset {p['_offset_inicio']}")

    return p


def _read_np_v6(f, dtype, count):
    arr = np.fromfile(f, dtype=np.dtype(dtype), count=int(count))
    if arr.size != int(count):
        raise EOFError(f"No se pudieron leer {count} elementos {dtype}. Leídos: {arr.size}")
    return arr


def leer_canales_reales_v6(f, n, sounder_type, transmission_mode):
    """Equivalente a Read_Real_Channels_From_File_Column_By_Column_v6.m."""
    pkw = {}

    pkw["KeyWord_Data1"] = _read_text_v6(f, HEADER_SIZE)
    Stage2 = _read_np_v6(f, "<i4", n)

    pkw["KeyWord_Data2"] = _read_text_v6(f, HEADER_SIZE)
    if int(transmission_mode) == TRANSMISSION_CHIRP:
        raw = _read_np_v6(f, "<f4", 2 * int(n))
        # S3 / Correlator en modo CHIRP viene como complejo:
        # real_0, imag_0, real_1, imag_1, ...
        # IMPORTANTE: no se guarda la magnitud acá. Se conserva complejo.
        Stage3 = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
        pkw["Data2_is_complex"] = True
        pkw["Data2_raw_layout"] = "interleaved float32: real, imag"
    else:
        Stage3 = _read_np_v6(f, "<i4", n)
        pkw["Data2_is_complex"] = False
        pkw["Data2_raw_layout"] = "int32 real"

    pkw["KeyWord_Data3"] = _read_text_v6(f, HEADER_SIZE)
    Stage5 = _read_np_v6(f, "<i2", n)

    Stage7 = None
    pkw["KeyWord_Data4"] = None
    if int(sounder_type) != ECHOSOUNDER_SPB:
        pkw["KeyWord_Data4"] = _read_text_v6(f, HEADER_SIZE)
        Stage7 = _read_np_v6(f, "<i2", n)

    return Stage2, Stage3, Stage5, Stage7, pkw


def leer_canal_virtual_v6(f, n):
    """Equivalente a Read_Virtual_Channels_From_File_Column_By_Column_v6.m."""
    pkw = {}
    pkw["KeyWord_Virtual_Stage5"] = _read_text_v6(f, HEADER_SIZE)
    Virtual_Stage5 = _read_np_v6(f, "<i2", n)
    pkw["KeyWord_Virtual_Stage7"] = _read_text_v6(f, HEADER_SIZE)
    Virtual_Stage7 = _read_np_v6(f, "<i2", n)
    return Virtual_Stage5, Virtual_Stage7, pkw


def _keyword_parece_dato_virtual_v6(txt):
    """
    Devuelve True si el texto de 24 bytes parece un keyword real de datos
    de canal virtual guardado en el archivo. No se usa para inventar nada.
    """
    t = (txt or "").upper().strip()
    if not t or "HEADER" in t:
        return False
    tokens = ["DB", "DATA", "TVG", "WITHOUT", "VIRTUAL", "VIRT", "STAGE"]
    return any(tok in t for tok in tokens)


def hay_canal_virtual_explicito_v6(f, n):
    """
    Mira sin avanzar el archivo. En SPB, algunos ID traen después de los
    canales reales dos bloques virtuales: keyword + int16[n] y keyword + int16[n].

    Regla importante:
      - Si el próximo bloque es HEADER, NO hay virtual.
      - Si no hay dos keywords plausibles de datos, NO hay virtual.
      - Nunca se genera/fusiona un virtual: solo se lee si está físicamente escrito.
    """
    pos = f.tell()
    try:
        kw1 = _read_text_v6(f, HEADER_SIZE)
        if not _keyword_parece_dato_virtual_v6(kw1):
            return False, kw1, None

        # saltar matriz virtual Stage5 tentativa
        f.seek(2 * int(n), 1)

        kw2 = _read_text_v6(f, HEADER_SIZE)
        if not _keyword_parece_dato_virtual_v6(kw2):
            return False, kw1, kw2

        return True, kw1, kw2
    except Exception:
        return False, None, None
    finally:
        f.seek(pos)


def _agregar_ping(datos_por_canal, ch, d1, d2, d3, d4, p):
    if ch not in datos_por_canal:
        datos_por_canal[ch] = {
            "Data1_list": [],
            "Data2_list": [],
            "Data3_list": [],
            "Data4_list": [],
            "Parameters": [],
        }
    datos_por_canal[ch]["Data1_list"].append(d1)
    datos_por_canal[ch]["Data2_list"].append(d2)
    datos_por_canal[ch]["Data3_list"].append(d3)
    datos_por_canal[ch]["Data4_list"].append(d4)
    datos_por_canal[ch]["Parameters"].append(p)


def _stack_por_canal(datos_por_canal):
    canales_ordenados = sorted(datos_por_canal.keys(), key=lambda x: (isinstance(x, str), str(x)))
    channels = []
    data1_stacks, data2_stacks, data3_stacks, data4_stacks = [], [], [], []

    for ch in canales_ordenados:
        info = datos_por_canal[ch]
        max_samples = max(len(x) for x in info["Data3_list"])
        n_pings = len(info["Data3_list"])

        # Dtype de Data2: normalmente int32; puede ser float32 si fue magnitud de chirp.
        dtype_d2 = np.result_type(*[np.asarray(x).dtype for x in info["Data2_list"]])

        A1 = np.zeros((max_samples, n_pings), dtype=np.int32)
        A2 = np.zeros((max_samples, n_pings), dtype=dtype_d2)
        A3 = np.zeros((max_samples, n_pings), dtype=np.int16)
        tiene_data4 = any(x is not None for x in info.get("Data4_list", []))
        A4 = np.zeros((max_samples, n_pings), dtype=np.int16) if tiene_data4 else None

        for i in range(n_pings):
            d1 = info["Data1_list"][i]
            d2 = info["Data2_list"][i]
            d3 = info["Data3_list"][i]
            d4 = info["Data4_list"][i]
            A1[:len(d1), i] = d1
            A2[:len(d2), i] = d2
            A3[:len(d3), i] = d3
            if A4 is not None and d4 is not None:
                A4[:len(d4), i] = d4

        # Diagnóstico al levantar S3 / Correlator desde el .ID
        imprimir_diagnostico_s3(f"{ch}_Data2_S3_CORRELATOR_DETECTOR", A2, etapa="LEVANTADO DEL ID")

        channel_name = ch if isinstance(ch, str) else f"CH{ch}"
        channels.append({
            "Channel": ch,
            "Channel_Name": channel_name,
            "IsVirtual": isinstance(ch, str) and str(ch).upper().startswith("VIRT"),
            "Parameters": info["Parameters"],
        })
        data1_stacks.append(A1)
        data2_stacks.append(A2)
        data3_stacks.append(A3)
        data4_stacks.append(A4)

    return canales_ordenados, channels, data3_stacks, data1_stacks, data2_stacks, data4_stacks


def leer_id_completo_robusto(path, max_bloques=None, verbose=True):
    """
    Lector principal corregido para Fish ID v6.

    Devuelve lo mismo que esperaba el resto del programa:
      header, channels, Z, Data1_stack, Data2_stack, Data4_stack, bloques
    """
    path = Path(path)
    datos_por_canal = {}
    bloques = []

    with open(path, "rb") as f:
        header = leer_cabecera_training_data_v6(f, path)
        n_pings_total = int(header["Pings_Amount"])
        n_channels = int(header["Channel_Amount"])
        sounder_type = int(header["Sounder_Type"])
        n_pings = n_pings_total if max_bloques is None else min(n_pings_total, int(max_bloques))

        if verbose:
            print("Modo lectura: v6_matlab_original")
            print("Versión:", header["Version_Data_Fish_ID"])
            print("Pings declarados:", n_pings_total)
            print("Pings a leer:", n_pings)
            print("Sounder_Type:", sounder_type, descripcion_sounder(sounder_type))
            print("Canales reales declarados:", n_channels)
            if sounder_type == ECHOSOUNDER_SPB:
                print("Canal virtual: se leerá solo si aparece explícitamente en el archivo")
            else:
                print("Canal virtual: NO")
            print("Max_Samples CH1:", header["Max_Samples_On_File_CH1"])
            print("Max_Samples CH2:", header["Max_Samples_On_File_CH2"])

        if sounder_type == ECHOSOUNDER_SIB:
            # Single beam / Size Indicator: header + datos para cada ping/canal.
            for i in range(n_pings):
                for _channel_loop in range(n_channels):
                    p = leer_data_header_v6(f)
                    n = int(p["Total_Scale_Samples"])
                    d1, d2, d3, d4, kw = leer_canales_reales_v6(
                        f, n, sounder_type, p["Transmission_Mode"]
                    )
                    p.update(kw)
                    p["_offset_fin_bloque"] = f.tell()
                    ch = int(p["Channel"])
                    _agregar_ping(datos_por_canal, ch, d1, d2, d3, d4, p)
                    bloques.append(p)

        elif sounder_type == ECHOSOUNDER_SPB:
            # Split beam: un header por ping, luego todos los canales reales y luego virtual S5/S7.
            for i in range(n_pings):
                p_base = leer_data_header_v6(f)
                n = int(p_base["Total_Scale_Samples"])

                for ch_idx in range(1, n_channels + 1):
                    d1, d2, d3, d4, kw = leer_canales_reales_v6(
                        f, n, sounder_type, p_base["Transmission_Mode"]
                    )
                    p = dict(p_base)
                    p.update(kw)
                    p["Channel"] = ch_idx
                    p["Channel_Name"] = f"CH{ch_idx}"
                    p["_offset_fin_bloque"] = f.tell()
                    _agregar_ping(datos_por_canal, ch_idx, d1, d2, d3, d4, p)
                    bloques.append(p)

                # Canal virtual: NO se inventa ni se calcula.
                # Solo se lee si el archivo lo trae explícitamente después
                # de los canales reales, como keyword + int16[n] + keyword + int16[n].
                hay_virtual, kwv1, kwv2 = hay_canal_virtual_explicito_v6(f, n)

                if hay_virtual:
                    v5, v7, vkw = leer_canal_virtual_v6(f, n)
                    p_v5 = dict(p_base)
                    p_v5.update(vkw)
                    p_v5["Channel"] = "VIRT_STAGE5"
                    p_v5["Channel_Name"] = "VIRT_STAGE5"
                    p_v5["IsVirtual"] = True
                    p_v5["KeyWord_Data3"] = vkw.get("KeyWord_Virtual_Stage5")
                    zeros = np.zeros(n, dtype=np.int32)
                    _agregar_ping(datos_por_canal, "VIRT_STAGE5", zeros, zeros, v5, None, p_v5)

                    p_v7 = dict(p_base)
                    p_v7.update(vkw)
                    p_v7["Channel"] = "VIRT_STAGE7"
                    p_v7["Channel_Name"] = "VIRT_STAGE7"
                    p_v7["IsVirtual"] = True
                    p_v7["KeyWord_Data3"] = vkw.get("KeyWord_Virtual_Stage7")
                    v7_scaled = (v7.astype(np.float32) * float(header.get("DB_By_Color_CH1", 1.0))).astype(np.int16)
                    _agregar_ping(datos_por_canal, "VIRT_STAGE7", zeros.copy(), zeros.copy(), v7_scaled, None, p_v7)

        else:
            raise ValueError(f"Sounder_Type no soportado: {sounder_type}")

    if not bloques:
        raise ValueError("No se pudo leer ningún bloque de datos.")

    canales_ordenados, channels, Z, Data1_stack, Data2_stack, Data4_stack = _stack_por_canal(datos_por_canal)

    header["_modo_lectura"] = "v6_matlab_original"
    header["_bloques_leidos"] = len(bloques)
    header["_canales_detectados"] = canales_ordenados
    header["_tiene_canal_virtual"] = any(
        isinstance(ch, str) and ch.upper().startswith("VIRT") for ch in canales_ordenados
    )

    if verbose:
        print("Bloques leídos:", len(bloques))
        print("Canales detectados:", canales_ordenados)
        for i, Zch in enumerate(Z):
            print(f"{channels[i]['Channel_Name']} Data3/Stage5 shape: {Zch.shape} (samples, pings)")

    return header, channels, Z, Data1_stack, Data2_stack, Data4_stack, bloques


# Para las JPG ya no usamos el lector viejo, porque estaba desalineado con v6.
# Mantenemos esta función solo para que procesar_id_con_json no tenga que cambiar.
def leer_id_completo_robusto_JPG(path, max_bloques=None, verbose=True):
    return leer_id_completo_robusto(path, max_bloques=max_bloques, verbose=verbose)

# ============================================================
# METADATA ÚTIL PARA JSON
# ============================================================





# ============================================================
# Lector dedicado SplitBeam: CH1-CH4 + canal virtual
# ============================================================

def _sb_u32(b,o): return struct.unpack_from("<I", b, o)[0]
def _sb_i32(b,o): return struct.unpack_from("<i", b, o)[0]
def _sb_f32(b,o): return struct.unpack_from("<f", b, o)[0]

def _sb_txt(b,o,n):
    raw = b[o:o+n]
    s = raw.decode("latin1", errors="ignore").replace("\x00", "")
    return "".join(c for c in s if 32 <= ord(c) <= 126).strip()

def _sb_buscar_headers(data: bytes):
    hs = []
    start = 0
    while True:
        i = data.find(b"HEADER", start)
        if i < 0:
            break
        hs.append(i)
        start = i + 6
    return hs

def _sb_leer_header_ping(data: bytes, base: int):
    o = base
    p = {}
    p["KeyWord_Header"] = _sb_txt(data, o, HEADER_SIZE); o += HEADER_SIZE
    p["Maneouver_Ping"] = _sb_u32(data, o); o += 4
    p["Channel"] = _sb_i32(data, o); o += 4
    p["Transmission_Mode"] = _sb_u32(data, o); o += 4
    p["Transducer_Name"] = _sb_txt(data, o, 100); o += 100
    p["Beam_1"] = _sb_f32(data, o); o += 4
    p["Beam_2"] = _sb_f32(data, o); o += 4
    p["TX_Center_Frequency"] = _sb_i32(data, o); o += 4
    p["TX_Band_Width"] = _sb_i32(data, o); o += 4
    p["Sampling_Frequency"] = _sb_i32(data, o); o += 4
    p["Center_Freq_After_FPGA"] = _sb_i32(data, o); o += 4
    p["Power"] = _sb_u32(data, o); o += 4
    p["Frequency_Modulation"] = _sb_u32(data, o); o += 4
    p["Window_RX_Name"] = _sb_txt(data, o, 8); o += 8
    p["Pulse_Width"] = _sb_u32(data, o); o += 4
    p["SamplesXMeter"] = _sb_f32(data, o); o += 4
    p["Sounder_Scale"] = _sb_u32(data, o); o += 4
    p["SubScale_1"] = _sb_u32(data, o); o += 4
    p["SubScale_2"] = _sb_u32(data, o); o += 4
    p["Total_Scale_Samples"] = _sb_u32(data, o); o += 4
    p["Date"] = _sb_i32(data, o); o += 4
    p["Time"] = _sb_i32(data, o); o += 4
    p["Latitude"] = _sb_f32(data, o); o += 4
    p["Card_N_S"] = _sb_txt(data, o, 4); o += 4
    p["Longitude"] = _sb_f32(data, o); o += 4
    p["Card_E_W"] = _sb_txt(data, o, 4); o += 4
    p["Vessel_Speed"] = _sb_f32(data, o); o += 4
    p["Vessel_Course"] = _sb_f32(data, o); o += 4
    p["Vessel_Heading"] = _sb_f32(data, o); o += 4
    p["Transducer_Temperature"] = _sb_f32(data, o); o += 4
    p["Net_Temperature"] = _sb_f32(data, o); o += 4
    p["Ping_Rate"] = _sb_f32(data, o); o += 4
    p["Depth_In_Samples"] = _sb_u32(data, o); o += 4
    p["Depth"] = _sb_f32(data, o); o += 4
    p["SV"] = _sb_f32(data, o); o += 4
    p["Roughness"] = _sb_i32(data, o); o += 4
    p["Hardness"] = _sb_i32(data, o); o += 4
    p["_header_len"] = o - base
    return p


def _sb_stat_array(vals):
    """Devuelve min/median/max y cantidad de valores válidos para diagnóstico."""
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": None, "min": None, "max": None, "n_valid": 0, "n_total": int(len(vals))}
    return {
        "median": float(np.nanmedian(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "n_valid": int(arr.size),
        "n_total": int(len(vals)),
    }

def _sb_diagnosticar_navegacion(data: bytes, headers: list, max_pings_diag: int = 200):
    """
    Diagnóstico específico de los campos de navegación del header SplitBeam.

    Sirve para distinguir dos casos:
    1) El archivo realmente trae Vessel_Speed/Vessel_Course = -1.
    2) El lector está corrido de offset.

    En este layout confirmado, Vessel_Speed está a offset relativo 224 bytes
    desde el comienzo de HEADER y Vessel_Course a 228 bytes.
    """
    n = int(min(len(headers), max_pings_diag))
    params_diag = []
    for h in headers[:n]:
        try:
            params_diag.append(_sb_leer_header_ping(data, h))
        except Exception:
            pass

    def col(name):
        out = []
        for p in params_diag:
            try:
                out.append(float(p.get(name, np.nan)))
            except Exception:
                out.append(np.nan)
        return out

    speed = col("Vessel_Speed")
    course = col("Vessel_Course")
    heading = col("Vessel_Heading")
    pingrate = col("Ping_Rate")
    depth = col("Depth")
    depth_s = col("Depth_In_Samples")

    diag = {
        "n_pings_checked": int(len(params_diag)),
        "vessel_speed_knots": _sb_stat_array(speed),
        "vessel_course_deg": _sb_stat_array(course),
        "vessel_heading_deg": _sb_stat_array(heading),
        "ping_rate_hz": _sb_stat_array(pingrate),
        "depth_m": _sb_stat_array(depth),
        "depth_in_samples": _sb_stat_array(depth_s),
        "speed_all_minus_one": bool(len(speed) > 0 and np.all(np.asarray(speed, dtype=float) == -1.0)),
        "course_all_minus_one": bool(len(course) > 0 and np.all(np.asarray(course, dtype=float) == -1.0)),
        "interpretation": None,
    }

    if diag["speed_all_minus_one"] and diag["course_all_minus_one"]:
        diag["interpretation"] = (
            "El archivo trae Vessel_Speed y Vessel_Course en -1 para los pings revisados. "
            "En estos datos -1 debe interpretarse como dato de navegación no disponible, "
            "no como velocidad real negativa."
        )
    else:
        diag["interpretation"] = (
            "Hay valores distintos de -1 en Vessel_Speed/Vessel_Course. "
            "Si son absurdos, revisar offset; si son razonables, usar esos valores."
        )

    return diag

def _sb_imprimir_diagnostico_navegacion(diag: dict):
    print("\n--- DIAGNÓSTICO NAVEGACIÓN SPLITBEAM ---")
    print("Pings revisados:", diag.get("n_pings_checked"))
    for k in ["vessel_speed_knots", "vessel_course_deg", "vessel_heading_deg", "ping_rate_hz", "depth_m", "depth_in_samples"]:
        st = diag.get(k, {})
        print(f"{k}: median={st.get('median')} min={st.get('min')} max={st.get('max')} n={st.get('n_valid')}/{st.get('n_total')}")
    print("Interpretación:", diag.get("interpretation"))
    print("----------------------------------------\n")

def _sb_i16(data, off, n):
    return np.frombuffer(data, dtype="<i2", count=int(n), offset=int(off)).copy()

def _sb_i32_arr(data, off, n):
    return np.frombuffer(data, dtype="<i4", count=int(n), offset=int(off)).copy()

def _sb_s3_complex(data, off, n):
    # CORREGIDO: en Fish ID v6.0, para CHIRP el S3/Correlator Data viene como single/float32
    # intercalado real, imag. Esto coincide con el lector MATLAB oficial:
    # fread(fid, 2*Total_Scale_Samples, 'single')
    raw = np.frombuffer(data, dtype="<f4", count=int(2*n), offset=int(off)).copy()
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

def _sb_estimar_mps(params):
    vals = []
    for p in params:
        try:
            d = float(p.get("Depth", np.nan))
            ds = float(p.get("Depth_In_Samples", np.nan))
            if np.isfinite(d) and np.isfinite(ds) and ds > 0:
                vals.append(d/ds)
        except Exception:
            pass
    if vals:
        return float(np.nanmedian(vals))
    try:
        sxm = float(params[0].get("SamplesXMeter", np.nan))
        if np.isfinite(sxm) and sxm > 0:
            return 1.0/sxm
    except Exception:
        pass
    return 1.0

def _sb_fondo_desde_params(params, n_ping, n_range):
    vals = np.full(int(n_ping), np.nan, dtype=np.float32)
    for i, p in enumerate(params[:int(n_ping)]):
        try:
            ds = float(p.get("Depth_In_Samples", np.nan))
            if np.isfinite(ds) and 1 <= ds < n_range:
                vals[i] = ds
        except Exception:
            pass
    ok = np.isfinite(vals)
    if ok.sum() < 2:
        vals[:] = int(0.78*n_range)
        return vals
    x = np.arange(n_ping, dtype=np.float32)
    vals = np.interp(x, x[ok], vals[ok]).astype(np.float32)
    # suavizado robusto simple
    w = 11
    r = w//2
    out = vals.copy()
    for i in range(n_ping):
        a = max(0, i-r); b = min(n_ping, i+r+1)
        v = vals[a:b]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = np.nanmedian(v)
    return np.clip(out, 1, n_range-1).astype(np.float32)

def _sb_preparar_imagen_seaman(M_ping_sample, quitar_luvia=True):
    M = np.asarray(M_ping_sample, dtype=np.float32).copy()
    M[~np.isfinite(M)] = np.nan
    if quitar_luvia:
        med_ping = np.nanmedian(M, axis=1, keepdims=True)
        M = M - med_ping
        med_range = np.nanmedian(M, axis=0, keepdims=True)
        M = M - 0.35 * med_range
    vals = M[np.isfinite(M)]
    if vals.size == 0:
        return np.zeros_like(M, dtype=np.float32)
    lo = np.nanpercentile(vals, 58)
    hi = np.nanpercentile(vals, 99.65)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanpercentile(vals, 2)
        hi = np.nanpercentile(vals, 99)
    X = np.clip((M - lo) / (hi - lo + 1e-6), 0, 1)
    X = np.log1p(8.0 * X) / np.log1p(8.0)
    X = np.power(X, 1.35)
    X[~np.isfinite(X)] = 0
    return X.astype(np.float32)

def _sb_guardar_jpg_sin_fondo(M_samples_pings, fondo, out_jpg, dpi=None):
    """
    Grafica sin la franja negra del fondo. Recorta la imagen hasta el fondo
    mínimo común y usa contraste tipo Seaman.
    Devuelve la misma sección cruda usada para la imagen, shape (samples, pings).
    """
    dpi = dpi_salida if dpi is None else dpi
    M = np.asarray(M_samples_pings)
    if np.iscomplexobj(M):
        M = np.abs(M)
    M = M.astype(np.float32, copy=False)
    n_range, n_ping = M.shape
    fondo = np.asarray(fondo, dtype=np.float32)
    margen = int(max(0, margen_quitar_fondo_samples if "margen_quitar_fondo_samples" in globals() else 0))
    valid = fondo[np.isfinite(fondo)]
    if valid.size:
        cut = int(np.nanpercentile(valid, 5)) - margen
    else:
        cut = int(0.78*n_range)
    cut = max(10, min(cut, n_range))
    M_sec = M[:cut, :].copy()
    M_ping_sample = M_sec.T
    M_vis = _sb_preparar_imagen_seaman(M_ping_sample, quitar_luvia=True)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(M_vis.T, aspect="auto", origin="upper", cmap=fish_cmap_oscura, vmin=0, vmax=1)
    ax.axis("off")
    plt.savefig(out_jpg, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return M_sec

def _sb_guardar_npy_misma_seccion(M_samples_pings, cut_samples, out_path):
    M = np.asarray(M_samples_pings)
    if M.ndim != 2:
        raise ValueError(f"Matriz 2D esperada, llegó {M.shape}")
    cut_samples = int(max(1, min(cut_samples, M.shape[0])))
    M_sec = M[:cut_samples, :]
    # Como el JPG se ve con eje horizontal=pings y vertical=samples,
    # guardamos en el mismo orden visual: (pings, samples).
    M_out = M_sec.T
    np.save(out_path, M_out)
    return M_out

def _sb_json_para_canal(path_id, maneouver_util, params, ch, n_pings, n_samples, cut_samples, es_virtual=False, jpg_path=None):
    """
    JSON completo para la rama SplitBeam dedicada.

    Antes esta función usaba solo p0 = primer ping, por eso el JSON SplitBeam salía
    mucho más pobre que el JSON del flujo general. Ahora usa todos los headers de
    ping en `params` y calcula median/min/max igual que `resumir_metadata_canal`.
    """
    params = params or []
    p0 = params[0] if params else {}

    freq_hz = median_param(params, "TX_Center_Frequency")
    bw_hz = median_param(params, "TX_Band_Width")
    fs_hz = median_param(params, "Sampling_Frequency")
    center_fpga_hz = median_param(params, "Center_Freq_After_FPGA")
    transmission_mode = median_param(params, "Transmission_Mode")
    frequency_modulation = median_param(params, "Frequency_Modulation")

    channel_id = "VIRTUAL_S7" if es_virtual else int(ch)
    channel_name = "VIRTUAL_S7" if es_virtual else f"CH{ch}"

    freq_khz = None if freq_hz is None else float(freq_hz) / 1000.0

    return {
        "maneouver_context": maneouver_util,

        # Mantengo este bloque porque es útil para trazabilidad, pero ahora
        # no reemplaza al resumen acústico completo.
        "file": {
            "source_file": Path(path_id).name,
            "sounder_type": 2,
            "sounder_type_description": "ECHOSOUNDER_SPB_BEAM - Split beam",
            "n_pings": int(n_pings),
            "n_samples_original": int(n_samples),
            "n_samples_section_without_bottom": int(cut_samples),
        },

        "file_acoustic_summary": {
            "frequencies_khz": [] if freq_khz is None else [freq_khz],
            "is_multifrequency": False,
            "has_chirp_or_fm": modo_acustico(transmission_mode, frequency_modulation, bw_hz) == "CHIRP",
            "channels_detected": [1, 2, 3, 4] if not es_virtual else ["VIRTUAL_S7"],
        },

        "channel_metadata": {
            "channel": channel_id,
            "channel_name": channel_name,
            "is_virtual": bool(es_virtual),

            "acoustic_setup": {
                "tx_center_frequency_hz": freq_hz,
                "tx_center_frequency_khz": freq_khz,
                "tx_band_width_hz": bw_hz,
                "tx_band_width_khz": None if bw_hz is None else float(bw_hz) / 1000.0,
                "sampling_frequency_hz": fs_hz,
                "center_freq_after_fpga_hz": center_fpga_hz,
                "transmission_mode": transmission_mode,
                "frequency_modulation": frequency_modulation,
                "mode": modo_acustico(transmission_mode, frequency_modulation, bw_hz),
                "Transducer_Beam_Angle_3dB_Port_Starboard": (
                    stat_param(params, "Transducer_Beam_Angle_3dB_Port_Starboard")
                    or stat_param(params, "Beam_1")
                ),
                "Transducer_Beam_Angle_3dB_Fore_Aft": (
                    stat_param(params, "Transducer_Beam_Angle_3dB_Fore_Aft")
                    or stat_param(params, "Beam_2")
                ),
                "transducer_index": median_param(params, "Transducer_Index"),
                "power": median_param(params, "Power"),
                "pulse_width_us": median_param(params, "Pulse_Width"),
                "window_rx_name": p0.get("Window_RX_Name"),
                "transducer_name": p0.get("Transducer_Name"),
            },

            "data_products_present": {
                "data1": "S2_FILTERED",
                "data2": "S3_CORRELATOR_DETECTOR_COMPLEX",
                "data3": "S5_DB_WITHOUT_TVG",
                "data4": "VIRTUAL_S7_DB" if es_virtual else "Data4_S7_DB_VIRTUAL.npy",
                "image_source_used": "VIRTUAL_S7" if es_virtual else "S5",
                "data4_note": (
                    "S7 pertenece a la sonda virtual SplitBeam; se copia con nombre por canal para mantener el dataset homogéneo."
                    if not es_virtual else None
                ),
            },

            "range_geometry": {
                "samples_x_meter": stat_param(params, "SamplesXMeter"),
                "total_scale_samples": stat_param(params, "Total_Scale_Samples"),
                "sounder_scale_m": stat_param(params, "Sounder_Scale"),
                "subscale_1": stat_param(params, "SubScale_1"),
                "subscale_2": stat_param(params, "SubScale_2"),
                "depth_m": stat_param(params, "Depth"),
                "depth_in_samples": stat_param(params, "Depth_In_Samples"),
            },

            "navigation_environment": {
                "date": stat_param(params, "Date"),
                "time": stat_param(params, "Time"),
                "latitude_raw": stat_param(params, "Latitude"),
                "longitude_raw": stat_param(params, "Longitude"),
                "vessel_speed_knots": stat_param(params, "Vessel_Speed"),
                "vessel_course_deg": stat_param(params, "Vessel_Course"),
                "ping_rate_hz": stat_param(params, "Ping_Rate"),
                "bottom_roughness": stat_param(params, "Roughness"),
                "bottom_hardness": stat_param(params, "Hardness"),
                "sv": stat_param(params, "SV"),
                "vessel_heading_deg": stat_param(params, "Vessel_Heading"),
                "transducer_temperature": stat_param(params, "Transducer_Temperature"),
                "net_temperature": stat_param(params, "Net_Temperature"),
                "latitude_cardinal": p0.get("Card_N_S"),
                "longitude_cardinal": p0.get("Card_E_W"),
            },
        },

        "processed_view": {
            "aligned_from_bottom": False,
            "vertical_reference": "surface_to_bottom_cut",
            "bottom_alignment_source": "Depth_In_Samples_header",
            "bottom_removed": True,
            "section": "same cut_samples applied to all saved matrices for this file",
            "image_geometry": (
                construir_geometria_vertical_surface_view(
                    path_jpg=jpg_path,
                    cut_samples=cut_samples,
                    params=params,
                )
                if jpg_path is not None
                else None
            ),
        },
    }

def procesar_splitbeam_v6_para_yolo(path_id, output_dir, maneouver_util=None, max_pings=None, verbose=True):
    """
    Procesa SOLO SplitBeam v6 con layout confirmado.
    Devuelve True si procesó el archivo, False si el layout no coincide y debe usarse el flujo viejo.
    """
    path_id = Path(path_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = path_id.stem

    data = path_id.read_bytes()
    headers = _sb_buscar_headers(data)
    if len(headers) < 2:
        return False

    ping_len = headers[1] - headers[0]
    if max_pings is not None:
        headers = headers[:int(max_pings)]

    p0 = _sb_leer_header_ping(data, headers[0])
    n = int(p0.get("Total_Scale_Samples", 0))
    header_len = int(p0.get("_header_len", 0))
    if not (1 <= n <= 300000) or header_len <= 0:
        return False

    ch_block = (HEADER_SIZE + 4*n) + (HEADER_SIZE + 8*n) + (HEADER_SIZE + 2*n)
    virtual_s5_rel = header_len + 4*ch_block
    virtual_s7_rel = virtual_s5_rel + HEADER_SIZE + 2*n
    expected_end = virtual_s7_rel + HEADER_SIZE + 2*n

    if expected_end != ping_len:
        if verbose:
            print(f"SplitBeam v6 no compatible para {path_id.name}: expected_end={expected_end}, ping_len={ping_len}. Se usa flujo viejo.")
        return False

    print(">>> ENTRANDO A RAMA SPLITBEAM v7 CONFIRMADA <<<")
    print("Archivo:", path_id.name, "pings:", len(headers), "samples:", n, "ping_len:", ping_len)

    # Diagnóstico automático: confirma si velocidad/curso vienen en -1 en el archivo
    # o si habría que sospechar offset corrido.
    diag_nav = _sb_diagnosticar_navegacion(data, headers, max_pings_diag=200)
    if verbose:
        _sb_imprimir_diagnostico_navegacion(diag_nav)

    S2 = {ch: [] for ch in range(1,5)}
    S3 = {ch: [] for ch in range(1,5)}
    S5 = {ch: [] for ch in range(1,5)}
    V5, V7, params = [], [], []

    for k, base_off in enumerate(headers):
        p = _sb_leer_header_ping(data, base_off)
        params.append(p)
        rel = header_len
        for ch in range(1,5):
            rel += HEADER_SIZE
            S2[ch].append(_sb_i32_arr(data, base_off + rel, n)); rel += 4*n
            rel += HEADER_SIZE
            S3[ch].append(_sb_s3_complex(data, base_off + rel, n)); rel += 8*n
            rel += HEADER_SIZE
            S5[ch].append(_sb_i16(data, base_off + rel, n)); rel += 2*n
        rel += HEADER_SIZE
        V5.append(_sb_i16(data, base_off + rel, n)); rel += 2*n
        rel += HEADER_SIZE
        V7.append(_sb_i16(data, base_off + rel, n)); rel += 2*n
        if verbose and (k % 200 == 0):
            print(f"  ping {k+1}/{len(headers)}")

    def _stack(lst, dtype=None):
        A = np.stack(lst, axis=1)
        return A.astype(dtype) if dtype is not None else A

    S2 = {ch: _stack(S2[ch], np.int32) for ch in range(1,5)}
    S3 = {ch: _stack(S3[ch], np.complex64) for ch in range(1,5)}
    S5 = {ch: _stack(S5[ch], np.int16) for ch in range(1,5)}
    V5 = _stack(V5, np.int16)
    V7 = _stack(V7, np.int16)

    fondo = _sb_fondo_desde_params(params, n_ping=len(headers), n_range=n)
    # JPG virtual S7 sin fondo y sección común
    virtual_jpg = output_dir / nombre_con_prefijo(f"{base}_VIRTUAL_S7.jpg")
    V7_sec = _sb_guardar_jpg_sin_fondo(V7, fondo, virtual_jpg)
    cut_samples = V7_sec.shape[0]
    copiar_a_carpeta_global(virtual_jpg)

    # Matrices virtuales
    virtual_npy_s5 = output_dir / nombre_con_prefijo(f"{base}_VIRTUAL_Data3_S5_DB_WITHOUT_TVG.npy")
    virtual_npy_s7 = output_dir / nombre_con_prefijo(f"{base}_VIRTUAL_Data4_S7_DB.npy")
    _sb_guardar_npy_misma_seccion(V5, cut_samples, virtual_npy_s5)
    _sb_guardar_npy_misma_seccion(V7, cut_samples, virtual_npy_s7)
    copiar_a_carpeta_global(virtual_npy_s5)
    copiar_a_carpeta_global(virtual_npy_s7)

    virtual_json = output_dir / nombre_con_prefijo(f"{base}_VIRTUAL_S7.json")
    with open(virtual_json, "w", encoding="utf-8") as f:
        json.dump(
            json_safe(
                _sb_json_para_canal(
                    path_id,
                    maneouver_util,
                    params,
                    "VIRTUAL",
                    len(headers),
                    n,
                    cut_samples,
                    es_virtual=True,
                    jpg_path=virtual_jpg,
                )
            ),
            f,
            indent=2,
            ensure_ascii=False,
            default=json_safe,
        )
    copiar_a_carpeta_global(virtual_json)

    # Canales reales: JPG de S5 sin fondo + NPY de S2/S3/S5 con la MISMA sección + JSON por canal
    for ch in range(1,5):
        ch_name = f"CH{ch}"
        jpg_path = output_dir / nombre_con_prefijo(f"{base}_{ch_name}.jpg")
        _ = _sb_guardar_jpg_sin_fondo(S5[ch], fondo, jpg_path)
        copiar_a_carpeta_global(jpg_path)

        files = [
            ("Data1_S2_FILTERED", S2[ch]),
            ("Data2_S3_CORRELATOR_DETECTOR", S3[ch]),
            ("Data3_S5_DB_WITHOUT_TVG", S5[ch]),
        ]
        for label, M in files:
            npy_path = output_dir / nombre_con_prefijo(f"{base}_{ch_name}_{label}.npy")
            M_out = _sb_guardar_npy_misma_seccion(M, cut_samples, npy_path)
            copiar_a_carpeta_global(npy_path)
            if verbose:
                print("NPY SplitBeam guardado:", npy_path.name, "shape=", M_out.shape, "dtype=", M_out.dtype)

        # También guardo una copia de la S7 virtual con nombre por canal para que el dataset tenga Data4 en CH1-CH4.
        # Nota: en SplitBeam real los canales reales llegan hasta S5; S7 pertenece a la sonda virtual.
        npy_s7_ch = output_dir / nombre_con_prefijo(f"{base}_{ch_name}_Data4_S7_DB_VIRTUAL.npy")
        _sb_guardar_npy_misma_seccion(V7, cut_samples, npy_s7_ch)
        copiar_a_carpeta_global(npy_s7_ch)

        json_path = output_dir / nombre_con_prefijo(f"{base}_{ch_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            meta = _sb_json_para_canal(
                path_id,
                maneouver_util,
                params,
                ch,
                len(headers),
                n,
                cut_samples,
                es_virtual=False,
                jpg_path=jpg_path,
            )
            meta["channel_metadata"]["data_products_present"]["data4_note"] = (
                "S7 pertenece a la sonda virtual SplitBeam; se copia con nombre por canal para mantener el dataset homogéneo."
            )
            json.dump(json_safe(meta), f, indent=2, ensure_ascii=False, default=json_safe)
        copiar_a_carpeta_global(json_path)

        print(f"OK SplitBeam {ch_name}: JPG + JSON + S2/S3/S5 + S7 virtual")

    print("OK VERIFICADO SplitBeam v7 guardado en:", output_dir.resolve())
    return True


def procesar_id_con_json(path_id, output_dir, maneouver_util, max_bloques=None, verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    path_id = Path(path_id)
    base = path_id.stem

    if verbose:
        print("\n========== PROCESANDO ==========")
        print(path_id.name)

    # ------------------------------------------------------------
    # 0) RAMA NUEVA SPLITBEAM v6
    # ------------------------------------------------------------
    # Si el archivo es SplitBeam con layout v6 confirmado, se procesa con
    # el lector dedicado que recupera CH1-CH4 + VIRTUAL_S7.
    # Si no coincide el layout, devuelve False y continúa el flujo original.
    try:
        if procesar_splitbeam_v6_para_yolo(
            path_id=path_id,
            output_dir=output_dir,
            maneouver_util=maneouver_util,
            max_pings=max_bloques,
            verbose=verbose,
        ):
            return
    except Exception as e:
        print("AVISO: falló rama SplitBeam dedicada; se intenta flujo original. Motivo:", repr(e))

    # ------------------------------------------------------------
    # 1) LECTOR NUEVO / COMPLETO: backend, JSON, Data1..Data4, virtual real
    # ------------------------------------------------------------
    header, channels, Z, Data1_stack, Data2_stack, Data4_stack, bloques = leer_id_completo_robusto(
        path_id,
        max_bloques=max_bloques,
        verbose=verbose,
    )

    n_channels = len(channels)
    if n_channels < 1:
        raise ValueError("No se detectaron canales.")

    # ------------------------------------------------------------
    # 2) LECTOR VIEJO SOLO PARA JPG
    # ------------------------------------------------------------
    # Esto es intencional: el lector viejo era el que producía las JPG lindas.
    # No se usa para JSON ni para guardar matrices nuevas.
    try:
        header_jpg, channels_jpg, Z_jpg, Data1_jpg, Data2_jpg, bloques_jpg = leer_id_completo_robusto_JPG(
            path_id,
            max_bloques=max_bloques,
            verbose=False,
        )
        usar_jpg_viejo = True
        if verbose:
            print("JPG: usando pipeline/lector viejo para mantener imagen idéntica.")
    except Exception as e:
        print("?? No se pudo usar lector viejo para JPG; se usa lector nuevo. Motivo:", e)
        channels_jpg = channels
        Z_jpg = Z
        usar_jpg_viejo = False

    if metodo_fondo.upper() == "A":
        detector_fondo = detectar_fondo_metodo_a
        nombre_metodo = "metodo_A"
    elif metodo_fondo.upper() == "B":
        detector_fondo = detectar_fondo_metodo_b
        nombre_metodo = "metodo_B"
    else:
        raise ValueError("metodo_fondo debe ser 'A' o 'B'")

    Sv_list = []
    fondo_list = []
    mps_list = []
    freq_list = []
    fondo_origen_list = []

    # ------------------------------------------------------------
    # Matriz usada para JPG: exactamente la del programa viejo.
    # Si por algún motivo no coincide la cantidad de canales, se cae al nuevo.
    # ------------------------------------------------------------
    for ch_idx in range(n_channels):
        if usar_jpg_viejo and ch_idx < len(channels_jpg):
            channels_img = channels_jpg
            Z_img = Z_jpg
            idx_img = ch_idx
        else:
            channels_img = channels
            Z_img = Z
            idx_img = ch_idx

        Sv = obtener_matriz_desde_stack(idx_img, Z_img, channels=channels_img)
        Sv[Sv == 0] = np.nan

        ch_name = channels[ch_idx].get("Channel_Name", f"CH{ch_idx + 1}")
        ch_name_img = channels_img[idx_img].get("Channel_Name", ch_name)

        # Para que la imagen sea igual al programa viejo, el fondo y m/px salen
        # del mismo conjunto channels/Z usado para la JPG.
        mps = calcular_metros_por_sample(channels_img, idx_img)
        freq = obtener_parametro_representativo(channels_img, idx_img, "TX_Center_Frequency")
        freq_khz = float(freq) / 1000.0 if np.isfinite(freq) else np.nan

        if verbose:
            print(f"{ch_name}: JPG usa {ch_name_img}, shape {Sv.shape}, freq {freq_khz:.1f} kHz, mps={mps:.6f}")

        fondo, origen_fondo = obtener_fondo_para_canal(
            Sv=Sv,
            channels=channels_img,
            ch_idx=idx_img,
            detector_fondo=detector_fondo,
        )

        if verbose:
            print(f"{ch_name}: fondo usado para JPG = {origen_fondo}, mediana={np.nanmedian(fondo):.1f} samples")

        Sv_list.append(Sv)
        fondo_list.append(fondo)
        mps_list.append(mps)
        freq_list.append(freq_khz)
        fondo_origen_list.append(origen_fondo)

    training_util = metadata_training_util(path_id, header, channels)
    base_metadata = {
        "maneouver_metadata": maneouver_util,
        "training_data_metadata": training_util,
    }

    fuentes = {
        "Data1_S2_FILTERED": Data1_stack,
        "Data2_S3_CORRELATOR_DETECTOR": Data2_stack,
        "Data3_S5_DB_WITHOUT_TVG": Z,
        "Data4_S7_DB": Data4_stack,
    }

    channel_summaries = training_util["channels_summary"]

    for ch_idx in range(n_channels):
        ch_name = channels[ch_idx].get("Channel_Name", f"CH{ch_idx + 1}")
        contraste = contraste_ch1 if ch_idx == 0 else contraste_ch2

        dz_ch = mps_list[ch_idx]
        hmax_fondo = float(np.nanpercentile(fondo_list[ch_idx] * mps_list[ch_idx], 5))
        hmax_ch = min(hmax_fondo, hmax_sobre_fondo_m)

        if not np.isfinite(hmax_ch) or hmax_ch <= 0:
            print(f"?? {ch_name}: altura sobre fondo inválida, se saltea")
            continue

        # --------------------------------------------------------
        # JPG: MISMO M, MISMA función, MISMO contraste del programa viejo
        # --------------------------------------------------------
        M, altura_m = alinear_desde_fondo_en_metros(
            Sv_list[ch_idx],
            fondo_list[ch_idx],
            metros_por_sample=mps_list[ch_idx],
            dz=dz_ch,
            hmax=hmax_ch,
        )

        # --------------------------------------------------------
        # JPG COMPLETA: SIN RECORTE, SIN ALINEAR, CON FONDO
        # Esta pasa a ser la JPG principal del canal. Usa los mismos
        # colores/contraste que la imagen recortada de referencia.
        # --------------------------------------------------------
        image_filename = nombre_con_prefijo(f"{base}_{ch_name}.jpg")
        image_path = Path(output_dir) / image_filename
        #guardar_full_con_contraste_del_recorte(
        #    Sv_list[ch_idx],
        #    M,
        #    image_path,
         #   contraste,
        #)
        #print("Imagen FULL completa guardada:", image_path)
        copiar_a_carpeta_global(image_path)

        # --------------------------------------------------------
        # JPG SIN FONDO: ARRANCA EN EL FONDO Y LLEGA HASTA ARRIBA.
        # No deja franja negra. Se recorta/alinea desde el fondo detectado.
        # Para que sea completa hacia arriba, usa hmax_fondo y NO el límite
        # hmax_sobre_fondo_m.
        # --------------------------------------------------------
        hmax_sin_fondo = hmax_fondo

        M_sin_fondo, altura_m_sin_fondo = alinear_desde_fondo_en_metros(
            Sv_list[ch_idx],
            fondo_list[ch_idx],
            metros_por_sample=mps_list[ch_idx],
            dz=dz_ch,
            hmax=hmax_sin_fondo,
        )

        sin_fondo_filename = nombre_con_prefijo(f"{base}_{ch_name}.jpg")
        sin_fondo_path = Path(output_dir) / sin_fondo_filename

        guardar_full_sin_fondo_con_contraste_del_recorte(
            M_sin_fondo,
            altura_m_sin_fondo,
            M,
            sin_fondo_path,
            contraste,
        )

        print("Imagen SIN_FONDO recortada desde el fondo guardada:", sin_fondo_path)
        copiar_a_carpeta_global(sin_fondo_path)

        # --------------------------------------------------------
        # NPY: EXACTAMENTE LA MISMA SECCIÓN QUE LA JPG _SIN_FONDO
        # --------------------------------------------------------
        # Importante:
        #   - La JPG _SIN_FONDO se generó con hmax_sin_fondo.
        #   - Por eso los .npy de S2, S3, S5 y S7 se guardan con el MISMO
        #     fondo, el MISMO dz y el MISMO hmax_sin_fondo.
        #   - Así la sección física guardada en .npy coincide con la sección
        #     que se ve en la imagen _SIN_FONDO.
        npy_files = {}
        for nombre_fuente, STACK in fuentes.items():
            M_src = obtener_matriz_desde_stack(ch_idx, STACK, channels=channels)
            if M_src is None:
                continue

            M_src_sin_fondo, _ = alinear_desde_fondo_en_metros(
                M_src,
                fondo_list[ch_idx],
                metros_por_sample=mps_list[ch_idx],
                dz=dz_ch,
                hmax=hmax_sin_fondo,
            )

            # ----------------------------------------------------
            # IMPORTANTE:
            # La JPG _SIN_FONDO ya fue generada arriba y NO se toca.
            # Para que el .npy corresponda exactamente a esa vista final,
            # remuestreamos la matriz de la MISMA sección al tamaño real
            # en pixeles de la JPG guardada.
            #
            # Resultado esperado:
            #   npy.shape[0] == ancho de la JPG
            #   npy.shape[1] == alto de la JPG
            # ----------------------------------------------------
            M_src_sin_fondo_pix = ajustar_matriz_a_pixeles_de_jpg(
                M_src_sin_fondo,
                sin_fondo_path,
            )

            # ----------------------------------------------------
            # CORRECCIÓN DE ORIENTACIÓN SOLO PARA LOS .NPY
            # ----------------------------------------------------
            # Las JPG ya están bien y NO se modifican.
            #
            # Los .npy se guardan como:
            #   axis=0 -> pings / ancho de la JPG
            #   axis=1 -> muestras / alto de la JPG
            #
            # Como S2, S3, S5 y S7 quedaban "patas para arriba"
            # respecto de la JPG, invertimos solamente el eje vertical.
            # Data1=S2, Data2=S3, Data3=S5 y Data4=S7.
            # Esto preserva la sección física y el tamaño del archivo.
            M_src_sin_fondo_pix = np.flip(M_src_sin_fondo_pix, axis=1)

            npy_name = nombre_con_prefijo(f"{base}_{ch_name}_{nombre_fuente}.npy")
            npy_path = Path(output_dir) / npy_name

            np.save(npy_path, M_src_sin_fondo_pix)
            npy_files[nombre_fuente] = npy_name

            print(
                "NPY SIN_FONDO guardado:",
                npy_path,
                "shape_seccion=", M_src_sin_fondo.shape,
                "shape_guardado=", M_src_sin_fondo_pix.shape,
                "dtype=", M_src_sin_fondo_pix.dtype,
                "es_complejo=", bool(np.iscomplexobj(M_src_sin_fondo_pix)),
            )

            imprimir_diagnostico_s3(
                nombre_fuente,
                M_src_sin_fondo_pix,
                etapa=f"GUARDADO EN {npy_name}",
            )
            copiar_a_carpeta_global(npy_path)

        depth_summary = (
            channel_summaries[ch_idx]
            .get("range_geometry", {})
            .get("depth_m", {})
        )
        profundidad_fondo_mediana = (
            depth_summary.get("median")
            if isinstance(depth_summary, dict)
            else None
        )

        geometria_imagen = construir_geometria_vertical_bottom_aligned(
            path_jpg=sin_fondo_path,
            altura_m=altura_m_sin_fondo,
            profundidad_fondo_m=profundidad_fondo_mediana,
        )

        json_img = metadata_imagen_canal(
            base_metadata,
            channel_summaries[ch_idx],
            extra={
                "height_above_bottom_m": hmax_sin_fondo,
                "bottom_alignment_source": fondo_origen_list[ch_idx],
                "image_geometry": geometria_imagen,
            },
        )
        json_name = nombre_con_prefijo(f"{base}_{ch_name}.json")
        json_img_path = Path(output_dir) / json_name
        with open(json_img_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(json_img), f, indent=2, ensure_ascii=False, default=json_safe)

        print("JSON canal guardado:", json_img_path)
        copiar_a_carpeta_global(json_img_path)

    del Z, Data1_stack, Data2_stack, Data4_stack, Sv_list
    gc.collect()
    plt.close("all")
# PROCESAR CARPETA
# ============================================================


def procesar_carpeta(input_dir, output_dir, max_bloques_por_archivo=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Carpeta entrada:", input_dir)
    print("Carpeta salida:", output_dir)

    training_files = sorted(input_dir.glob("Seaman_Fish_ID_Training_Data_*.ID"))
    if not training_files:
        raise FileNotFoundError("No se encontraron archivos Training_Data en la carpeta.")

    path_maneouver = buscar_archivo_maneouver(input_dir)
    if path_maneouver is None:
        print("ADVERTENCIA: no se encontró Maneouver. JSON irá sin metadata de maniobra.")
        maneouver_util = None
    else:
        print("Leyendo Maneouver una sola vez:", path_maneouver.name)
        maneouver_raw = leer_maneouver_id(path_maneouver)
        maneouver_util = extraer_maneouver_util(maneouver_raw, training_files_found=len(training_files))

    for i, path_id in enumerate(training_files, start=1):
        print(f"\n[{i}/{len(training_files)}] {path_id.name}")
        try:
            procesar_id_con_json(
                path_id=path_id,
                output_dir=output_dir,
                maneouver_util=maneouver_util,
                max_bloques=max_bloques_por_archivo,
                verbose=True,
            )
        except Exception as e:
            print("ERROR procesando", path_id.name, ":", e)


# ============================================================
# INTERFAZ DE LÍNEA DE COMANDOS
# ============================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convierte archivos Seaman FishID .ID en imágenes JPG, matrices NPY "
            "y metadata JSON por canal, preservando el procesamiento usado para "
            "el dataset original."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Carpeta que contiene Seaman_Fish_ID_Training_Data_*.ID. "
            "Puede contener además Seaman_Fish_ID_File_Features_Maneouver_*.ID."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Carpeta donde se escribirán JPG, NPY y JSON.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Prefijo agregado a todos los archivos de salida. Si se omite, "
            "se usa el nombre de la carpeta de entrada."
        ),
    )
    parser.add_argument(
        "--global-output-dir",
        default=None,
        help=(
            "Opcional. Copia además todos los archivos generados a esta carpeta "
            "global, manteniendo el prefijo."
        ),
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Procesa como máximo esta cantidad de bloques/pings por archivo (útil para pruebas).",
    )
    parser.add_argument(
        "--bottom-method",
        choices=["A", "B", "a", "b"],
        default="A",
        help="Método alternativo de detección de fondo cuando Depth_In_Samples no es utilizable (default: A).",
    )
    parser.add_argument(
        "--no-depth-in-samples",
        action="store_true",
        help="No usar Depth_In_Samples del encabezado para alinear el fondo.",
    )
    return parser


def main(argv=None):
    global guardar_copia_global
    global carpeta_todas_imagenes_yolo
    global prefijo_salida_global
    global max_bloques_por_archivo
    global metodo_fondo
    global usar_depth_in_samples_para_fondo

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        parser.error(f"La carpeta de entrada no existe o no es un directorio: {input_dir}")

    prefijo_salida_global = args.prefix or input_dir.name
    max_bloques_por_archivo = args.max_blocks
    metodo_fondo = args.bottom_method.upper()
    usar_depth_in_samples_para_fondo = not args.no_depth_in_samples

    if args.global_output_dir:
        guardar_copia_global = True
        carpeta_todas_imagenes_yolo = str(Path(args.global_output_dir).expanduser().resolve())
        Path(carpeta_todas_imagenes_yolo).mkdir(parents=True, exist_ok=True)
    else:
        guardar_copia_global = False
        carpeta_todas_imagenes_yolo = None

    print("=" * 80)
    print("SEAMAN FishID -> YOLO/NPY/JSON")
    print("Entrada:", input_dir)
    print("Salida:", output_dir)
    print("Prefijo:", prefijo_salida_global)
    print("Depth_In_Samples:", "sí" if usar_depth_in_samples_para_fondo else "no")
    print("Método de fondo alternativo:", metodo_fondo)
    print("Máximo de bloques:", max_bloques_por_archivo if max_bloques_por_archivo else "todos")
    if guardar_copia_global:
        print("Copia global:", carpeta_todas_imagenes_yolo)
    print("=" * 80)

    procesar_carpeta(
        input_dir=input_dir,
        output_dir=output_dir,
        max_bloques_por_archivo=max_bloques_por_archivo,
    )


if __name__ == "__main__":
    main()
