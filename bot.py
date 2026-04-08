import os
import json
import re
import telebot
import torch
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime

# ══════════════════════════════════════════════════════
# 1. CARGAR MODELO DESDE HUGGING FACE
# ══════════════════════════════════════════════════════
print("⏳ Descargando modelo de Hugging Face...")
MODEL_NAME = "DavidS95/chatbot-balcon-real"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
modelo = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
modelo.eval()
print("✅ Modelo cargado")

# Cargar diccionario de etiquetas
from huggingface_hub import hf_hub_download
dicc_path = hf_hub_download(repo_id=MODEL_NAME, filename="diccionario.json")
with open(dicc_path, 'r') as f:
    dicc = json.load(f)
DICCIONARIO_ETIQUETAS = dicc['etiquetas']
DICCIONARIO_INVERSO = {int(k): v for k, v in dicc['inverso'].items()}

# ══════════════════════════════════════════════════════
# 2. FIREBASE
# ══════════════════════════════════════════════════════
print("⏳ Conectando a Firebase...")
firebase_creds = json.loads(os.environ.get("FIREBASE_CREDENTIALS", "{}"))
if firebase_creds:
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_ENABLED = True
    print("✅ Firebase conectado")
else:
    FIREBASE_ENABLED = False
    print("⚠️ Firebase no configurado, continuando sin registro")

def registrar_reporte_firebase(user_id, tipo_reporte, descripcion, torre=None, depto=None):
    if not FIREBASE_ENABLED: return None
    try:
        doc_ref = db.collection('reportes_mantenimiento').document()
        doc_ref.set({
            'user_id': str(user_id), 'tipo': tipo_reporte, 'descripcion': descripcion,
            'torre': torre, 'depto': depto, 'estado': 'pendiente', 'notas_admin': '',
            'fecha_creacion': datetime.now().isoformat(), 'fecha_actualizacion': datetime.now().isoformat()
        })
        return doc_ref.id
    except Exception as e:
        print(f"⚠️ Error Firebase: {e}")
        return None

def registrar_reserva_firebase(user_id, area, torre, depto):
    if not FIREBASE_ENABLED: return None
    try:
        doc_ref = db.collection('solicitudes_reserva').document()
        doc_ref.set({
            'user_id': str(user_id), 'area': area, 'torre': torre, 'depto': depto,
            'estado': 'pendiente', 'fecha_creacion': datetime.now().isoformat(),
            'fecha_actualizacion': datetime.now().isoformat()
        })
        return doc_ref.id
    except Exception as e:
        print(f"⚠️ Error Firebase: {e}")
        return None

def registrar_emergencia_firebase(user_id, mensaje, palabra_detectada):
    if not FIREBASE_ENABLED: return None
    try:
        doc_ref = db.collection('alertas_emergencia').document()
        doc_ref.set({
            'user_id': str(user_id), 'mensaje': mensaje, 'palabra_clave': palabra_detectada,
            'fecha': datetime.now().isoformat(), 'atendida': False
        })
        return doc_ref.id
    except Exception as e:
        print(f"⚠️ Error Firebase: {e}")
        return None

# ══════════════════════════════════════════════════════
# 3. EXTRACTOR DE TORRE Y DEPARTAMENTO
# ══════════════════════════════════════════════════════
def extraer_torre_depto(texto):
    texto_limpio = texto.upper().strip()
    torre = None
    depto = None
    match_depto = re.search(r'(?:DEPTO|DEPARTAMENTO|DPTO|DEP)\.?\s*([0-9]+[A-Z]+)', texto_limpio)
    if match_depto:
        depto = match_depto.group(1)
    else:
        match_suelto = re.findall(r'\b(\d+[A-Z])\b', texto_limpio)
        if match_suelto:
            depto = match_suelto[-1]
    match_torre = re.search(r'(?:TORRE|T)\s*(\d+)', texto_limpio)
    if match_torre:
        torre = match_torre.group(1)
    else:
        match_la = re.search(r'\bLA\s+(\d+)\b', texto_limpio)
        if match_la and match_la.group(1) in ['1','2','3','4','5','6']:
            torre = match_la.group(1)
    return torre, depto

# ══════════════════════════════════════════════════════
# 4. CONSULTA DE SALDO (SIN EXCEL EN RAILWAY)
# ══════════════════════════════════════════════════════
def consultar_saldo_excel(torre, depto):
    return (f"🤖 Consulta de saldo recibida.\n"
            f"🏠 Torre {torre}, Depto {depto}\n"
            f"📋 Su consulta ha sido registrada. La administración le proporcionará su estado de cuenta.")

# ══════════════════════════════════════════════════════
# 5. ORQUESTADOR
# ══════════════════════════════════════════════════════
MEMORIA_CONTEXTO = {}
UMBRAL_CONFIANZA = 0.5

PALABRAS_EMERGENCIA = [
    'fuga de gas', 'incendio', 'fuego', 'atrapado', 'atrapada',
    'emergencia', 'auxilio', 'socorro', 'ambulancia', 'desmayó',
    'desmayo', 'electrocutó', 'inundación', 'inundacion', 'derrumbe',
    'explosión', 'explosion', 'herido', 'herida', 'accidente grave'
]

def detectar_emergencia(texto):
    texto_lower = texto.lower()
    for palabra in PALABRAS_EMERGENCIA:
        if palabra in texto_lower:
            return True, palabra
    return False, None

def clasificar_con_confianza(mensaje):
    inputs = tokenizer(mensaje, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = modelo(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confianza, pred_idx = torch.max(probs, dim=1)
    intencion = DICCIONARIO_INVERSO[pred_idx.item()]
    return intencion, confianza.item()

def orquestador_inteligente(mensaje_usuario, user_id):
    global MEMORIA_CONTEXTO
    texto_input = mensaje_usuario.strip()

    es_emergencia, palabra_detectada = detectar_emergencia(texto_input)
    if es_emergencia:
        MEMORIA_CONTEXTO[user_id] = None
        registrar_emergencia_firebase(user_id, texto_input, palabra_detectada)
        return ("🚨 ALERTA DE EMERGENCIA DETECTADA 🚨\n\n"
                f"Se detectó una posible emergencia: «{palabra_detectada}»\n\n"
                "⚠️ Este chatbot NO gestiona emergencias.\n"
                "📞 Contacte INMEDIATAMENTE al ECU 911\n"
                "📞 O comuníquese directamente con la administración.\n\n"
                "Su mensaje fue registrado y la administración será alertada.")

    if texto_input.lower() in ['cancelar', '/cancelar', 'salir', '/salir', 'reiniciar']:
        MEMORIA_CONTEXTO[user_id] = None
        return "🤖 Consulta cancelada. ¿En qué más puedo ayudarle?"

    if user_id in MEMORIA_CONTEXTO and MEMORIA_CONTEXTO[user_id] is not None:
        estado = MEMORIA_CONTEXTO[user_id]

        if estado['estado'] == 'esperando_depto':
            torre_nueva, depto_nuevo = extraer_torre_depto(texto_input)
            if depto_nuevo: estado['depto'] = depto_nuevo
            if torre_nueva: estado['torre'] = torre_nueva
            if estado['torre'] and estado['depto']:
                respuesta = consultar_saldo_excel(estado['torre'], estado['depto'])
                MEMORIA_CONTEXTO[user_id] = None
                return respuesta
            elif estado['depto'] and not estado['torre']:
                estado['estado'] = 'esperando_torre'
                MEMORIA_CONTEXTO[user_id] = estado
                return f"🤖 Perfecto, departamento {estado['depto']}. ¿De qué torre? (1 al 6)"
            else:
                return "🤖 No logré identificar el departamento. ¿Me lo indica? (Ej: 2C, 1A)\n💡 Escriba /cancelar para reiniciar."

        elif estado['estado'] == 'esperando_torre':
            torre_nueva, depto_nuevo = extraer_torre_depto(texto_input)
            if not torre_nueva:
                numeros = re.findall(r'\b(\d+)\b', texto_input)
                for n in numeros:
                    if n in ['1','2','3','4','5','6']:
                        torre_nueva = n
                        break
            if torre_nueva: estado['torre'] = torre_nueva
            if depto_nuevo: estado['depto'] = depto_nuevo
            if estado['torre'] and estado['depto']:
                respuesta = consultar_saldo_excel(estado['torre'], estado['depto'])
                MEMORIA_CONTEXTO[user_id] = None
                return respuesta
            else:
                return "🤖 No identifiqué la torre. ¿Cuál es? (1 al 6)\n💡 Escriba /cancelar para reiniciar."

        elif estado['estado'] == 'esperando_area_reserva':
            estado['area'] = texto_input
            estado['estado'] = 'esperando_torre_depto_reserva'
            MEMORIA_CONTEXTO[user_id] = estado
            return f"🤖 Quiere reservar: {texto_input}. ¿Me indica su Torre y Departamento? (Ej: Torre 3, 2A)"

        elif estado['estado'] == 'esperando_torre_depto_reserva':
            torre_nueva, depto_nuevo = extraer_torre_depto(texto_input)
            if torre_nueva and depto_nuevo:
                area = estado.get('area', 'área comunal')
                MEMORIA_CONTEXTO[user_id] = None
                registrar_reserva_firebase(user_id, area, torre_nueva, depto_nuevo)
                return (f"🤖 ✅ Solicitud de reserva registrada.\n"
                        f"📋 Área: {area}\n"
                        f"🏠 Torre {torre_nueva}, Depto {depto_nuevo}\n"
                        f"⏳ Estado: Pendiente de aprobación\n"
                        f"La administración revisará disponibilidad y le confirmará.")
            else:
                return "🤖 Necesito su Torre y Departamento. (Ej: Torre 3, 2A)\n💡 Escriba /cancelar para reiniciar."

    intencion, confianza = clasificar_con_confianza(mensaje_usuario)
    print(f"DEBUG: BERT → {intencion} | confianza: {confianza:.4f}")

    if confianza < UMBRAL_CONFIANZA:
        return ("🤖 No estoy completamente seguro de haber entendido su solicitud.\n"
                "Su mensaje ha sido derivado a la administración para revisión manual.\n"
                "💡 Si desea, intente reformular su mensaje con más detalle.")

    if intencion == "saludo":
        return "¡Hola! 👋 Soy el asistente virtual de Balcón Real. ¿En qué puedo ayudarle?"
    elif intencion == "agradecimiento":
        return "¡Con mucho gusto! Estoy aquí si necesita algo más. 😊"
    elif intencion == "consulta_saldo":
        torre, depto = extraer_torre_depto(mensaje_usuario)
        if torre and depto:
            return consultar_saldo_excel(torre, depto)
        else:
            MEMORIA_CONTEXTO[user_id] = {'estado': 'esperando_depto', 'torre': None, 'depto': None}
            return "🤖 Con gusto le ayudo con su saldo. ¿Me indica su Torre y Departamento? (Ej: Torre 3, 2C)"
    elif intencion == "envio_comprobante":
        return "🤖 📎 Recibido. Su comprobante será revisado por la administración."
    elif intencion == "consulta_metodo_pago":
        return "🤖 💳 Comuníquese con la administración para conocer métodos de pago y datos bancarios."
    elif intencion == "reclamo_pago":
        registrar_reporte_firebase(user_id, 'reclamo_pago', mensaje_usuario)
        return "🤖 ⚠️ Su reclamo ha sido registrado. La administración revisará su caso."
    elif intencion == "solicitud_documento":
        return "🤖 📄 Solicitud registrada. La administración preparará su documento."
    elif intencion == "reporte_fuga":
        registrar_reporte_firebase(user_id, 'reporte_fuga', mensaje_usuario)
        return "🤖 🔧 Reporte de fuga registrado con prioridad. La administración coordinará la revisión."
    elif intencion == "reporte_electrico":
        registrar_reporte_firebase(user_id, 'reporte_electrico', mensaje_usuario)
        return "🤖 ⚡ Reporte eléctrico registrado. Si hay riesgo inmediato, evite la zona afectada."
    elif intencion == "reporte_daño":
        registrar_reporte_firebase(user_id, 'reporte_daño', mensaje_usuario)
        return "🤖 🔧 Reporte de daño registrado. La administración coordinará la reparación."
    elif intencion == "solicitud_mantenimiento":
        registrar_reporte_firebase(user_id, 'solicitud_mantenimiento', mensaje_usuario)
        return "🤖 🔧 Solicitud de mantenimiento registrada."
    elif intencion == "seguimiento_reporte":
        return "🤖 📋 Vamos a consultar el estado de su reporte. Si tiene el número, compártalo."
    elif intencion == "solicitud_reserva":
        MEMORIA_CONTEXTO[user_id] = {'estado': 'esperando_area_reserva', 'torre': None, 'depto': None}
        return "🤖 🏢 ¡Con gusto! ¿Qué área desea reservar? (Salón comunal, BBQ, Turco, Terraza, Fogata)"
    elif intencion == "consulta_disponibilidad":
        return "🤖 📅 Indíqueme qué área y fecha le interesa."
    elif intencion == "cancelar_reserva":
        return "🤖 ❌ Indíqueme el área y fecha de su reserva para cancelar."
    elif intencion == "consulta_requisitos":
        return ("🤖 📋 Requisitos para reservar:\n• Estar al día con alícuotas\n"
                "• Reservar con 48h de anticipación\n• Respetar horarios establecidos")
    elif intencion == "queja_convivencia":
        registrar_reporte_firebase(user_id, 'queja_convivencia', mensaje_usuario)
        return "🤖 📝 Su queja ha sido registrada. La administración tomará las medidas correspondientes."
    elif intencion == "consulta_informacion":
        return "🤖 ℹ️ ¿Qué información necesita? Puedo orientarle sobre horarios, contactos o reglamento."
    elif intencion == "reporte_seguridad":
        registrar_reporte_firebase(user_id, 'reporte_seguridad', mensaje_usuario)
        return "🤖 🔒 Reporte de seguridad registrado. Si hay riesgo inmediato, contacte al ECU 911."
    elif intencion == "aviso_comunidad":
        return "🤖 📢 Gracias por el aviso. La administración lo difundirá por los canales oficiales."
    else:
        return "🤖 Su mensaje ha sido registrado. La administración le dará seguimiento."

# ══════════════════════════════════════════════════════
# 6. BOT DE TELEGRAM
# ══════════════════════════════════════════════════════
TOKEN = os.environ.get("TELEGRAM_TOKEN", "8494300636:AAEwkfpWFZbnCW-zZwOjQUX-RR9JGyn4KQ4")
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def enviar_bienvenida(message):
    nombre = message.from_user.first_name
    MEMORIA_CONTEXTO[message.chat.id] = None
    bot.send_message(message.chat.id, f"¡Hola {nombre}! Soy el asistente de Balcón Real. ¿En qué te puedo ayudar?")

@bot.message_handler(func=lambda message: True)
def responder(message):
    user_id = message.chat.id
    respuesta = orquestador_inteligente(message.text, user_id)
    bot.reply_to(message, respuesta)

print("📡 Bot de Telegram iniciado. Escuchando mensajes...")
bot.infinity_polling()
