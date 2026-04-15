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

def registrar_reserva_firebase(user_id, area, torre, depto, fecha_hora=None):
    if not FIREBASE_ENABLED: return None
    try:
        doc_ref = db.collection('solicitudes_reserva').document()
        doc_ref.set({
            'user_id': str(user_id), 'area': area, 'torre': torre, 'depto': depto,
            'fecha_hora_solicitada': fecha_hora,
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
# 4. CONSULTA DE SALDO 
# Protección de datos: NO muestra nombre del propietario
# ══════════════════════════════════════════════════════
def consultar_saldo_firebase(torre, depto):
    """Consulta el saldo en Firebase. NO muestra nombre del propietario."""
    if not FIREBASE_ENABLED:
        return (f"🏠 Torre {torre}, Departamento {depto}\n"
                f"📋 Su consulta ha sido registrada. La administración le proporcionará su estado de cuenta.")
    try:
        doc_id = f"T{torre}_{depto.upper()}"
        doc = db.collection('saldos').document(doc_id).get()
        if doc.exists:
            saldo = doc.to_dict().get('saldo', 0)
            if saldo < -0.50:
                return (f"🏠 Torre {torre}, Departamento {depto}\n"
                        f"💰 Saldo pendiente: ${abs(saldo):.2f}\n"
                        f"Le recordamos que puede acercarse a la administración para regularizar su situación.")
            else:
                return (f"🏠 Torre {torre}, Departamento {depto}\n"
                        f"✅ Se encuentra al día con sus obligaciones.")
        else:
            return f"🏠 No encontré registros para Torre {torre}, Depto {depto}. ¿Podría verificar los datos?"
    except Exception as e:
        print(f"Error consultando saldo: {e}")
        return (f"🏠 Torre {torre}, Departamento {depto}\n"
                f"📋 Su consulta ha sido registrada. La administración le proporcionará su estado de cuenta.")

# ══════════════════════════════════════════════════════
# 5. ORQUESTADOR v5
# - Protección de datos personales
# - Flujos completos: reserva (con fecha/hora), cancelación, disponibilidad
# - Detector de emergencias ampliado
# ══════════════════════════════════════════════════════
MEMORIA_CONTEXTO = {}
UMBRAL_CONFIANZA = 0.5

PALABRAS_EMERGENCIA = [
    'fuga de gas', 'incendio', 'fuego', 'atrapado', 'atrapada',
    'emergencia', 'auxilio', 'socorro', 'ambulancia', 'desmayó',
    'desmayo', 'electrocutó', 'inundación', 'inundacion', 'derrumbe',
    'explosión', 'explosion', 'herido', 'herida', 'accidente grave',
    'encerrado', 'encerrada', 'sangre', 'inconsciente'
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

    # ══════════════════════════════════════════════════════
    # PASO 0: DETECTOR DE EMERGENCIAS
    # ══════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════
    # PASO 0.5: COMANDO DE CANCELACIÓN
    # ══════════════════════════════════════════════════════
    if texto_input.lower() in ['cancelar', '/cancelar', 'salir', '/salir', 'reiniciar']:
        MEMORIA_CONTEXTO[user_id] = None
        return "🤖 Consulta cancelada. ¿En qué más puedo ayudarle?"

    # ══════════════════════════════════════════════════════
    # PASO 1: ¿HAY CONTEXTO ACTIVO?
    # ══════════════════════════════════════════════════════
    if user_id in MEMORIA_CONTEXTO and MEMORIA_CONTEXTO[user_id] is not None:
        estado = MEMORIA_CONTEXTO[user_id]

        # --- SALDO: ESPERANDO DEPTO ---
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

        # --- SALDO: ESPERANDO TORRE ---
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

        # --- RESERVA: ESPERANDO ÁREA ---
        elif estado['estado'] == 'esperando_area_reserva':
            estado['area'] = texto_input
            estado['estado'] = 'esperando_fecha_reserva'
            MEMORIA_CONTEXTO[user_id] = estado
            return f"🤖 Quiere reservar: {texto_input}. ¿Para qué fecha y hora? (Ej: sábado 20 de julio, 14:00 a 18:00)"

        # --- RESERVA: ESPERANDO FECHA/HORA ---
        elif estado['estado'] == 'esperando_fecha_reserva':
            estado['fecha_hora'] = texto_input
            estado['estado'] = 'esperando_torre_depto_reserva'
            MEMORIA_CONTEXTO[user_id] = estado
            return f"🤖 Perfecto, {estado['area']} para {texto_input}. ¿Me indica su Torre y Departamento? (Ej: Torre 3, 2A)"

        # --- RESERVA: ESPERANDO TORRE/DEPTO ---
        elif estado['estado'] == 'esperando_torre_depto_reserva':
            torre_nueva, depto_nuevo = extraer_torre_depto(texto_input)
            if torre_nueva and depto_nuevo:
                area = estado.get('area', 'área comunal')
                fecha_hora = estado.get('fecha_hora', 'por confirmar')
                MEMORIA_CONTEXTO[user_id] = None
                registrar_reserva_firebase(user_id, area, torre_nueva, depto_nuevo, fecha_hora)
                return (f"🤖 ✅ Solicitud de reserva registrada.\n"
                        f"📋 Área: {area}\n"
                        f"📅 Fecha/Hora: {fecha_hora}\n"
                        f"🏠 Torre {torre_nueva}, Depto {depto_nuevo}\n"
                        f"⏳ Estado: Pendiente de aprobación\n"
                        f"La administración revisará disponibilidad y le confirmará.")
            else:
                return "🤖 Necesito su Torre y Departamento. (Ej: Torre 3, 2A)\n💡 Escriba /cancelar para reiniciar."

        # --- CANCELAR RESERVA: ESPERANDO ÁREA ---
        elif estado['estado'] == 'esperando_area_cancelar':
            estado['area'] = texto_input
            estado['estado'] = 'esperando_fecha_cancelar'
            MEMORIA_CONTEXTO[user_id] = estado
            return f"🤖 Cancelar reserva de: {texto_input}. ¿Para qué fecha estaba la reserva?"

        # --- CANCELAR RESERVA: ESPERANDO FECHA ---
        elif estado['estado'] == 'esperando_fecha_cancelar':
            area = estado.get('area', 'área comunal')
            MEMORIA_CONTEXTO[user_id] = None
            registrar_reporte_firebase(user_id, 'cancelar_reserva', f"Cancelar {area} - Fecha: {texto_input}")
            return (f"🤖 ❌ Solicitud de cancelación registrada.\n"
                    f"📋 Área: {area}\n"
                    f"📅 Fecha: {texto_input}\n"
                    f"La administración procesará la cancelación y le confirmará.")

        # --- DISPONIBILIDAD: ESPERANDO ÁREA ---
        elif estado['estado'] == 'esperando_area_disponibilidad':
            estado['area'] = texto_input
            estado['estado'] = 'esperando_fecha_disponibilidad'
            MEMORIA_CONTEXTO[user_id] = estado
            return f"🤖 Consultar disponibilidad de: {texto_input}. ¿Para qué fecha?"

        # --- DISPONIBILIDAD: ESPERANDO FECHA ---
        elif estado['estado'] == 'esperando_fecha_disponibilidad':
            area = estado.get('area', 'área comunal')
            MEMORIA_CONTEXTO[user_id] = None
            return (f"🤖 📅 Consulta de disponibilidad registrada.\n"
                    f"📋 Área: {area}\n"
                    f"📅 Fecha: {texto_input}\n"
                    f"La administración le confirmará la disponibilidad a la brevedad.")

    # ══════════════════════════════════════════════════════
    # PASO 2: SIN CONTEXTO → BERT CLASIFICA CON CONFIANZA
    # ══════════════════════════════════════════════════════
    intencion, confianza = clasificar_con_confianza(mensaje_usuario)
    print(f"DEBUG: BERT → {intencion} | confianza: {confianza:.4f}")

    # ══════════════════════════════════════════════════════
    # PASO 2.5: VERIFICAR UMBRAL DE CONFIANZA
    # ══════════════════════════════════════════════════════
    if confianza < UMBRAL_CONFIANZA:
        return ("🤖 No estoy completamente seguro de haber entendido su solicitud.\n"
                "Su mensaje ha sido derivado a la administración para revisión manual.\n"
                "💡 Si desea, intente reformular su mensaje con más detalle.")

    # ══════════════════════════════════════════════════════
    # PASO 3: RESPUESTAS POR INTENCIÓN
    # ══════════════════════════════════════════════════════

    # SALUDOS
    if intencion == "saludo":
        return "¡Hola! 👋 Soy el asistente virtual de Balcón Real. ¿En qué puedo ayudarle?"
    elif intencion == "agradecimiento":
        return "¡Con mucho gusto! Estoy aquí si necesita algo más. 😊"

    # ALÍCUOTAS
    elif intencion == "consulta_saldo":
        torre, depto = extraer_torre_depto(mensaje_usuario)
        if torre and depto:
            return consultar_saldo_excel(torre, depto)
        elif depto:
            MEMORIA_CONTEXTO[user_id] = {'estado': 'esperando_torre', 'torre': None, 'depto': depto}
            return f"🤖 Departamento {depto}, entendido. ¿De qué torre? (1 al 6)"
        elif torre:
            MEMORIA_CONTEXTO[user_id] = {'estado': 'esperando_depto', 'torre': torre, 'depto': None}
            return f"🤖 Torre {torre}, perfecto. ¿Cuál es su departamento? (Ej: 1A, 2C)"
        else:
            MEMORIA_CONTEXTO[user_id] = {'estado': 'esperando_depto', 'torre': None, 'depto': None}
            return "🤖 Con gusto le ayudo con su saldo. ¿Me indica su Torre y Departamento? (Ej: Torre 3, 2C)"
    elif intencion == "envio_comprobante":
        return "🤖 📎 Recibido. Su comprobante será revisado por la administración. Si todo está en orden, el pago se reflejará en su estado de cuenta."
    elif intencion == "consulta_metodo_pago":
        return "🤖 💳 Para conocer los métodos de pago y datos bancarios vigentes, por favor comuníquese con la administración."
    elif intencion == "reclamo_pago":
        registrar_reporte_firebase(user_id, 'reclamo_pago', mensaje_usuario)
        return "🤖 ⚠️ Su reclamo ha sido registrado. La administración revisará su caso y se comunicará con usted."
    elif intencion == "solicitud_documento":
        return "🤖 📄 Solicitud registrada. La administración preparará su documento y le avisará cuando esté disponible."

    # MANTENIMIENTO
    elif intencion == "reporte_fuga":
        registrar_reporte_firebase(user_id, 'reporte_fuga', mensaje_usuario)
        return ("🤖 🔧 Reporte de fuga registrado con prioridad. "
                "La administración será informada para coordinar la revisión lo antes posible. "
                "Si la situación es grave, le recomendamos contactar directamente a la administración.")
    elif intencion == "reporte_electrico":
        registrar_reporte_firebase(user_id, 'reporte_electrico', mensaje_usuario)
        return ("🤖 ⚡ Reporte eléctrico registrado. "
                "La administración será informada para coordinar la revisión. "
                "Por seguridad, si hay riesgo inmediato, evite la zona afectada y contacte directamente a la administración.")
    elif intencion == "reporte_daño":
        registrar_reporte_firebase(user_id, 'reporte_daño', mensaje_usuario)
        return "🤖 🔧 Reporte de daño registrado. La administración lo revisará y coordinará la reparación correspondiente."
    elif intencion == "solicitud_mantenimiento":
        registrar_reporte_firebase(user_id, 'solicitud_mantenimiento', mensaje_usuario)
        return "🤖 🔧 Solicitud de mantenimiento registrada. La administración evaluará su solicitud y programará la intervención según disponibilidad."
    elif intencion == "seguimiento_reporte":
        return "🤖 📋 Entendido. Vamos a consultar el estado de su reporte con la administración. Si tiene el número de reporte, compártalo para agilizar la consulta."

    # RESERVAS
    elif intencion == "solicitud_reserva":
        MEMORIA_CONTEXTO[user_id] = {'estado': 'esperando_area_reserva', 'torre': None, 'depto': None}
        return "🤖 🏢 ¡Con gusto! ¿Qué área desea reservar? (Ej: Salón comunal, BBQ, Turco, Terraza, Fogata)"
    elif intencion == "consulta_disponibilidad":
        MEMORIA_CONTEXTO[user_id] = {'estado': 'esperando_area_disponibilidad'}
        return "🤖 📅 ¡Claro! ¿De qué área desea consultar disponibilidad? (Ej: Salón comunal, BBQ, Turco, Terraza)"
    elif intencion == "cancelar_reserva":
        MEMORIA_CONTEXTO[user_id] = {'estado': 'esperando_area_cancelar'}
        return "🤖 ❌ Entendido. ¿Qué área tenía reservada? (Ej: Salón comunal, BBQ, Turco)"
    elif intencion == "consulta_requisitos":
        return ("🤖 📋 Los requisitos generales para reservar áreas comunales son:\n"
                "• Estar al día con las alícuotas\n"
                "• Reservar con al menos 48 horas de anticipación\n"
                "• Respetar los horarios establecidos\n"
                "Para detalles específicos, consulte con la administración.")

    # FAQ
    elif intencion == "queja_convivencia":
        registrar_reporte_firebase(user_id, 'queja_convivencia', mensaje_usuario)
        return "🤖 📝 Su queja ha sido registrada. La administración la revisará conforme al reglamento de convivencia y tomará las medidas correspondientes."
    elif intencion == "consulta_informacion":
        return "🤖 ℹ️ Claro, ¿qué información necesita? Puedo orientarle sobre horarios, contactos de administración o reglamento del conjunto."
    elif intencion == "reporte_seguridad":
        registrar_reporte_firebase(user_id, 'reporte_seguridad', mensaje_usuario)
        return ("🤖 🔒 Reporte de seguridad registrado con prioridad. "
                "La administración y el personal de seguridad serán informados. "
                "Si hay riesgo inmediato, contacte directamente a seguridad o al ECU 911.")
    elif intencion == "aviso_comunidad":
        return "🤖 📢 Gracias por compartir este aviso. Ha sido registrado y la administración lo difundirá por los canales oficiales del conjunto."

    # FALLBACK
    else:
        return "🤖 Su mensaje ha sido registrado. La administración lo revisará y le dará seguimiento."

# ══════════════════════════════════════════════════════
# 6. BOT DE TELEGRAM
# ══════════════════════════════════════════════════════
TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
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
