import io
import os
import zipfile
import pandas as pd
import requests

ARCHIVO_SMS = "sms_spam_original_completo.csv"
ARCHIVO_ENRON = "enron_spam_original_completo.csv"
ARCHIVO_SALIDA = "dataset_pruebas_200.csv"

# ==========================================
# 1. GESTIÓN DATASET SMS (UCI)
# ==========================================
if os.path.exists(ARCHIVO_SMS):
  print(f"1. Cargando SMS desde archivo local existente ('{ARCHIVO_SMS}')...")
  df_sms = pd.read_csv(ARCHIVO_SMS)
else:
  print("1. Descargando SMS Spam Collection (UCI)...")
  url_sms = (
      "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
  )
  res_sms = requests.get(url_sms, timeout=30)
  res_sms.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(res_sms.content)) as z:
    with z.open("SMSSpamCollection") as f:
      df_sms = pd.read_csv(
          f,
          sep="\t",
          header=None,
          names=["etiqueta", "texto"],
          encoding="utf-8",
      )

  df_sms["etiqueta"] = df_sms["etiqueta"].str.strip().str.upper()
  df_sms["canal"] = "SMS"
  df_sms = df_sms[["canal", "etiqueta", "texto"]]
  df_sms.to_csv(ARCHIVO_SMS, index=False, encoding="utf-8")
  print(f"   -> Guardado '{ARCHIVO_SMS}'")

print(f"-> Total SMS disponibles: {len(df_sms)} filas")

# ==========================================
# 2. GESTIÓN DATASET ENRON (EMAILS)
# ==========================================
if os.path.exists(ARCHIVO_ENRON):
  print(f"\n2. Cargando Emails desde archivo local existente ('{ARCHIVO_ENRON}')...")
  df_email = pd.read_csv(ARCHIVO_ENRON)
else:
  print("\n2. Descargando Enron Spam (Emails)...")
  url_enron = "https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/train.jsonl"
  res_enron = requests.get(url_enron, timeout=60, allow_redirects=True)
  res_enron.raise_for_status()

  df_email = pd.read_json(io.BytesIO(res_enron.content), lines=True)
  df_email = df_email.rename(columns={"text": "texto", "label": "etiqueta"})
  df_email["etiqueta"] = df_email["etiqueta"].map({0: "HAM", 1: "SPAM"})
  df_email["canal"] = "EMAIL"
  df_email = df_email[["canal", "etiqueta", "texto"]]
  df_email.to_csv(ARCHIVO_ENRON, index=False, encoding="utf-8")
  print(f"   -> Guardado '{ARCHIVO_ENRON}'")

print(f"-> Total Emails disponibles: {len(df_email)} filas")

# ==========================================
# 3. EXTRACCIÓN DE LA MUESTRA (200 MENSAJES)
# ==========================================
print("\n3. Generando muestra estratificada (50 SMS HAM, 50 SMS SPAM, 50 EMAIL HAM, 50 EMAIL SPAM)...")

muestra_sms_ham = df_sms[df_sms["etiqueta"] == "HAM"].sample(50, random_state=42)
muestra_sms_spam = df_sms[df_sms["etiqueta"] == "SPAM"].sample(50, random_state=42)

muestra_email_ham = df_email[df_email["etiqueta"] == "HAM"].sample(50, random_state=42)
muestra_email_spam = df_email[df_email["etiqueta"] == "SPAM"].sample(50, random_state=42)

# Unir, mezclar con semilla fija y asignar IDs
df_final = pd.concat([muestra_sms_ham, muestra_sms_spam, muestra_email_ham, muestra_email_spam])
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
df_final.insert(0, "id", range(1, len(df_final) + 1))

# Guardar muestra definitiva
df_final.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8")
print(f"-> Archivo '{ARCHIVO_SALIDA}' generado con éxito.\n")

print("Resumen de la muestra generada:")
print(df_final.groupby(["canal", "etiqueta"]).size())