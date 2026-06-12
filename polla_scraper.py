#!/usr/bin/env python3
# Inicia sesion en pollamundial.org (Firebase) con TU correo+clave (leidos de
# variables de entorno locales, NUNCA escritos aqui) y baja los datos de tu grupo.
# Config Firebase = publica (esta en el JS de la web).
#
# Variables que debes tener puestas (setx, en tu PC):
#   POLLA_EMAIL, POLLA_PASS
import json, os, urllib.request, urllib.error

API_KEY = "AIzaSyDoO6RW1WD7yLpzIB9Qn7nNj7mxRMUbayQ"
DB = "https://pollamundialapp-5c605.firebaseio.com"
HERE = os.path.dirname(os.path.abspath(__file__))

def user_env(name):
    v = os.environ.get(name, "")
    if not v:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v = winreg.QueryValueEx(k, name)[0]
        except OSError:
            pass
    return v.strip('﻿').strip()  # elimina BOM y espacios (frecuente al copiar secrets)

def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    d = _post(url, {"email": email, "password": password, "returnSecureToken": True})
    return d["idToken"], d["localId"]

def get(path, token):
    url = f"{DB}/{path}.json?auth={token}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return f"<{e.code}>"

def explore(token, uid):
    # prueba rutas habituales para descubrir donde viven grupo, picks y tabla
    paths = ["", "users", f"users/{uid}", "grupos", "groups", "polla", "pollas",
             "predicciones", "predictions", "picks", "ranking", "estadisticas",
             "stats", "partidos", "matches", "subscriptions", "suscripciones"]
    print("=== explorando la base de datos (shallow) ===")
    for p in paths:
        url = f"{DB}/{p}.json?shallow=true&auth={token}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
            keys = list(data.keys())[:8] if isinstance(data, dict) else data
            print(f"  /{p or '(raiz)'}: {keys}")
        except urllib.error.HTTPError as e:
            print(f"  /{p or '(raiz)'}: <{e.code}>")

if __name__ == "__main__":
    email, pwd = user_env("POLLA_EMAIL"), user_env("POLLA_PASS")
    if not email or not pwd:
        print("Faltan POLLA_EMAIL / POLLA_PASS. Ponlas con setx en tu PC y reabre la terminal.")
        raise SystemExit(1)
    print(f"Iniciando sesion como {email[:3]}***...")
    try:
        token, uid = login(email, pwd)
    except urllib.error.HTTPError as e:
        print("ERROR de login:", json.load(e.fp).get("error", {}).get("message"))
        raise SystemExit(1)
    print(f"OK, sesion iniciada (uid {uid[:6]}***)\n")
    explore(token, uid)
    # guarda tu nodo de usuario (probablemente contiene tu grupo y tus picks)
    mine = get(f"users/{uid}", token)
    with open(os.path.join(HERE, "polla_raw.json"), "w", encoding="utf-8") as f:
        json.dump({"uid": uid, "user": mine}, f, ensure_ascii=False, indent=1)
    print("\nGuardado polla_raw.json (tu nodo de usuario, para ver la estructura)")
