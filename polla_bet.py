#!/usr/bin/env python3
# Escribe TU apuesta en pollamundial.org (con tu sesion). Formato/ruta sacados del
# codigo de la web: /bets/<matchId>/<groupId>/<userId>/<subsId>/prediction
#   valor: {gA, gB, w, ts, fs, sc}  (fs = "gA-gB-w")
import sys, json, time, urllib.request, urllib.error
from polla_scraper import login, user_env, DB

GID = "-OufgDiagwnoYiaufaac"
UID = "HZGu5zdBCJcpIV246oG5NbmCib63"
SUB = "F6yKZksrXApYB3xTTvb9"
SC = 1

def put(path, token, obj):
    url = f"{DB}/{path}.json?auth={token}"
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="PUT")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def get(path, token):
    try:
        with urllib.request.urlopen(f"{DB}/{path}.json?auth={token}", timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return f"<{e.code}>"

def place_bet(match_id, gA, gB):
    tok, uid = login(user_env("POLLA_EMAIL"), user_env("POLLA_PASS"))
    w = "teamA" if gA > gB else ("teamB" if gA < gB else "E")
    obj = {"gA": gA, "gB": gB, "w": w, "ts": int(time.time()*1000),
           "fs": f"{gA}-{gB}-{w}", "sc": SC}
    path = f"bets/{match_id}/{GID}/{uid}/{SUB}/prediction"
    print("Antes (lo que habia):", get(path, tok))
    res = put(path, tok, obj)
    print("Escrito:", json.dumps(res, ensure_ascii=False))
    print("Verificacion (releido de la web):", json.dumps(get(path, tok), ensure_ascii=False))

if __name__ == "__main__":
    mid = sys.argv[1]; gA = int(sys.argv[2]); gB = int(sys.argv[3])
    print(f"Apostando {gA}-{gB} en {mid}...")
    place_bet(mid, gA, gB)
