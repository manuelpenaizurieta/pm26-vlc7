#!/usr/bin/env python3
# REGLA DE AUTO-APUESTA (corre en cada actualizacion / cada hora en la nube):
# para CADA partido aun ABIERTO (no cerrado), compara tu apuesta colocada con el
# pick optimo actual. Si difiere o no hay apuesta -> la coloca/actualiza. Si ya es
# correcta -> la deja. Salta los partidos cerrados (no se pueden tocar).
# Cierre = saque - 30 min (la web bloquea las predicciones 30 min antes del saque).
# Ejecutado cada 30 min, la ultima pasada antes del cierre deja la apuesta definitiva;
# y como HOURS_AHEAD es amplio el pick ya esta colocado por si falla esa ultima pasada.
import sys, json, os, time, datetime, urllib.request, urllib.error
from polla_scraper import login, user_env, DB
from polla_sync import CODE
from polla_bet import GID, SUB, SC, put, get

HERE = os.path.dirname(os.path.abspath(__file__))
CLOSE_MS = 30 * 60 * 1000   # la web cierra las predicciones 30 min antes del saque
HOURS_AHEAD = 36.0     # coloca el pick hasta 36h antes del cierre; cada run lo actualiza si cambia
LOG_PATH = os.path.join(HERE, "apuestas_log.json")

def _log(entry):
    log = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)

def run(hours=HOURS_AHEAD):
    window = hours * 3600 * 1000
    tok, uid = login(user_env("POLLA_EMAIL"), user_env("POLLA_PASS"))
    matches = json.load(open(os.path.join(HERE, "polla_matches.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(HERE, "calendar_final.json"), encoding="utf-8"))
    picks = json.load(open(os.path.join(HERE, "picks.json"), encoding="utf-8"))
    try:
        override = json.load(open(os.path.join(HERE, "picks_override.json"), encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        override = {}
    cal_idx = {frozenset((c["home"], c["away"])): c for c in cal}
    now = time.time() * 1000
    ts_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
    placed, updated, left, skipped = 0, 0, 0, 0
    for cm, m in matches.items():
        ha, hb = CODE.get(m.get("tA")), CODE.get(m.get("tB"))
        if not ha or not hb:
            continue
        c = cal_idx.get(frozenset((ha, hb)))
        if not c:
            continue
        pk = picks.get(f"{c['home']}|{c['away']}")
        if not pk:
            continue
        ts = m.get("ts")
        if not ts or now > ts - CLOSE_MS:      # cerrado o sin hora -> no tocar
            skipped += 1; continue
        if ts - CLOSE_MS - now > window:       # aun lejos -> esperar
            skipped += 1; continue
        flip = (c["home"] != ha)               # orientar a teamA(local)=tA de la web
        gA, gB = (pk[1], pk[0]) if flip else (pk[0], pk[1])
        # ganador/avance: por defecto sale del marcador; si el override fija el equipo que
        # avanza (3er elemento, p.ej. eliminatoria con empate) se usa ese -> registra
        # campeon/quien pasa (vale para el bono). Se orienta a teamA/teamB de la web.
        w = "teamA" if gA > gB else ("teamB" if gA < gB else "E")
        ov = override.get(f"{c['home']}|{c['away']}")
        if ov and len(ov) >= 3 and ov[2] in ("teamA", "teamB"):
            w = ({"teamA": "teamB", "teamB": "teamA"}[ov[2]]) if flip else ov[2]
        path = f"bets/{cm}/{GID}/{uid}/{SUB}/prediction"
        cur = get(path, tok)
        if isinstance(cur, dict) and cur.get("gA") == gA and cur.get("gB") == gB and cur.get("w") == w:
            left += 1
            _log({"t": ts_str, "cm": cm, "match": f"{c['home']}|{c['away']}",
                  "pick": f"{gA}-{gB}", "accion": "ok"})
            continue
        obj = {"gA": gA, "gB": gB, "w": w, "ts": int(now),
               "fs": f"{gA}-{gB}-{w}", "sc": SC}
        put(path, tok, obj)
        accion = "actualizada" if isinstance(cur, dict) else "NUEVA"
        if isinstance(cur, dict): updated += 1
        else: placed += 1
        prev = f"{cur['gA']}-{cur['gB']}" if isinstance(cur, dict) else "-"
        _log({"t": ts_str, "cm": cm, "match": f"{c['home']}|{c['away']}",
              "pick": f"{gA}-{gB}", "prev": prev, "accion": accion})
        print(f"  {cm} {c['home']}-{c['away']}: {gA}-{gB} ({accion})")
    print(f"Auto-apuesta: {placed} nuevas, {updated} actualizadas, {left} ya correctas, {skipped} cerradas/sin hora")

if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else HOURS_AHEAD
    print(f"Revisando apuestas de partidos que cierran en las proximas {h}h...")
    run(h)
