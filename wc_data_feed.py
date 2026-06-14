#!/usr/bin/env python3
# Q6 del traspaso: cerrar el bucle de datos sin busqueda web manual.
#
# Dos APIs gratuitas (con registro):
#  1) football-data.org  -> resultados y calendario del Mundial (competition WC).
#     Registro: https://www.football-data.org/client/register (gratis, token por email).
#     Plan gratis: 10 req/min, incluye la World Cup. Guarda el token en la variable
#     de entorno FOOTBALL_DATA_TOKEN.
#  2) The Odds API       -> cuotas de campeon (outrights) de decenas de casas.
#     Registro: https://the-odds-api.com (gratis 500 req/mes). Variable ODDS_API_KEY.
#     Con esto se rellenan las cuotas de los 48 equipos y wc_model_v3 pasa
#     automaticamente al devig de Shin (len(implied)>=40).
#
# Uso diario (lo llama la tarea programada de las 08:00):
#   python wc_data_feed.py            -> baja resultados nuevos, actualiza Elo (Q5),
#                                        guarda elo_live.json y odds_latest.json
import json, os, time, urllib.request, datetime

HERE = os.path.dirname(os.path.abspath(__file__))

def user_env(name):
    """Variable de entorno con fallback al registro de Windows (HKCU\\Environment):
    setx guarda ahi y los procesos ya abiertos no lo ven en os.environ."""
    v = os.environ.get(name, "")
    if not v:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v = winreg.QueryValueEx(k, name)[0]
        except OSError:
            pass
    return v.strip('﻿').strip()  # elimina BOM y espacios (frecuente al copiar secrets)

FD_TOKEN = user_env("FOOTBALL_DATA_TOKEN")
ODDS_KEY = user_env("ODDS_API_KEY")

# mapeo nombre API -> nombre del modelo (rellenar al confirmar los nombres exactos)
NAME_MAP = {
 "Spain":"Espana","France":"Francia","Brazil":"Brasil","England":"Inglaterra",
 "Argentina":"Argentina","Portugal":"Portugal","Netherlands":"PaisesBajos",
 "Germany":"Alemania","Belgium":"Belgica","Croatia":"Croacia","Uruguay":"Uruguay",
 "Colombia":"Colombia","Morocco":"Marruecos","Norway":"Noruega","Japan":"Japon",
 "Senegal":"Senegal","Switzerland":"Suiza","Turkey":"Turkiye","Türkiye":"Turkiye",
 "United States":"EEUU","USA":"EEUU","Mexico":"Mexico","Ecuador":"Ecuador",
 "South Korea":"CoreaSur","Korea Republic":"CoreaSur","Austria":"Austria",
 "Czechia":"Chequia","Czech Republic":"Chequia","Egypt":"Egipto",
 "Ivory Coast":"CostaMarfil","Côte d'Ivoire":"CostaMarfil","Iran":"Iran",
 "Algeria":"Argelia","Sweden":"Suecia","Canada":"Canada","Australia":"Australia",
 "Paraguay":"Paraguay","Scotland":"Escocia","Ghana":"Ghana",
 "Bosnia and Herzegovina":"Bosnia","Bosnia & Herzegovina":"Bosnia","DR Congo":"RDCongo","Tunisia":"Tunez",
 "South Africa":"Sudafrica","Qatar":"Catar","Uzbekistan":"Uzbekistan",
 "Saudi Arabia":"ArabiaSaudi","Iraq":"Iraq","Panama":"Panama","Jordan":"Jordania",
 "Cape Verde":"CaboVerde","New Zealand":"NuevaZelanda","Curacao":"Curazao",
 "Curaçao":"Curazao","Haiti":"Haiti",
}

def _get(url, headers=None, retries=3):
    """GET con reintentos: si la API limita (429) o falla la red, espera y reintenta
    en vez de tumbar la actualizacion diaria."""
    import time
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < retries-1:
                print(f"  limite de peticiones, espero 30s (intento {i+1})...")
                time.sleep(30); continue
            raise
        except urllib.error.URLError:
            if i < retries-1:
                print(f"  red caida, reintento en 20s (intento {i+1})...")
                time.sleep(20); continue
            raise

def fetch_results():
    """Partidos del Mundial ya jugados (status FINISHED) desde football-data.org."""
    if not FD_TOKEN:
        print("AVISO: falta FOOTBALL_DATA_TOKEN (registro gratis en football-data.org)")
        return []
    data = _get("https://api.football-data.org/v4/competitions/WC/matches",
                headers={"X-Auth-Token": FD_TOKEN})
    out = []
    for m in data.get("matches", []):
        if m["status"] != "FINISHED": continue
        h = NAME_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"])
        a = NAME_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"])
        ft = m["score"]["fullTime"]
        out.append({"date": m["utcDate"][:10], "home": h, "away": a,
                    "gh": ft["home"], "ga": ft["away"], "stage": m["stage"]})
    return out

ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

def fetch_results_espn(days_back=5):
    """Resultados FINALIZADOS del Mundial desde ESPN (tiempo real, sin clave). Es la
    fuente mas rapida: marca el resultado al pitido final, sin esperar a football-data
    ni a que la polla puntue. Solo equipos que el modelo conoce (evita partidos fantasma)."""
    import datetime
    try:
        import wc_model_v3 as _M
        known = set(_M.R0) if hasattr(_M, "R0") else set()
    except Exception:
        known = set()
    out = []
    today = datetime.datetime.utcnow().date()
    for i in range(days_back + 1):
        d = (today - datetime.timedelta(days=i)).strftime("%Y%m%d")
        try:
            data = _get(f"{ESPN_SB}?dates={d}")
        except Exception:
            continue
        for ev in data.get("events", []):
            if ev.get("status", {}).get("type", {}).get("state") != "post":
                continue                     # solo finalizados (no 'in'=en vivo, no 'pre')
            cs = ev.get("competitions", [{}])[0].get("competitors", [])
            if len(cs) != 2:
                continue
            try:
                hc = next(c for c in cs if c.get("homeAway") == "home")
                ac = next(c for c in cs if c.get("homeAway") == "away")
                h = NAME_MAP.get(hc["team"]["displayName"], hc["team"]["displayName"])
                a = NAME_MAP.get(ac["team"]["displayName"], ac["team"]["displayName"])
                gh, ga = int(hc["score"]), int(ac["score"])
            except (KeyError, StopIteration, ValueError):
                continue
            if known and (h not in known or a not in known):
                continue                     # nombre no mapeado -> lo ignora
            out.append({"date": ev.get("date", "")[:10], "home": h, "away": a,
                        "gh": gh, "ga": ga, "stage": "GROUP_STAGE"})
    return out

def merge_results(*sources):
    """Une varias listas de resultados por (home,away); la primera fuente tiene prioridad."""
    seen, out = set(), []
    for src in sources:
        for r in src:
            key = (r["home"], r["away"])
            if key in seen:
                continue
            seen.add(key); out.append(r)
    return out

# Cache ADAPTATIVO de cuotas: el presupuesto de The Odds API es 500 req/mes, asi que
# no podemos rellamar cada pasada todo el dia. La estrategia (lo que pidio el usuario:
# pick base con anticipacion + revision de ultimo minuto con la mejor info):
#   - Si hay un partido por CERRAR pronto (<CRITICAL_MIN del cierre = saque-30): cuotas
#     MUY frescas (las cuotas ya incorporan alineaciones/lesiones de ultima hora).
#   - Si el proximo cierre esta lejos: cache largo, no gastar API en partidos lejanos.
CRITICAL_MIN   = 90          # ventana de "revision de ultimo minuto" antes del cierre
FRESH_CRITICAL = 12 * 60     # en ventana critica: refresca si el archivo tiene >12 min
FRESH_FAR      = 6 * 3600    # lejos del cierre: cache 6h (ahorra presupuesto de API)
CLOSE_BEFORE_MS = 30 * 60 * 1000   # la web cierra las predicciones 30 min antes del saque

def _next_close_minutes():
    """Minutos hasta el cierre (saque-30min) del proximo partido aun no cerrado.
    Lee polla_matches.json (existe de runs previos). None si no hay datos."""
    p = os.path.join(HERE, "polla_matches.json")
    if not os.path.exists(p):
        return None
    try:
        pm = json.load(open(p, encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return None
    now = time.time() * 1000
    closes = [m["ts"] - CLOSE_BEFORE_MS for m in pm.values()
              if isinstance(m, dict) and m.get("ts")]
    future = [c for c in closes if c > now]    # cierres que aun no pasaron
    return (min(future) - now) / 60000 if future else None

def _fresh_secs():
    """Segundos de cache vigente segun proximidad del proximo cierre (adaptativo)."""
    mins = _next_close_minutes()
    if mins is not None and mins < CRITICAL_MIN:
        return FRESH_CRITICAL
    return FRESH_FAR

def _is_fresh(path):
    return os.path.exists(path) and time.time() - os.path.getmtime(path) < _fresh_secs()

def fetch_outright_odds():
    """Cuotas de campeon (decimales, mediana entre casas) desde The Odds API."""
    p = os.path.join(HERE, "odds_latest.json")
    if _is_fresh(p):
        print(f"odds_latest.json fresco (cache {_fresh_secs()//60}min), omitiendo The Odds API")
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if not ODDS_KEY:
        print("AVISO: falta ODDS_API_KEY (registro gratis en the-odds-api.com)")
        return {}
    url = (f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup_winner/odds"
           f"/?apiKey={ODDS_KEY}&regions=eu&markets=outrights&oddsFormat=decimal")
    data = _get(url)
    prices = {}
    for ev in data:
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                for o in mk.get("outcomes", []):
                    t = NAME_MAP.get(o["name"], o["name"])
                    prices.setdefault(t, []).append(o["price"])
    import statistics
    return {t: round(statistics.median(v), 2) for t, v in prices.items()}

def fetch_match_odds():
    """Cuotas 1X2 + O/U totales + spreads (AH) — una sola peticion a The Odds API.
    Guarda match_odds.json (1X2) y match_odds_ext.json (datos completos con O/U y AH)."""
    p     = os.path.join(HERE, "match_odds.json")
    p_ext = os.path.join(HERE, "match_odds_ext.json")
    if _is_fresh(p):
        nm = _next_close_minutes()
        ctx = f"proximo cierre en {nm:.0f}min" if nm is not None else "sin partidos proximos"
        print(f"match_odds.json fresco (cache {_fresh_secs()//60}min, {ctx}), omitiendo The Odds API")
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if not ODDS_KEY: return []
    import statistics
    url = (f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
           f"/?apiKey={ODDS_KEY}&regions=eu&markets=h2h,totals,spreads&oddsFormat=decimal")
    out_h2h, out_ext = [], []
    for ev in _get(url):
        h = NAME_MAP.get(ev["home_team"], ev["home_team"])
        a = NAME_MAP.get(ev["away_team"], ev["away_team"])
        ph, pd_, pa = [], [], []
        ou_over_by_pt, ou_under_by_pt = {}, {}
        ah_home_by_pt, ah_away_by_pt  = {}, {}
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                key = mk["key"]
                if key == "h2h":
                    for o in mk["outcomes"]:
                        n = NAME_MAP.get(o["name"], o["name"])
                        if n == h: ph.append(o["price"])
                        elif n == a: pa.append(o["price"])
                        else: pd_.append(o["price"])
                elif key == "totals":
                    for o in mk["outcomes"]:
                        pt = o.get("point", 2.5)
                        if o["name"] == "Over": ou_over_by_pt.setdefault(pt, []).append(o["price"])
                        else: ou_under_by_pt.setdefault(pt, []).append(o["price"])
                elif key == "spreads":
                    for o in mk["outcomes"]:
                        pt = o.get("point", 0.0)
                        n  = NAME_MAP.get(o["name"], o["name"])
                        if n == h: ah_home_by_pt.setdefault(pt, []).append(o["price"])
                        else: ah_away_by_pt.setdefault(-pt, []).append(o["price"])
        if not (ph and pd_ and pa): continue
        h2h = {"home": h, "away": a, "oh": statistics.median(ph),
               "od": statistics.median(pd_), "oa": statistics.median(pa)}
        out_h2h.append(h2h)
        ext = dict(h2h)
        if ou_over_by_pt:
            best_pt = max(ou_over_by_pt, key=lambda k: len(ou_over_by_pt[k]))
            ov_list = ou_over_by_pt[best_pt]; un_list = ou_under_by_pt.get(best_pt, [])
            if ov_list and un_list:
                raw_o = 1/statistics.median(ov_list); raw_u = 1/statistics.median(un_list)
                s = raw_o + raw_u
                ext["ou_line"] = best_pt; ext["p_over"] = round(raw_o / s, 4)
        if ah_home_by_pt:
            best_ah = max(ah_home_by_pt, key=lambda k: len(ah_home_by_pt[k]))
            hh_list = ah_home_by_pt[best_ah]; ha_list = ah_away_by_pt.get(best_ah, [])
            if hh_list and ha_list:
                rh = 1/statistics.median(hh_list); ra = 1/statistics.median(ha_list)
                s = rh + ra
                ext["ah_line"] = best_ah; ext["p_ah_home"] = round(rh / s, 4)
        out_ext.append(ext)
    if out_h2h:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out_h2h, f, ensure_ascii=False, indent=1)
    if out_ext:
        with open(p_ext, "w", encoding="utf-8") as f:
            json.dump(out_ext, f, ensure_ascii=False, indent=1)
        ou_n = sum(1 for e in out_ext if "ou_line" in e)
        ah_n = sum(1 for e in out_ext if "ah_line" in e)
        print(f"match_odds_ext.json: {len(out_ext)} partidos | {ou_n} con O/U | {ah_n} con AH")
    return out_h2h


def track_odds_movement():
    """Detecta movimiento de linea vs la snapshot anterior. Guarda odds_movement.json."""
    import math
    p = os.path.join(HERE, "match_odds.json"); p_snap = os.path.join(HERE, "odds_snapshot.json")
    p_mov = os.path.join(HERE, "odds_movement.json")
    if not os.path.exists(p): return
    cur = {(o["home"], o["away"]): o for o in json.load(open(p, encoding="utf-8"))}
    if not os.path.exists(p_snap):
        json.dump(list(cur.values()), open(p_snap, "w", encoding="utf-8"), indent=1); return
    prev = {(o["home"], o["away"]): o for o in json.load(open(p_snap, encoding="utf-8"))}
    movement = []
    for key, c in cur.items():
        if key not in prev: continue
        old = prev[key]
        dH = math.log(old["oh"] / c["oh"]) if c["oh"] > 0 else 0.0
        dA = math.log(old["oa"] / c["oa"]) if c["oa"] > 0 else 0.0
        if abs(dH) > 0.04 or abs(dA) > 0.04:
            movement.append({"home": key[0], "away": key[1],
                             "dH": round(dH, 3), "dA": round(dA, 3),
                             "sharp": abs(dH) > 0.12 or abs(dA) > 0.12})
    json.dump(movement, open(p_mov, "w", encoding="utf-8"), indent=1)
    json.dump(list(cur.values()), open(p_snap, "w", encoding="utf-8"), indent=1)
    sharp = [m for m in movement if m["sharp"]]
    if movement:
        print(f"odds_movement.json: {len(movement)} movimientos ({len(sharp)} sharp)")
    if sharp:
        for m in sharp:
            print(f"  *** SHARP *** {m['home']} vs {m['away']}: dH={m['dH']:+.3f} dA={m['dA']:+.3f}")

def update_elo_with_results(results):
    """Q5: aplica elo_update de wc_model_v3 a los resultados nuevos y persiste."""
    import wc_model_v3 as M
    state_path = os.path.join(HERE, "elo_live.json")
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"ratings": dict(M.R0), "applied": []}
    ratings = state["ratings"]
    for r in results:
        key = f"{r['date']}|{r['home']}|{r['away']}"
        if key in state["applied"]: continue
        if r["home"] not in ratings or r["away"] not in ratings: continue
        K = 40 if "GROUP" in r.get("stage", "GROUP") else 50
        d = M.elo_update(ratings, r["home"], r["away"], r["gh"], r["ga"], K=K)
        state["applied"].append(key)
        print(f"  Elo: {r['home']} {r['gh']}-{r['ga']} {r['away']}  ({d:+.1f})")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    return ratings

if __name__ == "__main__":
    print(f"=== wc_data_feed {datetime.date.today()} ===")
    # ESPN (tiempo real, sin clave) PRIMERO; football-data como respaldo. Asi el modelo
    # aprende del resultado al pitido final, sin esperar a que la polla lo puntue.
    espn = fetch_results_espn()
    fd   = fetch_results()
    results = merge_results(espn, fd)          # ESPN tiene prioridad
    print(f"resultados: {len(espn)} ESPN + {len(fd)} football-data -> {len(results)} unicos")
    if results:
        # resultados crudos para que build_dashboard los embeba en el tablero
        with open(os.path.join(HERE, "results_live.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        update_elo_with_results(results)
        print("elo_live.json actualizado -> re-corre wc_model_v3.py con estos ratings")
    modds = fetch_match_odds()
    print(f"match_odds.json: cuotas 1X2 de {len(modds)} partidos")
    track_odds_movement()
    odds = fetch_outright_odds()
    if odds:
        with open(os.path.join(HERE, "odds_latest.json"), "w", encoding="utf-8") as f:
            json.dump(odds, f, ensure_ascii=False, indent=1)
        print(f"odds_latest.json: cuotas de {len(odds)} equipos "
              f"(con >=40, wc_model_v3 usa devig de Shin automaticamente)")
