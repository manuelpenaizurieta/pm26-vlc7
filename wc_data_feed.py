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
    if v: return v
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except OSError:
        return ""

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

ODDS_FRESH_SECS = 3 * 3600  # rellamar la API solo si el archivo tiene >3h de antiguedad

def _is_fresh(path):
    return os.path.exists(path) and time.time() - os.path.getmtime(path) < ODDS_FRESH_SECS

def fetch_outright_odds():
    """Cuotas de campeon (decimales, mediana entre casas) desde The Odds API."""
    p = os.path.join(HERE, "odds_latest.json")
    if _is_fresh(p):
        print("odds_latest.json reciente (<3h), omitiendo llamada a The Odds API")
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
    """Cuotas 1X2 de cada partido (mediana entre ~20 casas). Es LA señal de mercado
    fina: incorpora alineaciones y noticias al minuto. 1 peticion por ejecucion."""
    p = os.path.join(HERE, "match_odds.json")
    if _is_fresh(p):
        print("match_odds.json reciente (<3h), omitiendo llamada a The Odds API")
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    if not ODDS_KEY: return []
    import statistics
    url = (f"https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
           f"/?apiKey={ODDS_KEY}&regions=eu&markets=h2h&oddsFormat=decimal")
    out = []
    for ev in _get(url):
        h = NAME_MAP.get(ev["home_team"], ev["home_team"])
        a = NAME_MAP.get(ev["away_team"], ev["away_team"])
        ph, pd_, pa = [], [], []
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                if mk["key"] != "h2h": continue
                for o in mk["outcomes"]:
                    n = NAME_MAP.get(o["name"], o["name"])
                    if n == h: ph.append(o["price"])
                    elif n == a: pa.append(o["price"])
                    else: pd_.append(o["price"])
        if ph and pd_ and pa:
            out.append({"home": h, "away": a, "oh": statistics.median(ph),
                        "od": statistics.median(pd_), "oa": statistics.median(pa)})
    if out:
        with open(os.path.join(HERE, "match_odds.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return out

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
    results = fetch_results()
    print(f"{len(results)} partidos terminados descargados")
    if results:
        # resultados crudos para que build_dashboard los embeba en el tablero
        with open(os.path.join(HERE, "results_live.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        update_elo_with_results(results)
        print("elo_live.json actualizado -> re-corre wc_model_v3.py con estos ratings")
    modds = fetch_match_odds()
    print(f"match_odds.json: cuotas 1X2 de {len(modds)} partidos")
    odds = fetch_outright_odds()
    if odds:
        with open(os.path.join(HERE, "odds_latest.json"), "w", encoding="utf-8") as f:
            json.dump(odds, f, ensure_ascii=False, indent=1)
        print(f"odds_latest.json: cuotas de {len(odds)} equipos "
              f"(con >=40, wc_model_v3 usa devig de Shin automaticamente)")
