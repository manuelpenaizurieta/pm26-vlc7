#!/usr/bin/env python3
"""
Tarjetas y suspensiones WC 2026 via ESPN API (gratis, sin clave).

Flujo:
  1. Descarga el scoreboard ESPN del Mundial para obtener event IDs.
  2. Por cada partido terminado (no cacheado aun), baja el summary con stats
     de jugadores: YC (amarillas) y RC (rojas) por partido.
  3. Acumula amarillas por jugador a lo largo del torneo.
  4. Detecta suspendidos para el siguiente partido de cada equipo:
       - 2 amarillas en grupos -> suspension 1 partido
       - Tarjeta roja -> suspension 1 partido
  5. Guarda wc_suspensions.json:
       {"Mexico": {"n_susp": 1, "players": ["Brian Gutierrez"], "yellows": {...}}, ...}

build_dashboard.py carga este archivo y aplica -LAMBDA_PEN por cada suspension.

Ademas, si existe wc_injuries.json (editable a mano), lo incorpora:
  {"Mexico": ["Hirving Lozano"], "Brasil": ["Vinicius Jr"]} -> -LAMBDA_PEN por jugador

Penalizacion aplicada en build_dashboard: -10% lambda por cada suspension/lesion
(multiplicativa, cap -25%). El mercado 1X2 ya incorpora lesiones conocidas con
mas de 24h de antelacion; este modulo aporta cuando hay noticias de ultima hora.
"""
import json, os, time, datetime

HERE = os.path.dirname(os.path.abspath(__file__))

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"

ESPN_TEAM_MAP = {
    "Mexico": "Mexico", "South Africa": "Sudafrica", "South Korea": "CoreaSur",
    "Czech Republic": "Chequia", "Czechia": "Chequia",
    "United States": "EEUU", "USA": "EEUU",
    "Canada": "Canada", "Argentina": "Argentina", "Brazil": "Brasil",
    "France": "Francia", "England": "Inglaterra", "Spain": "Espana",
    "Germany": "Alemania", "Belgium": "Belgica", "Croatia": "Croacia",
    "Uruguay": "Uruguay", "Colombia": "Colombia", "Morocco": "Marruecos",
    "Norway": "Noruega", "Japan": "Japon", "Senegal": "Senegal",
    "Switzerland": "Suiza", "Turkey": "Turkiye", "Turkiye": "Turkiye",
    "Ecuador": "Ecuador", "Austria": "Austria", "Egypt": "Egipto",
    "Ivory Coast": "CostaMarfil", "Iran": "Iran", "Algeria": "Argelia",
    "Sweden": "Suecia", "Australia": "Australia", "Paraguay": "Paraguay",
    "Scotland": "Escocia", "Ghana": "Ghana",
    "Bosnia-Herzegovina": "Bosnia", "Bosnia and Herzegovina": "Bosnia",
    "DR Congo": "RDCongo", "Tunisia": "Tunez",
    "Saudi Arabia": "ArabiaSaudi", "Iraq": "Iraq", "Panama": "Panama",
    "Jordan": "Jordania", "Cape Verde": "CaboVerde",
    "New Zealand": "NuevaZelanda", "Curacao": "Curazao",
    "Haiti": "Haiti", "Qatar": "Catar", "Uzbekistan": "Uzbekistan",
    "Netherlands": "PaisesBajos", "Portugal": "Portugal",
    "South Korea": "CoreaSur", "Korea Republic": "CoreaSur",
}

LAMBDA_PEN = 0.10   # -10% lambda por cada jugador suspendido/lesionado
LAMBDA_CAP = 0.25   # maximo -25% (3+ jugadores)

def _get(url, retries=3):
    import urllib.request
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception as e:
            if i < retries-1:
                time.sleep(10)
            else:
                raise

def fetch_espn_events():
    """Obtiene todos los event IDs del WC 2026 (fase de grupos jugados)."""
    cache_path = os.path.join(HERE, "espn_events.json")
    cached = {}
    if os.path.exists(cache_path):
        try:
            cached = json.load(open(cache_path, encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            cached = {}

    # Fecha inicio WC 2026 = 8 jun; buscar hasta hoy
    today = datetime.date.today().strftime("%Y%m%d")
    try:
        data = _get(f"{ESPN_BASE}/scoreboard?dates=20260608-{today}&limit=200")
    except Exception as e:
        print(f"wc_cards: ESPN scoreboard error: {e}")
        return cached

    for ev in data.get("events", []):
        state = ev.get("status", {}).get("type", {}).get("state", "")
        if state != "post":
            continue
        comps = ev.get("competitions", [ev])
        for c in comps:
            competitors = c.get("competitors", [])
            if len(competitors) != 2:
                continue
            teams = sorted([competitors[0].get("team", {}).get("displayName", ""),
                            competitors[1].get("team", {}).get("displayName", "")])
            key = "|".join(teams)
            if key not in cached:
                cached[key] = {"espn_id": ev["id"],
                               "home": competitors[0].get("team", {}).get("displayName", ""),
                               "away": competitors[1].get("team", {}).get("displayName", "")}

    json.dump(cached, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return cached

def fetch_match_bookings(espn_id, events_cache):
    """Baja stats de jugadores (YC, RC) para un evento ESPN. Con cache."""
    book_cache_path = os.path.join(HERE, "espn_bookings_cache.json")
    book_cache = {}
    if os.path.exists(book_cache_path):
        try:
            book_cache = json.load(open(book_cache_path, encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            book_cache = {}

    sid = str(espn_id)
    if sid in book_cache:
        return book_cache[sid]

    try:
        summ = _get(f"{ESPN_BASE}/summary?event={espn_id}")
        time.sleep(1)   # respecta el rate-limit de ESPN
    except Exception as e:
        print(f"  ESPN summary {espn_id}: {e}")
        return {}

    result = {}   # team_model -> [{name, yc, rc}]
    for roster in summ.get("rosters", []):
        raw_name = roster.get("team", {}).get("displayName", "")
        team = ESPN_TEAM_MAP.get(raw_name, raw_name)
        players = []
        for p in roster.get("roster", []):
            athlete = p.get("athlete", {})
            name = athlete.get("displayName", "?")
            stats_dict = {s["abbreviation"]: s["value"] for s in p.get("stats", [])}
            yc = int(stats_dict.get("YC", 0))
            rc = int(stats_dict.get("RC", 0))
            if yc > 0 or rc > 0:
                players.append({"name": name, "yc": yc, "rc": rc})
        if players:
            result[team] = players

    book_cache[sid] = result
    json.dump(book_cache, open(book_cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return result

def build_suspensions():
    """
    Calcula tarjetas acumuladas y suspensiones actuales para todos los equipos.
    Devuelve dict: {team_model -> {n_susp, players, yellows_acc, next_opponent}}
    Guarda wc_suspensions.json.
    """
    events = fetch_espn_events()
    if not events:
        print("wc_cards: sin eventos ESPN, saltando")
        return {}

    # Acumular tarjetas por jugador a lo largo del torneo
    yellows = {}   # (team, player) -> total yellows
    reds    = {}   # (team, player) -> total reds

    n_fetched = 0
    for key, ev in events.items():
        bookings = fetch_match_bookings(ev["espn_id"], events)
        n_fetched += 1
        for team, players in bookings.items():
            for p in players:
                k = (team, p["name"])
                yellows[k] = yellows.get(k, 0) + p["yc"]
                reds[k]    = reds.get(k, 0)    + p["rc"]

    # Detectar suspendidos (2 amarillas o roja en grupo)
    suspended = {}   # team -> [player_name]
    for (team, player), yc in yellows.items():
        if yc >= 2:
            suspended.setdefault(team, []).append(player)
    for (team, player), rc in reds.items():
        if rc >= 1:
            if player not in suspended.get(team, []):
                suspended.setdefault(team, []).append(player)

    # Acumular amarillas simples (aviso para la proxima)
    yellow_acc = {}   # team -> {player: yc}
    for (team, player), yc in yellows.items():
        if yc == 1:
            yellow_acc.setdefault(team, {})[player] = yc

    # Resultado
    result = {}
    all_teams = set(t for t, _ in yellows) | set(t for t, _ in reds)
    for team in all_teams:
        susp_list = suspended.get(team, [])
        result[team] = {
            "n_susp": len(susp_list),
            "players": susp_list,
            "yellows_acc": yellow_acc.get(team, {}),   # 1 amarilla (en riesgo)
        }

    out_path = os.path.join(HERE, "wc_suspensions.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    n_teams_susp = sum(1 for v in result.values() if v["n_susp"] > 0)
    print(f"wc_cards: {n_fetched} partidos | {n_teams_susp} equipos con suspendidos")
    for team, d in result.items():
        if d["n_susp"] > 0:
            print(f"  {team}: SUSPENDIDO {d['players']}")
        elif d["yellows_acc"]:
            at_risk = list(d["yellows_acc"].keys())[:3]
            print(f"  {team}: en riesgo {at_risk}")
    return result

def get_lambda_pen(team_model):
    """Multiplicador de lambda por suspensiones + lesiones manuales.
    Devuelve un float <= 1.0 (1.0 = sin penalizacion).
    """
    susp_path = os.path.join(HERE, "wc_suspensions.json")
    inj_path  = os.path.join(HERE, "wc_injuries.json")
    n = 0
    if os.path.exists(susp_path):
        try:
            d = json.load(open(susp_path, encoding="utf-8")).get(team_model, {})
            n += d.get("n_susp", 0)
        except (json.JSONDecodeError, IOError):
            pass
    if os.path.exists(inj_path):
        try:
            inj = json.load(open(inj_path, encoding="utf-8"))
            n += len(inj.get(team_model, []))
        except (json.JSONDecodeError, IOError):
            pass
    penalty = min(n * LAMBDA_PEN, LAMBDA_CAP)
    return 1.0 - penalty

def get_alerts(team_model):
    """Lista de strings de alerta para mostrar en el dashboard."""
    alerts = []
    susp_path = os.path.join(HERE, "wc_suspensions.json")
    inj_path  = os.path.join(HERE, "wc_injuries.json")
    if os.path.exists(susp_path):
        try:
            d = json.load(open(susp_path, encoding="utf-8")).get(team_model, {})
            for p in d.get("players", []):
                alerts.append(f"🟥 {p} suspendido")
            at_risk = list(d.get("yellows_acc", {}).keys())
            if at_risk:
                alerts.append(f"🟨 en riesgo: {', '.join(at_risk[:2])}")
        except (json.JSONDecodeError, IOError):
            pass
    if os.path.exists(inj_path):
        try:
            for p in json.load(open(inj_path, encoding="utf-8")).get(team_model, []):
                alerts.append(f"🏥 {p} lesionado")
        except (json.JSONDecodeError, IOError):
            pass
    return alerts

if __name__ == "__main__":
    build_suspensions()

    # Crear wc_injuries.json de ejemplo si no existe
    inj_path = os.path.join(HERE, "wc_injuries.json")
    if not os.path.exists(inj_path):
        example = {
            "_instrucciones": "Anadir aqui jugadores clave lesionados: equipo -> [lista nombres]. "
                              "El modelo aplica -10% lambda por cada jugador. Editar a mano con noticias de prensa."
        }
        json.dump(example, open(inj_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wc_injuries.json creado (editar a mano con lesiones conocidas)")
