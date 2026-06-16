#!/usr/bin/env python3
# Genera polla_v4.html desde los datos (wc_probs_v3.json + calendar_final.json +
# modelo en vivo). El HTML nunca se edita a mano: se regenera con este script.
# Para cada partido del calendario real calcula:
#   - pick VALIENTE (politica B: max EV + bono unicidad esperado)  <- el que juegas
#   - pick SEGURO  (politica A: max EV puro)                       <- referencia
#   - prob local/empate/visitante y EVs
import json, os, datetime
import numpy as np
import wc_model_v3 as M
import wc_pool_strategy as S

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "wc_probs_v3.json"), encoding="utf-8") as f:
    PROBS = json.load(f)
C = PROBS["c"]
with open(os.path.join(HERE, "calendar_final.json"), encoding="utf-8") as f:
    CAL = json.load(f)

# cuotas 1X2 por partido (mediana de ~20 casas, bajadas por wc_data_feed.py)
MODDS = {}
try:
    with open(os.path.join(HERE, "match_odds.json"), encoding="utf-8-sig") as f:
        for o in json.load(f):
            MODDS[(o["home"], o["away"])] = (o["oh"], o["od"], o["oa"])
except FileNotFoundError:
    pass

# cuotas extendidas: O/U totales + spreads (Asian Handicap)
MODDS_EXT = {}
try:
    with open(os.path.join(HERE, "match_odds_ext.json"), encoding="utf-8-sig") as f:
        for o in json.load(f):
            MODDS_EXT[(o["home"], o["away"])] = o
except FileNotFoundError:
    pass

# movimiento de linea (dinero sharp detectado)
MOVEMENT = {}
try:
    with open(os.path.join(HERE, "odds_movement.json"), encoding="utf-8") as f:
        for o in json.load(f):
            MOVEMENT[(o["home"], o["away"])] = o
except FileNotFoundError:
    pass

# contexto de standings (presion por posicion en el grupo)
try:
    import wc_standings_context as _SC
    _SC.reset_cache()
except ImportError:
    _SC = None

# tarjetas y lesiones
try:
    import wc_cards as _WCC
except ImportError:
    _WCC = None

# noticias ESPN del Mundial
NEWS = []
try:
    import wc_news as _NEWS
    NEWS = _NEWS.fetch_news()
except Exception:
    pass

# Correccion de minnow: cuanto inflar lambda del favorito en un mismatch extremo.
# Se aplica solo si el favorito gana con prob > MINNOW_PWIN_THR; el boost crece con
# la extremidad (0 en el umbral, MINNOW_BOOST cuando P(gana)=100%). Conservador a
# proposito: corrige el sesgo sistematico de compresion de handicaps, no persigue
# outliers como el 7-1.
MINNOW_PWIN_THR = 0.80
MINNOW_BOOST    = 0.45   # subido de 0.22: Alemania-Curazao acabo 7-1 con el modelo en ~4;
                         # los grandes golean a los minnows MAS de lo que el mercado comprimido dice

def mat_from_lams(la, lb):
    x = np.arange(M.MAXG+1)
    pa_ = np.exp(-la)*la**x/M.FACT; pb_ = np.exp(-lb)*lb**x/M.FACT
    mat = np.outer(pa_, pb_)
    mat[0,0] *= 1-la*lb*M.RHO; mat[0,1] *= 1+la*M.RHO
    mat[1,0] *= 1+lb*M.RHO; mat[1,1] *= 1-M.RHO
    return mat/mat.sum()

def market_matrix(la0, lb0, tH, tA, ou_data=None, ah_data=None):
    """Ajusta (lambda_local, lambda_visit) para que la matriz Dixon-Coles reproduzca
    1X2 + O/U + AH del mercado. Con 4 constraints la solucion es casi unica."""
    from scipy.optimize import minimize
    W_OU, W_AH = 2.0, 1.5
    N = M.MAXG + 1
    if ou_data:
        ou_line = ou_data[0]
        ou_mask = np.array([[1.0 if i+j > ou_line else 0.0 for j in range(N)] for i in range(N)])
    if ah_data:
        ah_line = ah_data[0]
        # ah_line negativo = local favorito; cubre si i - j > -ah_line
        ah_mask = np.array([[1.0 if i-j > -ah_line else 0.0 for j in range(N)] for i in range(N)])
    def loss(p):
        la, lb = max(.15, p[0]), max(.15, p[1])
        mat = mat_from_lams(la, lb)
        err = (float(np.tril(mat,-1).sum())-tH)**2 + (float(np.triu(mat,1).sum())-tA)**2
        if ou_data:
            err += W_OU * (float((mat * ou_mask).sum()) - ou_data[1])**2
        if ah_data:
            err += W_AH * (float((mat * ah_mask).sum()) - ah_data[1])**2
        return err
    r = minimize(loss, [la0, lb0], method="Nelder-Mead",
                 options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 2000})
    return mat_from_lams(max(.15, r.x[0]), max(.15, r.x[1]))

def analyze(home, away):
    # presion de standings
    ph_mult = pa_mult = 1.0
    if _SC:
        ph_mult, pa_mult = _SC.get_lambda_mults(home, away)

    # penalizacion por tarjetas/lesiones
    card_pen_h = _WCC.get_lambda_pen(home) if _WCC else 1.0
    card_pen_a = _WCC.get_lambda_pen(away) if _WCC else 1.0

    la0, lb0 = M.lambdas(home, away, C)
    la0_adj   = la0 * ph_mult * card_pen_h
    lb0_adj   = lb0 * pa_mult * card_pen_a

    mat = M.build_matrix(home, away, C)
    if ph_mult != 1.0 or pa_mult != 1.0 or card_pen_h != 1.0 or card_pen_a != 1.0:
        mat = mat_from_lams(la0_adj, lb0_adj)

    ou_dt = None; ah_dt = None; mov = None
    mkt   = False
    flip  = False
    oo    = MODDS.get((home, away))
    if oo is None and (away, home) in MODDS:
        oo = MODDS[(away, home)]; flip = True
    if oo:
        rh, rd, ra = 1/oo[0], 1/oo[1], 1/oo[2]
        s = rh+rd+ra; tH, tA = rh/s, ra/s
        if flip: tH, tA = tA, tH

        ext = MODDS_EXT.get((home, away)) or (MODDS_EXT.get((away, home)) if flip else None)
        if ext and "ou_line" in ext:
            ou_dt = (ext["ou_line"], ext["p_over"])
        if ext and "ah_line" in ext:
            ah_line = -ext["ah_line"] if flip else ext["ah_line"]
            p_ah    = (1 - ext["p_ah_home"]) if flip else ext["p_ah_home"]
            # Solo lineas FRACCIONARIAS (±0.5, ±1.5...): ahi p_ah_home es prob absoluta.
            # En lineas ENTERAS (0.0, ±1...) el empate es push y p_ah_home es CONDICIONAL
            # (excluye el push); usarla como absoluta infla al local e invierte el favorito
            # en partidos parejos (bug detectado en CostaMarfil-Ecuador). Se omite el AH ahi.
            if abs(ah_line - round(ah_line)) > 0.01:
                ah_dt = (ah_line, p_ah)

        mov = MOVEMENT.get((home, away)) or (MOVEMENT.get((away, home)) if flip else None)
        if mov and mov.get("sharp"):
            boost = 0.06
            dH = -mov["dA"] if flip else mov["dH"]
            dA = -mov["dH"] if flip else mov["dA"]
            la0_adj *= (1 + boost * max(-1, min(1, dH)))
            lb0_adj *= (1 + boost * max(-1, min(1, dA)))

        mat = market_matrix(la0_adj, lb0_adj, tH, tA, ou_dt, ah_dt)
        mkt = True

    # CORRECCION DE MINNOW (mismatch extremo): las casas comprimen el handicap asiatico
    # por gestion de riesgo/liquidez, asi que el mercado (y el modelo calibrado a el)
    # SUBESTIMAN la goleada. Caso real: Alemania-Curazao acabo 7-1 con el mercado en
    # AH -3.5 (~4 goles). En mismatches claros inflamos lambda del favorito para
    # engrosar la cola de goleadas (4-0/5-0 en vez de quedarse en 3-0).
    pH_t = float(np.tril(mat, -1).sum()); pA_t = float(np.triu(mat, 1).sum())
    p_win = max(pH_t, pA_t)
    minnow = False
    if p_win > MINNOW_PWIN_THR:
        eh_ = sum(i*float(mat[i, :].sum()) for i in range(mat.shape[0]))
        ea_ = sum(j*float(mat[:, j].sum()) for j in range(mat.shape[1]))
        boost = MINNOW_BOOST * (p_win - MINNOW_PWIN_THR) / (1 - MINNOW_PWIN_THR)
        if pH_t >= pA_t:
            eh_ *= (1 + boost)
        else:
            ea_ *= (1 + boost)
        mat = mat_from_lams(max(.05, eh_), max(.05, ea_))
        minnow = True

    q = S.rival_pick_dist(home, away)
    pH = float(np.tril(mat, -1).sum()); pD = float(np.trace(mat)); pA = float(np.triu(mat, 1).sum())
    cand = []
    for px in range(M.MAXG+1):   # grid 0-8 (era 0-4)
        for py in range(M.MAXG+1):
            ev = sum(mat[ax, ay]*S.pts(px, py, ax, ay)
                     for ax in range(M.MAXG+1) for ay in range(M.MAXG+1))
            # bono unicidad: +2 SOLO si aciertas exacto Y eres unico
            # E[bonus] = P(exacto) * 2 * P(ningun rival lo tiene)
            uniq = mat[min(px,M.MAXG), min(py,M.MAXG)] * 2 * (1 - q.get((px, py), 0.0))**S.N_RIVALS
            cand.append((ev, uniq, px, py))
    evA, _, axp, ayp = max(cand, key=lambda t: t[0])
    evB, uqB, bxp, byp = max(cand, key=lambda t: t[0]+t[1])
    g6 = [[round(sum(mat[ax, ay]*S.pts(px, py, ax, ay)
                     for ax in range(M.MAXG+1) for ay in range(M.MAXG+1)), 3)
           for py in range(M.MAXG+1)] for px in range(M.MAXG+1)]
    p6 = [[round(float(mat[min(px, M.MAXG), min(py, M.MAXG)]), 4)
           for py in range(M.MAXG+1)] for px in range(M.MAXG+1)]

    signals = []
    if ph_mult > 1.05: signals.append("presion " + home)
    if ph_mult < 0.95: signals.append("clasificado " + home)
    if pa_mult > 1.05: signals.append("presion " + away)
    if pa_mult < 0.95: signals.append("clasificado " + away)
    if ou_dt: signals.append("O/U " + str(ou_dt[0]) + " (" + str(round(100*ou_dt[1])) + "% over)")
    if ah_dt: signals.append("AH " + ("+"+str(ah_dt[0]) if ah_dt[0]>=0 else str(ah_dt[0])) + " (" + str(round(100*ah_dt[1])) + "%)")
    if mov and mov.get("sharp"): signals.append("SHARP en linea")
    if _WCC:
        for alert in _WCC.get_alerts(home) + _WCC.get_alerts(away):
            signals.append(alert)

    return {"ph": round(100*pH, 1), "pd": round(100*pD, 1), "pa": round(100*pA, 1),
            "bx": bxp, "by": byp, "evb": round(evB+uqB, 2),
            "ax": axp, "ay": ayp, "eva": round(evA, 2), "mkt": mkt, "g6": g6, "p6": p6,
            "pex": round(100*float(mat[bxp, byp]), 1), "signals": signals,
            "la_h": round(la0, 3), "la_a": round(lb0, 3),
            "la_h_adj": round(la0_adj, 3), "la_a_adj": round(lb0_adj, 3),
            "ph_mult": round(ph_mult, 3), "pa_mult": round(pa_mult, 3),
            "card_pen_h": round(card_pen_h, 3), "card_pen_a": round(card_pen_a, 3)}

# resultados reales bajados por wc_data_feed.py (football-data.org)
RES = {}
try:
    with open(os.path.join(HERE, "results_live.json"), encoding="utf-8-sig") as f:
        for r in json.load(f):
            RES[(r["home"], r["away"])] = (r["gh"], r["ga"])
except FileNotFoundError:
    pass

# picks REALES de tu grupo (polla_sync.py). PRIVADO: solo para calcular tu pick,
# NO se embeben los datos crudos en el HTML publico.
GROUP = {}
try:
    with open(os.path.join(HERE, "group_stats.json"), encoding="utf-8") as f:
        GROUP = json.load(f)
except FileNotFoundError:
    pass

# OVERRIDES manuales: picks fijados a mano que el piloto automatico DEBE respetar
# (no recalcular). Formato: {"Home|Away": [gx, gy]}. Para jugadas deliberadas como
# aprovechar un hueco de goleada que el grupo no tiene. Se commitea (lo usa la nube).
OVERRIDE = {}
try:
    with open(os.path.join(HERE, "picks_override.json"), encoding="utf-8") as f:
        OVERRIDE = json.load(f)
except FileNotFoundError:
    pass

# PICK DE LA VUELTA ANTERIOR (picks.json de la pasada previa; archivo NO trackeado, asi
# que sobrevive al git reset del bucle en la nube). Sirve para la HISTERESIS: no cambiar
# un pick ya colocado por ruido de cuotas de ultimo minuto si el nuevo optimo apenas lo
# supera (evita el flip-flop 2-0<->1-0 entre marcadores casi empatados, sin valor real).
PREV_PICKS = {}
try:
    with open(os.path.join(HERE, "picks.json"), encoding="utf-8") as f:
        PREV_PICKS = json.load(f)
except (FileNotFoundError, ValueError):
    pass
HYST = 0.05   # margen de E[pts]: solo cambiar el pick si el nuevo lo supera por >0.05

# tabla de posiciones del grupo (standings.json). Privado: solo en tu URL secreta.
STANDINGS = []
try:
    with open(os.path.join(HERE, "standings.json"), encoding="utf-8") as f:
        STANDINGS = json.load(f)
except FileNotFoundError:
    pass

def group_optimal(a, taken):
    """PICK = maximiza el VALOR ESPERADO REAL de puntos dado el grupo, SIN heuristicas
    artificiales (sin penalizaciones de separacion ni umbrales de 'marcador creible',
    que sesgaban el pick — ej. Francia 3-1 cuando el 2-0 daba mas puntos).

    Para CADA marcador: E[pts] = EV del modelo (exacto 5 + ganador 2 + goles 1/equipo,
    ya ponderado por la matriz de probabilidad) + bono de unicidad (2*P(exacto)) SOLO si
    el marcador esta LIBRE (ningun rival del grupo lo tiene). Se elige el de mayor E[pts].

    Es el maximizador EXACTO de puntos esperados, y se comporta como queremos sin reglas
    ad-hoc: clava el marcador mas probable cuando ningun hueco libre lo supera (un
    marcador de cola libre tiene P baja -> EV bajo y bono pequeño -> no gana, ej. Francia
    2-0); se desvia a un hueco libre SOLO cuando su E[pts] real supera al pico (ej. Iraq
    0-1 libre supera al 0-2 saturado de 4 rivales; Ghana 1-1; Argentina 1-0)."""
    g6 = a["g6"]; p6 = a.get("p6", [[0]*9 for _ in range(9)])
    n = len(g6)
    def cnt_of(px, py):
        return taken.get(f"{px}-{py}", 0) if isinstance(taken, dict) else (1 if f"{px}-{py}" in taken else 0)
    best = None
    for px in range(M.MAXG+1):
        for py in range(M.MAXG+1):
            ev = g6[px][py] if px < n and py < n else 0
            pe = p6[px][py] if px < n and py < n else 0
            uniq_bonus = pe * 2 if cnt_of(px, py) == 0 else 0   # +2 si aciertas exacto Y eres unico
            tot = ev + uniq_bonus
            if best is None or tot > best[0]: best = (tot, px, py)
    return best[1], best[2]

def epts_of(a, px, py, taken):
    """E[pts] de jugar (px,py) con el mismo criterio que group_optimal: EV del modelo +
    bono de unicidad (2*P) si el marcador esta LIBRE en el grupo. Para la histeresis."""
    g6 = a["g6"]; p6 = a.get("p6", [[0]*9 for _ in range(9)]); n = len(g6)
    if px >= n or py >= n:
        return 0.0
    def cnt_of(i, j):
        return taken.get(f"{i}-{j}", 0) if isinstance(taken, dict) else (1 if f"{i}-{j}" in taken else 0)
    free = (taken is not None) and cnt_of(px, py) == 0
    return g6[px][py] + (2 * p6[px][py] if free else 0.0)

matches = []
for m in CAL:
    a = analyze(m["home"], m["away"])
    rr = RES.get((m["home"], m["away"]))
    if rr is None:
        rv = RES.get((m["away"], m["home"]))
        rr = (rv[1], rv[0]) if rv else (None, None)
    taken = GROUP.get(f"{m['home']}|{m['away']}")
    # PICK = maximizador de E[pts] dado el grupo (group_optimal): clava el marcador de mayor
    # valor esperado real (EV + bono de unicidad si esta libre). Se desvia a un hueco libre
    # SOLO cuando su E[pts] real supera al pico. Sin datos de grupo -> probabilidad pura.
    if taken:
        a["bx"], a["by"] = group_optimal(a, taken)
    else:
        a["bx"], a["by"] = a["ax"], a["ay"]
    # HISTERESIS: si ya habia un pick colocado y el nuevo optimo apenas lo supera (<HYST en
    # E[pts]), MANTENER el previo -> evita el flip-flop 2-0<->1-0 por ruido de cuotas de
    # ultimo minuto (un cambio REAL por alineaciones mueve el E[pts] mucho mas que HYST).
    prev = PREV_PICKS.get(f"{m['home']}|{m['away']}")
    if prev and len(prev) == 2:
        new_ep  = epts_of(a, a["bx"], a["by"], taken)
        prev_ep = epts_of(a, int(prev[0]), int(prev[1]), taken)
        if prev_ep >= new_ep - HYST:
            a["bx"], a["by"] = int(prev[0]), int(prev[1])
    auto = bool(taken)
    ovr = OVERRIDE.get(f"{m['home']}|{m['away']}")
    if ovr:                       # pick fijado a mano: manda sobre todo lo demas
        a["bx"], a["by"] = int(ovr[0]), int(ovr[1])
        a["override"] = True
    matches.append({**{k: m[k] for k in ("g", "home", "away", "date", "time", "dow", "dlabel", "venue")},
                    **a, "rx": rr[0], "ry": rr[1], "auto": auto, "grp": taken or None})

rows = [{k: r.get(k) for k in ("team", "R32", "R16", "QF", "SF", "FINAL", "CAMPEON", "R32_sd", "CAMPEON_sd")}
        for r in PROBS["rows"]]

# bonos de avance (advance_strategy.py) — TODO-O-NADA, marginales (EV ~1 pt total)
try:
    with open(os.path.join(HERE, "advance_picks.json"), encoding="utf-8") as f:
        ADV = json.load(f)
except FileNotFoundError:
    ADV = {"stages": {}, "max_pts": 29, "total_exp_bonus": 0, "note": ""}

HTML = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polla Mundial 2026 · v4</title>
<meta name="robots" content="noindex, nofollow">
<style>
:root{--bg:#eef1f6;--card:#fff;--ink:#13245c;--mut:#5d6b8a;--line:#e6eaf1;--acc:#2563eb;--ok:#0e9f6e;--warn:#d97706;--bad:#dc2626;}
@media(prefers-color-scheme:dark){:root{--bg:#0e1422;--card:#16203a;--ink:#e8ecf6;--mut:#9aa7c4;--line:#243155;--acc:#60a5fa;--ok:#34d399;--warn:#fbbf24;--bad:#f87171;}}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
.hero{background:linear-gradient(135deg,#0b1b44,#1d4ed8 70%,#2563eb);color:#fff;padding:22px 0 18px}
header{padding:0 16px;max-width:980px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px;font-weight:700;color:#fff}
.sub{color:#cfe0ff;font-size:13px;margin:0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;max-width:980px;margin:14px auto 0;padding:0 16px}
.stat{background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.22);border-radius:10px;padding:8px 12px}
.stat b{display:block;font-size:20px;color:#fff}
.stat span{font-size:12px;color:#cfe0ff}
.risk{display:inline-block;background:#FAEEDA;color:#854F0B;border-radius:6px;padding:0 7px;font-size:12px;font-weight:600;margin-left:6px;white-space:nowrap}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px}
.chip{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:9px;padding:8px 12px;font-size:15px;font-weight:600;cursor:pointer;min-width:46px;transition:all .12s}
.chip.on{background:#d97706;border-color:#d97706;color:#fff}
.sgrid{display:grid;grid-template-columns:auto repeat(6,1fr);gap:4px;margin-top:8px;max-width:380px;align-items:center}
.sgrid .cell{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:7px;padding:9px 0;font-size:14px;font-weight:600;cursor:pointer;text-align:center;transition:all .1s}
.sgrid .cell.on{background:#d97706;border-color:#d97706;color:#fff}
.sgrid .axis{font-size:11px;color:var(--mut);text-align:center;font-weight:600}
@media(prefers-color-scheme:dark){.risk{background:#633806;color:#FAC775}}
.navwrap{position:sticky;top:0;z-index:20;background:var(--bg);border-bottom:1px solid var(--line)}
nav{display:flex;gap:8px;max-width:980px;margin:0 auto;padding:10px 16px;overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:none}
nav::-webkit-scrollbar{display:none}
nav button{border:1px solid var(--line);background:var(--card);padding:8px 14px;border-radius:9px;font-size:14px;cursor:pointer;white-space:nowrap;flex:0 0 auto;transition:background .12s}
nav button.on{background:#2563eb;border-color:#2563eb;color:#fff;box-shadow:0 2px 8px rgba(37,99,235,.35)}
main{max-width:980px;margin:14px auto 60px;padding:0 16px;display:none}
main.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:12px}
.day{font-weight:600;font-size:15px;margin:18px 4px 8px}
.day.today{color:var(--acc)}
.mx{display:grid;grid-template-columns:64px 1fr auto;gap:10px;align-items:center;padding:9px 0;border-top:1px solid var(--line)}
.mx:first-of-type{border-top:0}
.t{font-size:13px;color:var(--mut)}
.tm{font-size:15px}
.pick{font-weight:700;color:#1d4ed8;background:#dbeafe;border-radius:7px;padding:1px 8px;white-space:nowrap}
@media(prefers-color-scheme:dark){.pick{color:#bfdbfe;background:#1e3a8a}}
.alt{font-size:12px;color:var(--mut);white-space:nowrap}
.bar{height:6px;border-radius:3px;display:flex;overflow:hidden;margin-top:4px;max-width:260px}
.bar i{display:block;height:100%}
.res{width:64px;border:1px solid var(--line);border-radius:7px;padding:4px 6px;font-size:14px;text-align:center}
.ptsb{font-size:12px;color:var(--warn);min-width:52px;text-align:right}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:380px}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--line)}
th{cursor:pointer;color:var(--mut);font-weight:600;font-size:12px}
td:first-child,th:first-child{text-align:left}
.tag{display:inline-block;background:#EEEDFE;color:#3C3489;border-radius:6px;padding:1px 7px;font-size:12px;margin-left:6px}
.note{font-size:13px;color:var(--mut)}
.sigs{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
.sig{font-size:11px;padding:2px 8px;border-radius:999px;font-weight:500;white-space:nowrap}
.sig-red{background:#FCEBEB;color:#A32D2D}
.sig-yellow{background:#FAEEDA;color:#633806}
.sig-orange{background:#FAEEDA;color:#854F0B}
.sig-green{background:#EAF3DE;color:#3B6D11}
.sig-blue{background:#E6F1FB;color:#185FA5}
.sig-sharp{background:#EEEDFE;color:#534AB7}
.sig-gray{background:#F1EFE8;color:#5F5E5A}
.lam{font-size:12px;color:var(--mut);margin-top:3px}
.lam b{color:var(--txt)}
.tot{font-size:15px;font-weight:600}
select{border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:14px;background:var(--card)}
@media(max-width:560px){.mx{grid-template-columns:48px 1fr;gap:8px}.mx>.right{grid-column:1/3;justify-self:end;margin-top:4px}
 body{font-size:15px}.card{padding:12px 14px}h1{font-size:20px}
 .stats{grid-template-columns:1fr 1fr;gap:8px}.stat{padding:9px 11px}.stat b{font-size:19px}
 nav button{padding:8px 12px;font-size:13px}
 .pick{font-size:16px!important}.bar{max-width:none}}
#updateBtn{background:none;border:1px solid var(--line);border-radius:8px;padding:5px 11px;font-size:12px;cursor:pointer;color:var(--sub);margin-top:6px;transition:all .15s}
#updateBtn:hover{border-color:var(--acc);color:var(--acc)}
#updateBtn:disabled{opacity:.5;cursor:default}
</style></head><body>
<div class="hero"><header><h1>⚽ Polla Mundial 2026 <span class="tag">modelo v4</span></h1>
<p class="sub">Generado __GEN__ · bracket oficial 2026 · calibrado a 48 cuotas reales (devig Shin, c=__C__) · picks = max EV (exactos + parciales + unicidad real)</p>
<button id="updateBtn" onclick="triggerUpdate()">Actualizar ahora</button></header>
<div class="stats">
<div class="stat"><b>__D_FINAL__</b><span>días para la final</span></div>
<div class="stat"><b id="stJug">0/72</b><span>resultados metidos</span></div>
<div class="stat"><b>__P1__%</b><span>P(ganar) en vivo <span style="font-size:10px;opacity:.7">__P1_CI__pp · pos.__RANK__</span></span></div>
<div class="stat"><b>Brier __BRIER__</b><span>calibración modelo (↓ mejor)</span></div>
<div class="stat"><b><span id="nextrunStat">—</span></b><span>próxima actualización</span></div>
</div></div>
<div class="navwrap"><nav><button data-t="hoy" class="on">Hoy: qué hacer</button><button data-t="tabla">Clasificación</button><button data-t="cal">Calendario y picks</button><button data-t="avanza">Quién avanza</button><button data-t="news">Noticias</button><button data-t="probs">Probabilidades</button><button data-t="strat">Estrategia</button><button data-t="rules">Reglas</button></nav></div>
<main id="hoy" class="on">
<div class="card" style="border-left:3px solid var(--acc);border-radius:0 14px 14px 0"><b>🔒 Cierra apuestas en</b> <span id="countdown" style="font-weight:700;color:var(--acc)">—</span><div id="nextMatch" class="note" style="margin-top:2px"></div></div>
<div class="card"><b>1 · Las apuestas se colocan solas</b> <span class="note">El sistema apuesta con mucha anticipación y revisa cada ~15 min hasta el cierre (saque −30 min). Si el modelo cambia el pick antes del cierre, se actualiza. No tienes que hacer nada.</span>
<div id="todayList"></div></div>
<div class="card"><b>2 · Tu posición en la polla</b> <span class="note">(se actualiza automáticamente con los resultados oficiales)</span>
<div id="posbox" style="margin-top:6px"></div>
<div id="rec" class="note" style="margin-top:8px"></div></div>
__ANALYSIS_TODAY__
<div class="card"><b>3 · Qué hace el sistema cada 15 minutos (sin que toques nada)</b>
<ul style="margin:8px 0 6px;padding-left:20px;font-size:14px;line-height:1.9">
<li>📊 Descarga cuotas <b>1X2 + O/U + Asian Handicap</b> (~23 casas) y detecta movimiento sharp</li>
<li>🎯 Ajusta las lambdas Dixon-Coles a esas cuotas (4 constraints) → la mejor estimación de goles del partido</li>
<li>💥 En goleadas (grande vs equipo muy débil) <b>infla los goles del favorito</b>: el mercado los subestima</li>
<li>🟥 Baja <b>tarjetas y suspensiones</b> (ESPN) → −10% lambda por jugador suspendido</li>
<li>⚡ Ajusta por <b>presión de grupo</b>: 0pts en jornada 3 = +15% goles; ya clasificado = −10%</li>
<li>🔬 <b>Calibración Bayesiana online</b>: aprende de cada gol del Mundial</li>
<li>🎲 Simula <b>30.000 Mundiales</b> (Monte Carlo, bracket oficial FIFA, Elo en vivo)</li>
<li>📈 Elige el marcador de <b>mayor puntos esperados</b> = probabilidad de acierto + bono si nadie de tu grupo lo tiene; suele ser el más probable, y se mueve a un hueco libre solo si ahí gana más</li>
<li>⚡ Resultados en <b>TIEMPO REAL</b> (ESPN, al pitido final) → actualiza tu clasificación</li>
<li>🤖 <b>Apuesta por ti</b> con anticipación y revisa cada ~15 min hasta el cierre (saque −30 min)</li>
</ul>
<div class="note">Última actualización: <b>__GEN__</b> · estado de las conexiones:</div>
<ul style="margin:6px 0 0;padding-left:20px;font-size:14px">__SETUP__</ul></div></main>
<main id="tabla">
<div class="card"><b>Clasificación de tu grupo</b> <span class="note">"Polla Mundial 2026" · actualizada automáticamente con los puntos oficiales de la web.</span>
<div class="tablewrap" style="margin-top:8px"><table id="standtbl"><thead><tr><th>#</th><th>Jugador</th><th>Exa</th><th>Gan</th><th>Gol</th><th>Úni</th><th>Pts</th></tr></thead><tbody></tbody></table></div></div></main>
<main id="cal">
<div class="card"><span class="tot" id="score">0 pts</span> <span class="note" id="scoren">— introduce resultados (ej. 2-1) y calculo los puntos de tu pick valiente. El bono unicidad (+2) no es calculable aquí: súmalo cuando lo veas en la web.</span><br>
<label class="note">Ver: <select id="statef"><option value="pend">⏳ pendientes</option><option value="played">✅ jugados</option><option value="all">todos</option></select></label>
<label class="note" style="margin-left:12px">Grupo: <select id="gf"><option value="">todos</option></select></label>
<label class="note" style="margin-left:12px"><input type="checkbox" id="onlyClose"> parejos ⚖</label></div>
<div id="days"></div></main>
<main id="avanza">
<div class="card"><b>Los bonos de avance son marginales.</b> <span class="note">Regla REAL de tu grupo: cada bono es <b>todo-o-nada</b> — solo lo cobras si aciertas <b>TODOS</b> los equipos que pasan de esa ronda (acertar 7 de 8 = 0 pts). Máximo 29 pts en total y casi imposibles de cobrar (EV realista ~1 pt). <b>La polla se gana en los MARCADORES, no aquí.</b></span></div>
<div class="card"><b>Jugada óptima: favoritos, NO diferenciarse.</b> <span class="note">En un bono todo-o-nada, un pick contrarian que falla te borra el bono entero. Por eso pones los más probables. El único con algo de valor es el campeón (EV ~0.9 pts).</span><div id="advlist" style="margin-top:8px"></div></div></main>
<main id="news">__NEWS__</main>
<main id="probs"><div class="card"><p class="note">Probabilidad de cada equipo de llegar a cada fase. Ordenado por "pasa de grupos (R32)". Pulsa una columna para reordenar.</p>
<div style="position:relative;height:300px;margin-bottom:14px"><canvas id="champChart"></canvas></div>
<div class="tablewrap"><table id="pt"><thead><tr><th data-k="team">Equipo</th><th data-k="R32">R32</th><th data-k="R16">Octavos</th><th data-k="QF">Cuartos</th><th data-k="SF">Semis</th><th data-k="FINAL">Final</th><th data-k="CAMPEON">Campeón</th></tr></thead><tbody></tbody></table></div></div></main>
<main id="strat"><div class="card">
<p><b>La clave: marcadores exactos.</b> Acertar el marcador exacto da 5 pts — más que ganar ganador+ambos goles. El sistema elige el marcador de <b>mayor puntos esperados (E[pts])</b>, calculado de forma exacta: E[pts] = valor del modelo (exacto 5 + ganador 2 + goles por equipo, ponderado por la probabilidad de cada resultado posible) + <b>bono de unicidad</b> (+2 × probabilidad de acierto) si ningún rival de tu grupo tiene ese marcador. En la mayoría de partidos eso coincide con el marcador más probable; solo se desvía a un hueco libre cuando ese hueco da MÁS puntos esperados (su probabilidad casi igual + el bono por ser único). No hay reglas inventadas ni umbrales: es el maximizador exacto de puntos, así que nunca persigue marcadores improbables (su baja probabilidad hunde su E[pts]).</p>
<p><b>Qué datos usa el modelo en cada pick:</b></p>
<ul>
<li><b>Cuotas 1X2 + O/U + Asian Handicap</b> (~23 casas) — 4 constraints determinan λ_local y λ_visitante casi únicamente. Sin O/U, infinitas combinaciones de goles satisfacen el mismo 1X2.</li>
<li><b>Tarjetas y suspensiones</b> (ESPN, automático) — cada jugador suspendido aplica −10% lambda (cap −25%). Señal 🟥 en el calendario.</li>
<li><b>Presión de grupo</b> — 0pts en jornada 3 = +15% goles (todo o nada). Ya clasificado = −10% (posible rotación). Señal ⚡ en el calendario.</li>
<li><b>Movimiento de línea sharp</b> — cuotas que se mueven >12% entre ciclos = dinero informado. El modelo ajusta +6% al lado que se acorta.</li>
<li><b>Calibración Bayesiana online</b> — aprende de cada gol del Mundial. ATT/DEF por equipo se ajustan con resultados reales.</li>
<li><b>Elo en vivo</b> — K=40 en grupos, actualizado tras cada resultado.</li>
</ul>
<p><b>Bono unicidad (+2):</b> se cobra SOLO si aciertas el exacto Y eres el único del grupo. Entra directo en el cálculo de E[pts]: un marcador libre (que nadie tiene) suma +2×P(acierto), y el sistema lo elige cuando eso lo hace ganar al marcador más probable (ej. si el favorito obvio lo tienen 4 rivales y hay un marcador casi igual de probable libre). Si el hueco libre es poco probable, su bono pequeño no compensa y se queda en el más probable.</p>
<p><b>Sin bandazos de último minuto (histéresis):</b> una vez colocado un pick, el sistema no lo cambia por ruido de cuotas — solo si el nuevo marcador supera al puesto por un margen real de puntos esperados. Así no ves el marcador oscilando entre dos casi-empatados (ej. 2-0 ↔ 1-0); sí reacciona a cambios de verdad, como una alineación confirmada.</p>
<p><b>Donde se gana de verdad:</b> los MARCADORES, sobre todo los exactos (5 pts, o 7 si eres único). Los bonos de avance son todo-o-nada y marginales (EV ~1 pt total). No los persigas con picks arriesgados.</p>
<p><b>De dónde viene la ventaja:</b> vas último, así que copiar al grupo te deja último. La ventaja es doble: (1) el modelo usa cuotas reales del mercado — lo mejor que existe — mientras los rivales pican a ojo; (2) cuando un marcador casi igual de probable está libre, el sistema lo prefiere (vía el bono de unicidad en el E[pts]) para sumar exactos que nadie más tiene. En un Mundial tan variable, clavar exactos y separarte cuando sale a cuenta es lo que rompe el empate a tu favor.</p>
<p><b>Alineaciones de último minuto:</b> se captan vía las cuotas — cuando se confirma que un crack es suplente, las casas mueven la línea ~1h antes y el sistema baja cuotas frescas y reajusta el pick antes del cierre.</p>
</div></main>
<main id="rules"><div class="card"><table><tbody>
<tr><td>Marcador exacto</td><td>5</td></tr><tr><td>Ganador o empate acertado</td><td>2</td></tr>
<tr><td>Gol acertado (por equipo)</td><td>1</td></tr><tr><td>Predicción única</td><td>2</td></tr>
<tr><td>Bono dieciseisavos</td><td>10</td></tr><tr><td>Bono octavos</td><td>8</td></tr>
<tr><td>Bono cuartos</td><td>4</td></tr><tr><td>Bono semifinales</td><td>2</td></tr>
<tr><td>Bono final</td><td>5</td></tr><tr><td>Premios</td><td>70% / 20% / 10%</td></tr>
</tbody></table><p class="note"><b>Bonos de avance = TODO-O-NADA</b>: solo se cobran si aciertas TODOS los equipos que pasan de esa ronda (acertar 7 de 8 = 0 pts). Por eso son marginales. Fuente: pollamundial.org (reglas verificadas en la base de datos del grupo).</p></div></main>
<script>
"use strict";
var DATA=__DATA__;
var PROBS=__PROBS__;
var ADV=__ADV__;
var STANDINGS=__STANDINGS__;
var store={};
try{ store=JSON.parse(localStorage.getItem("polla_v4")||"{}"); }catch(e){ store={}; }
function save(){ try{ localStorage.setItem("polla_v4",JSON.stringify(store)); }catch(e){} }
function pts(px,py,ax,ay){
 if(px===ax&&py===ay)return 5;          // exacto: solo 5 (no se acumula)
 var p=0;if(Math.sign(px-py)===Math.sign(ax-ay))p+=2; if(px===ax)p+=1; if(py===ay)p+=1; return p;}
function todayStr(){ try{ return new Date().toLocaleDateString("en-CA",{timeZone:"Europe/Madrid"}); }catch(e){ return new Date().toISOString().slice(0,10);} }
var tabs=document.querySelectorAll("nav button");
tabs.forEach(function(b){ b.addEventListener("click",function(){
 tabs.forEach(function(x){x.classList.remove("on")}); b.classList.add("on");
 document.querySelectorAll("main").forEach(function(m){m.classList.remove("on")});
 document.getElementById(b.dataset.t).classList.add("on"); }); });
var gf=document.getElementById("gf");
"ABCDEFGHIJKL".split("").forEach(function(g){var o=document.createElement("option");o.value=g;o.textContent="Grupo "+g;gf.appendChild(o);});
gf.addEventListener("change",render);
var onlyClose=document.getElementById("onlyClose");
onlyClose.addEventListener("change",render);
var statef=document.getElementById("statef");
statef.addEventListener("change",render);
function isClose(m){return Math.abs(m.ph-m.pa)<8;}
function isPlayed(m){ return (m.rx!=null&&m.ry!=null) || !!store[m.date+"|"+m.home+"|"+m.away]; }
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;");}
function render(){
 var hoy=todayStr(); var cont=document.getElementById("days"); cont.innerHTML="";
 var st=statef.value;
 var byDay={}; DATA.forEach(function(m){
  if(gf.value&&m.g!==gf.value)return;
  if(onlyClose.checked&&!isClose(m))return;
  if(st==="pend"&&isPlayed(m))return;
  if(st==="played"&&!isPlayed(m))return;
  (byDay[m.date]=byDay[m.date]||[]).push(m); });
 var total=0,nres=0;
 Object.keys(byDay).sort().forEach(function(d){
  var h=document.createElement("div"); h.className="day"+(d===hoy?" today":"");
  var first=byDay[d][0]; h.textContent=first.dow+" "+first.dlabel+(d===hoy?"  ·  HOY":"");
  cont.appendChild(h);
  var card=document.createElement("div"); card.className="card";
  byDay[d].forEach(function(m){
   var key=m.date+"|"+m.home+"|"+m.away;
   var row=document.createElement("div"); row.className="mx";
   var api=(m.rx!=null&&m.ry!=null)?m.rx+"-"+m.ry:"";
   var val=store[key]||api;
   var p="";
   if(val){ var r=val.split("-"); var got=pts(m.bx,m.by,+r[0],+r[1]); total+=got; nres++;
    p=got+" pts"+((api&&!store[key])?" · auto":""); }
   // badges de señales con colores semanticos
   var sigHtml='';
   if(m.signals&&m.signals.length){
    var badges=m.signals.map(function(s){
     var c=s.indexOf('suspendido')>=0?'sig-red':
           s.indexOf('riesgo')>=0?'sig-yellow':
           s.indexOf('presion')>=0?'sig-orange':
           s.indexOf('clasificado')>=0?'sig-green':
           s.indexOf('O/U')>=0||s.indexOf('AH ')>=0?'sig-blue':
           s.indexOf('SHARP')>=0?'sig-sharp':'sig-gray';
     return '<span class="sig '+c+'">'+esc(s)+'</span>';
    });
    sigHtml='<div class="sigs">'+badges.join('')+'</div>';
   }
   // linea de lambdas ajustadas (solo si hay ajuste real)
   var lamHtml='';
   if(m.la_h!==undefined){
    var pctH=Math.round((m.la_h_adj/m.la_h-1)*100);
    var pctA=Math.round((m.la_a_adj/m.la_a-1)*100);
    var fmtH=(pctH>0?'+':'')+pctH+'%';
    var fmtA=(pctA>0?'+':'')+pctA+'%';
    var lamParts=[];
    if(pctH!==0) lamParts.push('λ '+esc(m.home)+' '+fmtH+' ('+m.la_h_adj.toFixed(2)+')');
    if(pctA!==0) lamParts.push('λ '+esc(m.away)+' '+fmtA+' ('+m.la_a_adj.toFixed(2)+')');
    if(lamParts.length) lamHtml='<div class="lam">'+lamParts.join(' · ')+'</div>';
   }
   row.innerHTML='<div class="t">'+m.time+'<br>['+m.g+']</div>'
    +'<div><span class="tm">'+esc(m.home)+' – '+esc(m.away)+'</span> '
    +'<span class="pick">'+m.bx+'-'+m.by+'</span> <span class="alt">(EV '+m.evb.toFixed(2)+' pts · P(exacto) '+m.pex+'%'+(m.bx!==m.ax||m.by!==m.ay?' · alternativa '+m.ax+'-'+m.ay:'')+' )</span>'
    +(isClose(m)?'<span class="risk">⚖ parejo</span>':'')
    +'<div class="bar"><i style="width:'+m.ph+'%;background:#1D9E75"></i><i style="width:'+m.pd+'%;background:#B4B2A9"></i><i style="width:'+m.pa+'%;background:#D85A30"></i></div>'
    +'<div class="t">'+m.ph+'% / '+m.pd+'% / '+m.pa+'%'+(m.mkt?' · mercado (O/U+AH+1X2)':'')+'  '+esc(m.venue)+'</div>'
    +sigHtml+lamHtml+'</div>'
    +'<div class="right"><input class="res" placeholder="res" value="'+val+'"> <span class="ptsb">'+p+'</span></div>';
   var inp=row.querySelector("input");
   inp.addEventListener("change",function(){
    var v=inp.value.trim();
    if(/^\\d+-\\d+$/.test(v)){ store[key]=v; } else { delete store[key]; inp.value=""; }
    save(); render(); });
   card.appendChild(row);
  });
  cont.appendChild(card);
 });
 // total y contador SIEMPRE sobre TODOS los partidos jugados (no solo el filtro)
 var gtot=0,gres=0,done=0;
 DATA.forEach(function(m){
  var v=store[m.date+"|"+m.home+"|"+m.away]||((m.rx!=null&&m.ry!=null)?m.rx+"-"+m.ry:"");
  if(v){ var r=v.split("-"); gtot+=pts(m.bx,m.by,+r[0],+r[1]); gres++; done++; }
 });
 document.getElementById("score").textContent=gtot+" pts";
 document.getElementById("scoren").textContent="("+gres+" partidos jugados; sin contar bonos de unicidad ni de avance — esos los ves en la pestaña Clasificación)";
 document.getElementById("stJug").textContent=done+"/72";
}
var sortK="R32",sortAsc=false;
function pct(v){return (Math.round(v*10)/10).toFixed(1)+"%";}
function renderProbs(){
 var tb=document.querySelector("#pt tbody"); tb.innerHTML="";
 var rs=PROBS.slice().sort(function(a,b){
  if(sortK==="team")return sortAsc?a.team.localeCompare(b.team):b.team.localeCompare(a.team);
  return sortAsc?a[sortK]-b[sortK]:b[sortK]-a[sortK];});
 rs.forEach(function(r){
  var tr=document.createElement("tr");
  tr.innerHTML="<td>"+esc(r.team)+"</td><td>"+pct(r.R32)+"</td><td>"+pct(r.R16)+"</td><td>"+pct(r.QF)
   +"</td><td>"+pct(r.SF)+"</td><td>"+pct(r.FINAL)+"</td><td>"+pct(r.CAMPEON)+"</td>";
  tb.appendChild(tr); });
}
document.querySelectorAll("#pt th").forEach(function(th){ th.addEventListener("click",function(){
 if(sortK===th.dataset.k)sortAsc=!sortAsc; else {sortK=th.dataset.k;sortAsc=false;} renderProbs(); }); });
function addDays(s,n){var d=new Date(s+"T12:00:00");d.setDate(d.getDate()+n);return d.toISOString().slice(0,10);}
function parseGroup(txt){
 var taken={}; if(!txt) return taken;
 var re=/(\\d+)\\s*-\\s*(\\d+)/g, mm;
 while((mm=re.exec(txt))!==null){ taken[mm[1]+"-"+mm[2]]=true; }
 return taken;
}
function parseGroupList(txt){ return Object.keys(parseGroup(txt)); }
function optimalWithGroup(m, gtxt){
 var taken=parseGroup(gtxt); var hasGroup=Object.keys(taken).length>0;
 // sin datos del grupo: pick del sistema (politica B, con unicidad estimada)
 if(!hasGroup||!m.g6){
  var n=m.auto?"✅ ajustado a tu grupo (automático) · P(exacto) "+m.pex+"%":"P(exacto) "+m.pex+"%"+(m.bx!==m.ax||m.by!==m.ay?" · alternativa "+m.ax+"-"+m.ay:"");
  return {px:m.bx, py:m.by, note:n};
 }
 // con datos del grupo: bono unico = P(exacto)*2 SOLO cuando aciertas el marcador exacto
 var best=null;
 var gMax=(m.g6&&m.g6.length)||9;
 for(var px=0;px<gMax;px++) for(var py=0;py<gMax;py++){
  var pex_cell=(m.p6&&m.p6[px]&&m.p6[px][py]!=null)?m.p6[px][py]:0;
  var total=m.g6[px][py]+(!taken[px+"-"+py]?pex_cell*2:0);
  if(!best||total>best.total) best={px:px,py:py,total:total};
 }
 var nT=Object.keys(taken).length;
 var note=taken[best.px+"-"+best.py]
  ? "tu grupo ya lo tiene; aun así es el de más EV"
  : "unico en tu grupo (P exacto "+(((m.p6&&m.p6[best.px]&&m.p6[best.px][best.py])||0)*100).toFixed(1)+"%) · "+nT+" pick"+(nT>1?"s":"")+" suyos descartados";
 return {px:best.px, py:best.py, note:note};
}
function renderHoy(){
 var hoy=todayStr(), man=addDays(hoy,1);
 var list=document.getElementById("todayList"); list.innerHTML="";
 var nowD=new Date();
 // solo partidos AUN ABIERTOS (no jugados ni cerrados) de las proximas 26 h
 var prox=DATA.filter(function(m){return (m.date===hoy||(m.date===man&&m.time<="08:00"))&&!isPlayed(m)&&closeTime(m)>nowD;});
 if(!prox.length){ prox=DATA.filter(function(m){return !isPlayed(m)&&closeTime(m)>nowD;}).slice(0,4);
  list.innerHTML='<p class="note">No hay partidos abiertos en las próximas 26 h. Los siguientes:</p>'; }
 prox.forEach(function(m){
  var closed=closeTime(m)<new Date();
  var closeLbl=closeTime(m).toLocaleTimeString("es-ES",{timeZone:"Europe/Madrid",hour:"2-digit",minute:"2-digit"});
  var div=document.createElement("div"); div.style.padding="12px 0"; div.style.borderTop="1px solid var(--line)";
  if(closed)div.style.opacity="0.5";
  // PICK = el que de verdad se apuesta (m.bx-m.by = maxima probabilidad). Es EXACTAMENTE
  // lo que el autobet coloca en la web: el dashboard nunca debe mostrar otro marcador.
  div.innerHTML='<div style="display:flex;justify-content:space-between;align-items:baseline">'
   +'<span class="tm">'+esc(m.home)+' – '+esc(m.away)+'</span>'
   +'<span class="t">'+(closed?'<b style="color:#dc2626">CERRADO</b>':'🔒 cierra '+closeLbl)+' · saque '+m.time+'</span></div>'
   +'<div style="margin:4px 0"><span class="pick" style="font-size:24px">'+m.bx+' - '+m.by+'</span>'
   +' <span class="alt gnote">P(exacto) '+m.pex+'% · marcador de mayor puntos esperados (probabilidad + bono si nadie del grupo lo tiene)</span></div>'
   +'<div class="t">gana '+esc(m.home)+' '+m.ph+'% / empate '+m.pd+'% / gana '+esc(m.away)+' '+m.pa+'%</div>';
  list.appendChild(div); });
 function rec(){
  var out=document.getElementById("rec"); var box=document.getElementById("posbox");
  var me=STANDINGS.filter(function(r){return r.me;}).sort(function(a,b){return b.pts-a.pts;})[0];
  var lead=STANDINGS[0];
  if(!me||!STANDINGS.length){ box.textContent=""; out.textContent="Aún sin puntos (no se ha jugado nada). En cuanto haya resultados, aquí verás tu posición y qué marcha usar."; return; }
  var mi=me.pts, li=lead.pts;
  box.innerHTML="Vas <b>"+me.pos+"º</b> de "+STANDINGS.length+" · <b>"+mi+" pts</b> · líder ("+esc(lead.name)+") "+li+" pts";
  var d=li-mi;
  if(d<=0){ out.innerHTML="<b style='color:var(--ok)'>Vas LÍDER (+"+(-d)+").</b> El sistema apuesta el marcador de mayor puntos esperados en cada partido. No tienes que tocar nada."; }
  else { out.innerHTML="<b>A "+d+" pts del líder.</b> El sistema apuesta el marcador de <b>mayor puntos esperados</b>: la probabilidad real (cuotas del mercado) más un bono cuando un marcador casi igual de probable está libre en tu grupo. La vía para remontar es acertar más <b>exactos</b> que los rivales (que pican a ojo). No tienes que tocar nada."; }
 }
 rec();
}
function renderAvanza(){
 var el=document.getElementById("advlist"); if(!el)return;
 var order=["b32","b16","bQ","bS","bF"];
 var stages=ADV.stages||{};
 var html=order.filter(function(k){return stages[k];}).map(function(k){
  var s=stages[k];
  var pl=(s.picks||[]).map(function(t){return esc(t);}).join(", ");
  return '<div style="padding:6px 0;border-top:1px solid var(--line);font-size:14px">'
   +'<b>'+esc(s.label)+'</b> <span class="note">(vale '+s.pts+' pts · acierta '+s.n
   +' · P(todos) '+s.p_all+'% · EV '+(s.exp_bonus||0).toFixed(2)+' pts)</span><br>'
   +'<span style="color:var(--ink)">'+pl+'</span></div>';
 }).join("");
 el.innerHTML=html+'<div class="note" style="margin-top:8px">EV total de los bonos: '
  +(ADV.total_exp_bonus||0).toFixed(1)+' pts (máx '+(ADV.max_pts||29)+'). Marginal frente a los marcadores.</div>';
}
function renderStandings(){
 var tb=document.querySelector("#standtbl tbody"); if(!tb)return; tb.innerHTML="";
 STANDINGS.forEach(function(r){
  var tr=document.createElement("tr");
  if(r.me)tr.style.background="#dbeafe";
  tr.innerHTML="<td>"+r.pos+"</td><td>"+(r.me?"<b>":"")+esc(r.name)+(r.sc>1?" ("+r.sc+")":"")+(r.me?" ⬅</b>":"")
   +"</td><td>"+r.ceS+"</td><td>"+r.ccW+"</td><td>"+r.ccG+"</td><td>"+(r.cuP||0)+"</td><td><b>"+r.pts+"</b></td>";
  tb.appendChild(tr);
 });
 if(!STANDINGS.length)tb.innerHTML='<tr><td colspan="7" class="note">Aún sin puntos (no se ha jugado ningún partido).</td></tr>';
}
render(); renderProbs(); renderHoy(); renderAvanza(); renderStandings();
// las apuestas cierran 30 min antes del saque (la web bloquea las predicciones ahi)
function kickoff(m){ return new Date(m.date+"T"+(m.time.length===5?m.time:"0"+m.time)+":00+02:00"); }
function closeTime(m){ return new Date(kickoff(m).getTime()-30*60000); }
function hm(ms){ var h=Math.floor(ms/3.6e6), mi=Math.floor((ms%3.6e6)/6e4), d=Math.floor(h/24);
 return (d>0?d+"d ":"")+(h%24)+"h "+mi+"m"; }
// cuenta atras al CIERRE de apuestas del proximo partido aun abierto
function tickCountdown(){
 var now=new Date(), next=null;
 for(var i=0;i<DATA.length;i++){ if(closeTime(DATA[i])>now){ next=DATA[i]; break; } }
 var cd=document.getElementById("countdown"), nm=document.getElementById("nextMatch");
 if(!next){ cd.textContent="—"; nm.textContent="No quedan partidos por apostar"; return; }
 var diff=Math.max(0,closeTime(next)-now);
 var urgent=diff<3.6e6;  // menos de 1h
 cd.textContent=hm(diff); cd.style.color=urgent?"#dc2626":"var(--acc)";
 nm.innerHTML="<b>"+esc(next.home)+" – "+esc(next.away)+"</b> · cierra a las "
  +closeTime(next).toLocaleTimeString("es-ES",{timeZone:"Europe/Madrid",hour:"2-digit",minute:"2-digit"})
  +" (saque "+next.time+")"+(urgent?" · <b style='color:#dc2626'>¡corre!</b>":"");
}
tickCountdown(); setInterval(tickCountdown,20000);
function tickNextRun(){
  var now=new Date();
  var ms=now.getUTCMinutes()*60000+now.getUTCSeconds()*1000+now.getUTCMilliseconds();
  var interval=15*60000;
  var left=interval-(ms%interval);
  var s=Math.ceil(left/1000);
  var m=Math.floor(s/60), ss=s%60;
  var el=document.getElementById('nextrunStat');
  if(s<=3){ if(el) el.textContent='Recargando...'; location.reload(); return; }
  if(el) el.textContent=(m>0?m+'m ':'')+ss+'s';
}
tickNextRun(); setInterval(tickNextRun,1000);
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
"use strict";
(function(){
 if(!window.Chart) return;  // sin internet: la tabla sigue funcionando
 var topTeams=PROBS.slice().sort(function(a,b){return b.CAMPEON-a.CAMPEON;}).slice(0,12);
 var dark=window.matchMedia&&matchMedia("(prefers-color-scheme: dark)").matches;
 var tc=dark?"#a3a29a":"#6b6a64";
 new Chart(document.getElementById("champChart"),{type:"bar",
  data:{labels:topTeams.map(function(r){return r.team;}),
   datasets:[{label:"P(campeón) %",data:topTeams.map(function(r){return r.CAMPEON;}),
    backgroundColor:dark?"#AFA9EC":"#534AB7",borderRadius:4}]},
  options:{responsive:true,maintainAspectRatio:false,
   plugins:{legend:{display:false},tooltip:{callbacks:{label:function(c){
    var r=topTeams[c.dataIndex];return "P(campeón) "+r.CAMPEON+"% ± "+r.CAMPEON_sd;}}}},
   scales:{y:{ticks:{color:tc,callback:function(v){return v+"%";}},grid:{color:dark?"rgba(255,255,255,.07)":"rgba(0,0,0,.06)"}},
    x:{ticks:{color:tc,maxRotation:45,autoSkip:false},grid:{display:false}}}}});
})();
</script>
<script>
function triggerUpdate(){
  var btn=document.getElementById('updateBtn');
  var tok=localStorage.getItem('gh_pat');
  if(!tok){
    tok=prompt('Token de GitHub (guardado en tu navegador). Crealo en: github.com/settings/tokens > Fine-grained > repo pm26-vlc7 > Actions: Read and write');
    if(!tok)return;
    localStorage.setItem('gh_pat',tok.trim());
    tok=tok.trim();
  }
  btn.textContent='Lanzando...'; btn.disabled=true;
  fetch('https://api.github.com/repos/manuelpenaizurieta/pm26-vlc7/actions/workflows/update.yml/dispatches',{
    method:'POST',
    headers:{'Authorization':'token '+tok,'Content-Type':'application/json'},
    body:JSON.stringify({ref:'main'})
  }).then(function(r){
    if(r.status===204){
      var secs=180;
      function tick(){
        btn.textContent='Recargando en '+secs+'s...';
        if(secs<=0){location.reload();return;}
        secs--; setTimeout(tick,1000);
      }
      tick();
    } else if(r.status===401||r.status===403){
      localStorage.removeItem('gh_pat');
      btn.textContent='Actualizar ahora'; btn.disabled=false;
      alert('Token invalido - borrado. Vuelve a intentarlo.');
    } else {
      btn.textContent='Actualizar ahora'; btn.disabled=false;
      alert('Error '+r.status);
    }
  }).catch(function(){btn.textContent='Actualizar ahora';btn.disabled=false;alert('Error de red');});
}
</script></body></html>"""

def news_html(articles, matches_data):
    """Genera la pestaña Noticias: feed ESPN agrupado por partido próximo."""
    import datetime as _dt
    today = _dt.date.today()
    window_days = 3
    upcoming = [m for m in matches_data
                if m.get("date","") >= today.isoformat()
                and m.get("date","") <= (today + _dt.timedelta(days=window_days)).isoformat()
                and m.get("rx") is None]  # no jugados

    tag_label = {"lesion": "🏥 Lesión", "tarjeta": "🟨 Tarjeta", "alineacion": "📋 Alineación",
                 "tactica": "🧠 Táctica", "general": "📰 General"}
    tag_color = {"lesion": "sig-red", "tarjeta": "sig-yellow", "alineacion": "sig-blue",
                 "tactica": "sig-green", "general": "sig-gray"}

    if not articles:
        return '<div class="card"><p class="note">Sin noticias disponibles. Se actualizan cada 30 min.</p></div>'

    # Separar noticias relevantes para próximos partidos vs resto
    seen = set()
    blocks = ""

    # Primero: noticias por partido próximo
    for m in upcoming[:6]:
        home, away = m["home"], m["away"]
        rel = [a for a in articles
               if (home in a["teams"] or away in a["teams"]) and id(a) not in seen]
        if not rel:
            continue
        date_label = "HOY" if m["date"] == today.isoformat() else m["dlabel"]
        blocks += (f'<div class="card" style="border-left:3px solid var(--acc);border-radius:0 14px 14px 0">'
                   f'<div style="font-weight:600;margin-bottom:8px">'
                   f'{home} – {away} <span class="note">{date_label} {m["time"]}'
                   f' · {m["ph"]}%/{m["pd"]}%/{m["pa"]}%'
                   f' · pick {m["bx"]}-{m["by"]}</span></div>')
        for a in rel[:4]:
            seen.add(id(a))
            tags_html = "".join(
                f'<span class="sig {tag_color.get(t,"sig-gray")}" style="font-size:11px">{tag_label.get(t,t)}</span>'
                for t in a["tags"]
            )
            img_html = (f'<img src="{a["img"]}" style="width:80px;height:54px;object-fit:cover;'
                        f'border-radius:6px;flex-shrink:0" onerror="this.style.display=\'none\'">'
                        if a["img"] else "")
            link_open  = f'<a href="{a["link"]}" target="_blank" style="text-decoration:none;color:inherit">' if a["link"] else "<span>"
            link_close = "</a>" if a["link"] else "</span>"
            blocks += (
                f'<div style="display:flex;gap:10px;padding:8px 0;border-top:1px solid var(--line);align-items:flex-start">'
                f'{img_html}'
                f'<div style="min-width:0">'
                f'{link_open}<div style="font-weight:500;font-size:14px;line-height:1.3;margin-bottom:3px">{a["headline"]}</div>{link_close}'
                f'<div style="font-size:12px;color:var(--sub);margin-bottom:4px">{a["description"]}</div>'
                f'<div style="display:flex;gap:5px;flex-wrap:wrap;align-items:center">'
                f'{tags_html}<span class="note" style="font-size:11px">{a["age"]}</span></div>'
                f'</div></div>'
            )
        blocks += '</div>'

    # Resto de noticias generales
    rest = [a for a in articles if id(a) not in seen]
    if rest:
        blocks += '<div class="card"><div style="font-weight:600;margin-bottom:8px">📰 Más del Mundial</div>'
        for a in rest[:10]:
            tags_html = "".join(
                f'<span class="sig {tag_color.get(t,"sig-gray")}" style="font-size:11px">{tag_label.get(t,t)}</span>'
                for t in a["tags"]
            )
            img_html = (f'<img src="{a["img"]}" style="width:72px;height:48px;object-fit:cover;'
                        f'border-radius:6px;flex-shrink:0" onerror="this.style.display=\'none\'">'
                        if a["img"] else "")
            link_open  = f'<a href="{a["link"]}" target="_blank" style="text-decoration:none;color:inherit">' if a["link"] else "<span>"
            link_close = "</a>" if a["link"] else "</span>"
            teams_str = ", ".join(a["teams"][:3]) if a["teams"] else ""
            blocks += (
                f'<div style="display:flex;gap:10px;padding:8px 0;border-top:1px solid var(--line);align-items:flex-start">'
                f'{img_html}'
                f'<div style="min-width:0">'
                f'{link_open}<div style="font-weight:500;font-size:14px;line-height:1.3;margin-bottom:3px">{a["headline"]}</div>{link_close}'
                f'<div style="display:flex;gap:5px;flex-wrap:wrap;align-items:center">'
                f'{tags_html}'
                f'{"<span class=note style=font-size:11px>"+teams_str+"</span>" if teams_str else ""}'
                f'<span class="note" style="font-size:11px">{a["age"]}</span></div>'
                f'</div></div>'
            )
        blocks += '</div>'

    import time as _time
    age_mins = int((_time.time() - os.path.getmtime(os.path.join(HERE,"wc_news.json"))) / 60) if os.path.exists(os.path.join(HERE,"wc_news.json")) else 0
    header = (f'<div class="card" style="background:var(--card2)">'
              f'<b>Noticias del Mundial</b> '
              f'<span class="note">ESPN · {len(articles)} artículos · actualizado hace {age_mins} min · se refresca cada 30 min automáticamente</span>'
              f'</div>')
    return header + blocks


def analysis_today_html(matches_data):
    """Genera el HTML de la sección '2b · Análisis del modelo para los partidos de hoy/mañana'."""
    import datetime as _dt
    today = _dt.date.today()
    tomorrow = today + _dt.timedelta(days=1)
    window = {today.isoformat(), tomorrow.isoformat()}
    relevant = [m for m in matches_data if m.get("date", "") in window]
    if not relevant:
        return ""

    rows_html = ""
    for m in relevant:
        home, away = m["home"], m["away"]
        date_label = "HOY" if m.get("date", "") == today.isoformat() else "MAÑANA"

        # encabezado del partido
        rows_html += (
            f'<div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--line)">'
            f'<div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap">'
            f'<b style="font-size:15px">{home} – {away}</b>'
            f'<span class="note">{date_label} {m.get("time","")}</span>'
            f'<span class="pick">{m["bx"]}-{m["by"]}</span>'
            f'<span class="note">EV {m["evb"]:.2f} · P(exacto) {m["pex"]}%</span>'
            f'</div>'
        )

        # tabla de factores aplicados
        rows_html += '<table style="width:100%;font-size:13px;margin-top:6px;min-width:0"><tbody>'

        # lambdas base y ajustadas
        la_h = m.get("la_h", 0); la_a = m.get("la_a", 0)
        la_ha = m.get("la_h_adj", la_h); la_aa = m.get("la_a_adj", la_a)
        pct_h = round((la_ha / la_h - 1) * 100) if la_h else 0
        pct_a = round((la_aa / la_a - 1) * 100) if la_a else 0
        sign_h = "+" if pct_h >= 0 else ""
        sign_a = "+" if pct_a >= 0 else ""
        rows_html += (
            f'<tr><td style="color:var(--mut);padding:2px 8px 2px 0">λ esperados</td>'
            f'<td>{home}: <b>{la_h:.2f}</b> → <b>{la_ha:.2f}</b> ({sign_h}{pct_h}%)'
            f' · {away}: <b>{la_a:.2f}</b> → <b>{la_aa:.2f}</b> ({sign_a}{pct_a}%)</td></tr>'
        )

        # probabilidades resultado
        rows_html += (
            f'<tr><td style="color:var(--mut);padding:2px 8px 2px 0">Probabilidades</td>'
            f'<td>Local {m["ph"]}% · Empate {m["pd"]}% · Visita {m["pa"]}%'
            f'{"  · <i>mercado 1X2+O/U+AH</i>" if m.get("mkt") else "  · <i>modelo base</i>"}</td></tr>'
        )

        # signals: suspensiones, presión, O/U, AH, sharp
        sigs = m.get("signals", [])
        susp  = [s for s in sigs if "suspendido" in s or "riesgo" in s or "lesionado" in s]
        press = [s for s in sigs if "presion" in s or "clasificado" in s]
        mkt_s = [s for s in sigs if "O/U" in s or "AH " in s or "SHARP" in s]

        if susp:
            badges = "".join(
                f'<span class="sig {"sig-red" if "suspendido" in s or "lesionado" in s else "sig-yellow"}">{s}</span>'
                for s in susp
            )
            rows_html += (
                f'<tr><td style="color:var(--mut);padding:2px 8px 2px 0">Tarjetas</td>'
                f'<td><div class="sigs">{badges}</div></td></tr>'
            )

        if press:
            badges = "".join(
                f'<span class="sig {"sig-orange" if "presion" in s else "sig-green"}">{s}</span>'
                for s in press
            )
            rows_html += (
                f'<tr><td style="color:var(--mut);padding:2px 8px 2px 0">Standings</td>'
                f'<td><div class="sigs">{badges}</div></td></tr>'
            )

        if mkt_s:
            badges = "".join(
                f'<span class="sig {"sig-sharp" if "SHARP" in s else "sig-blue"}">{s}</span>'
                for s in mkt_s
            )
            rows_html += (
                f'<tr><td style="color:var(--mut);padding:2px 8px 2px 0">Mercado</td>'
                f'<td><div class="sigs">{badges}</div></td></tr>'
            )

        rows_html += '</tbody></table></div>'

    if not rows_html:
        return ""

    return (
        '<div class="card"><b>2b · Análisis del modelo para hoy y mañana</b>'
        '<span class="note"> — factores aplicados en cada pick</span>'
        + rows_html + '</div>'
    )


def setup_items():
    import time as _time
    def li(ok, txt_ok, txt_falta):
        mark = "✅" if ok else "⬜"
        return f"<li>{mark} {txt_ok if ok else txt_falta}</li>"
    def age_str(path):
        if not os.path.exists(path): return ""
        mins = (_time.time() - os.path.getmtime(path)) / 60
        if mins < 90: return f" <span class='note'>({mins:.0f} min)</span>"
        return f" <span class='note'>({mins/60:.1f} h)</span>"
    from wc_data_feed import user_env
    p_ext  = os.path.join(HERE, "match_odds_ext.json")
    p_mov  = os.path.join(HERE, "odds_movement.json")
    p_susp = os.path.join(HERE, "wc_suspensions.json")
    p_elo  = os.path.join(HERE, "elo_live.json")
    p_adj  = os.path.join(HERE, "wc_live_adj.json")
    p_sty  = os.path.join(HERE, "style_adj.json")

    has_ext = os.path.exists(p_ext)
    has_mov = os.path.exists(p_mov)
    has_susp = os.path.exists(p_susp)

    # cuenta suspendidos activos
    n_susp = 0
    susp_teams = []
    if has_susp:
        try:
            sd = json.load(open(p_susp, encoding="utf-8"))
            for team, d in sd.items():
                if d.get("n_susp", 0) > 0:
                    n_susp += d["n_susp"]
                    susp_teams.append(f"{team} ({d['n_susp']})")
        except Exception: pass

    items = [
        li(bool(user_env("FOOTBALL_DATA_TOKEN")),
           "Resultados automáticos conectados (football-data.org)" + age_str(p_elo),
           "<b>Falta:</b> token de football-data.org — resultados a mano"),
        li(os.path.exists(p_elo),
           "Elo en vivo activo (K=40 grupos)" + age_str(p_elo),
           "Elo en vivo: se activa con los primeros resultados"),
        li(os.path.exists(p_adj),
           "Calibración Bayesiana WC2026 activa (ATT/DEF/BASE online)" + age_str(p_adj),
           "Calibración online: pendiente de resultados"),
        li(bool(user_env("ODDS_API_KEY")),
           "Cuotas 1X2 conectadas (~23 casas, devig Shin activo)",
           "<b>Falta:</b> clave de the-odds-api.com"),
        li(has_ext,
           "O/U + Asian Handicap activos (4 constraints → λ casi únicos)" + age_str(p_ext),
           "O/U + AH: pendiente (se genera en la próxima bajada de cuotas)"),
        li(has_mov,
           "Detección de línea sharp activa" + age_str(p_mov),
           "Movimiento sharp: pendiente (requiere 2 snapshots de odds)"),
        li(has_susp,
           (f"Tarjetas/suspensiones ESPN activas — {n_susp} suspendido(s): {', '.join(susp_teams)}" if n_susp
            else "Tarjetas/suspensiones ESPN activas — sin suspendidos ahora") + age_str(p_susp),
           "Tarjetas/suspensiones: se activan en la próxima ejecución"),
        li(_SC is not None,
           "Presión de grupo activa (standings context — mult por jornada/pts)",
           "Standings context: módulo no encontrado"),
        li(os.path.exists(p_sty),
           "Estilos MLE activos (ATT/DEF histórico 4 años)" + age_str(p_sty),
           "Estilos MLE: ejecutar fit_ratings.py"),
    ]
    return "".join(items)

# P(1o) en vivo (live_p1.py lo genera antes que este script en daily_update)
LIVE = {}
try:
    with open(os.path.join(HERE, "live_stats.json"), encoding="utf-8") as f:
        LIVE = json.load(f)
except FileNotFoundError:
    pass

p1_live = LIVE.get("p1", "?")
p1_ci   = f"±{round((LIVE.get('ci_high', 0) - LIVE.get('ci_low', 0)) / 2, 1)}" if LIVE else ""
p1_rank = LIVE.get("my_rank", "?")
brier   = LIVE.get("brier")
brier_str = f"{brier:.3f}" if brier is not None else "—"

fav = max(PROBS["rows"], key=lambda r: r["CAMPEON"])
dias_final = max(0, (datetime.date(2026, 7, 19) - datetime.date.today()).days)

_now_cest = datetime.datetime.utcnow() + datetime.timedelta(hours=2)
html = (HTML.replace("__GEN__", _now_cest.strftime("%d %b %Y %H:%M") + " (Valencia)")
            .replace("__D_FINAL__", str(dias_final))
            .replace("__P1__", str(p1_live))
            .replace("__P1_CI__", p1_ci)
            .replace("__RANK__", str(p1_rank))
            .replace("__BRIER__", brier_str)
            .replace("__FAV__", fav["team"]).replace("__FAVP__", str(fav["CAMPEON"]))
            .replace("__C__", str(C))
            .replace("__DATA__", json.dumps(matches, ensure_ascii=False))
            .replace("__PROBS__", json.dumps(rows, ensure_ascii=False))
            .replace("__ADV__", json.dumps(ADV, ensure_ascii=False))
            .replace("__STANDINGS__", json.dumps(STANDINGS, ensure_ascii=False))
            .replace("__SETUP__", setup_items())
            .replace("__ANALYSIS_TODAY__", analysis_today_html(matches))
            .replace("__NEWS__", news_html(NEWS, matches)))
# picks finales (para la auto-apuesta): "Home|Away" -> [bx, by]
with open(os.path.join(HERE, "picks.json"), "w", encoding="utf-8") as f:
    json.dump({f"{m['home']}|{m['away']}": [m["bx"], m["by"]] for m in matches}, f, ensure_ascii=False, indent=1)

out = os.path.join(HERE, "polla_v4.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"polla_v4.html generado ({len(html)} bytes, {len(matches)} partidos)")

# NOTA: la publicacion la hace SOLO la nube (GitHub Actions cada 30 min): copia
# polla_v4.html -> index.html y hace push. Aqui NO se copia al Escritorio ni se hace
# push local a proposito: asi el PC nunca interfiere con la nube (evita copias viejas
# y conflictos de git push PC-vs-nube). Todo es automatico y no depende de tu PC.
print("\nPicks de HOY y manana:")
hoy = datetime.date.today().isoformat()
for m in matches[:6]:
    tag = " <- HOY" if m["date"] == hoy else ""
    alt = f" / alt {m['ax']}-{m['ay']}" if (m['ax'] != m['bx'] or m['ay'] != m['by']) else ""
    print(f"  {m['date']} {m['time']} [{m['g']}] {m['home']} {m['bx']}-{m['by']} {m['away']}"
          f"  (P(exacto)={m['pex']}%{alt}; L{m['ph']}/X{m['pd']}/V{m['pa']}){tag}")
