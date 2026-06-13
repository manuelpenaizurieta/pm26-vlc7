#!/usr/bin/env python3
# Modelo Mundial 2026 v3 — responde a las preguntas 1, 2, 3, 5 y 7 del traspaso.
#  Q1: ratings ataque/defensa (multiplicativos) + funcion de ajuste MLE Dixon-Coles
#      (fit_dixon_coles) lista para usar con datos de football-data.org.
#  Q2: bracket OFICIAL 2026 (partidos 73-104, fuente: Wikipedia/FIFA) con asignacion
#      de mejores terceros a sus llaves por backtracking (sustituye al shuffle aleatorio).
#  Q3: devig de Shin implementado (shin_devig) + estructura para cuotas de los 48;
#      mientras solo haya 6 cuotas se mantiene el ancla del 62% como en v2.
#  Q5: actualizacion Elo durante el torneo (elo_update, formula World Football Elo
#      con factor de margen de goles) — la tarea diaria la llama con cada resultado.
#  Q7: incertidumbre total = Monte Carlo + sensibilidad a ratings (bootstrap
#      parametrico: se perturban los ratings con ruido N(0, sigma) y se re-simula).
import numpy as np
from collections import defaultdict
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
FACT = np.array([math.factorial(i) for i in range(9)], dtype=float)
rng = np.random.default_rng(7)

R0 = {
 "Espana":2105,"Francia":2080,"Brasil":2025,"Inglaterra":2015,"Argentina":2005,
 "Portugal":1990,"PaisesBajos":1965,"Alemania":1955,"Belgica":1930,"Croacia":1895,
 "Uruguay":1900,"Colombia":1885,"Marruecos":1875,"Noruega":1850,"Japon":1845,
 "Senegal":1845,"Suiza":1830,"Turkiye":1820,"EEUU":1820,"Mexico":1810,
 "Ecuador":1800,"CoreaSur":1790,"Austria":1790,"Chequia":1775,"Egipto":1780,
 "CostaMarfil":1770,"Iran":1765,"Argelia":1760,"Suecia":1775,"Canada":1750,
 "Australia":1740,"Paraguay":1740,"Escocia":1745,"Ghana":1730,"Bosnia":1725,
 "RDCongo":1720,"Tunez":1710,"Sudafrica":1680,"Catar":1680,"Uzbekistan":1680,
 "ArabiaSaudi":1670,"Iraq":1660,"Panama":1660,"Jordania":1640,"CaboVerde":1640,
 "NuevaZelanda":1620,"Curazao":1600,"Haiti":1600,
}
HOST_BOOST = {"Mexico":40,"EEUU":40,"Canada":40}
GROUPS = {
 "A":["Mexico","CoreaSur","Sudafrica","Chequia"],"B":["Suiza","Canada","Catar","Bosnia"],
 "C":["Brasil","Marruecos","Haiti","Escocia"],"D":["EEUU","Paraguay","Australia","Turkiye"],
 "E":["Alemania","Curazao","Ecuador","CostaMarfil"],"F":["PaisesBajos","Suecia","Tunez","Japon"],
 "G":["Belgica","Egipto","Iran","NuevaZelanda"],"H":["Espana","CaboVerde","ArabiaSaudi","Uruguay"],
 "I":["Francia","Senegal","Iraq","Noruega"],"J":["Argentina","Argelia","Austria","Jordania"],
 "K":["Portugal","RDCongo","Uzbekistan","Colombia"],"L":["Inglaterra","Croacia","Ghana","Panama"],
}
TEAMS = [t for g in GROUPS.values() for t in g]
RBAR = float(np.mean([R0[t] for t in TEAMS]))

# ---------------------------------------------------------------- Q1: ataque/defensa
# Ajustes manuales (o salidos de fit_dixon_coles) sobre la descomposicion del Elo.
# att>0 = marca mas de lo que dice su Elo; dfn>0 = encaja menos. En logs de goles.
ATT_ADJ = defaultdict(float)   # ej.: ATT_ADJ["Noruega"]=+0.10 (Haaland), DEF_ADJ["Noruega"]=-0.05
DEF_ADJ = defaultdict(float)

# estilo ajustado por MLE sobre resultados reales (fit_ratings.py); la fuerza
# total sigue anclada al mercado, esto solo inclina ataque vs defensa
_style = os.path.join(HERE, "style_adj.json")
if os.path.exists(_style):
    with open(_style, encoding="utf-8") as f:
        _sa = json.load(f)
    for _t, _d in _sa["adj"].items():
        ATT_ADJ[_t] = _d; DEF_ADJ[_t] = -_d
    print(f"Estilo ataque/defensa cargado de style_adj.json ({len(_sa['adj'])} equipos, ajustado {_sa['fitted']})")

MAXG = 8
RHO = -0.08
BASE = 1.32

# carga elo en vivo si existe: se actualiza despues de cada resultado en elo_live.json
_elo_path = os.path.join(HERE, "elo_live.json")
if os.path.exists(_elo_path):
    with open(_elo_path, encoding="utf-8") as _f:
        _elo_live = json.load(_f)
    R0.update(_elo_live["ratings"])
    RBAR = float(np.mean([R0[t] for t in TEAMS]))   # recalcular con ratings vivos
    print(f"Ratings en vivo: {len(_elo_live['applied'])} partidos aplicados a R0")

# calibracion Bayesiana online: ajustes de ataque/defensa aprendidos de goles WC 2026
_WC26_BASE_FACTOR = 1.0
_live_adj_path = os.path.join(HERE, "wc_live_adj.json")
if os.path.exists(_live_adj_path):
    with open(_live_adj_path, encoding="utf-8") as _f:
        _la = json.load(_f)
    _WC26_BASE_FACTOR = _la.get("base_factor", 1.0)
    for _t, _d in _la.get("att_delta", {}).items():
        ATT_ADJ[_t] = ATT_ADJ.get(_t, 0.0) + _d
    for _t, _d in _la.get("def_delta", {}).items():
        DEF_ADJ[_t] = DEF_ADJ.get(_t, 0.0) + _d
    print(f"Calibracion WC26: {_la['n_matches']} partidos | base x{_WC26_BASE_FACTOR:.3f}")

def R(t): return R0[t] + HOST_BOOST.get(t, 0)

def lambdas(a, b, c):
    """Goles esperados multiplicativos: la = BASE*exp(att_a - def_b).
    att y def se siembran iguales desde el Elo ((R-RBAR)*c/2) y se separan
    con ATT_ADJ/DEF_ADJ (a mano o por MLE)."""
    att_a = (R(a)-RBAR)*c/2 + ATT_ADJ[a]; dfn_a = (R(a)-RBAR)*c/2 + DEF_ADJ[a]
    att_b = (R(b)-RBAR)*c/2 + ATT_ADJ[b]; dfn_b = (R(b)-RBAR)*c/2 + DEF_ADJ[b]
    la = BASE*_WC26_BASE_FACTOR*math.exp(att_a - dfn_b)
    lb = BASE*_WC26_BASE_FACTOR*math.exp(att_b - dfn_a)
    return max(0.15, min(4.5, la)), max(0.15, min(4.5, lb))

def build_matrix(a, b, c):
    la, lb = lambdas(a, b, c)
    x = np.arange(MAXG+1)
    pa = np.exp(-la)*la**x/FACT; pb = np.exp(-lb)*lb**x/FACT
    M = np.outer(pa, pb)
    M[0,0]*=1-la*lb*RHO; M[0,1]*=1+la*RHO; M[1,0]*=1+lb*RHO; M[1,1]*=1-RHO
    return M/M.sum()

def fit_dixon_coles(matches, teams, xi=0.0018):
    """MLE Dixon-Coles completo. matches: lista de dicts con keys
    home, away, gh, ga, days_ago (dias hasta hoy), neutral (bool).
    Devuelve (att, dfn, home_adv, rho). Pensado para alimentarse del CSV de
    resultados internacionales (p.ej. martj42/international_results en GitHub)
    o de los partidos que devuelva football-data.org (ver wc_data_feed.py)."""
    from scipy.optimize import minimize
    idx = {t:i for i,t in enumerate(teams)}; n = len(teams)
    def tau(x, y, la, lb, rho):
        if x==0 and y==0: return 1-la*lb*rho
        if x==0 and y==1: return 1+la*rho
        if x==1 and y==0: return 1+lb*rho
        if x==1 and y==1: return 1-rho
        return 1.0
    def nll(p):
        att, dfn = p[:n], p[n:2*n]; mu, home, rho = p[2*n], p[2*n+1], p[2*n+2]
        ll = 0.0
        for m in matches:
            i, j = idx[m["home"]], idx[m["away"]]
            h = 0.0 if m.get("neutral") else home
            la = math.exp(mu + att[i] - dfn[j] + h)
            lb = math.exp(mu + att[j] - dfn[i])
            w = math.exp(-xi*m.get("days_ago", 0))   # decaimiento temporal
            x, y = m["gh"], m["ga"]
            ll += w*(x*math.log(la)-la + y*math.log(lb)-lb
                     + math.log(max(1e-9, tau(x, y, la, lb, rho))))
        # penalizacion suave para identificabilidad (sum att = sum dfn = 0)
        return -ll + 100*(sum(att)**2 + sum(dfn)**2)
    p0 = np.concatenate([np.zeros(2*n), [math.log(BASE), 0.25, -0.08]])
    res = minimize(nll, p0, method="L-BFGS-B", options={"maxiter":400})
    att, dfn = res.x[:n], res.x[n:2*n]
    return ({t:float(att[idx[t]]) for t in teams}, {t:float(dfn[idx[t]]) for t in teams},
            float(res.x[2*n+1]), float(res.x[2*n+2]))

# ---------------------------------------------------------------- Q3: devig de Shin
ODDS = {"Espana":475,"Francia":500,"Inglaterra":650,"Brasil":850,"Portugal":800,"Argentina":900}
def am_to_p(o): return 100/(o+100) if o>0 else (-o)/(-o+100)

def shin_devig(implied):
    """Devig de Shin: corrige el sesgo favorito-tapado mejor que el multiplicativo.
    implied: dict equipo -> prob implicita de un LIBRO COMPLETO (los 48 equipos).
    Resuelve z (proporcion de dinero informado) por biseccion."""
    pis = np.array(list(implied.values())); s = pis.sum()
    def probs(z):
        return (np.sqrt(z*z + 4*(1-z)*pis*pis/s) - z) / (2*(1-z))
    lo, hi = 0.0, 0.4
    for _ in range(60):
        z = (lo+hi)/2
        if probs(z).sum() > 1: lo = z
        else: hi = z
    p = probs((lo+hi)/2); p = p/p.sum()
    return dict(zip(implied.keys(), p.tolist()))

# cuotas en vivo (decimales, bajadas por wc_data_feed.py) si existen; si no, las 6 de jun-2026
_odds_live = os.path.join(HERE, "odds_latest.json")
implied = {}
if os.path.exists(_odds_live):
    with open(_odds_live, encoding="utf-8") as f:
        _dec = json.load(f)
    implied = {t: 1.0/p for t, p in _dec.items() if t in R0 and p > 1.01}
if len(implied) >= 40:      # libro completo -> devig de Shin
    TARGET = shin_devig(implied)
    print(f"Calibracion con libro completo: {len(implied)} cuotas reales (devig Shin)")
else:                       # libro parcial: ancla al 62% como en v2 (documentado)
    implied = {k: am_to_p(v) for k, v in ODDS.items()}
    s = sum(implied.values()); TARGET = {k: v/s*0.62 for k, v in implied.items()}

def brier_and_reliability(preds, outcomes, bins=10):
    """Q3: test de calibracion sobre partidos ya jugados.
    preds: lista de probs pronosticadas; outcomes: lista de 0/1."""
    p = np.array(preds); o = np.array(outcomes)
    brier = float(np.mean((p-o)**2))
    edges = np.linspace(0, 1, bins+1); rel = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i+1])
        if m.sum() > 0:
            rel.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(m.sum()),
                        "pred": round(float(p[m].mean()), 3), "real": round(float(o[m].mean()), 3)})
    return brier, rel

# ---------------------------------------------------------------- Q2: bracket oficial
# Fuente: Wikipedia "2026 FIFA World Cup knockout stage" (partidos 73-104).
# "3" = mejor tercero asignado a esa llave; SLOT_OK = grupos elegibles por llave.
R32_T = {73:("2A","2B"), 74:("1E","3"), 75:("1F","2C"), 76:("1C","2F"),
         77:("1I","3"), 78:("2E","2I"), 79:("1A","3"), 80:("1L","3"),
         81:("1D","3"), 82:("1G","3"), 83:("2K","2L"), 84:("1H","2J"),
         85:("1B","3"), 86:("1J","2H"), 87:("1K","3"), 88:("2D","2G")}
SLOT_OK = {74:set("ABCDF"), 77:set("CDFGH"), 79:set("CEFHI"), 80:set("EHIJK"),
           81:set("BEFIJ"), 82:set("AEHIJ"), 85:set("EFGIJ"), 87:set("DEIJL")}
R16_T = {89:(74,77), 90:(73,75), 91:(76,78), 92:(79,80),
         93:(83,84), 94:(81,82), 95:(86,88), 96:(85,87)}
QF_T  = {97:(89,90), 98:(93,94), 99:(91,92), 100:(95,96)}
SF_T  = {101:(97,98), 102:(99,100)}

def allocate_thirds(third_groups):
    """Asigna los 8 grupos de los mejores terceros a sus llaves (backtracking).
    Aproxima el Anexo C de FIFA: respeta los grupos elegibles de cada llave."""
    slots = sorted(SLOT_OK, key=lambda s: len(SLOT_OK[s] & set(third_groups)))
    assign = {}
    def bt(i, remaining):
        if i == len(slots): return True
        s = slots[i]
        for g in sorted(remaining):
            if g in SLOT_OK[s]:
                assign[s] = g; remaining.remove(g)
                if bt(i+1, remaining): return True
                remaining.add(g); del assign[s]
        return False
    if bt(0, set(third_groups)): return assign
    rem = list(third_groups); assign = {}          # fallback (no deberia ocurrir)
    for s in slots:
        cand = [g for g in rem if g in SLOT_OK[s]] or rem
        assign[s] = cand[0]; rem.remove(cand[0])
    return assign

# ---------------------------------------------------------------- simulacion
def make_sampler(cache, a, b, c):
    key = (a, b)
    if key not in cache:
        M = build_matrix(a, b, c)
        cache[key] = (M.cumsum().reshape(-1), M.shape)
    return cache[key]

def sample(cache, a, b, c):
    cum, shape = make_sampler(cache, a, b, c)
    idx = np.searchsorted(cum, rng.random())
    return idx//shape[1], idx % shape[1]

def winner(cache, a, b, c):
    ga, gb = sample(cache, a, b, c)
    if ga == gb:
        pa = 1/(1+10**((R(b)-R(a))/600)); return a if rng.random() < pa else b
    return a if ga > gb else b

def simulate(N, c, cache, ratings_override=None):
    global R0
    saved = None
    if ratings_override is not None:
        saved, R0 = R0, ratings_override
    reach = defaultdict(lambda: defaultdict(int))
    for _ in range(N):
        slot_team = {}; thirds = []; third_team = {}
        for g, tm in GROUPS.items():
            pts = {t:0 for t in tm}; gd = {t:0 for t in tm}; gf = {t:0 for t in tm}
            for i in range(4):
                for j in range(i+1, 4):
                    x, y = tm[i], tm[j]; gx, gy = sample(cache, x, y, c)
                    gd[x] += gx-gy; gd[y] += gy-gx; gf[x] += gx; gf[y] += gy
                    if gx > gy: pts[x] += 3
                    elif gy > gx: pts[y] += 3
                    else: pts[x] += 1; pts[y] += 1
            o = sorted(tm, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
            slot_team["1"+g] = o[0]; slot_team["2"+g] = o[1]
            for t in (o[0], o[1]): reach[t]["R32"] += 1
            thirds.append((g, o[2], pts[o[2]], gd[o[2]], gf[o[2]]))
        thirds.sort(key=lambda z: (z[2], z[3], z[4], rng.random()), reverse=True)
        best8 = thirds[:8]
        for g, t, *_ in best8:
            reach[t]["R32"] += 1; third_team[g] = t
        alloc = allocate_thirds([g for g, *_ in best8])
        # bracket oficial
        win32 = {}
        for m, (sa, sb) in R32_T.items():
            a = slot_team[sa]
            b = third_team[alloc[m]] if sb == "3" else slot_team[sb]
            w = winner(cache, a, b, c); win32[m] = w; reach[w]["R16"] += 1
        win16 = {}
        for m, (ma, mb) in R16_T.items():
            w = winner(cache, win32[ma], win32[mb], c); win16[m] = w; reach[w]["QF"] += 1
        winqf = {}
        for m, (ma, mb) in QF_T.items():
            w = winner(cache, win16[ma], win16[mb], c); winqf[m] = w; reach[w]["SF"] += 1
        winsf = {}
        for m, (ma, mb) in SF_T.items():
            w = winner(cache, winqf[ma], winqf[mb], c); winsf[m] = w; reach[w]["FINAL"] += 1
        champ = winner(cache, winsf[101], winsf[102], c)
        reach[champ]["CAMPEON"] += 1
    if saved is not None: R0 = saved
    return reach

# ---------------------------------------------------------------- Q5: Elo en vivo
def elo_update(ratings, a, b, ga, gb, K=40):
    """Actualizacion tipo World Football Elo con factor de margen de goles.
    K=40 fase de grupos, K=50 eliminatorias. La tarea diaria llama esto con cada
    resultado real y luego re-corre simulate() con los ratings actualizados y
    los partidos ya jugados fijados."""
    dr = ratings[a] + HOST_BOOST.get(a, 0) - ratings[b] - HOST_BOOST.get(b, 0)
    We = 1/(1+10**(-dr/400))
    W = 1.0 if ga > gb else (0.5 if ga == gb else 0.0)
    d = abs(ga-gb)
    G = 1.0 if d <= 1 else (1.5 if d == 2 else (11+d)/8)
    delta = K*G*(W-We)
    ratings[a] += delta; ratings[b] -= delta
    return delta

# ---------------------------------------------------------------- Q7: sensibilidad
def rating_sensitivity(c, B=20, sigma=30, N=1500):
    """Bootstrap parametrico: perturba cada rating con N(0, sigma) y re-simula.
    Devuelve por equipo la desviacion estandar de P(R32) y P(CAMPEON) atribuible
    a la incertidumbre de los ratings (se suma en cuadratura con el error MC)."""
    acc = {t: {"R32": [], "CAMPEON": []} for t in TEAMS}
    for k in range(B):
        pert = {t: r + rng.normal(0, sigma) for t, r in R0.items()}
        rch = simulate(N, c, {}, ratings_override=pert)
        for t in TEAMS:
            acc[t]["R32"].append(rch[t]["R32"]/N)
            acc[t]["CAMPEON"].append(rch[t]["CAMPEON"]/N)
    return {t: {"R32_sd": round(100*float(np.std(acc[t]["R32"])), 1),
                "CAMPEON_sd": round(100*float(np.std(acc[t]["CAMPEON"])), 1)}
            for t in TEAMS}

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    # calibracion de c (sustituye a gamma; multiplicativo)
    best = None
    for c in [0.0020, 0.0024, 0.0028, 0.0032, 0.0036, 0.0040]:
        rch = simulate(2500, c, {})
        err = sum((rch[t]["CAMPEON"]/2500 - TARGET[t])**2 for t in TARGET)
        if best is None or err < best[1]: best = (c, err)
    c = best[0]
    print(f"Calibracion -> c optimo = {c} (error vs mercado = {best[1]:.5f})")

    N = 30000
    cache = {}
    reach = simulate(N, c, cache)
    stages = ["R32", "R16", "QF", "SF", "FINAL", "CAMPEON"]
    rows = []
    for t in TEAMS:
        d = {"team": t}
        for s in stages:
            p = reach[t][s]/N
            d[s] = round(100*p, 1); d[s+"_mc"] = round(100*1.96*math.sqrt(p*(1-p)/N), 1)
        rows.append(d)
    rows.sort(key=lambda r: r["CAMPEON"], reverse=True)

    print("\nSensibilidad a ratings (bootstrap, esto tarda ~1-2 min)...")
    sens = rating_sensitivity(c)
    for r in rows:
        r["R32_sd"] = sens[r["team"]]["R32_sd"]
        r["CAMPEON_sd"] = sens[r["team"]]["CAMPEON_sd"]

    print(f"\n{'Equipo':<13}" + "".join(f"{s:>9}" for s in stages) + f"{'±R32':>7}{'±CAM':>7}")
    for r in rows[:16]:
        print(f"{r['team']:<13}" + "".join(f"{r[s]:>7}% " for s in stages)
              + f"{r['R32_sd']:>6} {r['CAMPEON_sd']:>6}")
    print("\nChequeo calibracion (modelo vs mercado, % campeon):")
    for t in TARGET:
        print(f"  {t:<11} modelo {next(r for r in rows if r['team']==t)['CAMPEON']:>5}%"
              f"  | mercado ~{100*TARGET[t]:.1f}%")

    out = {"n": N, "c": c, "stages": stages, "rows": rows, "groups": GROUPS,
           "targets": {k: round(100*v, 1) for k, v in TARGET.items()},
           "bracket": "oficial-2026"}
    with open(os.path.join(HERE, "wc_probs_v3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nGuardado wc_probs_v3.json")
