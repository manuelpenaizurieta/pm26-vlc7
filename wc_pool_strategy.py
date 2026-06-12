#!/usr/bin/env python3
# Q4 del traspaso: pasar de "maximizar EV por partido" a "maximizar P(quedar 1o)".
#
# Idea central: como NO vemos los picks de los rivales (5-15 jugadores), se modela
# un RIVAL SINTETICO: apuesta al favorito percibido con marcadores tipicos
# (1-0, 2-1, 2-0...). Con esa distribucion q(s) por partido:
#   1) La "prediccion unica" (+2) se vuelve calculable SIN conocer a los rivales:
#      E[bono] = 2 * (1 - q(s))^n_rivales  ->  se suma al EV del marcador s.
#   2) Se simula la polla completa (verdad del modelo + 9 rivales sinteticos) y se
#      compara P(1o) de tres politicas:
#        A) EV-max puro (lo que hace v2 hoy)
#        B) EV + bono de unicidad esperado (uniqueness-aware)
#        C) B + contrarian en partidos parejos (|pH-pA|<8pp elige el lado/marcador
#           que menos pisan los rivales) -> mas varianza = mas P(1o) con 70% al lider
#
# Como los picks son partido a partido, tambien imprime la REGLA ADAPTATIVA:
# percentiles de la distribucion de deficits para saber cuantos "coin flips"
# tienes que flipear segun lo lejos que vayas del lider.
import numpy as np, json, math, os
from collections import defaultdict
import wc_model_v3 as M

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11)
N_RIVALS = 9
C = 0.0028          # se sobreescribe con el c calibrado si existe wc_probs_v3.json
try:
    with open(os.path.join(HERE, "wc_probs_v3.json"), encoding="utf-8") as f:
        C = json.load(f)["c"]
except FileNotFoundError:
    pass

FIXTURES = [(g, tm[i], tm[j]) for g, tm in M.GROUPS.items()
            for i in range(4) for j in range(i+1, 4)]

# datos REALES de rivales si existen (Camino A); si no, modelo sintetico
try:
    import rivals as _RV
    _REAL = _RV.load()
    if _REAL:
        print(f"Rivales REALES cargados ({len(_REAL['picks'])} partidos con datos)")
except Exception:
    _REAL = None

# ---------------- modelo de rival sintetico: distribucion q(s) por partido ----------------
WIN_SCORES  = [((1,0),.40), ((2,1),.30), ((2,0),.20), ((3,1),.10)]
DRAW_SCORES = [((1,1),.70), ((0,0),.15), ((2,2),.15)]

OTHER_SCORES = [(x, y) for x in range(5) for y in range(5)
                if (x, y) not in dict(WIN_SCORES) and (y, x) not in dict(WIN_SCORES)
                and (x, y) not in dict(DRAW_SCORES)]

def rival_pick_dist(a, b):
    """Distribucion de marcadores de los rivales en a-b. Usa los picks REALES del
    grupo si estan disponibles para ese partido; si no, el modelo sintetico."""
    if _REAL is not None:
        real = _RV.real_pick_dist(a, b, _REAL)
        if real:
            return real
    pf = 1/(1+10**((M.R(b)-M.R(a))/400))          # prob percibida de que gane a
    w_a = min(.80, max(.10, pf*1.05 - .06))        # sesgo al favorito
    w_d = .10; w_o = .12
    w_b = max(.0, 1 - w_a - w_d - w_o)
    q = defaultdict(float)
    for (x, y), p in WIN_SCORES:
        q[(x, y)] += w_a*p
        q[(y, x)] += w_b*p
    for (x, y), p in DRAW_SCORES:
        q[(x, y)] += w_d*p
    for s in OTHER_SCORES:
        q[s] += w_o/len(OTHER_SCORES)
    return q

def sample_from(qdict):
    ks = list(qdict.keys()); ps = np.array([qdict[k] for k in ks]); ps /= ps.sum()
    return ks[rng.choice(len(ks), p=ps)]

# ---------------- puntos segun reglas REALES de la polla (verificadas en su codigo) ----------------
# IMPORTANTE: el marcador exacto vale 5 y NO se acumula con ganador/goles (es 5, no 9).
# Formula web: exacto*5 + ganador*2 + gol*1 + unico*2 + bonos.
def pts(px, py, ax, ay):
    if px == ax and py == ay:
        return 5                       # exacto: solo 5
    p = 0
    if np.sign(px-py) == np.sign(ax-ay): p += 2   # ganador o empate
    if px == ax: p += 1                # goles del local
    if py == ay: p += 1                # goles del visitante
    return p

# ---------------- politicas de pick ----------------
def policy_picks(a, b, mode):
    """Devuelve el marcador (px,py) segun la politica. mode: 'A', 'B' o 'C'."""
    mat = M.build_matrix(a, b, C)
    q = rival_pick_dist(a, b)
    pH = float(np.tril(mat, -1).sum()); pA = float(np.triu(mat, 1).sum())
    cand = []
    for px in range(5):
        for py in range(5):
            ev = sum(mat[ax, ay]*pts(px, py, ax, ay)
                     for ax in range(M.MAXG+1) for ay in range(M.MAXG+1))
            uniq = 2*(1 - q.get((px, py), 0.0))**N_RIVALS   # E[bono unicidad]
            cand.append((ev, uniq, px, py))
    if mode == "A":
        ev, uniq, px, py = max(cand, key=lambda t: t[0])
    elif mode == "B":
        ev, uniq, px, py = max(cand, key=lambda t: t[0]+t[1])
    else:  # C: contrarian en partidos parejos
        if abs(pH-pA) < 0.08:
            # entre los 8 mejores por EV+unicidad, el que menos pisan los rivales
            top = sorted(cand, key=lambda t: t[0]+t[1], reverse=True)[:8]
            ev, uniq, px, py = max(top, key=lambda t: t[1] - 0.0)
        else:
            ev, uniq, px, py = max(cand, key=lambda t: t[0]+t[1])
    return px, py

# ---------------- prediccion de clasificados (bono R32 = 10 pts/equipo) ----------------
def my_r32_prediction():
    """Manu: los 32 con mayor PROBABILIDAD REAL de pasar (del Monte Carlo), no por
    fama. Cae a rating solo si aun no hay wc_probs_v3.json."""
    try:
        with open(os.path.join(HERE, "wc_probs_v3.json"), encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        return set(r["team"] for r in sorted(rows, key=lambda r: r["R32"], reverse=True)[:32])
    except (FileNotFoundError, KeyError):
        qual, thirds = [], []
        for g, tm in M.GROUPS.items():
            o = sorted(tm, key=M.R, reverse=True)
            qual += o[:2]; thirds.append(o[2])
        thirds.sort(key=M.R, reverse=True)
        return set(qual + thirds[:8])

def rival_r32_prediction():
    """Rival: igual pero con ruido (a veces invierte 2o/3o del grupo)."""
    qual, thirds = [], []
    for g, tm in M.GROUPS.items():
        o = sorted(tm, key=lambda t: M.R(t) + rng.normal(0, 55), reverse=True)
        qual += o[:2]; thirds.append(o[2])
    thirds.sort(key=lambda t: M.R(t) + rng.normal(0, 55), reverse=True)
    return set(qual + thirds[:8])

# ---------------- simulacion de la polla (fase de grupos + bono R32) ----------------
def simulate_pool(T=1500):
    cache = {}
    my_picks = {p: {f: policy_picks(f[1], f[2], p) for f in FIXTURES} for p in "ABC"}
    my_r32 = my_r32_prediction()
    qdists = {f: rival_pick_dist(f[1], f[2]) for f in FIXTURES}
    wins  = {p: 0 for p in "ABC"}; podium = {p: 0 for p in "ABC"}
    totals = {p: [] for p in "ABC"}; deficits = []
    for _ in range(T):
        # verdad simulada: 72 marcadores + clasificados reales
        truth = {}; slot = {}; thirds = []
        for g, tm in M.GROUPS.items():
            ptsg = {t:0 for t in tm}; gd = {t:0 for t in tm}; gf = {t:0 for t in tm}
            for i in range(4):
                for j in range(i+1, 4):
                    x, y = tm[i], tm[j]
                    gx, gy = M.sample(cache, x, y, C)
                    truth[(g, x, y)] = (gx, gy)
                    gd[x] += gx-gy; gd[y] += gy-gx; gf[x] += gx; gf[y] += gy
                    if gx > gy: ptsg[x] += 3
                    elif gy > gx: ptsg[y] += 3
                    else: ptsg[x] += 1; ptsg[y] += 1
            o = sorted(tm, key=lambda t: (ptsg[t], gd[t], gf[t], rng.random()), reverse=True)
            slot[g] = o; thirds.append((o[2], ptsg[o[2]], gd[o[2]], gf[o[2]]))
        thirds.sort(key=lambda z: (z[1], z[2], z[3], rng.random()), reverse=True)
        real_r32 = set(t for g in M.GROUPS for t in slot[g][:2]) | set(t for t, *_ in thirds[:8])
        # picks de los rivales en esta realizacion
        rpicks = [{f: sample_from(qdists[f]) for f in FIXTURES} for _ in range(N_RIVALS)]
        rr32 = [rival_r32_prediction() for _ in range(N_RIVALS)]
        rbase = np.zeros(N_RIVALS)                 # puntos sin bono de unicidad
        for k in range(N_RIVALS):
            for f in FIXTURES:
                ax, ay = truth[f]; px, py = rpicks[k][f]
                rbase[k] += pts(px, py, ax, ay)
            rbase[k] += 10*len(rr32[k] & real_r32)
        for p in "ABC":
            # unicidad para TODOS (cada pick debe ser unico entre los 10 jugadores)
            from collections import Counter
            runiq = np.zeros(N_RIVALS); my_uniq = 0
            for f in FIXTURES:
                picks_all = [rpicks[k][f] for k in range(N_RIVALS)] + [my_picks[p][f]]
                cnt = Counter(picks_all)
                for k in range(N_RIVALS):
                    if cnt[rpicks[k][f]] == 1: runiq[k] += 2
                if cnt[my_picks[p][f]] == 1: my_uniq += 2
            rscores = rbase + runiq
            sc = my_uniq + 10*len(my_r32 & real_r32)
            for f in FIXTURES:
                ax, ay = truth[f]; px, py = my_picks[p][f]
                sc += pts(px, py, ax, ay)
            totals[p].append(sc)
            best_r = rscores.max()
            if sc > best_r: wins[p] += 1
            if sc >= np.sort(rscores)[-3]: podium[p] += 1
            if p == "B": deficits.append(best_r - sc)
    return wins, podium, totals, deficits, my_picks

if __name__ == "__main__":
    T = 1500
    print(f"Simulando la polla: {T} torneos x {N_RIVALS} rivales sinteticos...")
    wins, podium, totals, deficits, my_picks = simulate_pool(T)
    print(f"\n{'Politica':<42}{'P(1o)':>8}{'P(podio)':>10}{'pts medios':>12}{'sd':>6}")
    names = {"A": "A) EV-max puro (v2 actual)",
             "B": "B) EV + bono unicidad esperado",
             "C": "C) B + contrarian en partidos parejos"}
    for p in "ABC":
        t = np.array(totals[p])
        print(f"{names[p]:<42}{100*wins[p]/T:>7.1f}%{100*podium[p]/T:>9.1f}%"
              f"{t.mean():>12.1f}{t.std():>6.1f}")
    d = np.array(deficits)
    print(f"\nDeficit vs lider (politica B): mediana {np.median(d):.0f} pts, "
          f"p25 {np.percentile(d,25):.0f}, p75 {np.percentile(d,75):.0f}")
    print("Regla adaptativa (picks partido a partido):")
    print("  - si vas a <8 pts del lider  -> politica B (no regales EV)")
    print("  - si vas a 8-20 pts          -> politica C (contrarian en parejos)")
    print("  - si vas a >20 pts           -> ademas flipea el GANADOR en los 3-4")
    print("    partidos mas parejos que queden (max varianza, el 2o/3o premio es poco)")
    diff = [(f, my_picks["A"][f], my_picks["C"][f]) for f in FIXTURES
            if my_picks["A"][f] != my_picks["C"][f]]
    print(f"\nPartidos donde C difiere de A ({len(diff)}):")
    for (g, a, b), pa, pc in diff[:15]:
        print(f"  [{g}] {a} vs {b}: A={pa[0]}-{pa[1]}  ->  C={pc[0]}-{pc[1]}")
    out = {"T": T, "n_rivals": N_RIVALS,
           "policies": {p: {"win": round(100*wins[p]/T, 1), "podium": round(100*podium[p]/T, 1),
                            "mean": round(float(np.mean(totals[p])), 1),
                            "sd": round(float(np.std(totals[p])), 1)} for p in "ABC"},
           "picks_C": [{"g": f[0], "home": f[1], "away": f[2],
                        "px": my_picks["C"][f][0], "py": my_picks["C"][f][1]} for f in FIXTURES]}
    with open(os.path.join(HERE, "wc_pool_strategy.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nGuardado wc_pool_strategy.json")
