#!/usr/bin/env python3
# Ajuste de ATAQUE/DEFENSA por seleccion con datos reales (MLE Poisson ponderado).
#
# Datos: martj42/international_results (todos los internacionales oficiales).
# Ventana: ultimos 4 anos, con decaimiento temporal exp(-xi*dias) (mitad de peso
# a ~13 meses). Equipos fuera de los 48 del Mundial con pocos partidos se agrupan
# en un pseudo-equipo "OTROS" para mantener la identificabilidad.
#
# Salida: style_adj.json -> wc_model_v3 lo carga automaticamente.
#   - El sistema mantiene la FUERZA total anclada al mercado (devig Shin);
#     de aqui solo se toma el ESTILO (sesgo ataque vs defensa), que el mercado
#     de campeon no revela: p.ej. Noruega golea pero encaja.
#   - Tambien imprime las discrepancias de fuerza dato-vs-Elo mas grandes,
#     como sugerencia de correccion manual de R0.
import csv, json, math, os, datetime
import numpy as np
from scipy.optimize import minimize
from wc_data_feed import NAME_MAP
from wc_model_v3 import TEAMS, R0

HERE = os.path.dirname(os.path.abspath(__file__))
LP = "\\\\?\\"   # prefijo de ruta larga de Windows
XI = 0.0018      # decaimiento temporal
WINDOW_DAYS = 4*365
MIN_MATCHES = 8  # un equipo no-mundialista necesita esto para tener parametros propios
CAP = 0.20       # tope del ajuste de estilo (en log-goles)

today = datetime.date.today()
rows = []
with open(LP + os.path.join(HERE, "intl_results.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            d = datetime.date.fromisoformat(r["date"])
        except ValueError:
            continue
        days = (today - d).days
        if days < 0 or days > WINDOW_DAYS or not r["home_score"].isdigit() or not r["away_score"].isdigit():
            continue
        h = NAME_MAP.get(r["home_team"], r["home_team"])
        a = NAME_MAP.get(r["away_team"], r["away_team"])
        rows.append((h, a, int(r["home_score"]), int(r["away_score"]),
                     math.exp(-XI*days), r["neutral"] == "TRUE"))

count = {}
for h, a, *_ in rows:
    count[h] = count.get(h, 0) + 1; count[a] = count.get(a, 0) + 1
def canon(t):
    if t in TEAMS: return t
    return t if count.get(t, 0) >= MIN_MATCHES else "OTROS"

teams = sorted({canon(t) for h, a, *_ in rows for t in (h, a)})
idx = {t: i for i, t in enumerate(teams)}
n = len(teams)
hi = np.array([idx[canon(h)] for h, a, x, y, w, nt in rows])
ai = np.array([idx[canon(a)] for h, a, x, y, w, nt in rows])
gx = np.array([r[2] for r in rows], dtype=float)
gy = np.array([r[3] for r in rows], dtype=float)
w  = np.array([r[4] for r in rows])
hadv = np.array([0.0 if r[5] else 1.0 for r in rows])
print(f"{len(rows)} partidos ({today - datetime.timedelta(days=WINDOW_DAYS)} a hoy), {n} equipos (48 mundialistas + resto)")

def unpack(p): return p[:n], p[n:2*n], p[2*n], p[2*n+1]

def nll(p):
    att, dfn, mu, home = unpack(p)
    lh = np.exp(mu + att[hi] - dfn[ai] + home*hadv)
    la = np.exp(mu + att[ai] - dfn[hi])
    ll = np.sum(w*(gx*np.log(lh) - lh + gy*np.log(la) - la))
    return -ll + 100*(att.sum()**2 + dfn.sum()**2)

def grad(p):
    att, dfn, mu, home = unpack(p)
    lh = np.exp(mu + att[hi] - dfn[ai] + home*hadv)
    la = np.exp(mu + att[ai] - dfn[hi])
    rh = w*(gx - lh); ra = w*(gy - la)
    datt = np.zeros(n); ddfn = np.zeros(n)
    np.add.at(datt, hi, rh); np.add.at(datt, ai, ra)
    np.add.at(ddfn, ai, -rh); np.add.at(ddfn, hi, -ra)
    dmu = rh.sum() + ra.sum(); dhome = (rh*hadv).sum()
    g = -np.concatenate([datt, ddfn, [dmu, dhome]])
    g[:n] += 200*att.sum(); g[n:2*n] += 200*dfn.sum()
    return g

p0 = np.zeros(2*n + 2); p0[2*n] = math.log(1.3); p0[2*n+1] = 0.25
res = minimize(nll, p0, jac=grad, method="L-BFGS-B", options={"maxiter": 800})
att, dfn, mu, home = unpack(res.x)
print(f"Convergencia: {res.success} ({res.nit} iteraciones) | mu={mu:.3f} (base {math.exp(mu):.2f} goles) | ventaja local={home:.3f}")

# --- estilo (lo que se inyecta al modelo) y diagnostico de fuerza ---
style = {}; strength = {}
for t in TEAMS:
    if t not in idx: continue
    i = idx[t]
    style[t] = float(att[i] - dfn[i])
    strength[t] = float(att[i] + dfn[i])
m_style = np.mean(list(style.values()))
adj = {t: float(np.clip((s - m_style)/2, -CAP, CAP)) for t, s in style.items()}
with open(os.path.join(HERE, "style_adj.json"), "w", encoding="utf-8") as f:
    json.dump({"fitted": str(today), "window_days": WINDOW_DAYS, "xi": XI,
               "adj": adj}, f, ensure_ascii=False, indent=1)
print(f"\nstyle_adj.json guardado ({len(adj)} equipos). Estilos mas marcados:")
for t, d in sorted(adj.items(), key=lambda z: -abs(z[1]))[:8]:
    lado = "ataca mas de lo que defiende" if d > 0 else "defiende mas de lo que ataca"
    print(f"  {t:<13} {d:+.3f}  ({lado})")

# fuerza segun datos vs Elo manual (sugerencias, no se aplican solas)
elo = {t: R0[t] for t in strength}
e_arr = np.array([elo[t] for t in strength]); s_arr = np.array([strength[t] for t in strength])
slope = float(np.polyfit(s_arr, e_arr, 1)[0]); inter = float(np.polyfit(s_arr, e_arr, 1)[1])
print("\nMayores discrepancias fuerza-datos vs Elo manual (sugerencia de revision):")
diffs = {t: (inter + slope*strength[t]) - elo[t] for t in strength}
for t, d in sorted(diffs.items(), key=lambda z: -abs(z[1]))[:8]:
    print(f"  {t:<13} Elo {elo[t]}  datos sugieren ~{elo[t]+d:.0f}  ({d:+.0f})")
