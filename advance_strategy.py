#!/usr/bin/env python3
# BONOS DE AVANCE — modelo CORRECTO (todo-o-nada), corregido 2026-06-14.
#
# Reglas REALES del grupo (bettinggroups/<GID>/rules -> polla_rules.json):
#   b32=10  b16=8  bQ=4  bS=2  bF=5   (max 29 pts en total)
# Cada bono es TODO-O-NADA: solo se cobra si aciertas TODOS los equipos que pasan
# de esa ronda. Acertar 7 de 8 = 0 pts. El bono se entrega al cerrar la fase.
#
# CONSECUENCIA ESTRATEGICA (lo contrario de lo que hacia el sistema viejo):
#   - Como es todo-o-nada, la jugada optima es poner los FAVORITOS (maximizar la
#     probabilidad de acertarlos TODOS). Un "diferenciador" contrarian que falla
#     te borra el bono entero -> meter diferenciadores aqui es CONTRAPRODUCENTE.
#   - El EV del bono es marginal: acertar los 16 ganadores de dieciseisavos es
#     casi imposible. La polla se gana en los MARCADORES, no aqui.
#
# Salida: advance_picks.json (lo consume el tablero, ahora con la narrativa correcta).
import json, os
import wc_model_v3 as M

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "wc_probs_v3.json"), encoding="utf-8") as f:
    PROBS = json.load(f)
ROWS = {r["team"]: r for r in PROBS["rows"]}

# reglas reales (con respaldo por si falta el archivo)
try:
    with open(os.path.join(HERE, "polla_rules.json"), encoding="utf-8") as f:
        RULES = json.load(f)
except FileNotFoundError:
    RULES = {"b32": 10, "b16": 8, "bQ": 4, "bS": 2, "bF": 5}

# cuantos equipos hay que acertar por ronda + a que columna del Monte Carlo miran +
# el valor real del bono. R32->llegan 32 (la columna R32), pero el bono se cobra por
# los que GANAN la ronda; usamos P(llegar a la SIGUIENTE ronda) como proxy de ganarla.
STAGES = [
    ("b32", "dieciseisavos", 32, "R32", "R16", RULES.get("b32", 10)),
    ("b16", "octavos",       16, "R16", "QF",  RULES.get("b16", 8)),
    ("bQ",  "cuartos",        8, "QF",  "SF",  RULES.get("bQ", 4)),
    ("bS",  "semifinales",    4, "SF",  "FINAL", RULES.get("bS", 2)),
    ("bF",  "final/campeon",  2, "FINAL", "CAMPEON", RULES.get("bF", 5)),
]

def optimize_stage(key, label, n_in, col_reach, col_advance, pts):
    """Cobras el bono si TODOS los equipos que eliges llegan a la siguiente ronda.
    Todo-o-nada -> pones los n_advance mas probables de llegar ahi (favoritos).
    P(cobrar) = producto de P(llegar a la siguiente ronda) de tus picks (aprox
    independiente; el real es algo menor por correlacion de cruces)."""
    n_advance = n_in // 2 if key != "bF" else 1
    picks = sorted(M.TEAMS, key=lambda t: ROWS[t].get(col_advance, 0), reverse=True)[:n_advance]
    p_all = 1.0
    for t in picks:
        p_all *= ROWS[t].get(col_advance, 0) / 100.0   # P(absoluta de llegar a la ronda destino)
    return {"stage": key, "label": label, "pts": pts, "n": n_advance,
            "picks": picks, "p_all": round(100 * p_all, 2),
            "exp_bonus": round(p_all * pts, 3)}

def main():
    stages = {}
    total_exp = 0.0
    max_pts = 0
    for key, label, n_in, cr, ca, pts in STAGES:
        r = optimize_stage(key, label, n_in, cr, ca, pts)
        stages[key] = r
        total_exp += r["exp_bonus"]
        max_pts += pts

    out = {"stages": stages, "groups": M.GROUPS,
           "max_pts": max_pts, "total_exp_bonus": round(total_exp, 2),
           "note": ("Bonos TODO-O-NADA: solo cobras si aciertas TODOS los equipos de "
                    "la ronda. Jugada optima = favoritos (no diferenciadores). EV "
                    "marginal; la polla se gana en los marcadores.")}
    with open(os.path.join(HERE, "advance_picks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"Bonos de avance (TODO-O-NADA) — max {max_pts} pts, EV realista {total_exp:.1f} pts")
    print("La polla se gana en los MARCADORES; estos bonos son marginales.\n")
    for key, label, *_ in STAGES:
        r = stages[key]
        print(f"  {label:16} (vale {r['pts']}): acierta {r['n']} equipos -> "
              f"P(todos)~{r['p_all']}% -> EV {r['exp_bonus']:.2f} pts")
        print(f"     favoritos: {', '.join(r['picks'][:8])}")
    print("\nGuardado advance_picks.json")

if __name__ == "__main__":
    main()
