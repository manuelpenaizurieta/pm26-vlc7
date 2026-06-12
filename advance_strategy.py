#!/usr/bin/env python3
# OPTIMIZADOR DE BONOS DE AVANCE — donde de verdad se gana la polla.
#
# Por que importa: acertar los 32 que pasan = 32 x 10 = 320 pts potenciales, mas que
# TODOS los marcadores juntos. El bono se cobra por equipo que predices y alcanza la
# fase (R32 10, R16 8, QF 4, SF 2, Final 5).
#
# Que hace distinto al sistema viejo (que elegia "los mas famosos por rating"):
#   1. Elige por PROBABILIDAD REAL de avanzar (P(R32) del Monte Carlo), que ya tiene
#      en cuenta la dificultad del grupo y los rivales — no por fama. Un equipo fuerte
#      en un grupo brutal puede tener menos opciones que uno flojo en un grupo facil;
#      el rating no lo ve, la simulacion si.
#   2. Marca los equipos BURBUJA (cerca del corte) y calcula su LEVERAGE: equipos que
#      probablemente pasan pero que los rivales NO van a poner (porque van por fama).
#      Acertar uno de esos = +10 que casi nadie tiene -> te separa. Ahi se gana.
#
# Salida: advance_picks.json (lo consume el tablero).
import json, os
import wc_model_v3 as M

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "wc_probs_v3.json"), encoding="utf-8") as f:
    PROBS = json.load(f)
ROWS = {r["team"]: r for r in PROBS["rows"]}

# cuantos equipos se predicen por fase, y puntos del bono
STAGE_N = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "FINAL": 2}
STAGE_PTS = {"R32": 10, "R16": 8, "QF": 4, "SF": 2, "FINAL": 5}
STAGE_LABEL = {"R32": "dieciseisavos", "R16": "octavos", "QF": "cuartos",
               "SF": "semifinales", "FINAL": "final"}

# --- modelo de lo que pondra el rival tipico: ordena por FAMA (rating) ---
def field_pick_prob(stage):
    """Prob. de que un rival 'medio' incluya a cada equipo en su prediccion de la
    fase. Modelo: el rival ordena por rating y nombra los N de mayor fama (con ruido).
    Devuelve dict equipo -> prob de ser nombrado por un rival."""
    n = STAGE_N[stage]
    fame = sorted(M.TEAMS, key=M.R, reverse=True)
    rank = {t: i for i, t in enumerate(fame)}
    out = {}
    for t in M.TEAMS:
        # logistica centrada en el corte n: dentro del top-n -> alta prob, fuera -> baja
        d = (n - rank[t]) / 4.0
        out[t] = 1 / (1 + 2.718281828 ** (-d))
    return out

def optimize_stage(stage):
    n = STAGE_N[stage]
    field = field_pick_prob(stage)
    # P(alcanzar la fase) de cada equipo, del Monte Carlo
    p = {t: ROWS[t][stage] / 100.0 for t in M.TEAMS}
    ranked = sorted(M.TEAMS, key=lambda t: p[t], reverse=True)
    picks = ranked[:n]                       # optimo de bono ESPERADO: top-N por P real
    cut_lo, cut_hi = max(0, n - 6), n + 6     # zona burbuja alrededor del corte
    bubble = ranked[cut_lo:cut_hi]
    # leverage = prob de que pase * (1 - prob de que el rival lo ponga)
    lev = []
    for t in bubble:
        leverage = p[t] * (1 - field[t])
        lev.append({"team": t, "p": round(100 * p[t], 1),
                    "field": round(100 * field[t]), "in": t in picks,
                    "leverage": round(100 * leverage, 1)})
    lev.sort(key=lambda z: z["leverage"], reverse=True)
    # diferenciadores: pasan con buena prob pero el campo los ignora (no estan en chalk top)
    fame_rank = {t: i for i, t in enumerate(sorted(M.TEAMS, key=M.R, reverse=True))}
    diff = [d for d in lev if p[d["team"]] >= 0.45 and fame_rank[d["team"]] >= n]
    exp_bonus = round(sum(p[t] for t in picks) * STAGE_PTS[stage], 1)
    return {"stage": stage, "label": STAGE_LABEL[stage], "pts": STAGE_PTS[stage],
            "n": n, "picks": picks, "exp_bonus": exp_bonus,
            "bubble": lev, "differentiators": diff[:5]}

def main():
    out = {"stages": {}, "groups": M.GROUPS}
    total_exp = 0
    for s in STAGE_N:
        r = optimize_stage(s)
        out["stages"][s] = r
        total_exp += r["exp_bonus"]
    out["total_exp_bonus"] = round(total_exp, 1)
    with open(os.path.join(HERE, "advance_picks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"Bono de avance ESPERADO total: {total_exp:.0f} pts "
          f"(vs ~150 de todos los marcadores juntos)\n")
    r32 = out["stages"]["R32"]
    print("=== TUS 32 CLASIFICADOS (por probabilidad real, no por fama) ===")
    locks = [t for t in r32["picks"] if ROWS[t]["R32"] >= 80]
    print(f"  Seguros (P>=80%): {len(locks)} equipos")
    print(f"\n=== ZONA BURBUJA — aqui se decide la polla ===")
    print(f"  {'equipo':<13}{'P(pasa)':>9}{'rival lo pone':>14}{'en tu pick':>12}")
    for b in r32["bubble"]:
        mark = "si" if b["in"] else "NO"
        print(f"  {b['team']:<13}{b['p']:>8}%{b['field']:>13}%{mark:>12}")
    print(f"\n=== DIFERENCIADORES (pasan pero el campo los ignora -> te separan) ===")
    if r32["differentiators"]:
        for d in r32["differentiators"]:
            print(f"  {d['team']:<13} pasa {d['p']}% · solo {d['field']}% de rivales lo pondra "
                  f"-> +10 casi en exclusiva")
    else:
        print("  (ninguno claro esta vez: la burbuja coincide con la fama)")
    print("\nGuardado advance_picks.json")

if __name__ == "__main__":
    main()
