#!/usr/bin/env python3
"""
Calibracion Bayesiana online con resultados WC 2026.

Metodo: Poisson-Gamma conjugado con shrinkage hacia el prior del modelo.
  - Prior: lambdas de wc_model_v3 (Elo + mercado + style_adj)
  - Likelihood: goles observados en WC 2026 (results_live.json)
  - Posterior: lambda_post = (k*lambda_prior + goals_obs) / (k + n_matches)
    donde k=PRIOR_WEIGHT (el prior equivale a k partidos "virtuales")

Salida: wc_live_adj.json con:
  - att_delta[equipo]: ajuste log-multiplicativo del ataque (suma a ATT_ADJ)
  - def_delta[equipo]: ajuste log-multiplicativo de la defensa (suma a DEF_ADJ)
  - base_factor: correccion global del nivel de goles WC 2026 vs modelo
  - n_matches: partidos usados

wc_model_v3 carga este archivo al importarse (despues de elo_live y style_adj).
"""
import json, math, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Prior weight: el modelo base equivale a PRIOR_WEIGHT partidos virtuales.
# Muy bajo (2) -> aprende rapido pero inestable en los primeros partidos.
# Muy alto (20) -> aprende lento pero suave.
# 8 es un buen balance: con 8 partidos jugados el peso data=prior.
PRIOR_WEIGHT = 8

def run():
    import wc_model_v3 as M

    res_path = os.path.join(HERE, "results_live.json")
    if not os.path.exists(res_path):
        print("wc_live_calibrate: sin resultados aun, nada que calibrar.")
        return

    results = json.load(open(res_path, encoding="utf-8"))
    if not results:
        print("wc_live_calibrate: 0 resultados, saltando.")
        return

    # --- acumular goles observados y lambdas esperadas por equipo (como atacante) ---
    att_obs   = {}  # equipo -> [goles marcados en cada partido WC 2026]
    att_prior = {}  # equipo -> [lambda esperada (como atacante) en ese partido]
    def_obs   = {}  # equipo -> [goles encajados]
    def_prior = {}  # equipo -> [lambda del rival en ese partido]

    c = 0.0028
    try:
        with open(os.path.join(HERE, "wc_probs_v3.json"), encoding="utf-8") as f:
            c = json.load(f)["c"]
    except (FileNotFoundError, KeyError):
        pass

    for r in results:
        h, a, gh, ga = r["home"], r["away"], r["gh"], r["ga"]
        if h not in M.R0 or a not in M.R0:
            continue
        lh, la = M.lambdas(h, a, c)

        att_obs.setdefault(h, []).append(gh)
        att_prior.setdefault(h, []).append(lh)
        att_obs.setdefault(a, []).append(ga)
        att_prior.setdefault(a, []).append(la)

        def_obs.setdefault(h, []).append(ga)  # goles encajados por h
        def_prior.setdefault(h, []).append(la)
        def_obs.setdefault(a, []).append(gh)
        def_prior.setdefault(a, []).append(lh)

    # --- posterior por equipo (Poisson-Gamma shrinkage) ---
    # lambda_post = (prior_weight * lambda_prior + goals_obs) / (prior_weight + n)
    # delta_att = log(lambda_post / lambda_prior_mean)
    att_delta = {}
    def_delta = {}

    for team in att_obs:
        n = len(att_obs[team])
        g_obs = sum(att_obs[team])
        lp_mean = np.mean(att_prior[team])  # lambda prior media del equipo en sus partidos

        # posterior mean del ratio multiplicativo (Gamma posterior)
        # Gamma prior: mean=1, k=PRIOR_WEIGHT -> alpha=PRIOR_WEIGHT, beta=PRIOR_WEIGHT/lp_mean
        # Posterior mean = (PRIOR_WEIGHT + g_obs) / (PRIOR_WEIGHT/lp_mean + n)
        post_lam = (PRIOR_WEIGHT * lp_mean + g_obs) / (PRIOR_WEIGHT + n)
        att_delta[team] = float(np.clip(math.log(post_lam / lp_mean), -0.30, 0.30))

    for team in def_obs:
        n = len(def_obs[team])
        g_enc = sum(def_obs[team])
        lp_mean = np.mean(def_prior[team])

        post_lam = (PRIOR_WEIGHT * lp_mean + g_enc) / (PRIOR_WEIGHT + n)
        # mayor encaje -> peor defensa -> def_delta positivo (encaja mas de lo esperado)
        def_delta[team] = float(np.clip(math.log(post_lam / lp_mean), -0.30, 0.30))

    # --- correccion global de nivel de goles WC 2026 ---
    total_obs = sum(r["gh"] + r["ga"] for r in results)
    total_exp = sum(sum(M.lambdas(r["home"], r["away"], c))
                    for r in results if r["home"] in M.R0 and r["away"] in M.R0)
    n_games = len(results)
    # base_factor con shrinkage fuerte hasta tener >20 partidos
    shrink = min(n_games, 20) / 20.0
    raw_factor = total_obs / max(total_exp, 0.1)
    base_factor = float(1.0 + shrink * (raw_factor - 1.0))

    out = {
        "n_matches": n_games,
        "base_factor": round(base_factor, 4),
        "att_delta": {t: round(v, 4) for t, v in att_delta.items()},
        "def_delta": {t: round(v, 4) for t, v in def_delta.items()},
    }
    out_path = os.path.join(HERE, "wc_live_adj.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"wc_live_calibrate: {n_games} partidos | base_factor={base_factor:.3f} "
          f"| {len(att_delta)} equipos actualizados")
    for t in sorted(att_delta, key=lambda x: abs(att_delta[x]), reverse=True)[:6]:
        print(f"  {t:15} att={att_delta[t]:+.3f}  def={def_delta.get(t,0):+.3f}")

if __name__ == "__main__":
    run()
