#!/usr/bin/env python3
# P(quedar 1o) en tiempo real — combina:
#  (a) standings actuales (pts reales de cada jugador)
#  (b) simula partidos RESTANTES con las matrices del modelo
#  (c) mis picks = picks.json (ya optimizados con el grupo)
#  (d) picks de rivales = distribucion empirica de group_stats (o sintetico)
#  (e) bono de unicidad exacto en cada simulacion
#  (f) bonos de avance R32 (mi pred real vs sintetico de rivales)
#  (g) Brier score corriente (calibracion) sobre los partidos ya jugados
# Salida: live_stats.json
import json, os, math, datetime, time
import numpy as np
from collections import Counter
import wc_model_v3 as M
import wc_pool_strategy as S

HERE = os.path.dirname(os.path.abspath(__file__))
N_SIM = 5000
RNG = np.random.default_rng(42)

def load(name, default=None):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def sample_mat(mat):
    flat = mat.ravel().astype(float)
    flat /= flat.sum()
    idx = int(RNG.choice(len(flat), p=flat))
    return divmod(idx, mat.shape[1])

def rival_dist(key, group_stats, home, away):
    """Distribucion de picks de rivales. Empirica si disponible, sintetica si no."""
    raw = group_stats.get(key)
    if raw:
        total = sum(raw.values())
        return {tuple(map(int, sc.split("-"))): n / total for sc, n in raw.items()}
    return {k: float(v) for k, v in S.rival_pick_dist(home, away).items()}

def sample_pick(dist):
    ks = list(dist.keys())
    ps = np.array([dist[k] for k in ks], dtype=float)
    ps /= ps.sum()
    return ks[int(RNG.choice(len(ks), p=ps))]

def r32_from_results(all_results):
    """Top 2 de cada grupo + mejores 8 terceros dado un dict completo de resultados."""
    group_for = {t: g for g, tms in M.GROUPS.items() for t in tms}
    gpts = {g: {t: 0 for t in tms} for g, tms in M.GROUPS.items()}
    ggd  = {g: {t: 0 for t in tms} for g, tms in M.GROUPS.items()}
    ggf  = {g: {t: 0 for t in tms} for g, tms in M.GROUPS.items()}
    for (h, a), (gh, ga) in all_results.items():
        g = group_for.get(h)
        if g is None:
            continue
        ggd[g][h] += gh - ga; ggd[g][a] += ga - gh
        ggf[g][h] += gh;      ggf[g][a] += ga
        if   gh > ga: gpts[g][h] += 3
        elif gh == ga: gpts[g][h] += 1; gpts[g][a] += 1
        else:          gpts[g][a] += 3
    qualified = set()
    thirds = []
    for g, tms in M.GROUPS.items():
        o = sorted(tms, key=lambda t: (gpts[g][t], ggd[g][t], ggf[g][t], float(RNG.random())), reverse=True)
        qualified |= {o[0], o[1]}
        thirds.append((o[2], gpts[g][o[2]], ggd[g][o[2]], ggf[g][o[2]]))
    thirds.sort(key=lambda z: (z[1], z[2], z[3], float(RNG.random())), reverse=True)
    qualified |= {t for t, *_ in thirds[:8]}
    return qualified

def brier_score(results, C_val):
    """Brier score (3 resultados: H/D/A) sobre partidos ya jugados."""
    bs_list = []
    for r in results:
        try:
            mat = M.build_matrix(r["home"], r["away"], C_val)
        except Exception:
            continue
        pH = float(np.tril(mat, -1).sum())
        pD = float(np.trace(mat))
        pA = float(np.triu(mat, 1).sum())
        gh, ga = r["gh"], r["ga"]
        if   gh > ga: actual = (1.0, 0.0, 0.0)
        elif gh == ga: actual = (0.0, 1.0, 0.0)
        else:         actual = (0.0, 0.0, 1.0)
        bs_list.append(sum((p - o)**2 for p, o in zip((pH, pD, pA), actual)) / 3)
    if not bs_list:
        return None
    return round(sum(bs_list) / len(bs_list), 4)

def run(n_sim=N_SIM):
    standings    = load("standings.json", [])
    picks_raw    = load("picks.json", {})
    group_stats  = load("group_stats.json", {})
    results_list = load("results_live.json", [])
    cal          = load("calendar_final.json", [])

    C_val = 0.0028
    try:
        C_val = load("wc_probs_v3.json")["c"]
    except (TypeError, KeyError):
        pass

    # mis picks: (home, away) -> (bx, by)
    picks_me = {}
    for key, v in (picks_raw or {}).items():
        h, a = key.split("|")
        picks_me[(h, a)] = (int(v[0]), int(v[1]))

    my_r32 = S.my_r32_prediction()

    # partidos ya jugados (orden canonico = el del calendario, igual al de polla_sync)
    completed_dict = {}
    for r in (results_list or []):
        completed_dict[(r["home"], r["away"])] = (r["gh"], r["ga"])

    # partidos RESTANTES del calendario de grupos
    # (busca en ambas orientaciones por si results_live usa orden inverso)
    remaining = []
    for m in (cal or []):
        h, a = m["home"], m["away"]
        if (h, a) not in completed_dict and (a, h) not in completed_dict:
            remaining.append((h, a))

    me_entry    = next((r for r in standings if r.get("me")), None)
    rivals_list = [r for r in standings if not r.get("me")]
    n_rivals    = len(rivals_list)

    if not standings or me_entry is None:
        return None

    if not remaining:
        others = [r["pts"] for r in rivals_list]
        p1 = 1.0 if (not others or me_entry["pts"] > max(others)) else 0.0
        return {"p1": round(p1*100, 1), "ci_low": round(p1*100, 1), "ci_high": round(p1*100, 1),
                "brier": brier_score(results_list or [], C_val),
                "n_remaining": 0, "my_base": me_entry["pts"], "my_rank": me_entry.get("pos", "?")}

    # matrices de marcador para partidos restantes
    matrices = {}
    for (h, a) in remaining:
        try:
            matrices[(h, a)] = M.build_matrix(h, a, C_val)
        except Exception:
            matrices[(h, a)] = None

    # distribuciones de picks de rivales para partidos restantes
    rival_dists = {
        (h, a): rival_dist(f"{h}|{a}", group_stats, h, a)
        for h, a in remaining
    }

    my_base      = me_entry["pts"]
    rival_bases  = np.array([r["pts"] for r in rivals_list], dtype=float)

    wins = 0
    for _ in range(n_sim):
        # simular resultados de partidos restantes
        sim_res = {}
        for (h, a) in remaining:
            mat = matrices.get((h, a))
            sim_res[(h, a)] = sample_mat(mat) if mat is not None else (1, 0)

        # R32 a partir de resultados reales + simulados
        all_res = dict(completed_dict)
        all_res.update(sim_res)
        real_r32 = r32_from_results(all_res)

        # picks de rivales para partidos restantes (muestreados independientemente)
        rival_picks = [
            {(h, a): sample_pick(rival_dists[(h, a)]) for (h, a) in remaining}
            for _ in range(n_rivals)
        ]

        # predicciones R32 de rivales (sintetico con ruido)
        rival_r32_preds = [S.rival_r32_prediction() for _ in range(n_rivals)]

        # puntos de marcadores restantes
        my_match    = 0
        rival_match = np.zeros(n_rivals)
        uniq_pool   = {}

        for (h, a) in remaining:
            ax, ay    = sim_res[(h, a)]
            my_pk     = picks_me.get((h, a), (1, 0))
            my_match += S.pts(my_pk[0], my_pk[1], ax, ay)
            r_pks     = [rival_picks[k][(h, a)] for k in range(n_rivals)]
            for k in range(n_rivals):
                rival_match[k] += S.pts(r_pks[k][0], r_pks[k][1], ax, ay)
            uniq_pool[(h, a)] = r_pks + [my_pk]   # rival picks primero, yo al final

        # bono de unicidad sobre partidos restantes
        my_uniq    = 0
        rival_uniq = np.zeros(n_rivals)
        for (h, a) in remaining:
            pool = uniq_pool[(h, a)]
            cnt  = Counter(pool)
            if cnt[pool[-1]] == 1:          # mi pick unico
                my_uniq += 2
            for k in range(n_rivals):
                if cnt[pool[k]] == 1:
                    rival_uniq[k] += 2

        # bonos de avance R32
        my_r32b    = 10 * len(my_r32 & real_r32)
        rival_r32b = np.array([10 * len(rp & real_r32) for rp in rival_r32_preds])

        # totales finales
        my_total     = my_base + my_match + my_uniq + my_r32b
        rival_totals = rival_bases + rival_match + rival_uniq + rival_r32b

        if my_total > rival_totals.max():
            wins += 1

    p1  = wins / n_sim
    ci  = 1.96 * math.sqrt(p1 * (1 - p1) / n_sim)
    bs  = brier_score(results_list or [], C_val)

    return {
        "p1":          round(p1 * 100, 1),
        "ci_low":      round(max(0.0,   (p1 - ci) * 100), 1),
        "ci_high":     round(min(100.0, (p1 + ci) * 100), 1),
        "brier":       bs,
        "n_remaining": len(remaining),
        "n_played":    len(results_list or []),
        "my_base":     my_base,
        "my_rank":     me_entry.get("pos", "?"),
    }

if __name__ == "__main__":
    print(f"Calculando P(1o) en vivo — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    t0 = time.time()
    stats = run()
    if stats is None:
        print("Error: faltan standings.json o picks.json (corre primero polla_sync + build_dashboard)")
    else:
        elapsed = time.time() - t0
        bs_str = f"{stats['brier']:.4f}" if stats.get("brier") is not None else "—"
        print(f"P(1o) en vivo:  {stats['p1']}%  [{stats['ci_low']}%–{stats['ci_high']}%]")
        print(f"Brier score:    {bs_str}  (baseline ~0.222 para 3 resultados equi-probables)")
        print(f"Partidos:       {stats['n_played']} jugados · {stats['n_remaining']} restantes")
        print(f"Tu posicion:    {stats['my_rank']}o · {stats['my_base']} pts actuales · {elapsed:.1f}s")
        out = os.path.join(HERE, "live_stats.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"Guardado {out}")
