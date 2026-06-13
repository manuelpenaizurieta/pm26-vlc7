#!/usr/bin/env python3
# Pipeline diario completo (lo llama la tarea de las 08:00 o se corre a mano):
#   1. wc_data_feed     -> resultados nuevos + cuotas (si hay claves de API)
#   2. wc_model_v3      -> recalibra y re-simula con ratings actualizados
#   3. wc_pool_strategy -> re-evalua politicas con el c nuevo
#   4. build_dashboard  -> regenera polla_v4.html
import subprocess, sys, os, json, datetime

HERE = os.path.dirname(os.path.abspath(__file__))

def run(script):
    print(f"\n=== {script} ===")
    r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE)
    if r.returncode != 0:
        print(f"ERROR en {script} (codigo {r.returncode}) — sigo con lo demas")
    return r.returncode == 0

def should_run_fit_ratings():
    """Auto-trigger fit_ratings cuando hay suficientes datos (>=12 partidos de grupo)
    y ha pasado mas de 12h desde el ultimo ajuste."""
    results_path = os.path.join(HERE, "results_live.json")
    ts_path = os.path.join(HERE, "fit_ratings_last.json")
    if not os.path.exists(results_path):
        return False
    try:
        results = json.load(open(results_path, encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return False
    group_matches = [r for r in results if "GROUP" in r.get("stage", "GROUP_STAGE")]
    if len(group_matches) < 12:
        return False
    if os.path.exists(ts_path):
        try:
            ts = json.load(open(ts_path, encoding="utf-8")).get("ts", 0)
            hours_since = (datetime.datetime.now().timestamp() - ts) / 3600
            if hours_since < 12:
                print(f"fit_ratings: ultimo ajuste hace {hours_since:.1f}h (<12h), omitiendo")
                return False
        except (json.JSONDecodeError, IOError):
            pass
    return True

if __name__ == "__main__":
    print(f"Pipeline polla — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    run("wc_data_feed.py")       # descarga resultados + cuotas O/U+AH -> elo_live.json
    run("wc_cards.py")           # tarjetas y suspensiones via ESPN -> wc_suspensions.json
    run("wc_live_calibrate.py")  # Bayesian update ATT/DEF/BASE desde goles WC2026
    if should_run_fit_ratings():
        print("\n=== fit_ratings.py (auto-trigger: >=12 partidos de grupo) ===")
        if run("fit_ratings.py"):
            ts_data = {"ts": datetime.datetime.now().timestamp(),
                       "date": datetime.datetime.now().isoformat()}
            json.dump(ts_data, open(os.path.join(HERE, "fit_ratings_last.json"), "w"), indent=1)
    run("wc_model_v3.py")        # carga elo_live + wc_live_adj + style_adj automaticamente
    run("advance_strategy.py")
    run("wc_pool_strategy.py")
    # picks reales de tu grupo desde pollamundial.org (privado, local)
    if run("polla_fetch.py"):
        run("polla_sync.py")
    run("live_p1.py")         # P(1o) en vivo: standings + simulacion restantes -> live_stats.json
    run("build_dashboard.py") # carga live_stats.json recien generado
    run("polla_autobet.py")   # revisa/actualiza tus apuestas (regla: ~1h antes del cierre)
    print("\nListo: abre polla_v4.html")
