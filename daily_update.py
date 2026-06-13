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

if __name__ == "__main__":
    print(f"Pipeline polla — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    run("wc_data_feed.py")       # descarga resultados -> actualiza elo_live.json
    run("wc_live_calibrate.py")  # Bayesian update ATT/DEF/BASE desde goles WC2026
    run("wc_model_v3.py")        # carga elo_live + wc_live_adj automaticamente
    run("advance_strategy.py")
    run("wc_pool_strategy.py")
    # picks reales de tu grupo desde pollamundial.org (privado, local)
    if run("polla_fetch.py"):
        run("polla_sync.py")
    run("live_p1.py")         # P(1o) en vivo: standings + simulacion restantes -> live_stats.json
    run("build_dashboard.py") # carga live_stats.json recien generado
    run("polla_autobet.py")   # revisa/actualiza tus apuestas (regla: ~1h antes del cierre)
    print("\nListo: abre polla_v4.html")
