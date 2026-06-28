#!/usr/bin/env python3
# Reconstruye en el calendario TODA la fase de ELIMINATORIA (32 partidos) desde la polla
# (polla_matches.json), para que el sistema represente el torneo completo (72 grupos + 32
# = 104). Los cruces con equipos YA definidos llevan pick (los procesa el pipeline normal:
# polla_sync -> group_stats, build_dashboard -> pick, polla_autobet -> apuesta). Los que aun
# tienen equipos "Por Definir" se agregan como PLACEHOLDER (tbd=True): se muestran en el
# calendario como estructura, sin pick ni apuesta, y se convierten en cruce real solos en
# cuanto se definan (este script corre cada vuelta). Idempotente: reconstruye desde cero la
# parte de eliminatoria cada ejecucion; los 72 de grupos no se tocan.
import json, os, datetime
from polla_sync import CODE

HERE = os.path.dirname(os.path.abspath(__file__))
DOW = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
STAGE = {"D": "16avos", "R": "Octavos", "Q": "Cuartos", "S": "Semis", "T": "3er puesto", "F": "Final"}
GROUPS = set("ABCDEFGHIJKL")


def build():
    cal = json.load(open(os.path.join(HERE, "calendar_final.json"), encoding="utf-8"))
    pm = json.load(open(os.path.join(HERE, "polla_matches.json"), encoding="utf-8"))
    # conservar SOLO los 72 de grupos; la eliminatoria se reconstruye entera desde la polla
    cal = [c for c in cal if c.get("g") in GROUPS]
    n_grp = len(cal)
    real = ph = 0
    for mid, m in sorted(pm.items(), key=lambda kv: kv[1].get("ts", 0)):
        st = m.get("st")
        if st == "G" or st not in STAGE or m.get("pf"):
            continue
        ha, hb = CODE.get(m.get("tA")), CODE.get(m.get("tB"))
        ts = m.get("ts")
        if ts:
            d = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc)
            date, tm = d.strftime("%Y-%m-%d"), d.strftime("%H:%M")
            dow, dlabel = DOW[d.weekday()], f"{d.day} {MES[d.month - 1]}"
        else:
            date = tm = dow = dlabel = ""
        base = {"g": STAGE[st], "date": date, "time": tm, "dow": dow, "dlabel": dlabel,
                "venue": "", "px": 0, "py": 0, "ph": 0, "pd": 0, "pa": 0, "ev": 0}
        if ha and hb:                                  # cruce definido -> con pick
            cal.append({**base, "home": ha, "away": hb}); real += 1
        else:                                          # equipos por definir -> placeholder
            cal.append({**base, "home": "Por definir", "away": "Por definir",
                        "koid": mid, "tbd": True}); ph += 1
    json.dump(cal, open(os.path.join(HERE, "calendar_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"build_ko_calendar: {n_grp} grupos + {real} cruces definidos + {ph} por definir = {len(cal)} total")
    return real + ph


if __name__ == "__main__":
    build()
