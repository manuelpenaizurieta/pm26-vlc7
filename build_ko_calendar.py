#!/usr/bin/env python3
# Agrega al calendario los partidos de ELIMINATORIA cuyos equipos YA estan definidos en la
# polla (polla_matches.json), para que el pipeline existente (polla_sync -> group_stats,
# build_dashboard -> picks, polla_autobet -> apuestas) los procese igual que los de grupos.
# Se ejecuta cada vuelta: a medida que se resuelven cruces (PD -> equipo real) se agregan
# solos. Idempotente: no duplica los que ya estan. El calendario de grupos (72) no se toca.
import json, os, time, datetime
from polla_sync import CODE

HERE = os.path.dirname(os.path.abspath(__file__))
DOW = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
# etiqueta de ronda segun el campo 'st' de la polla
STAGE = {"D": "16avos", "R": "Octavos", "Q": "Cuartos", "S": "Semis", "T": "3er puesto", "F": "Final"}


def build():
    cal = json.load(open(os.path.join(HERE, "calendar_final.json"), encoding="utf-8"))
    pm = json.load(open(os.path.join(HERE, "polla_matches.json"), encoding="utf-8"))
    have = {frozenset((c["home"], c["away"])) for c in cal}
    added = 0
    for mid, m in pm.items():
        st = m.get("st")
        if st == "G" or st not in STAGE:       # solo eliminatoria
            continue
        if m.get("pf"):                          # ya jugado -> no apostar
            continue
        ha, hb = CODE.get(m.get("tA")), CODE.get(m.get("tB"))
        if not ha or not hb:                     # algun lado aun "PD" (por definir) -> esperar
            continue
        key = frozenset((ha, hb))
        if key in have:                          # ya esta en el calendario
            continue
        ts = m.get("ts")
        if ts:
            d = datetime.datetime.fromtimestamp(ts / 1000, datetime.timezone.utc)
            date = d.strftime("%Y-%m-%d"); tm = d.strftime("%H:%M")
            dow = DOW[d.weekday()]; dlabel = f"{d.day} {MES[d.month - 1]}"
        else:
            date = tm = dow = dlabel = ""
        cal.append({"g": STAGE[st], "home": ha, "away": hb, "date": date, "time": tm,
                    "dow": dow, "dlabel": dlabel, "venue": "",
                    "px": 1, "py": 0, "ph": 0, "pd": 0, "pa": 0, "ev": 0})
        have.add(key)
        added += 1
    if added:
        json.dump(cal, open(os.path.join(HERE, "calendar_final.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"build_ko_calendar: {added} cruces de eliminatoria agregados (calendario ahora {len(cal)})")
    return added


if __name__ == "__main__":
    build()
