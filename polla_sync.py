#!/usr/bin/env python3
# Convierte lo bajado de la web (polla_matches.json + polla_groupstats.json) al
# formato de tu calendario -> group_stats.json (PRIVADO, se queda en tu PC).
# Clave: "Home|Away" (tus nombres) -> {"x-y": cuantos del grupo lo pusieron}.
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

CODE = {
 "ALG":"Argelia","ARG":"Argentina","AUS":"Australia","AUT":"Austria","BEL":"Belgica",
 "BIH":"Bosnia","BRA":"Brasil","CAN":"Canada","CIV":"CostaMarfil","COD":"RDCongo",
 "COL":"Colombia","CPV":"CaboVerde","CRO":"Croacia","CUR":"Curazao","CZE":"Chequia",
 "ECU":"Ecuador","EGY":"Egipto","ENG":"Inglaterra","ESP":"Espana","FRA":"Francia",
 "GER":"Alemania","GHA":"Ghana","HAI":"Haiti","IRN":"Iran","IRQ":"Iraq","JOR":"Jordania",
 "JPN":"Japon","KOR":"CoreaSur","KSA":"ArabiaSaudi","MAR":"Marruecos","MEX":"Mexico",
 "NED":"PaisesBajos","NOR":"Noruega","NZL":"NuevaZelanda","PAN":"Panama","PAR":"Paraguay",
 "POR":"Portugal","QAT":"Catar","RSA":"Sudafrica","SCO":"Escocia","SEN":"Senegal",
 "SUI":"Suiza","SWE":"Suecia","TUN":"Tunez","TUR":"Turkiye","URU":"Uruguay","USA":"EEUU",
 "UZB":"Uzbekistan",
}

# valores de puntos (verificados en el codigo de la web)
PT = {"ceS": 5, "ccW": 2, "ccG": 1, "cuP": 2}
BONUS = {"bonusRoundr32": 10, "countcorrectRoundr32": 10}  # placeholders por si aparecen

def sync():
    matches = json.load(open(os.path.join(HERE, "polla_matches.json"), encoding="utf-8"))
    gstats = json.load(open(os.path.join(HERE, "polla_groupstats.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(HERE, "calendar_final.json"), encoding="utf-8"))
    cal_idx = {frozenset((c["home"], c["away"])): c for c in cal}

    # 1) picks de tu grupo -> group_stats.json
    out, results = {}, []
    for mid, m in matches.items():
        ha, hb = CODE.get(m.get("tA")), CODE.get(m.get("tB"))
        if not ha or not hb:
            continue
        c = cal_idx.get(frozenset((ha, hb)))
        if not c:
            continue
        flip = (c["home"] != ha)
        taken = gstats.get(mid)
        if taken:
            oriented = {}
            for sc, n in taken.items():
                x, y = sc.split("-")
                key = f"{y}-{x}" if flip else f"{x}-{y}"
                oriented[key] = oriented.get(key, 0) + n
            out[f"{c['home']}|{c['away']}"] = oriented
        # 2) resultados oficiales (partidos jugados, pf=true) -> results_live.json
        if m.get("pf") and m.get("gA") is not None:
            gh, ga = m["gA"], m["gB"]
            if flip: gh, ga = ga, gh
            results.append({"home": c["home"], "away": c["away"], "gh": gh, "ga": ga,
                            "stage": "GROUP"})
    with open(os.path.join(HERE, "group_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "results_live.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # 3) tabla de posiciones del grupo -> standings.json
    members = json.load(open(os.path.join(HERE, "polla_members.json"), encoding="utf-8"))
    rows = []
    for muid, mem in members.items():
        subs = [(sid, s) for sid, s in (mem.get("subscription") or {}).items()
                if s.get("s") == "accepted"]
        # quita suscripciones duplicadas VACIAS (sin tabla = nunca jugaron, siempre 0);
        # si la persona tiene alguna con datos, usa solo esas; si no, deja una
        with_table = [(sid, s) for sid, s in subs if s.get("table")]
        use = with_table if with_table else subs[:1]
        for sid, s in use:
            t = s.get("table") or {}
            base = sum(PT[k] * t.get(k, 0) for k in PT)
            bonus = sum(v for k, v in t.items() if k.startswith("bonus") or k.startswith("countcorrectRound"))
            rows.append({"name": mem.get("n", "?"), "sc": s.get("sc", 1),
                         "pts": base + bonus,
                         "ceS": t.get("ceS", 0), "ccW": t.get("ccW", 0),
                         "ccG": t.get("ccG", 0), "cuP": t.get("cuP", 0),
                         "me": (muid == "HZGu5zdBCJcpIV246oG5NbmCib63")})
    rows.sort(key=lambda r: -r["pts"])
    for i, r in enumerate(rows, 1):
        r["pos"] = i
    with open(os.path.join(HERE, "standings.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return out, results, rows

if __name__ == "__main__":
    out, results, rows = sync()
    print(f"group_stats.json: {len(out)} partidos · results_live.json: {len(results)} resultados "
          f"· standings.json: {len(rows)} entradas")
    print("\nTabla de posiciones:")
    for r in rows[:14]:
        me = " <- TU" if r["me"] else ""
        print(f"  {r['pos']:>2}. {r['name'][:24]:24} {r['pts']:>3} pts "
              f"(exa{r['ceS']} gan{r['ccW']} gol{r['ccG']} uni{r['cuP']}){me}")
