#!/usr/bin/env python3
# Baja de pollamundial.org (con TU sesion): partidos, resultados oficiales,
# y la distribucion de marcadores de TU GRUPO por partido. Privado: se queda local.
import json, os, urllib.request, urllib.error, urllib.parse
from polla_scraper import login, user_env, DB

GID = "-OufgDiagwnoYiaufaac"
HERE = os.path.dirname(os.path.abspath(__file__))

def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return None

def fetch_all():
    tok, uid = login(user_env("POLLA_EMAIL"), user_env("POLLA_PASS"))
    matches = _get(f"{DB}/matches.json?auth={tok}") or {}
    out_matches, group_stats = {}, {}
    for mid, m in matches.items():
        out_matches[mid] = m
        gs = _get(f"{DB}/groupstatistics/{GID}/{mid}.json?auth={tok}"
                  f"&orderBy=%22count%22&limitToLast=30")
        if gs:
            # "1-0-teamA" -> "1-0"; cualquiera con count>0 = lo tiene el grupo
            taken = {}
            for k, v in gs.items():
                sc = "-".join(k.split("-")[:2])
                taken[sc] = taken.get(sc, 0) + (v.get("count", 0) if isinstance(v, dict) else 0)
            group_stats[mid] = taken
    members = _get(f"{DB}/bettinggroups/{GID}/members.json?auth={tok}") or {}
    return out_matches, group_stats, members

if __name__ == "__main__":
    matches, gstats, members = fetch_all()
    for fn, data in [("polla_matches.json", matches), ("polla_groupstats.json", gstats),
                     ("polla_members.json", members)]:
        with open(os.path.join(HERE, fn), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"{len(matches)} partidos, {len(gstats)} con picks de tu grupo, {len(members)} miembros")
    # muestra los primeros con datos de grupo
    for mid in ["CM1", "CM2", "CM3"]:
        m = matches.get(mid, {})
        print(f"  {mid}: {m.get('tA')}-{m.get('tB')} gr{m.get('gr')} ts{m.get('ts')} "
              f"res {m.get('gA')}-{m.get('gB')} st={m.get('st')} | grupo: {gstats.get(mid)}")
