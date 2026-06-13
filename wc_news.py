#!/usr/bin/env python3
"""
Noticias del Mundial WC2026 via ESPN API (gratis, sin clave).
Guarda wc_news.json con articulos relevantes para los próximos partidos.
build_dashboard.py los embebe en la pestaña Noticias.
"""
import json, os, time, datetime, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
CACHE_SECS = 1800  # 30 min

# Nombres de equipos en noticias ESPN -> nombre modelo
NEWS_TEAM_MAP = {
    "brazil": "Brasil", "france": "Francia", "spain": "España", "espana": "Espana",
    "england": "Inglaterra", "argentina": "Argentina", "portugal": "Portugal",
    "netherlands": "PaisesBajos", "germany": "Alemania", "belgium": "Belgica",
    "croatia": "Croacia", "uruguay": "Uruguay", "colombia": "Colombia",
    "morocco": "Marruecos", "norway": "Noruega", "japan": "Japon",
    "senegal": "Senegal", "switzerland": "Suiza", "turkey": "Turkiye",
    "united states": "EEUU", "usa": "EEUU", "mexico": "Mexico",
    "ecuador": "Ecuador", "south korea": "CoreaSur", "korea": "CoreaSur",
    "austria": "Austria", "czechia": "Chequia", "czech": "Chequia",
    "egypt": "Egipto", "ivory coast": "CostaMarfil", "côte d'ivoire": "CostaMarfil",
    "iran": "Iran", "algeria": "Argelia", "sweden": "Suecia", "canada": "Canada",
    "australia": "Australia", "paraguay": "Paraguay", "scotland": "Escocia",
    "ghana": "Ghana", "bosnia": "Bosnia", "dr congo": "RDCongo", "tunisia": "Tunez",
    "south africa": "Sudafrica", "qatar": "Catar", "uzbekistan": "Uzbekistan",
    "saudi arabia": "ArabiaSaudi", "iraq": "Iraq", "panama": "Panama",
    "jordan": "Jordania", "cape verde": "CaboVerde", "new zealand": "NuevaZelanda",
    "curacao": "Curazao", "haiti": "Haiti",
}

KEYWORDS_INJURY  = ["injur", "lesion", "fitness", "doubt", "miss", "ruled out", "baja", "lesionado"]
KEYWORDS_LINEUP  = ["lineup", "starting", "eleven", "squad", "roster", "alineacion", "titular"]
KEYWORDS_SUSPEND = ["suspend", "ban", "card", "tarjeta", "sancion"]
KEYWORDS_TACTIK  = ["tactic", "formation", "strategy", "press conference", "coach", "manager"]

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)

def _tag_article(text_lower):
    """Clasifica un artículo según su contenido."""
    tags = []
    if any(k in text_lower for k in KEYWORDS_INJURY):   tags.append("lesion")
    if any(k in text_lower for k in KEYWORDS_SUSPEND):  tags.append("tarjeta")
    if any(k in text_lower for k in KEYWORDS_LINEUP):   tags.append("alineacion")
    if any(k in text_lower for k in KEYWORDS_TACTIK):   tags.append("tactica")
    return tags or ["general"]

def _teams_mentioned(text_lower):
    """Detecta qué equipos se mencionan en el artículo."""
    found = set()
    for keyword, model_name in NEWS_TEAM_MAP.items():
        if keyword in text_lower:
            found.add(model_name)
    return sorted(found)

def fetch_news():
    """Descarga noticias ESPN del Mundial. Retorna lista de artículos procesados."""
    cache_path = os.path.join(HERE, "wc_news.json")
    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < CACHE_SECS:
            return json.load(open(cache_path, encoding="utf-8"))

    try:
        data = _get(f"{ESPN_BASE}/news?limit=50")
    except Exception as e:
        print(f"wc_news: ESPN error: {e}")
        if os.path.exists(cache_path):
            return json.load(open(cache_path, encoding="utf-8"))
        return []

    articles = []
    for a in data.get("articles", []):
        headline    = a.get("headline", "")
        description = a.get("description", "") or ""
        pub         = a.get("published", "")[:16].replace("T", " ")
        link        = a.get("links", {}).get("web", {}).get("href", "")
        categories  = [c.get("description", "") for c in a.get("categories", [])]
        img         = ""
        for img_obj in a.get("images", []):
            if img_obj.get("url"):
                img = img_obj["url"]; break

        full_text = (headline + " " + description + " " + " ".join(categories)).lower()
        tags      = _tag_article(full_text)
        teams     = _teams_mentioned(full_text)

        # calcular hace cuánto
        try:
            pub_dt  = datetime.datetime.fromisoformat(a.get("published","")[:19])
            delta   = datetime.datetime.utcnow() - pub_dt
            mins    = int(delta.total_seconds() / 60)
            age_str = f"hace {mins}m" if mins < 60 else f"hace {mins//60}h {mins%60}m"
        except Exception:
            age_str = pub

        articles.append({
            "headline":    headline,
            "description": description[:280],
            "pub":         pub,
            "age":         age_str,
            "link":        link,
            "img":         img,
            "tags":        tags,
            "teams":       teams,
        })

    json.dump(articles, open(cache_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"wc_news.json: {len(articles)} noticias descargadas")
    return articles

def relevant_for(home, away, articles, max_articles=8):
    """Filtra artículos relevantes para un partido específico."""
    out = []
    for a in articles:
        if home in a["teams"] or away in a["teams"]:
            out.append(a)
    # si hay pocas noticias específicas, añadir generales
    if len(out) < 3:
        for a in articles:
            if a not in out and "general" in a["tags"]:
                out.append(a)
                if len(out) >= max_articles:
                    break
    return out[:max_articles]

if __name__ == "__main__":
    articles = fetch_news()
    print(f"\nMuestra de noticias:")
    for a in articles[:8]:
        print(f"  [{a['age']}] {a['headline']}")
        print(f"    Teams: {a['teams']} | Tags: {a['tags']}")
