#!/usr/bin/env python3
"""
Contexto de posicion en el grupo para cada equipo: calcula multiplicadores
de lambda (mas presion -> juego mas abierto -> mas goles esperados).

Mecanismo: lee polla_matches.json (resultados del WC2026), construye la tabla
de cada grupo y estima un multiplicador de intensidad ofensiva basado en cuanto
NECESITA el equipo el resultado del siguiente partido.

  0pts con 2 jugados: mult=1.15 (desesperado, todo o nada)
  6pts con 2 jugados: mult=0.90 (clasificado, puede rotar)
  default:            mult=1.00

Uso en build_dashboard.py:
    from wc_standings_context import get_lambda_mults
    ph_mult, pa_mult = get_lambda_mults("Mexico", "CoreaSur")
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))

POLLA_TO_MODEL = {
    'MEX': 'Mexico',     'RSA': 'Sudafrica',  'KOR': 'CoreaSur',   'CZE': 'Chequia',
    'USA': 'EEUU',       'CAN': 'Canada',     'ARG': 'Argentina',  'BRA': 'Brasil',
    'FRA': 'Francia',    'ENG': 'Inglaterra', 'ESP': 'Espana',     'GER': 'Alemania',
    'POR': 'Portugal',   'NED': 'PaisesBajos','BEL': 'Belgica',    'CRO': 'Croacia',
    'URU': 'Uruguay',    'COL': 'Colombia',   'MAR': 'Marruecos',  'NOR': 'Noruega',
    'JPN': 'Japon',      'SEN': 'Senegal',    'SUI': 'Suiza',      'TUR': 'Turkiye',
    'ECU': 'Ecuador',    'AUT': 'Austria',    'EGY': 'Egipto',     'CIV': 'CostaMarfil',
    'IRN': 'Iran',       'ALG': 'Argelia',    'SWE': 'Suecia',     'AUS': 'Australia',
    'PAR': 'Paraguay',   'SCO': 'Escocia',    'GHA': 'Ghana',      'BIH': 'Bosnia',
    'COD': 'RDCongo',    'TUN': 'Tunez',      'KSA': 'ArabiaSaudi','IRQ': 'Iraq',
    'PAN': 'Panama',     'JOR': 'Jordania',   'CPV': 'CaboVerde',  'NZL': 'NuevaZelanda',
    'CUR': 'Curazao',    'HAI': 'Haiti',      'QAT': 'Catar',      'UZB': 'Uzbekistan',
}

def _pressure_mult(pts, n_played):
    """Multiplicador de intensidad ofensiva segun necesidad de resultado.

    La logica: un equipo en apuros abre el juego, asume riesgos, marca mas
    Y encaja mas. Aplicamos el mismo mult a ambos lambdas del partido.
    Un equipo ya clasificado tiende a rotar y guardar energias.
    """
    if n_played == 0:
        return 1.0
    if n_played == 1:
        if pts >= 3: return 0.97   # gano, juega con tranquilidad
        if pts >= 1: return 1.02   # empato, ligera presion
        return 1.07                # perdio, necesita recuperar
    # n_played >= 2: jornada 3 o partido ya en curso
    if pts >= 6: return 0.90       # clasificado 1o seguro, puede rotar
    if pts >= 4: return 0.97       # muy probable clasificar, juega para 1o
    if pts >= 3: return 1.03       # en el limite, necesita al menos empate
    if pts >= 1: return 1.10       # en serios apuros, necesita ganar
    return 1.15                    # eliminado si no gana: todo o nada

_cache = None

def _build_standings():
    """Parsea polla_matches.json y construye tabla de posiciones de cada grupo."""
    global _cache
    if _cache is not None:
        return _cache

    path = os.path.join(HERE, "polla_matches.json")
    if not os.path.exists(path):
        _cache = {}
        return _cache

    matches = json.load(open(path, encoding="utf-8"))
    raw = {}  # group -> {team_code -> {pts, gf, ga, n}}

    for m in matches.values():
        if m.get('st') != 'G':
            continue
        tA, tB, gr = m['tA'], m['tB'], m['gr']
        if tA == 'PD' or tB == 'PD':
            continue
        grp = raw.setdefault(gr, {})
        for t in (tA, tB):
            grp.setdefault(t, {'pts': 0, 'gf': 0, 'ga': 0, 'n': 0})
        if not m.get('pf'):
            continue
        gA, gB, w = m['gA'], m['gB'], m['w']
        grp[tA]['gf'] += gA; grp[tA]['ga'] += gB; grp[tA]['n'] += 1
        grp[tB]['gf'] += gB; grp[tB]['ga'] += gA; grp[tB]['n'] += 1
        if   w == 'teamA': grp[tA]['pts'] += 3
        elif w == 'teamB': grp[tB]['pts'] += 3
        else: grp[tA]['pts'] += 1; grp[tB]['pts'] += 1

    result = {}
    for gr, grp in raw.items():
        ranked = sorted(grp.items(),
                        key=lambda x: (-x[1]['pts'], -(x[1]['gf']-x[1]['ga']), -x[1]['gf']))
        for rank, (code, st) in enumerate(ranked, 1):
            model = POLLA_TO_MODEL.get(code, code)
            result[model] = {
                'pts': st['pts'], 'gf': st['gf'], 'ga': st['ga'],
                'gd': st['gf'] - st['ga'], 'n': st['n'],
                'group': gr, 'rank': rank,
                'lambda_mult': _pressure_mult(st['pts'], st['n']),
            }
    _cache = result
    return result

def reset_cache():
    global _cache
    _cache = None

def get_lambda_mults(home_model, away_model):
    """Devuelve (mult_home, mult_away) para multiplicar sus lambdas."""
    st = _build_standings()
    mh = st.get(home_model, {}).get('lambda_mult', 1.0)
    ma = st.get(away_model, {}).get('lambda_mult', 1.0)
    return mh, ma

def get_match_context(home_model, away_model):
    """Contexto de standings para debug o display."""
    st = _build_standings()
    return {'home': st.get(home_model, {}), 'away': st.get(away_model, {})}

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    st = _build_standings()
    print(f"Contexto standings: {len(st)} equipos\n")
    for name, d in sorted(st.items(), key=lambda x: x[1].get('lambda_mult', 1.0), reverse=True):
        mult = d['lambda_mult']
        bar = "^" if mult > 1.05 else ("v" if mult < 0.95 else " ")
        print(f"  {bar} {name:16} Gr{d['group']} | pts={d['pts']} n={d['n']} "
              f"rank={d['rank']} | mult={mult:.2f}")
