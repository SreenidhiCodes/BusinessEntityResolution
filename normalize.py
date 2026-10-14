"""Text normalisation for business names / addresses.

Everything is local, rule based (no external lookups). Non-Latin scripts
(Devanagari etc.) are transliterated to ASCII with `unidecode` so that they can
be compared with their Latin-script counterparts; a consonant "skeleton" is
then used to absorb transliteration / vowel-spelling variants.
"""
import re
import unicodedata

from unidecode import unidecode

NAME_ABBREV = {
    "corp": "corporation", "co": "company", "inc": "incorporated", "incorp": "incorporated",
    "ltd": "limited", "ltda": "limited", "pvt": "private", "pte": "private", "priv": "private",
    "intl": "international", "mfg": "manufacturing", "assoc": "associates", "svcs": "services",
    "svc": "service", "grp": "group", "bros": "brothers", "mgmt": "management",
    "tech": "technologies", "techs": "technologies", "technology": "technologies",
    "ent": "enterprises", "entp": "enterprises", "enterprise": "enterprises",
    "sol": "solutions", "solns": "solutions", "solution": "solutions",
    "indus": "industries", "inds": "industries", "industry": "industries",
    "engg": "engineering", "eng": "engineering", "hosp": "hospital", "natl": "national",
    "dept": "department", "univ": "university", "assn": "association", "st": "saint",
    "ste": "sainte", "cie": "compagnie", "ets": "etablissements", "et": "and", "und": "and",
}
# Legal / generic tokens removed to build the "core" name used for blocking.
LEGAL = set("""private limited public company incorporated corporation llc llp lp plc pllc pc
l.l.c limited liability partnership opc the and of sarl sas sasu eurl sa sci snc selarl scop
societe compagnie etablissements groupe group holding holdings international india us usa
de du des la le les et d l""".split())

ADDR_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court",
    "apt": "apartment", "bldg": "building", "flr": "floor", "fl": "floor", "hwy": "highway",
    "sq": "square", "nr": "near", "opp": "opposite", "rte": "route", "ste": "suite",
    "pl": "place", "pkwy": "parkway", "cir": "circle", "ter": "terrace", "trl": "trail",
    "mg": "mahatma gandhi", "no": "number", "num": "number", "ho": "house", "hno": "house number",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "marg": "road", "nagar": "nagar", "chowk": "chowk",
    "mt": "mount", "ft": "fort", "hts": "heights", "jn": "junction", "jct": "junction",
    "imp": "impasse", "all": "allee", "ch": "chemin", "sec": "sector", "ph": "phase",
    "dist": "district", "tal": "taluka", "tq": "taluka", "po": "post", "ps": "police station",
    "ind": "industrial", "est": "estate", "gnd": "ground", "grd": "ground", "gr": "ground",
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4",
}
ADDR_STOP = set("road street avenue lane drive near opposite floor number building house the of de du des la le les a".split())


US_STATES = {"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co","connecticut":"ct","delaware":"de","florida":"fl","georgia":"ga","hawaii":"hi","idaho":"id","illinois":"il","indiana":"in","iowa":"ia","kansas":"ks","kentucky":"ky","louisiana":"la","maine":"me","maryland":"md","massachusetts":"ma","michigan":"mi","minnesota":"mn","mississippi":"ms","missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj","new mexico":"nm","new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok","oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc","south dakota":"sd","tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt","virginia":"va","washington":"wa","west virginia":"wv","wisconsin":"wi","wyoming":"wy","district of columbia":"dc"}
IN_STATES = {"maharashtra":"mh","delhi":"dl","uttar pradesh":"up","west bengal":"wb","gujarat":"gj","telangana":"tg","karnataka":"ka","tamil nadu":"tn","kerala":"kl","rajasthan":"rj","madhya pradesh":"mp","andhra pradesh":"ap","haryana":"hr","punjab":"pb","bihar":"br","odisha":"od","orissa":"od","assam":"as","jharkhand":"jh","chhattisgarh":"cg","uttarakhand":"uk","himachal pradesh":"hp","goa":"ga","jammu and kashmir":"jk","chandigarh":"ch","puducherry":"py","pondicherry":"py","tripura":"tr","meghalaya":"ml","manipur":"mn","nagaland":"nl","mizoram":"mz","sikkim":"sk","arunachal pradesh":"ar"}
_STATE_RE = {}
for _d, _p in ((US_STATES, "s_"), (IN_STATES, "s_")):
    for _k, _v in _d.items():
        _STATE_RE[_k] = _p + _v
_STATE_PAT = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _STATE_RE), key=len, reverse=True)) + r")\b")
_STATE_ABBR = set(US_STATES.values()) | set(IN_STATES.values())
_JUNK = {"null", "na", "none", "nan"}

_PUNCT_RE = re.compile(r"[^0-9a-z_ ]+")
_WS_RE = re.compile(r"\s+")
_ORD_RE = re.compile(r"\b(\d+)(st|nd|rd|th|er|e|eme)\b")
_DIGIT_RE = re.compile(r"\d+")
_POSTAL_RE = re.compile(r"(?<!\d)(\d{6}|\d{5}|\d{3} \d{3})(?!\d)")
_VOWELS = str.maketrans("", "", "aeiouy")


def _ascii(s):
    if not s:
        return ""
    if s.isascii():
        return s.lower()
    return unidecode(unicodedata.normalize("NFKC", s)).lower()


def _clean(s):
    s = _ascii(s)
    s = s.replace("&", " and ").replace("@", " at ")
    s = s.replace("'", "")
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def skeleton_token(t):
    """Consonant skeleton: collapses doubled letters and drops inner vowels.
    'ttreddrs' (unidecode of Devanagari 'traders') and 'traders' -> 'trdrs'."""
    if not t or t.isdigit():
        return t
    out = [t[0]]
    for ch in t[1:]:
        if ch != out[-1]:
            out.append(ch)
    t = "".join(out)
    first, rest = t[0], t[1:].translate(_VOWELS)
    r = first + rest
    r = r.replace("ph", "f").replace("kh", "k").replace("sh", "s").replace("th", "t").replace("bh", "b").replace("dh", "d").replace("gh", "g").replace("ch", "c").replace("w", "v").replace("z", "j")
    out = [r[0]]
    for ch in r[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


LEGAL_SKEL = None


def norm_name(raw):
    """Returns (full_name, core_name, skeleton_core)."""
    global LEGAL_SKEL
    if LEGAL_SKEL is None:
        LEGAL_SKEL = {skeleton_token(w) for w in LEGAL if len(w) > 3} | {"prvt", "lmtd", "elp", "prvtlmtd", "pvt", "ltd", "lp"}
    s = _clean(raw)
    toks = [NAME_ABBREV.get(t, t) for t in s.split()]
    full = " ".join(toks)
    core_t = [t for t in " ".join(toks).split()
              if t not in LEGAL and skeleton_token(t) not in LEGAL_SKEL]
    if not core_t:
        core_t = full.split()
    core = " ".join(core_t)
    skel = " ".join(skeleton_token(t) for t in core_t)
    return full, core, skel


def norm_addr(raw):
    """Returns (address, digits_string, postal)."""
    s = _ascii(raw)
    # comma-separated components that are exactly a 2-letter state code, or junk
    parts = []
    for p in s.split(","):
        q = p.strip()
        if q in _STATE_ABBR:
            q = "s_" + q
        elif q in ("n/a", "na", "null", "none", "-", "nan"):
            q = ""
        parts.append(q)
    s = ", ".join(parts)
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    s = _ORD_RE.sub(r"\1", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _STATE_PAT.sub(lambda m: _STATE_RE[m.group(1)], s)
    toks = s.split()
    toks = [ADDR_ABBREV.get(t, t) for t in toks if t not in _JUNK]
    a = " ".join(toks)
    digits = " ".join(sorted(set(_DIGIT_RE.findall(a))))
    m = _POSTAL_RE.findall(a)
    postal = m[-1].replace(" ", "") if m else ""
    return a, digits, postal


def normalize_frame(df):
    """Adds normalised columns to a source dataframe (in place, returns df)."""
    names = [norm_name(x) for x in df["business_name"].tolist()]
    addrs = [norm_addr(x) for x in df["business_address"].tolist()]
    df["n_full"] = [x[0] for x in names]
    df["n_core"] = [x[1] for x in names]
    df["n_skel"] = [x[2] for x in names]
    df["a_norm"] = [x[0] for x in addrs]
    df["a_dig"] = [x[1] for x in addrs]
    df["postal"] = [x[2] for x in addrs]
    df["nonascii"] = [0 if (isinstance(x, str) and x.isascii()) else 1 for x in df["business_name"].tolist()]
    return df
