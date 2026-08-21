import json
import re
from pathlib import Path
from typing import NamedTuple

from app.config import settings


class NicheConfig(NamedTuple):
    key: str
    display_name: str
    system_context: str  # Injected into every keyword-generation prompt
    banned_terms: list[str]  # Hard denylist checked post-generation
    positive_terms: list[str]  # If present alongside a banned term in stock metadata, don't hard-reject (see candidate_violates_niche)
    safe_fallback_keyword: str  # Substituted in when a keyword violates the denylist
    # Archive.org's catalog is deep for historical/archival footage (government
    # films, NASA, military training reels) but has ~nothing for contemporary
    # lifestyle niches — only True for niches where it's worth querying at all
    # (see app.documentary_pipeline.search_clip).
    use_archive_org: bool = False
    # Subfolder name under settings.local_clips_dir (e.g. "bodybuilding" for
    # clips/local/bodybuilding/) holding a curated, pre-indexed local clip
    # library for this niche (see app.stock.local_library_index /
    # app.stock.local_library). Empty means this niche has no local library —
    # gates app.stock.local_library the same way use_archive_org gates
    # Internet Archive (see documentary_pipeline.search_clip).
    local_library_path: str = ""
    # Key of the niche this one narrows (e.g. "players" -> parent_key
    # "tennis") — None for a top-level niche. Purely a display/generation
    # hint: sub-niches are otherwise ordinary NicheConfig entries in NICHES,
    # matched and filtered exactly like any other niche.
    parent_key: str | None = None


# NOTE: banned_terms should only contain substrings that are NEVER valid for
# a given niche regardless of beat context (e.g. "ocean" for a pool channel).
# Do NOT add terms that are sometimes valid depending on what the specific
# beat is describing (e.g. "laundry aisle" IS correct when a beat is about
# where to buy a product, but NOT correct as a generic disconnected visual).
# Context-dependent judgment calls belong in system_context, where the LLM
# can read the actual beat text — a blind substring filter cannot make that
# distinction and will incorrectly reject legitimate shots.


NICHES: dict[str, NicheConfig] = {
    "trucks": NicheConfig(
        key="trucks",
        display_name="Trucking / Semi Trucks / Commercial Vehicles",
        system_context=(
            "This content is strictly about SEMI TRUCKS, COMMERCIAL TRUCKING, "
            "and the trucking industry — long-haul rigs, tractor-trailers, "
            "18-wheelers, truck cabs, diesel engines, cargo/freight, truck "
            "stops, highway driving from a trucker's perspective, trucking "
            "regulations, and truck maintenance/mechanics. "
            "Every generated keyword MUST depict trucks or the trucking "
            "industry specifically. "
            "NEVER generate keywords for: passenger cars, sedans, SUVs, "
            "motorcycles, bicycles, pedestrians/people with no truck in "
            "frame, generic 'road' or 'highway' shots with no visible truck, "
            "trains, airplanes, boats, or any other vehicle type. "
            "If the script beat mentions a general concept (e.g. 'the economy', "
            "'workers', 'the industry'), reframe the keyword to show that "
            "concept THROUGH a trucking lens (e.g. 'trucker at freight "
            "terminal', 'truck driver in cab', 'semi truck at loading dock') "
            "rather than a generic/unrelated visual."
        ),
        banned_terms=[
            "car", "sedan", "suv", "motorcycle", "motorbike", "bicycle",
            "bike", "train", "airplane", "plane", "boat", "ship", "scooter",
            "pedestrian", "person walking", "family", "commuter",
        ],
        # Real stock-footage tags for legitimate truck highway footage routinely
        # co-mention "cars" (e.g. Pixabay tags "highway, traffic, cars, truck,
        # transportation" on an actual semi-truck video — verified empirically).
        # A banned term found alongside one of these isn't treated as a
        # violation; only a banned term with NO truck signal at all (e.g. a
        # candidate tagged/titled "cars in the road" for a "highway" search)
        # gets hard-rejected — see candidate_violates_niche.
        # "rig"/"freight"/"cargo"/"diesel"/"trailer" were removed (2026-07-23
        # verification run): each is generic enough on its own to appear on
        # totally off-niche footage — an industrial pipeline valve tagged
        # "diesel, equipment, industry" and a 1950s hot-rod CAR video tagged
        # "...auto, truck, rust, 1950s" both slipped through as false
        # negatives because one weak positive tag was enough to excuse an
        # otherwise all-car/all-industrial tag list. Only terms that
        # unambiguously identify the vehicle itself remain.
        positive_terms=[
            "truck", "trucks", "trucking", "trucker", "semi", "semi-truck",
            "semi truck", "18-wheeler", "eighteen-wheeler", "tractor-trailer",
            "tractor trailer", "big rig", "cab",
        ],
        safe_fallback_keyword="semi truck highway driving",
        use_archive_org=False,
    ),
    "pool_maintenance": NicheConfig(
        key="pool_maintenance",
        display_name="Swimming Pool Maintenance & Pool Chemistry",
        system_context=(
            "This content is strictly about SWIMMING POOL MAINTENANCE, "
            "POOL CHEMISTRY, and residential or commercial pool care. "
            "Relevant subjects include chlorine, pH and alkalinity balancing, "
            "borax and borates, algae prevention and treatment, chemical dosing, "
            "water-testing kits, pool filtration systems, pumps, saltwater cells, "
            "skimming, vacuuming, brushing, pool opening, closing, winterizing, "
            "covers, and maintenance tools. "
            "Every generated keyword MUST clearly depict a swimming pool, "
            "pool water, pool equipment, pool chemicals, or a person performing "
            "pool maintenance — WITH ONE EXCEPTION: if the script beat is "
            "specifically about WHERE or HOW a product is purchased/sourced "
            "(e.g. a store shelf, a laundry aisle, a specific box or product on "
            "a shelf), a store/aisle/product-shelf visual is correct and "
            "expected for THAT beat, since the visual should match what the "
            "narration is actually describing. "
            "For all OTHER beats, when borax or a kitchen/laundry ingredient is "
            "mentioned as something being USED (not purchased), show it being "
            "measured, prepared, or added beside a swimming pool — do not "
            "generate a standalone kitchen, laundry room, or grocery store "
            "visual for beats about USING the product, only for beats "
            "specifically about buying/sourcing it. "
            "NEVER generate keywords for ocean or beach swimming, lakes, rivers, "
            "water parks, generic people swimming with no maintenance context, "
            "unrelated backyard landscaping, bathtubs, or unrelated household "
            "cleaning tasks that have no connection to pool product sourcing or use. "
            "If a script beat mentions a broad concept such as safety, savings, "
            "panic, prevention, stability, or relaxation, depict that concept "
            "through pool maintenance. Examples include a pool technician testing "
            "pH, clear treated pool water, algae-covered pool walls, chemical "
            "containers beside a pool, a filtration system inspection, or a "
            "homeowner measuring pool chemicals."
        ),
        # Only unambiguous, always-invalid terms live here — see the NOTE above
        # NicheConfig. "laundry"/"kitchen"/"grocery" were removed: they're
        # correct whenever the beat is about sourcing the product (e.g. the
        # borax script's laundry-aisle opening hook) and wrong only as a
        # generic disconnected visual — a distinction only system_context's
        # LLM read of the beat text can make, not a substring match.
        banned_terms=[
            "ocean",
            "beach",
            "lake",
            "river",
            "water park",
            "waterpark",
            "bathtub",
            "bath tub",
        ],
        # Real stock-footage metadata for legitimate pool content can
        # co-mention a banned term (e.g. "lake house with pool" or "beach
        # resort pool deck"); only reject when there's no pool signal at all.
        positive_terms=[
            "pool", "swimming pool", "chlorine", "algae", "ph balance",
            "alkalinity", "borax", "borate", "pool chemical", "pool pump",
            "filtration", "skimmer", "pool cover", "pool technician",
            "saltwater cell", "pool deck", "pool water",
        ],
        safe_fallback_keyword="pool technician testing water chemistry",
        use_archive_org=False,
    ),
    "bodybuilding": NicheConfig(
        key="bodybuilding",
        display_name="Bodybuilding & Strength Training",
        system_context=(
            "This content is strictly about BODYBUILDING, WEIGHT TRAINING, and "
            "STRENGTH SPORTS. Relevant subjects include gym workouts, weightlifting, "
            "muscle-building exercises, bodybuilding competitions/posing, protein "
            "nutrition, supplements, gym equipment, and physique training. "
            "Every generated keyword MUST clearly depict weight training, a gym "
            "setting, or a bodybuilder/athlete performing a strength exercise. "
            "NEVER generate keywords for unrelated fitness (yoga, running, cardio-only "
            "content) unless the script beat specifically mentions it, generic "
            "wellness/lifestyle content, or unrelated sports."
        ),
        banned_terms=["yoga class", "marathon running", "swimming pool"],
        # Real stock/local footage for gym content routinely co-mentions general
        # fitness words; only reject when there's no bodybuilding/strength signal
        # at all (same rationale as the other niches' positive_terms).
        positive_terms=[
            "gym", "weightlifting", "weight lifting", "bodybuilder", "bodybuilding",
            "muscle", "barbell", "dumbbell", "bench press", "deadlift", "squat",
            "strength training", "physique", "posing", "protein", "workout",
        ],
        safe_fallback_keyword="bodybuilder lifting weights in gym",
        use_archive_org=False,
        local_library_path="bodybuilding",
    ),
    "prison": NicheConfig(
        key="prison",
        display_name="Prison & Incarceration",
        system_context=(
            "This content is strictly about PRISONS, INCARCERATION, and the "
            "criminal justice/corrections system — prison cells, cell blocks, "
            "correctional facilities, inmates, prison guards/correctional officers, "
            "prison yards, handcuffs, bars, courtrooms tied to sentencing, and "
            "prison life. Every generated keyword MUST clearly depict a prison/jail "
            "setting or a person/scene directly tied to incarceration. "
            "NEVER generate keywords for unrelated crime content (police chases, "
            "street crime, detective work) unless the script beat specifically "
            "mentions it, or generic unrelated urban/legal content."
        ),
        banned_terms=["police chase", "street crime", "detective office"],
        positive_terms=[
            "prison", "jail", "inmate", "incarceration", "correctional",
            "cell block", "prison guard", "prison yard", "handcuffs", "bars",
            "penitentiary", "sentencing", "prisoner",
        ],
        safe_fallback_keyword="prison cell block interior",
        use_archive_org=False,
        local_library_path="prison",
    ),
}

DEFAULT_NICHE = "trucks"

# Captured before custom_niches.json is merged in below — the fixed set of
# hardcoded niches (trucks/pool_maintenance/bodybuilding/prison) that the
# delete/rename endpoints must never touch, since they don't live in
# custom_niches.json and have no file entry to remove or edit.
BUILTIN_NICHE_KEYS = frozenset(NICHES.keys())

# User-added niches (via POST /niches in app/main.py) — same NicheConfig
# shape as the hardcoded entries above, persisted here so they survive a
# restart. Loaded once at import time and merged into NICHES below;
# add_custom_niche() keeps both the file and the live registry in sync
# afterwards so no restart is needed to use a newly-added niche.
CUSTOM_NICHES_PATH = Path(__file__).resolve().parent / "custom_niches.json"


def _load_custom_niches_file() -> dict[str, dict]:
    if not CUSTOM_NICHES_PATH.exists():
        return {}
    return json.loads(CUSTOM_NICHES_PATH.read_text(encoding="utf-8"))


def _save_custom_niches_file(data: dict[str, dict]) -> None:
    CUSTOM_NICHES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


for _custom_key, _custom_data in _load_custom_niches_file().items():
    NICHES[_custom_key] = NicheConfig(**_custom_data)


def _find_matching_local_library_folder(key: str) -> str:
    """Case-insensitive match of an existing clips/local/<folder>/ against a
    niche key, so a folder the user already dropped footage into (e.g.
    clips/local/American_Secrets/ for niche key "american_secrets") gets
    auto-linked instead of defaulting to no local library — folders don't
    reliably match a niche key's exact case. Returns the real on-disk folder
    name (preserving its actual case) so local_library_path resolves
    correctly regardless of the OS's filesystem case-sensitivity, or "" if
    no matching folder exists yet."""
    if not settings.local_clips_dir.exists():
        return ""
    for entry in settings.local_clips_dir.iterdir():
        if entry.is_dir() and entry.name.lower() == key.lower():
            return entry.name
    return ""


def add_custom_niche(
    key: str,
    display_name: str,
    system_context: str,
    banned_terms: list[str],
    parent_key: str | None = None,
) -> NicheConfig:
    """Persists a new user-generated niche to custom_niches.json and merges
    it into the live NICHES registry immediately (no restart required).
    positive_terms is left empty and safe_fallback_keyword is a generic
    default — an LLM-generated starting config, expected to be refined by
    hand once the user has run a real video through it. local_library_path
    auto-links to a same-named clips/local/ folder if one already exists
    (see _find_matching_local_library_folder); otherwise an empty
    clips/local/<key>/ folder is created on the spot and linked, so there's
    always a ready-made place to drop footage into right after creating a
    niche — no manual folder creation needed."""
    local_library_path = _find_matching_local_library_folder(key)
    if not local_library_path:
        (settings.local_clips_dir / key).mkdir(parents=True, exist_ok=True)
        local_library_path = key
    config = NicheConfig(
        key=key,
        display_name=display_name,
        system_context=system_context,
        banned_terms=banned_terms,
        positive_terms=[],
        safe_fallback_keyword=f"{display_name} b-roll footage",
        use_archive_org=False,
        local_library_path=local_library_path,
        parent_key=parent_key,
    )
    data = _load_custom_niches_file()
    data[key] = config._asdict()
    _save_custom_niches_file(data)
    NICHES[key] = config
    return config


def is_custom_niche(key: str) -> bool:
    return key in NICHES and key not in BUILTIN_NICHE_KEYS


def delete_custom_niche(key: str) -> None:
    """Removes a custom niche from custom_niches.json and the live NICHES
    registry. Footage on disk (local_library_path's folder) is left
    untouched — only the niche configuration entry is removed, so it can be
    re-linked later without re-indexing. Blocks deletion (rather than
    cascading) when other niches still reference this key as their
    parent_key — the caller must delete those sub-niches first, so a parent
    is never removed out from under children that still point at it."""
    if key not in NICHES:
        raise ValueError(f"unknown niche '{key}'")
    if key in BUILTIN_NICHE_KEYS:
        raise ValueError(f"'{key}' is a built-in niche and cannot be deleted")
    children = [n.key for n in NICHES.values() if n.parent_key == key]
    if children:
        raise ValueError(
            f"cannot delete '{key}': sub-niche(s) {', '.join(children)} still reference it as their "
            f"parent — delete those first"
        )
    data = _load_custom_niches_file()
    data.pop(key, None)
    _save_custom_niches_file(data)
    del NICHES[key]


def rename_custom_niche(key: str, display_name: str) -> NicheConfig:
    """Edits a custom niche's display_name only. Renaming the key itself is
    deliberately unsupported — it would need to rewrite every parent_key
    cross-reference in custom_niches.json and would silently break any
    already-generated project whose content_niche still points at the old
    key, so this stays display_name-only until that's handled."""
    if key not in NICHES:
        raise ValueError(f"unknown niche '{key}'")
    if key in BUILTIN_NICHE_KEYS:
        raise ValueError(f"'{key}' is a built-in niche and cannot be renamed")
    if not display_name:
        raise ValueError("display_name is required")
    data = _load_custom_niches_file()
    if key not in data:
        raise ValueError(f"'{key}' is not a custom niche")
    data[key]["display_name"] = display_name
    _save_custom_niches_file(data)
    config = NICHES[key]._replace(display_name=display_name)
    NICHES[key] = config
    return config


def get_niche(key: str | None) -> NicheConfig:
    return NICHES.get(key or DEFAULT_NICHE, NICHES[DEFAULT_NICHE])


def resolve_sub_niches(keys: list[str]) -> tuple[NicheConfig, list[str]]:
    """Resolves a multi-select sub-niche picker's chosen keys into one
    synthetic NicheConfig for keyword-generation/denylist purposes, plus the
    separate list of each selected niche's local_library_path (a NicheConfig
    only has room for one, but multi-select's whole point is letting
    search_clip query every selected sub-niche's local library in one run —
    see app.documentary_pipeline._safe_search).

    A single key resolves directly off NICHES (unlike get_niche, this raises
    on an unknown key rather than silently falling back to DEFAULT_NICHE —
    callers that want the silent-fallback behavior for a lone niche should
    keep calling get_niche() directly; this function is for the multi-select
    path, where a bad key is a real client error worth surfacing).

    2+ keys must all share the same non-None parent_key (mixing sub-niches
    from different parent categories isn't supported by this design) — the
    merged config borrows the shared parent's system_context/
    safe_fallback_keyword/use_archive_org, and unions each selected
    sub-niche's banned_terms/positive_terms (deduped, order-preserving)."""
    unique_keys = list(dict.fromkeys(keys))
    if not unique_keys:
        raise ValueError("no niche selected")
    missing = [k for k in unique_keys if k not in NICHES]
    if missing:
        raise ValueError(f"unknown niche key(s): {', '.join(missing)}")
    configs = [NICHES[k] for k in unique_keys]

    if len(configs) == 1:
        cfg = configs[0]
        return cfg, ([cfg.local_library_path] if cfg.local_library_path else [])

    parent_keys = {c.parent_key for c in configs}
    if len(parent_keys) != 1 or None in parent_keys:
        detail = ", ".join(f"{c.key} (parent={c.parent_key!r})" for c in configs)
        raise ValueError(f"multi-select niches must all share the same parent category — got: {detail}")
    parent = NICHES.get(parent_keys.pop())
    if parent is None:
        raise ValueError(f"parent niche of {[c.key for c in configs]!r} no longer exists")

    banned_terms = list(dict.fromkeys(t for c in configs for t in c.banned_terms))
    positive_terms = list(dict.fromkeys(t for c in configs for t in c.positive_terms))
    library_paths = list(dict.fromkeys(c.local_library_path for c in configs if c.local_library_path))

    merged = NicheConfig(
        key=parent.key,
        display_name=f"{parent.display_name} ({', '.join(c.display_name for c in configs)})",
        system_context=parent.system_context,
        banned_terms=banned_terms,
        positive_terms=positive_terms,
        safe_fallback_keyword=parent.safe_fallback_keyword,
        use_archive_org=parent.use_archive_org,
        local_library_path=library_paths[0] if library_paths else "",
        parent_key=parent.parent_key,
    )
    return merged, library_paths


def sorted_niches() -> list[NicheConfig]:
    """Every niche in NICHES (built-in + custom), grouped for display: each
    top-level niche (no parent, or a parent_key that no longer exists)
    sorted by display name, immediately followed by its own sub-niches
    (also sorted by display name) — so the category dropdown can render
    sub-niches nested directly under their parent instead of as unrelated
    flat entries."""
    all_niches = list(NICHES.values())
    top_level = sorted(
        (n for n in all_niches if not n.parent_key or n.parent_key not in NICHES),
        key=lambda n: n.display_name.lower(),
    )
    children_by_parent: dict[str, list[NicheConfig]] = {}
    for n in all_niches:
        if n.parent_key and n.parent_key in NICHES:
            children_by_parent.setdefault(n.parent_key, []).append(n)

    ordered: list[NicheConfig] = []
    for parent in top_level:
        ordered.append(parent)
        ordered.extend(sorted(children_by_parent.get(parent.key, []), key=lambda n: n.display_name.lower()))
    return ordered


def _term_pattern(term: str) -> str:
    # Optional trailing "s" so a singular banned/positive term (e.g. "car",
    # "truck") also matches its simple plural ("cars", "trucks") — a bare
    # \bterm\b would NOT match "cars" at all, since \b requires the match to
    # end where the word itself ends, not partway through a longer word.
    return rf"\b{re.escape(term)}s?\b"


def violates_niche(keyword: str, niche_config: NicheConfig) -> bool:
    """Whole-word match against the niche's denylist — plain substring
    containment would false-positive on legitimate in-niche terms that
    happen to contain a banned word, e.g. "cargo" or "carrier" both contain
    "car" as a substring but are exactly the kind of keyword this niche
    wants to keep. Used for OUR OWN generated keywords (canva_keyword,
    fallback_keyword, semantic search phrases) — those are short, purpose-
    built strings that should never mention a banned term at all, so any
    match is a violation."""
    lowered = keyword.lower()
    return any(re.search(_term_pattern(term), lowered) for term in niche_config.banned_terms)


def candidate_violates_niche(text: str, niche_config: NicheConfig) -> str | None:
    """Checks raw stock-provider metadata (tags/title/alt-text/URL slug)
    ahead of candidate scoring. Unlike violates_niche, real-world stock
    footage metadata legitimately co-mentions banned terms alongside the
    niche itself — ordinary highway footage is tagged with both "cars" and
    "truck" at once. Only rejects when a banned term appears with NO
    positive in-niche signal anywhere in the same text, which is a much
    stronger signal the footage isn't about the niche at all. Returns the
    matched banned term (for logging) if it should be rejected, else None —
    including when text is empty, since there's nothing to check.

    Known limitation (verification run, 2026-07-23): this is a denylist, not
    an allowlist, so it can't catch a candidate that mentions no banned term
    at all yet still isn't the niche (e.g. an industrial pipeline valve
    tagged "diesel, equipment, industry" for a truck-engine search — no
    "car"/"train" to trigger on, but also zero truck signal). A stricter
    "require a positive_terms match for any non-empty text" rule was tried
    and reverted: it closed that case but rejected real, sparsely-tagged
    legitimate footage too (confirmed by 8 unrelated pipeline tests breaking
    on terse test-fixture text with no niche signal), while still not
    catching a banned term sitting alongside one incidental, technically-true
    positive term (e.g. a vintage car video's 19-tag list that happens to
    include "truck" once). Accepted as a metadata-dependent gap alongside the
    empty-text gap _check_candidate_niche already documents."""
    lowered = text.lower().strip()
    if not lowered:
        return None
    if any(re.search(_term_pattern(term), lowered) for term in niche_config.positive_terms):
        return None
    for term in niche_config.banned_terms:
        if re.search(_term_pattern(term), lowered):
            return term
    return None
