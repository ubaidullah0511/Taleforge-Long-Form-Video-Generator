"""Offline tests for app.niches — no network/API keys needed.
Run with: pytest tests/test_niches.py
"""
import pytest

from app.config import settings
from app.niches import (
    CUSTOM_NICHES_PATH,
    DEFAULT_NICHE,
    NICHES,
    NicheConfig,
    _find_matching_local_library_folder,
    _load_custom_niches_file,
    _save_custom_niches_file,
    add_custom_niche,
    candidate_violates_niche,
    delete_custom_niche,
    get_niche,
    is_custom_niche,
    rename_custom_niche,
    resolve_sub_niches,
    sorted_niches,
    violates_niche,
)


def test_get_niche_returns_default_for_none():
    assert get_niche(None) is NICHES[DEFAULT_NICHE]


def test_get_niche_returns_requested_config():
    assert get_niche("trucks").key == "trucks"


def test_get_niche_falls_back_to_default_for_unknown_key():
    assert get_niche("not_a_real_niche") is NICHES[DEFAULT_NICHE]


def test_violates_niche_catches_banned_word():
    niche = get_niche("trucks")
    assert violates_niche("a car passing on the highway", niche) is True
    assert violates_niche("motorcycle rider on a mountain road", niche) is True


def test_violates_niche_is_case_insensitive():
    niche = get_niche("trucks")
    assert violates_niche("Vintage SEDAN parked outside", niche) is True


def test_violates_niche_does_not_false_positive_on_substrings():
    """Plain substring containment would flag "cargo"/"carrier" just because
    they contain "car" — exactly the kind of legitimate trucking keyword this
    niche wants to keep, so the check must be whole-word only."""
    niche = get_niche("trucks")
    assert violates_niche("cargo carrier at freight terminal", niche) is False
    assert violates_niche("trucking career opportunities", niche) is False


def test_violates_niche_allows_clean_keyword():
    niche = get_niche("trucks")
    assert violates_niche("semi truck climbing a mountain grade", niche) is False


def test_candidate_violates_niche_rejects_pure_non_truck_metadata():
    """Real Pexels example: a "truck highway" search returned a hit titled
    (via URL slug) "cars in the road" — no truck signal anywhere, this is
    exactly the leak this filter exists to catch."""
    niche = get_niche("trucks")
    assert candidate_violates_niche("cars in the road", niche) == "car"


def test_candidate_violates_niche_allows_mixed_metadata_with_truck_signal():
    """Real Pixabay tags for an actual semi-truck highway video:
    "highway, traffic, cars, road, transportation, truck, speed" — "cars"
    is present but so is "truck", so this must NOT be rejected; ordinary
    highway footage legitimately shows both."""
    niche = get_niche("trucks")
    assert candidate_violates_niche("highway, traffic, cars, road, transportation, truck, speed", niche) is None


def test_candidate_violates_niche_returns_none_for_empty_text():
    niche = get_niche("trucks")
    assert candidate_violates_niche("", niche) is None
    assert candidate_violates_niche("   ", niche) is None


def test_candidate_violates_niche_allows_clean_truck_metadata():
    niche = get_niche("trucks")
    assert candidate_violates_niche("semi truck climbing a mountain grade", niche) is None


def test_sorted_niches_nests_children_directly_under_their_parent():
    """A sub-niche (parent_key set) must sort immediately after its parent,
    not wherever its own display_name would otherwise land alphabetically —
    that's what lets the dropdown render it as visually nested."""
    from app.niches import NicheConfig

    NICHES["zzz_test_parent"] = NicheConfig(
        key="zzz_test_parent", display_name="Zzz Parent", system_context="",
        banned_terms=[], positive_terms=[], safe_fallback_keyword="x",
    )
    NICHES["zzz_test_child"] = NicheConfig(
        key="zzz_test_child", display_name="Aaa Child", system_context="",
        banned_terms=[], positive_terms=[], safe_fallback_keyword="x",
        parent_key="zzz_test_parent",
    )
    try:
        ordered_keys = [n.key for n in sorted_niches()]
        parent_idx = ordered_keys.index("zzz_test_parent")
        assert ordered_keys[parent_idx + 1] == "zzz_test_child"
    finally:
        NICHES.pop("zzz_test_parent", None)
        NICHES.pop("zzz_test_child", None)


def test_sorted_niches_treats_orphaned_parent_key_as_top_level():
    from app.niches import NicheConfig

    NICHES["zzz_orphan"] = NicheConfig(
        key="zzz_orphan", display_name="Zzz Orphan", system_context="",
        banned_terms=[], positive_terms=[], safe_fallback_keyword="x",
        parent_key="does_not_exist",
    )
    try:
        assert "zzz_orphan" in [n.key for n in sorted_niches()]
    finally:
        NICHES.pop("zzz_orphan", None)


def _install_test_niches():
    """A parent with two sub-niches (for the merge-happy-path tests) plus an
    unrelated second parent+child pair (for the mismatched-parent test)."""
    NICHES["zzz_parent"] = NicheConfig(
        key="zzz_parent", display_name="Zzz Parent", system_context="Zzz parent context.",
        banned_terms=["parent_banned"], positive_terms=["parent_positive"],
        safe_fallback_keyword="parent fallback", use_archive_org=True, local_library_path="",
    )
    NICHES["zzz_child_a"] = NicheConfig(
        key="zzz_child_a", display_name="Zzz Child A", system_context="child a context",
        banned_terms=["child_a_banned"], positive_terms=["child_a_positive"],
        safe_fallback_keyword="child a fallback", local_library_path="child_a_lib", parent_key="zzz_parent",
    )
    NICHES["zzz_child_b"] = NicheConfig(
        key="zzz_child_b", display_name="Zzz Child B", system_context="child b context",
        banned_terms=["child_b_banned", "child_a_banned"], positive_terms=["child_b_positive"],
        safe_fallback_keyword="child b fallback", local_library_path="child_b_lib", parent_key="zzz_parent",
    )
    NICHES["zzz_other_parent"] = NicheConfig(
        key="zzz_other_parent", display_name="Zzz Other Parent", system_context="",
        banned_terms=[], positive_terms=[], safe_fallback_keyword="x",
    )
    NICHES["zzz_other_child"] = NicheConfig(
        key="zzz_other_child", display_name="Zzz Other Child", system_context="",
        banned_terms=[], positive_terms=[], safe_fallback_keyword="x", parent_key="zzz_other_parent",
    )


def _remove_test_niches():
    for key in ["zzz_parent", "zzz_child_a", "zzz_child_b", "zzz_other_parent", "zzz_other_child"]:
        NICHES.pop(key, None)


def test_resolve_sub_niches_single_key_resolves_directly():
    _install_test_niches()
    try:
        config, paths = resolve_sub_niches(["zzz_child_a"])
        assert config.key == "zzz_child_a"
        assert config.banned_terms == ["child_a_banned"]
        assert paths == ["child_a_lib"]
    finally:
        _remove_test_niches()


def test_resolve_sub_niches_dedupes_repeated_keys_to_single_key_behavior():
    _install_test_niches()
    try:
        config, paths = resolve_sub_niches(["zzz_child_a", "zzz_child_a"])
        assert config.key == "zzz_child_a"
        assert paths == ["child_a_lib"]
    finally:
        _remove_test_niches()


def test_resolve_sub_niches_merges_same_parent_sub_niches():
    """Multi-select's core contract: shared parent's system_context, unioned
    (deduped) banned/positive terms across the selected sub-niches, and every
    selected sub-niche's own local_library_path preserved separately (for
    search_clip to query each one, not just the first)."""
    _install_test_niches()
    try:
        config, paths = resolve_sub_niches(["zzz_child_a", "zzz_child_b"])
        assert config.key == "zzz_parent"
        assert config.system_context == "Zzz parent context."
        assert config.banned_terms == ["child_a_banned", "child_b_banned"]
        assert config.positive_terms == ["child_a_positive", "child_b_positive"]
        assert config.safe_fallback_keyword == "parent fallback"
        assert config.use_archive_org is True
        assert paths == ["child_a_lib", "child_b_lib"]
    finally:
        _remove_test_niches()


def test_resolve_sub_niches_raises_on_mismatched_parents():
    _install_test_niches()
    try:
        with pytest.raises(ValueError, match="must all share the same parent"):
            resolve_sub_niches(["zzz_child_a", "zzz_other_child"])
    finally:
        _remove_test_niches()


def test_resolve_sub_niches_raises_on_unknown_key():
    with pytest.raises(ValueError, match="unknown niche key"):
        resolve_sub_niches(["not_a_real_niche"])


def test_resolve_sub_niches_raises_on_empty_list():
    with pytest.raises(ValueError, match="no niche selected"):
        resolve_sub_niches([])


def _cleanup_custom(key: str):
    NICHES.pop(key, None)
    data = _load_custom_niches_file()
    if key in data:
        del data[key]
        _save_custom_niches_file(data)


def test_is_custom_niche_true_for_custom_false_for_builtin(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    assert is_custom_niche("trucks") is False
    assert is_custom_niche("not_a_real_niche") is False
    _cleanup_custom("zzz_custom")
    try:
        add_custom_niche("zzz_custom", "Zzz Custom", "ctx", [])
        assert is_custom_niche("zzz_custom") is True
    finally:
        _cleanup_custom("zzz_custom")


def test_find_matching_local_library_folder_is_case_insensitive(tmp_path, monkeypatch):
    """Folders on disk don't reliably match a niche key's exact case (e.g.
    clips/local/American_Secrets/ for key "american_secrets") — the match
    must ignore case and return the real on-disk folder name."""
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    (tmp_path / "local" / "American_Secrets").mkdir(parents=True)
    assert _find_matching_local_library_folder("american_secrets") == "American_Secrets"
    assert _find_matching_local_library_folder("no_such_niche") == ""


def test_add_custom_niche_auto_links_matching_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    (tmp_path / "local" / "zzz_new_niche").mkdir(parents=True)
    _cleanup_custom("zzz_new_niche")
    try:
        config = add_custom_niche("zzz_new_niche", "Zzz New Niche", "ctx", [])
        assert config.local_library_path == "zzz_new_niche"
    finally:
        _cleanup_custom("zzz_new_niche")


def test_add_custom_niche_creates_folder_when_no_matching_folder_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup_custom("zzz_no_folder")
    try:
        config = add_custom_niche("zzz_no_folder", "Zzz No Folder", "ctx", [])
        assert config.local_library_path == "zzz_no_folder"
        assert (tmp_path / "local" / "zzz_no_folder").is_dir()
    finally:
        _cleanup_custom("zzz_no_folder")


def test_delete_custom_niche_removes_from_registry_and_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup_custom("zzz_deletable")
    add_custom_niche("zzz_deletable", "Zzz Deletable", "ctx", [])
    delete_custom_niche("zzz_deletable")
    assert "zzz_deletable" not in NICHES
    assert "zzz_deletable" not in _load_custom_niches_file()


def test_delete_custom_niche_blocks_builtin():
    with pytest.raises(ValueError, match="built-in"):
        delete_custom_niche("trucks")


def test_delete_custom_niche_blocks_when_sub_niches_reference_it(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup_custom("zzz_parent_del")
    _cleanup_custom("zzz_child_del")
    try:
        add_custom_niche("zzz_parent_del", "Zzz Parent Del", "ctx", [])
        add_custom_niche("zzz_child_del", "Zzz Child Del", "ctx", [], parent_key="zzz_parent_del")
        with pytest.raises(ValueError, match="zzz_child_del"):
            delete_custom_niche("zzz_parent_del")
        assert "zzz_parent_del" in NICHES  # not deleted
    finally:
        _cleanup_custom("zzz_child_del")
        _cleanup_custom("zzz_parent_del")


def test_rename_custom_niche_updates_display_name_only(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup_custom("zzz_renamable")
    try:
        add_custom_niche("zzz_renamable", "Old Name", "ctx", [])
        config = rename_custom_niche("zzz_renamable", "New Name")
        assert config.display_name == "New Name"
        assert config.key == "zzz_renamable"
        assert NICHES["zzz_renamable"].display_name == "New Name"
        assert _load_custom_niches_file()["zzz_renamable"]["display_name"] == "New Name"
    finally:
        _cleanup_custom("zzz_renamable")


def test_rename_custom_niche_blocks_builtin():
    with pytest.raises(ValueError, match="built-in"):
        rename_custom_niche("trucks", "New Trucks")


if __name__ == "__main__":
    test_get_niche_returns_default_for_none()
    test_get_niche_returns_requested_config()
    test_get_niche_falls_back_to_default_for_unknown_key()
    test_violates_niche_catches_banned_word()
    test_violates_niche_is_case_insensitive()
    test_violates_niche_does_not_false_positive_on_substrings()
    test_violates_niche_allows_clean_keyword()
    test_candidate_violates_niche_rejects_pure_non_truck_metadata()
    test_candidate_violates_niche_allows_mixed_metadata_with_truck_signal()
    test_sorted_niches_nests_children_directly_under_their_parent()
    test_sorted_niches_treats_orphaned_parent_key_as_top_level()
    test_candidate_violates_niche_returns_none_for_empty_text()
    test_candidate_violates_niche_allows_clean_truck_metadata()
    test_resolve_sub_niches_single_key_resolves_directly()
    test_resolve_sub_niches_dedupes_repeated_keys_to_single_key_behavior()
    test_resolve_sub_niches_merges_same_parent_sub_niches()
    test_resolve_sub_niches_raises_on_mismatched_parents()
    test_resolve_sub_niches_raises_on_unknown_key()
    test_resolve_sub_niches_raises_on_empty_list()
    print("OK")
