from lit_monitor.server.nav import NAV_GROUPS, active_group_for_path, breadcrumb_trail


def test_groups_in_order_with_final_labels():
    labels = [g.label for g in NAV_GROUPS]
    assert labels == ["Monitor", "Semantics", "Explore", "Tune", "Setup"]


def test_monitor_holds_discovery_and_schedule():
    monitor = next(g for g in NAV_GROUPS if g.label == "Monitor")
    hrefs = [i.href for i in monitor.items]
    assert hrefs == ["/discovery", "/schedule"]


def test_semantics_order_ask_brainbuild_corpushealth():
    # P4: Semantics renders Ask, Brain-build, Corpus Health in that exact order
    # with the renamed final label (hrefs unchanged).
    semantics = next(g for g in NAV_GROUPS if g.label == "Semantics")
    assert [i.label for i in semantics.items] == [
        "Ask", "Brain-build", "Corpus Health"
    ]
    assert [i.href for i in semantics.items] == ["/ask", "/brain-build", "/corpus"]


def test_active_group_resolves_nested_paths():
    assert active_group_for_path("/discovery/2") == "Monitor"
    assert active_group_for_path("/corpus/10.1/x") == "Semantics"
    # /settings is gone (folded into setup step-9); Domain still anchors Tune.
    assert active_group_for_path("/domain") == "Tune"


def test_no_nav_item_points_at_settings_page():
    """The standalone /settings page is removed — no nav item may link to it."""
    for group in NAV_GROUPS:
        for item in group.items:
            assert item.href != "/settings", f"{group.label} still links to /settings"


def test_no_banner_prefixes_drops_settings():
    """/settings is no longer a banner-excluded prefix (the page is gone);
    setup remains excluded, which now covers step-9 too."""
    from lit_monitor.server.nav import _NO_BANNER_PREFIXES

    assert "/settings" not in _NO_BANNER_PREFIXES
    assert "/setup" in _NO_BANNER_PREFIXES


def test_breadcrumb_list_page():
    assert breadcrumb_trail("/corpus") == [
        ("Semantics", None), ("Corpus Health", None)
    ]


def test_breadcrumb_detail_page_links_section():
    assert breadcrumb_trail("/corpus/10.1/x", detail="A paper") == [
        ("Semantics", None), ("Corpus Health", "/corpus"), ("A paper", None)
    ]


def test_breadcrumb_home_is_empty():
    assert breadcrumb_trail("/") == []


def test_show_stats_banner_excludes_home_setup_dev():
    from lit_monitor.server.nav import show_stats_banner
    assert show_stats_banner("/discovery") is True
    assert show_stats_banner("/corpus/10.1/x") is True
    assert show_stats_banner("/") is False
    assert show_stats_banner("/setup") is False
    assert show_stats_banner("/setup/step-1") is False
    # Step 9 (Tuning) lives under /setup, so it stays banner-excluded.
    assert show_stats_banner("/setup/step-9") is False
    assert show_stats_banner("/dev") is False
    assert show_stats_banner("/dev/anything") is False
