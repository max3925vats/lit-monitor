from lit_monitor.server.nav import NAV_GROUPS, active_group_for_path, breadcrumb_trail


def test_groups_in_order_with_final_labels():
    labels = [g.label for g in NAV_GROUPS]
    assert labels == ["Monitor", "Semantics", "Explore", "Tune", "Setup"]


def test_monitor_holds_discovery_and_schedule():
    monitor = next(g for g in NAV_GROUPS if g.label == "Monitor")
    hrefs = [i.href for i in monitor.items]
    assert hrefs == ["/discovery", "/schedule"]


def test_active_group_resolves_nested_paths():
    assert active_group_for_path("/discovery/2") == "Monitor"
    assert active_group_for_path("/corpus/10.1/x") == "Semantics"
    assert active_group_for_path("/settings") == "Tune"


def test_breadcrumb_list_page():
    assert breadcrumb_trail("/corpus") == [("Semantics", None), ("Corpus", None)]


def test_breadcrumb_detail_page_links_section():
    assert breadcrumb_trail("/corpus/10.1/x", detail="A paper") == [
        ("Semantics", None), ("Corpus", "/corpus"), ("A paper", None)
    ]


def test_breadcrumb_home_is_empty():
    assert breadcrumb_trail("/") == []


def test_show_stats_banner_excludes_home_setup_settings_dev():
    from lit_monitor.server.nav import show_stats_banner
    assert show_stats_banner("/discovery") is True
    assert show_stats_banner("/corpus/10.1/x") is True
    assert show_stats_banner("/") is False
    assert show_stats_banner("/setup") is False
    assert show_stats_banner("/setup/step-1") is False
    assert show_stats_banner("/settings") is False
    assert show_stats_banner("/dev") is False
    assert show_stats_banner("/dev/anything") is False
