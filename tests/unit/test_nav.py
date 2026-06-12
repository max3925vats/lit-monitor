from lit_monitor.server.nav import NAV_GROUPS, active_group_for_path


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
