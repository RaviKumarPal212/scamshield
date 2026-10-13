from scripts.prepare_sms_splits import assign_group_splits


def test_group_split_assignment_is_deterministic_and_keeps_groups_whole() -> None:
    group_rows = {
        **{f"ham-{index}": (0, 1) for index in range(90)},
        **{f"spam-{index}": (1, 1) for index in range(30)},
        "ham-template-group": (0, 12),
        "spam-template-group": (1, 4),
    }

    first = assign_group_splits(group_rows, seed=42)
    second = assign_group_splits(group_rows, seed=42)

    assert first == second
    assert set(first) == set(group_rows)
    assert set(first.values()) == {"train", "validation", "test"}
