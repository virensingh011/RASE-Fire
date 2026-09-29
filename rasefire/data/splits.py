import random


def make_splits(events, test_year=2021, val_frac=0.15, seed=0, group_key="region"):
    """events: list of dict(event_id, year, region). Test = held-out year. Validation = whole held-out groups
    (regions) from the remaining years, so neither pixels nor fires nor (when possible) regions leak between splits."""
    test = [e["event_id"] for e in events if e["year"] == test_year]
    rest = [e for e in events if e["year"] != test_year]
    groups = sorted({e.get(group_key, e["event_id"]) for e in rest}, key=str)
    random.Random(seed).shuffle(groups)
    val_groups, n_val = set(), 0
    for g in groups:
        if n_val >= val_frac * len(rest):
            break
        val_groups.add(g)
        n_val += sum(1 for e in rest if e.get(group_key, e["event_id"]) == g)
    val = [e["event_id"] for e in rest if e.get(group_key, e["event_id"]) in val_groups]
    train = [e["event_id"] for e in rest if e.get(group_key, e["event_id"]) not in val_groups]
    return {"train": train, "val": val, "test": test}
