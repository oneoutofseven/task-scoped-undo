"""Offline fact-extraction scoring. Private annotations never enter prompts."""

from collections import Counter


def score_facts(predicted, expected):
    """Score schema-valid predictions, or None for a missing/invalid extraction.

Task names are free-form and ignored. Ownership is the chronological task index;
both sequence equality and event counts are reported. Citation accuracy is separate.
"""
    def events(facts, citation=False):
        return [(index, event["kind"], event["object_id"], event["target_surface"],
                 *([event["line"]] if citation else []))
                for index, task in enumerate(facts["tasks"]) for event in task["events"]]

    gold = events(expected)
    actual = events(predicted) if predicted is not None else []
    matched = sum((Counter(actual) & Counter(gold)).values())
    precision = matched / len(actual) if actual else 0.0
    recall = matched / len(gold) if gold else 0.0
    initial_ok = predicted is not None and predicted["initial_supports"] == expected["initial_supports"]
    target_ok = predicted is not None and predicted["target_task_index"] == expected["target_task_index"]
    task_count_ok = predicted is not None and len(predicted["tasks"]) == len(expected["tasks"])
    events_ok = predicted is not None and task_count_ok and actual == gold
    citations_ok = predicted is not None and task_count_ok and events(predicted, True) == events(expected, True)
    return {"facts_exact": bool(initial_ok and target_ok and events_ok),
            "initial_supports_correct": initial_ok, "target_task_correct": target_ok,
            "task_count_correct": task_count_ok, "event_sequence_correct": events_ok,
            "cited_event_sequence_correct": citations_ok,
            "event_precision": precision, "event_recall": recall,
            "event_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
