"""Prospective factorial design and failure accounting, independent of the policy.

候选先固定，结果不能决定分母。这里只构造评测，不改变被测控制器。
"""
from copy import deepcopy
from itertools import product

from task_undo.evaluation.compositions import compile_history
from task_undo.evaluation.splits import audit_partition

AXES = ("scene", "assets", "relation")
BRANCHES = ("book_adopted", "cup_adopted")
KNOWN_FAMILY = "two_prepared_objects_later_use"
NEW_FAMILY = "prior_boundary_target_revisits_later_return"
OBJECTS = {"book": "book", "cup": "cup", "plate": "plate"}
SURFACES = {"storage_surface": "storage table", "dining_table": "activity table"}


def scene_family(name):
    # 与既有暴露表保持保守的建筑家族分组；同一家族不跨集合。
    return name.split("_")[0]


def select_catalog(names, registry, blocks=2, installed_scenes=(), installed_assets=()):
    """Lexicographic selection from ZIP names, before reading new geometry."""
    seen_scenes = {scene_family(s) for r in registry["records"] for s in r["scenes"]}
    seen_scenes.update(map(scene_family, installed_scenes))
    seen_assets = {a for r in registry["records"] for a in r["assets"]} | set(installed_assets)
    scenes = sorted({n.split("/")[1] for n in names
                     if n.startswith("scenes/") and len(n.split("/")) > 2
                     and n.split("/")[1] and "garden" not in n.split("/")[1]})
    selected = []
    for scene in scenes:
        family = scene_family(scene)
        if family not in seen_scenes:
            selected.append(scene)
            seen_scenes.add(family)
            if len(selected) == blocks:
                break
    models = {}
    for category in ("plate", "mug", "hardback"):
        pool = sorted({n.split("/")[2] for n in names
                       if n.startswith(f"objects/{category}/") and len(n.split("/")) > 3
                       and n.split("/")[2]
                       and f"{category}/{n.split('/')[2]}" not in seen_assets})
        models[category] = pool[:blocks]
    if len(selected) != blocks or any(len(v) != blocks for v in models.values()):
        raise ValueError("Insufficient unseen catalog families or models; do not substitute inspected assets")
    return [{"block": i, "novel_scene": scene,
             "novel_models": {c: v[i] for c, v in models.items()}}
            for i, scene in enumerate(selected)]


def factorial_candidates(blocks, known_scene, known_models):
    rows = []
    for block in blocks:
        for levels in product((0, 1), repeat=3):
            factors = dict(zip(AXES, levels))
            rows.append({"id": f"b{block['block']}_" + "".join(map(str, levels)),
                         "block": block["block"], "factors": factors,
                         "scene": block["novel_scene"] if factors["scene"] else known_scene,
                         "models": deepcopy(block["novel_models"] if factors["assets"] else known_models),
                         "history_family": NEW_FAMILY if factors["relation"] else KNOWN_FAMILY,
                         "seed": block["block"], "variants": list(BRANCHES)})
    return rows


def task_spec(relation, block):
    """Two semantic recipes; the new family combines three known primitives.

    The new family's book must return to its pre-target dining support, not the
    initial storage support. The later plate return remains protected. Target
    revisits combine with both earlier and later work. No motion is invented.
    """
    def task(name, moves, instruction, actor="user"):
        return {"task_id": name, "initiator": actor, "executor": "robot",
                "instruction": instruction, "moves": [list(move) for move in moves]}
    if not relation:
        return [task("A", [("book", "dining_table"), ("cup", "dining_table")],
                     "Set out the book and cup on the activity table for reading."),
                task("B", [("plate", "dining_table")],
                     "Place my plate on the activity table for my separate snack activity.", "family_member")], "A"
    # 两组使用同一家族的不同次序；不把它们标为两个独立历史家族。
    middle = [("book", "storage_surface"), ("book", "dining_table"),
              ("book", "storage_surface"), ("cup", "dining_table"), ("plate", "dining_table")]
    if block % 2:
        middle = [middle[3], *middle[:3], middle[4]]
    return [task("P", [("book", "dining_table")],
                 "Put the book on the activity table for the afternoon reference activity."),
            task("A", middle,
                 "Prepare the shared reading activity in this order: "
                 + "; ".join(f"move the {obj} to the {SURFACES[surface]}" for obj, surface in middle) + "."),
            task("B", [("plate", "storage_surface")],
                 "Return the plate to the storage table for my separate plate-organizing activity.", "family_member")], "A"


def variants(relation):
    result = {}
    for adopted, other in (("book", "cup"), ("cup", "book")):
        result[adopted + "_adopted"] = {
            "followup_task": {"task_id": "C", "initiator": "family_member", "executor": "robot",
                "instruction": f"Reserve the {adopted} for my independent activity without moving it.",
                "moves": [], "use_claim": {"object_id": adopted, "actor": "family_member",
                    "purpose": "independent_use", "utterance": f"I am keeping this {adopted} for my own use. Leave it where it is."}},
            "restore": {other: "dining_table" if relation and other == "book" else "storage_surface"}}
    return result


def symbolic_condition(relation, block, branch):
    tasks, target = task_spec(relation, block)
    follow = variants(relation)[branch]["followup_task"]
    tasks.append(follow)
    tuples = [[("place", o, s) for o, s in t["moves"]]
              + ([("claim", t["use_claim"]["object_id"], None)] if "use_claim" in t else [])
              for t in tasks]
    return compile_history({o: "storage_surface" for o in OBJECTS}, tuples,
                           next(i for i, t in enumerate(tasks) if t["task_id"] == target), OBJECTS, SURFACES)


def combined_relation(facts):
    """A topology predicate, unaffected by names, task padding or text language."""
    index = facts["target_task_index"]
    tasks = facts["tasks"]
    writes = [e["object_id"] for e in tasks[index]["events"] if e["kind"] == "place"]
    active = set(writes)
    prior = any(e["kind"] == "place" and e["object_id"] in active
                for t in tasks[:index] for e in t["events"])
    later = any(e["kind"] == "place" and e["object_id"] in active
                for t in tasks[index + 1:] for e in t["events"])
    return len(writes) > len(active) and prior and later


def lineage_record(candidate, member_id, *, parent=None, snapshot=None, exposed=False):
    """All translations, branch claims and repeats inherit their source group."""
    record = {"id": member_id, "kind": "condition", "exposed_to_development": exposed,
              "ancestors": [candidate["id"]], "snapshots": [snapshot] if snapshot else [],
              "designs": [candidate["id"]], "public_inputs": [], "scenes": [candidate["scene"]],
              "assets": [f"{c}/{m}" for c, m in sorted(candidate["models"].items())],
              "templates": [candidate["history_family"]]}
    if parent is not None:
        for key in ("ancestors", "snapshots", "designs", "templates"):
            record[key] = list(parent[key])
    return record


def audit_lineage(records, assignments, *, whole_family=False):
    return audit_partition(records, assignments, ["templates"] if whole_family else [])


def summarize(candidates, rows):
    """Exact planned denominators. Missing work stays pending, never a success.

    An admitted source owes BOTH branches even if the first branch fails or the
    second worker never starts. Conditional denominator uses obligations, not
    the number of available score files. Unscorable C/V remain null upstream.
    """
    ids = [c["id"] for c in candidates]
    if len(ids) != len(set(ids)) or len(rows) != len({r["id"] for r in rows}):
        raise ValueError("Duplicate candidate or outcome")
    lookup = {r["id"]: r for r in rows}
    if not set(lookup) <= set(ids):
        raise ValueError("Unplanned outcome")
    normalized = []
    for candidate in candidates:
        row = lookup.get(candidate["id"], {})
        branches = row.get("branches", [])
        names = [b["condition"] for b in branches]
        if len(names) != len(set(names)) or not set(names) <= set(candidate["variants"]):
            raise ValueError("Duplicate or unplanned undo branch")
        source = row.get("source_success", False) is True
        if branches and not source:
            raise ValueError("Undo without an admitted source")
        successes = sum(b.get("joint_success") is True for b in branches)
        pipeline = source and len(branches) == len(candidate["variants"]) and successes == len(branches)
        if row.get("pipeline_success", pipeline) != pipeline:
            raise ValueError("Pipeline success disagrees with branch outcomes")
        normalized.append({"id": candidate["id"], "factors": candidate["factors"],
            "status": row.get("status", "pending"), "source_success": source,
            "construction_passed": row.get("construction_passed", False) is True,
            "preview_passed": row.get("preview_passed", False) is True,
            "source_attempted": "source" in row, "undo_eligible": len(candidate["variants"]) if source else 0,
            "undo_attempted": sum(b.get("attempted") is True for b in branches),
            "undo_successes": successes, "pipeline_success": pipeline,
            "failure_stage": row.get("failure_stage")})
    def counts(items):
        return {"selected_sources": len(items),
                "completed_candidates": sum(r["status"] == "complete" for r in items),
                "construction_passed": sum(r["construction_passed"] for r in items),
                "preview_passed": sum(r["preview_passed"] for r in items),
                "source_attempts": sum(r["source_attempted"] for r in items),
                "source_successes": sum(r["source_success"] for r in items),
                "eligible_undo_conditions": sum(r["undo_eligible"] for r in items),
                "attempted_undo_conditions": sum(r["undo_attempted"] for r in items),
                "successful_undo_conditions": sum(r["undo_successes"] for r in items),
                "successful_pipelines": sum(r["pipeline_success"] for r in items)}
    cells = {"".join(map(str, level)): counts([r for r in normalized
                if tuple(r["factors"][a] for a in AXES) == level])
             for level in product((0, 1), repeat=3)}
    total = counts(normalized)
    return {"status": "complete" if total["completed_candidates"] == len(candidates) else "incomplete",
            "counts": total, "cells": cells, "rows": normalized,
            "independence": "Branches, translations, repeats and shared factor levels are not IID samples."}
