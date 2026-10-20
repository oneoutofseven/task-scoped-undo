"""Dataset lineage and split checks. No inference, simulator, or private labels.

Lineage and template names are curator declarations, checked against saved source
files where possible. Hashes detect copies, not undisclosed semantic paraphrases.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from task_undo.agents.narrative import public_input
from task_undo.common.storage import read_json
from task_undo.domain.case_layout import scene_settings
from task_undo.domain.history import check_directory

BASE_AXES = ("ancestors", "snapshots", "designs", "public_inputs")
OPTIONAL_AXES = ("scenes", "assets", "templates")
SOURCE_FILES = ("case.json", "result.json", "before_undo.json",
                "before_undo_scene.json", "public_history.jsonl")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Expected existing repository file: {name}")
    return path


def verify_hashes(root, hashes):
    if not hashes:
        raise ValueError("Empty frozen file manifest")
    for name, expected in hashes.items():
        if digest(local_file(root, name)) != expected:
            raise ValueError(f"Frozen file changed: {name}")


def _names(value, field):
    if (not isinstance(value, list) or not value
            or not all(isinstance(x, str) and x.strip() for x in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"Nonempty unique names required: {field}")
    return value


def build_registry(root, index_path, bindings_path, *, exposed):
    """Register accepted physical sources and explicitly mapped language members.

    The bindings list is exhaustive for each named dataset, including symbolic
    conditions with no physical source. It never guesses ancestry from filenames.
    """
    root = Path(root).resolve()
    hashes = {}

    def read(name):
        path = local_file(root, name)
        hashes[name] = digest(path)
        return read_json(path)

    index, bindings = read(index_path), read(bindings_path)
    if bindings.get("schema_version") != 1:
        raise ValueError("Unsupported binding schema")
    sources, observations, records = {}, {}, []
    scene_members = index["scene_groups"]
    for entry in index["prefixes"]:
        source = entry["source_episode"]
        source = str(local_file(root, source + "/case.json").parent.relative_to(root))
        if source in sources or ("history_generation_accepted" in entry
                                  and entry["history_generation_accepted"] is not True):
            raise ValueError(f"Duplicate or unaccepted source: {source}")
        verify_hashes(root, {f"{source}/{name}": sha for name, sha in entry["source_sha256"].items()})
        hashes.update({f"{source}/{name}": sha for name, sha in entry["source_sha256"].items()})
        case = read(source + "/case.json")
        result = read(source + "/result.json")
        if result.get("timed_out") or result.get("interrupted"):
            raise ValueError(f"Interrupted source: {source}")
        check_directory(root / source, write_report=False)
        observation = read(source + "/before_undo.json")
        if observation != result["final"] or not observation["hand_empty"]:
            raise ValueError(f"Undo starting observation disagrees with completed history: {source}")
        observations[source] = observation
        for name in SOURCE_FILES:
            path = source + "/" + name
            hashes[path] = digest(local_file(root, path))
        scenes = [name for name, members in scene_members.items() if source in members]
        if len(scenes) != 1:
            raise ValueError(f"Source must have exactly one scene: {source}")
        if scene_settings(case)["scene_model"] != scenes[0]:
            raise ValueError(f"Scene metadata disagrees with case: {source}")
        assets = sorted({f"{o['category']}/{o['model']}" for o in case["objects"]
                         if o.get("model") and o.get("category")})
        templates = _names(bindings["source_templates"][source], "source_templates")
        record = {"id": "physical:" + source, "kind": "physical_source",
                  "exposed_to_development": exposed, "ancestors": [source],
                  "snapshots": [hashes[source + "/before_undo_scene.json"]],
                  "designs": [entry.get("paired_design_group", entry["episode_id"])],
                  "public_inputs": [], "scenes": scenes, "assets": assets,
                  "templates": templates}
        sources[source] = record
        records.append(record)
    if set(bindings["source_templates"]) != set(sources):
        raise ValueError("Source template bindings must cover exactly the source index")
    datasets = set()
    for dataset in bindings["datasets"]:
        name = dataset["name"]
        if name in datasets:
            raise ValueError(f"Duplicate dataset: {name}")
        datasets.add(name)
        cases = read(dataset["cases"])
        read(dataset["provenance"])
        ids = [c["id"] for c in cases]
        if len(ids) != len(set(ids)) or set(ids) != set(dataset["members"]):
            raise ValueError(f"Bindings must cover exactly all conditions: {name}")
        for case in cases:
            binding = dataset["members"][case["id"]]
            ancestors = binding["ancestors"]
            if (not isinstance(ancestors, list) or len(set(ancestors)) != len(ancestors)
                    or not set(ancestors) <= set(sources)):
                raise ValueError(f"Unknown or duplicate ancestor: {name}/{case['id']}")
            public = public_input(case)
            for ancestor in ancestors:
                observed = observations[ancestor]["objects"]
                if (set(public["objects"]) != set(observed) or any(
                        support not in observed[obj]["on"]
                        for obj, support in public["current_supports"].items())):
                    raise ValueError(f"Public current state disagrees with physical source: {name}/{case['id']}")
            records.append({"id": name + ":" + case["id"], "kind": "condition",
                "exposed_to_development": exposed, "ancestors": ancestors,
                "public_inputs": [hashlib.sha256(json.dumps(public, sort_keys=True,
                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()],
                "designs": _names(binding["designs"], "designs"),
                "templates": _names(binding["templates"], "templates"),
                **{axis: sorted({v for a in ancestors for v in sources[a][axis]})
                   for axis in ("snapshots", "scenes", "assets")}})
    # Failed generation still exposes a scene/design to the developer. Keep
    # these records separate from admitted physical ancestors and conditions.
    for entry in bindings.get("additional_exposures", []):
        if not exposed:
            raise ValueError("Inspected candidates cannot enter an unexposed registry")
        case = read(entry["case"])
        for evidence in _names(entry["evidence"], "exposure evidence"):
            read(evidence)
        records.append({"id": "inspected:" + entry["case"],
            "kind": "inspected_candidate", "exposed_to_development": True,
            "ancestors": [], "snapshots": [], "public_inputs": [],
            "designs": [case["episode_id"]],
            "scenes": [scene_settings(case)["scene_model"]],
            "assets": sorted({f"{o['category']}/{o['model']}" for o in case["objects"]
                              if o.get("model") and o.get("category")}),
            "templates": _names(entry["templates"], "exposure templates")})
    if len({r['id'] for r in records}) != len(records):
        raise ValueError("Duplicate registry member")
    return {"schema_version": 1, "records": records, "sha256": hashes,
            "source_index": index_path, "bindings": bindings_path}


def audit_partition(records, assignments, disjoint_axes=()):
    """Check complete membership, development exposure, and shared group keys.

    Extra axes state the claimed generalization boundary. Physical ancestry,
    checkpoint copies, paired design, and identical public inputs never cross.
    """
    if not set(disjoint_axes) <= set(OPTIONAL_AXES):
        raise ValueError("Unknown held-out axis")
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)) or set(assignments) != set(ids):
        raise ValueError("Every unique member requires exactly one split assignment")
    if not set(assignments.values()) <= {"development", "validation", "test"}:
        raise ValueError("Unknown split name")
    axes = BASE_AXES + tuple(dict.fromkeys(disjoint_axes))
    violations, links = [], defaultdict(list)
    parents = {name: name for name in ids}

    def find(name):
        while name != parents[name]:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    for record in records:
        if record["exposed_to_development"] and assignments[record["id"]] != "development":
            violations.append({"reason": "development_exposure", "members": [record["id"]]})
        for axis in axes:
            for value in record[axis]:
                links[axis, value].append(record["id"])
    for (axis, value), members in sorted(links.items()):
        for member in members[1:]:
            parents[find(member)] = find(members[0])
        if len({assignments[m] for m in members}) > 1:
            violations.append({"reason": "cross_split_" + axis, "value": value,
                               "members": members})
    groups = defaultdict(list)
    for name in sorted(ids):
        groups[find(name)].append(name)
    test_conditions = [r for r in records if r["kind"] == "condition" and assignments[r["id"]] == "test"]
    return {"valid": not violations, "violations": violations, "disjoint_axes": list(axes),
            "counts": {split: sum(v == split for v in assignments.values())
                       for split in ("development", "validation", "test")},
            "connected_groups": sorted(groups.values()),
            "test_conditions": len(test_conditions),
            "test_has_physical_sources": bool(test_conditions) and all(r["ancestors"] for r in test_conditions),
            "duplicate_public_inputs": [members for (axis, _), members in sorted(links.items())
                                        if axis == "public_inputs" and len(members) > 1]}


def check_holdout(root, exposure_path, candidate_index, candidate_bindings, *, disjoint_axes):
    """Read-only preflight; validation does not imply feasibility or human review."""
    if not disjoint_axes:
        raise ValueError("Declare at least one generalization axis: scenes/assets/templates")
    exposure = read_json(local_file(root, exposure_path))
    verify_hashes(root, exposure["sha256"])
    # Rebuild instead of trusting edited records in a JSON registry.
    rebuilt = build_registry(root, exposure["source_index"], exposure["bindings"], exposed=True)
    if rebuilt != exposure:
        raise ValueError("Exposure registry differs from its bound evidence")
    candidate = build_registry(root, candidate_index, candidate_bindings, exposed=False)
    records = exposure["records"] + candidate["records"]
    assignments = {r["id"]: "development" for r in exposure["records"]}
    assignments.update({r["id"]: "test" for r in candidate["records"]})
    report = audit_partition(records, assignments, disjoint_axes)
    if not report["valid"] or not report["test_has_physical_sources"]:
        raise ValueError("Held-out admission rejected: " + json.dumps(report, ensure_ascii=False))
    return report, candidate
