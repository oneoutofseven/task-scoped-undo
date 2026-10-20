"""Lossless object-wise goal encoding; no history, ownership rules, or gold labels."""


def table_to_goal(table, objects, surfaces):
    if not isinstance(table, dict) or set(table) != set(objects):
        raise ValueError('Decision table must cover every object exactly once')
    restore, protect = {}, []
    for name in sorted(table):
        row = table[name]
        if not isinstance(row, dict) or set(row) != {'decision', 'target_surface'}:
            raise ValueError('Decision rows require decision and target_surface only')
        decision, target = row['decision'], row['target_surface']
        if decision == 'protect' and target is None:
            protect.append(name)
        elif decision == 'restore' and isinstance(target, str) and target in surfaces:
            restore[name] = target
        else:
            raise ValueError('Invalid decision or target surface')
    return {'restore': restore, 'protect': protect}


def goal_to_table(goal, objects, surfaces):
    if not isinstance(goal, dict) or set(goal) != {'restore', 'protect'}:
        raise ValueError('Goal requires restore and protect only')
    restore, protect = goal['restore'], goal['protect']
    if (not isinstance(restore, dict) or not isinstance(protect, list)
            or not all(isinstance(x, str) for x in protect)
            or len(protect) != len(set(protect)) or set(restore) & set(protect)):
        raise ValueError('Goal must partition objects without duplicates')
    table = {name: {'decision': 'restore', 'target_surface': target} for name, target in restore.items()}
    table.update({name: {'decision': 'protect', 'target_surface': None} for name in protect})
    table_to_goal(table, objects, surfaces)
    return table
