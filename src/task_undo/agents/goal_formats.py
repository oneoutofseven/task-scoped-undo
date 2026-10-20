"""Object-wise output instructions, without case-specific decisions or defaults."""
TABLE_SCHEMA = '''
Output a decisions object whose keys are exactly all object IDs in required_object_ids.
Every value must contain decision and target_surface.
decision must be restore or protect:
- restore: target_surface must be a destination ID from the input surface catalog.
- protect: target_surface must be null.
Decide explicitly for every object, including objects unrelated to the cancelled task and those taken over by later activities.
Do not omit objects, add objects, or output empty rows. Return JSON only, without explanations. Use English for any generated text.'''


def table_prompts(rules):
    # Preserve the semantic rules byte-for-byte; replace only the output encoding.
    semantics = rules.split('Use the input object and surface IDs.', 1)[0]
    direct = semantics + TABLE_SCHEMA + '\nThe outer object must contain exactly one field: decisions.'
    review = direct + '\nIndependently review the candidate and derive the final complete decision table from the input. No external correctness feedback is available.'
    shared = semantics + TABLE_SCHEMA + '''
Given the recorder's extracted_history_raw, select the requested cancelled activity from its tasks.
Use this extracted history as the factual basis. Do not add, remove, or modify events. Object catalogs and current supports are also provided.
The outer object must contain exactly two fields: target_task and decisions.
target_task must exactly copy the cancelled activity's name; decisions is the object-wise decision table.'''
    return direct, review, shared


def decision_schema(public, shared=False):
    """Constrain coverage and syntax using catalogs only, never task ownership."""
    def object_row(decision, target):
        return {'type': 'object', 'properties': {
            'decision': {'type': 'string', 'enum': [decision]}, 'target_surface': target},
            'required': ['decision', 'target_surface'], 'additionalProperties': False}
    row = {'anyOf': [object_row('restore', {'type':'string', 'enum':sorted(public['surfaces'])}),
                     object_row('protect', {'type':'null'})]}
    decisions = {'type':'object', 'properties':{name:row for name in sorted(public['objects'])},
                 'required':sorted(public['objects']), 'additionalProperties':False}
    fields = {'target_task': {'type':'string'}, 'decisions':decisions} if shared else {'decisions':decisions}
    return {'type':'object', 'properties':fields, 'required':list(fields), 'additionalProperties':False}
