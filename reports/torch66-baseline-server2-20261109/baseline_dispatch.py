"""Entry point for the reviewed 66-case replacement baselines.

The canonical CPU references stay separate. Call bind only after constructing
original inputs/private in-place reset, as demonstrated by the guarded workers.
ROUTES.json is emitted only after all selected raw evidence passes final audit.
"""
import importlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent

def bind(case,args,previous):
    key=[case._call.signature.name,case.id]
    route=next((r for r in json.loads((P/'ROUTES.json').read_text()) if r['key']==key),None)
    if route is None:raise KeyError('case outside reviewed baseline scope: '+str(key))
    return importlib.import_module(route['module']).bind(case,args,previous)
