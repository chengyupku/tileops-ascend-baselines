"""Build a private performance adapter by removing explicitly audited clone statements.

Original reference remains untouched for correctness. No generic clone suppression:
only specified direct assignments in the selected function are transformed.
"""
import ast,copy,hashlib,inspect,textwrap,types

def remove_protection_clones(bound_method, names):
    original=bound_method.__func__ if inspect.ismethod(bound_method) else bound_method
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source)
    requested=set(names);seen=set()
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)!=1:return node
            target=node.targets[0]
            targets=target.elts if isinstance(target,ast.Tuple) else [target]
            values=node.value.elts if isinstance(node.value,ast.Tuple) else [node.value]
            if len(targets)!=len(values):return node
            values=copy.deepcopy(values)
            for i,(t,v) in enumerate(zip(targets,values)):
                if (isinstance(t,ast.Name) and t.id in requested and isinstance(v,ast.Call)
                    and isinstance(v.func,ast.Attribute) and v.func.attr=='clone'
                    and isinstance(v.func.value,ast.Name) and v.func.value.id==t.id
                    and not v.args and not v.keywords):
                    values[i]=ast.Name(id=t.id,ctx=ast.Load());seen.add(t.id)
            node.value=ast.Tuple(elts=values,ctx=ast.Load()) if isinstance(target,ast.Tuple) else values[0]
            return node
    tree=Rewrite().visit(tree);ast.fix_missing_locations(tree)
    if seen!=requested:raise ValueError(f'audited clone sites mismatch: requested={requested},found={seen}')
    namespace=dict(original.__globals__)
    exec(compile(tree,'<explicit-protection-adapter>','exec'),namespace)
    fn=namespace[original.__name__]
    if inspect.ismethod(bound_method):fn=types.MethodType(fn,bound_method.__self__)
    proof={'original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
           'removed_clone_assignments':sorted(seen),'adapted_source':ast.unparse(tree),
           'requires_private_inputs_and_reset':True}
    return fn,proof
