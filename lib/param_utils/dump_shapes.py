import jax
import json

def dump_shapes(params):
    arr2shape = lambda x: str(x.shape).replace(',)', ')')
    d = jax.tree_map(arr2shape, params)
    print(json.dumps(d, indent=2))
