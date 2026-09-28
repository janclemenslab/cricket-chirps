import dataclasses as dc
import dataclasses
from jax import tree_util as jt
import yaml


def register_jax_dataclass(cls):
    """Registers a dataclass as a JAX pytree."""
    if not dc.is_dataclass(cls):
        raise TypeError('%s is not a dataclass.' % cls)

    keys: float = [field.name for field in dc.fields(cls)]

    def _flatten(obj):
        return [getattr(obj, key) for key in keys], None

    def _unflatten(_, children):
        return cls(**dict(zip(keys, children)))

    jt.register_pytree_node(cls, _flatten, _unflatten)
    return cls


def jax_dataclass(cls):
    """Decorator function to define a dataclass with JAX bindings."""
    return register_jax_dataclass(dc.dataclass(cls))


@jax_dataclass
class Params:

    def get_bounds(self):
        param_dict = self.to_dict()
        bounds = []
        for key in param_dict.keys():
            if 'tau' in key or 'sigma' in key:
                bounds.append((0, 10_000))
            elif 'delay' in key:
                bounds.append((0, 21))
            elif 'dur' in key:
                bounds.append((2, 10_000))
            else:
                bounds.append((None, None))
        return bounds

    def to_yaml(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filename: str):
        with open(filename, 'r') as f:
            return cls(**yaml.full_load(f))

    def to_m(self, filename: str):
        """Save as matlab file."""

        with open(filename, "w") as f:
            for key, val in self.to_dict().items():
                print(f"p.{key} = {val:0.4f};", file=f)

    def to_dict(self):
        dct = dataclasses.asdict(self)
        # cast all to floats so yaml looks proper
        for key, val in dct.items():
            dct[key] = float(val)
        return dct

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_list(self):
        return list(self.to_dict().values())

    @classmethod
    def from_list(cls, list):
        return cls(*list)
