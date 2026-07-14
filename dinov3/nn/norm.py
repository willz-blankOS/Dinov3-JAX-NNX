import jax
import flax.nnx as nnx

class LayerScale(nnx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class LayerNorm(nnx.Module):
    def __init__(self):
        super().__init__()
        

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class RMSNorm(nnx.Module):
    def __init__(self):
        super().__init__()
        

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)