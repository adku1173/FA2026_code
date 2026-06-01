# FA2026_code

Source code for the Forum Acusticum 2026 paper.

## Setup

```bash
uv sync                              # install base deps + dev tools
uv sync --extra torch                # add PyTorch
uv sync --extra tf                   # add TensorFlow
uv sync --extra torch --extra tf     # add both
```

## Run tests

```bash
uv run pytest
```

## Run the examples

```bash
uv run python examples/example_generator.py   # core numpy engine
uv run python examples/example_tf.py         # TensorFlow integration
uv run python examples/example_torch.py       # PyTorch integration with FISTA
```

## Vogel subarray

`vogel_subarray(n_mics)` selects the *n_mics* innermost channels of the TUB Vogel 64 array by ascending radial distance. Positions are normalised by the full-array aperture.

![Vogel subarray – 16 and 64 mics](docs/vogel_subarray.png)

## Package layout

```
src/fa2026/
    __init__.py
    generators/
        __init__.py     # CMFDataGenerator, CMFTorchDataset, make_tf_dataset
        generator.py    # core numpy engine (CMFDataGenerator)
        generator_torch.py  # PyTorch IterableDataset wrapper (CMFTorchDataset)
        generator_tf.py     # TensorFlow tf.data.Dataset factory (make_tf_dataset)
    physical.py         # sensing matrix & Vogel array utilities
    optim/
        __init__.py     # ISTA, FISTA, and callback utilities
        ista.py         # ISTA & FISTA sparse recovery algorithms
        callbacks.py     # iteration callbacks for optimization
tests/
    test_generator.py
examples/
    example_generator.py
    example_tf.py
    example_torch.py
```
