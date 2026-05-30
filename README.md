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

## Run the example

```bash
uv run python examples/example_generator.py
```

## Vogel subarray

`vogel_subarray(n_mics)` selects the *n_mics* innermost channels of the TUB Vogel 64 array by ascending radial distance. Positions are normalised by the full-array aperture.

![Vogel subarray – 16 and 64 mics](docs/vogel_subarray.png)

## Package layout

```
src/fa2026/
    generator.py        # core numpy engine (CMFDataGenerator)
    generator_torch.py  # PyTorch IterableDataset wrapper
    generator_tf.py     # TensorFlow tf.data.Dataset factory
tests/
    test_generator.py
examples/
    example_generator.py
```
