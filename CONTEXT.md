# CONTEXT

## Glossary

**source power vector** (`x`)
A real-valued, non-negative vector of length `N` representing the acoustic power contributed
by each candidate source location on the focus grid. Inactive sources have zero entries.
Drawn from a Bernoulli × Rayleigh distribution; never normalised — that is the responsibility
of the downstream input pipeline.

**sensing matrix** (`A`)
A real-valued matrix of shape `(M_eff, N)` mapping the source power vector `x` to the
vectorized CSM `y`. Constructed from the Kronecker product of acoustic transfer functions
(steering vectors) following the PhysicalModel / BeamformerCMF convention.
Shape depends on `r_diag`:
- `r_diag=False` : `M_eff = M²`
- `r_diag=True`  : `M_eff = M(M−1)`

The caller is responsible for building `A` (from acoular, random, or any other source) and
injecting it into the generator. The generator is agnostic about how `A` was constructed.

**cross-spectral matrix** (CSM)
The `M×M` Hermitian positive semi-definite covariance matrix of microphone signals at a
single frequency. Under the infinite-snapshot signal model:
`CSM = H diag(x) H^H + σ² I`
where `H` is the steering matrix and `σ²` is the sensor noise power.

**vectorized CSM** (`y`)
The measurement vector obtained by serialising the CSM into a real-valued vector following
the PhysicalModel / BeamformerCMF indexing convention (real parts of the lower triangle of
`CSM.T` first, then off-diagonal imaginary parts). Length `M_eff`.

**diagonal removal** (`r_diag`)
Boolean flag. When `True`, the main diagonal of the CSM is excluded from `y` and the
corresponding rows from `A`. This eliminates incoherent sensor self-noise from the forward
model. Matches `BeamformerCMF.r_diag` in acoular.

**sensor noise** (`σ²`)
Incoherent noise power per microphone appearing only on the diagonal of the CSM.
Only modelled when `r_diag=False`. Parameterised via
`snr_db = 10 log₁₀(sum(x) / σ²)` where `sum(x)` is the total source power (per sample).

**sparsity** (`nsources`)
Integer controlling the expected number of active sources. Internally converted to a
Bernoulli activation probability `pnz = nsources / N`.

**Rayleigh amplitude**
Source amplitudes are drawn from a Rayleigh(σ=1) distribution via the inverse-CDF method,
ensuring non-negative values consistent with the source-power interpretation.

**he** (Helmholtz number)
Dimensionless frequency `he = f · aperture / c`, where `aperture` is the physical aperture
of the microphone array in metres. Used to index frequency when constructing the physical
sensing matrix via acoular's `SteeringVector`.
_Avoid_: treating `he` as a normalised or unitless index detached from a physical aperture.

**n_grid**
Number of source grid points per axis on the rectangular focus grid. The grid always spans
[−0.5·ap, 0.5·ap] × [−0.5·ap, 0.5·ap] where `ap` is the physical aperture of the
microphone array. Total grid size is `n_grid²`; increment is `ap / (n_grid − 1)`.
_Avoid_: specifying the grid extent or increment directly.
