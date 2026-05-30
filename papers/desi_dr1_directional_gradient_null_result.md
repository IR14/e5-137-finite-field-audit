# A DESI DR1 Search for Directional Gradients in Galaxy Population and High-Redshift Quasar Observables: A Null Result

Draft date: 2026-05-28

Author: I. Mikhailov, with computational assistance from Codex

## Abstract

We present a first-pass, reproducible search for a large-scale directional gradient in DESI Data Release 1 (DR1) galaxy and quasar observables. The motivating hypothesis is deliberately tested only as an empirical anisotropy claim: at fixed redshift, tracer properties or tracer densities might show a weak dependence on sky direction after accounting for survey geometry and selection effects. We use DESI DR1 Large-Scale Structure (LSS) clustering catalogs and random catalogs for density tests, and the DESI DR1 FastSpecFit Iron value-added catalog for galaxy population and quasar emission-line observables. The tested families include an LRG `DN4000_MODEL` residual test at `0.4 <= z < 0.6`, high-redshift QSO density tests over `1.5 <= z < 3.5`, raw QSO emission-line equivalent-width residuals, and a preregistered QSO line-ratio family using `CIV_1549_EW`, `CIII_1908_EW`, and `MGII_2796_EW`. All tests use HEALPix sky maps, DESI random catalogs where appropriate, residualization against redshift and nuisance variables, external imaging/systematics templates, and spatial block-null tests. Across the primary controlled observational family, the minimum block-null p-value is `p = 0.409`; in the preregistered QSO line-ratio family the minimum primary p-value is `p = 0.880`. We find no robust, repeated, or statistically significant directional axis in the tested DESI DR1 observables. We therefore report an active null result and define a stop condition for further DESI DR1 scanning without new data, official property mocks, or a preregistered new observable family.

## 1. Introduction

The standard cosmological model assumes that the Universe is statistically homogeneous and isotropic on sufficiently large scales. Within that framework, observed angular structure in galaxy and quasar catalogs is expected to arise from cosmological clustering, survey geometry, selection functions, local systematics, and statistical noise. A meaningful search for a preferred direction must therefore be framed as a test against an explicitly corrected null model rather than as a fit to raw sky counts.

This work asks a narrow observational question: do public DESI DR1 catalogs contain evidence for a weak dipolar directional gradient in either tracer density or tracer properties, after the dominant survey and selection effects are accounted for?

The study was motivated by a broader speculative idea in which a preferred direction might appear not as a visible center but as a weak statistical anisotropy in the properties of distant tracers. That theoretical motivation is not used as an assumption in the analysis below. The present paper is restricted to a falsifiable observational test:

**H0:** after correcting for survey geometry, selection effects, random catalogs, and available systematics templates, the tested DESI DR1 tracer distributions and residual observables are compatible with statistical isotropy.

**H1:** one or more independent tracer/property families show a stable directional dipole that cannot be explained by survey geometry, systematics, or statistical noise, and that is repeated across redshift, region, or tracer splits.

The result is a null detection. The pipeline does not find a stable directional gradient in the DESI DR1 tests considered here.

## 2. Data

### 2.1 DESI DR1 LSS catalogs

We use public DESI DR1 Large-Scale Structure (LSS) catalogs. DESI DR1 includes data from the first 13 months of the main survey as well as uniformly reprocessed Survey Validation data; the DESI DR1 documentation describes the public data products, including LSS catalogs and value-added catalogs. The LSS catalogs used here are the DR1 Iron `LSScats/v1.5` clustering catalogs, with corresponding random catalogs.

The density analyses use QSO clustering catalogs in NGC and SGC regions, together with the matching random catalogs. The random catalogs are essential: raw counts alone would mostly measure survey footprint and targeting geometry, not a cosmological anisotropy.

### 2.2 FastSpecFit value-added catalog

For population and emission-line observables, we use the DESI DR1 FastSpecFit Iron VAC. FastSpecFit models DESI spectra and broadband photometry to provide stellar-continuum, emission-line, and derived spectral quantities. We use it here for:

- `DN4000_MODEL` for LRG population residuals;
- QSO emission-line equivalent widths, especially `CIV_1549_EW`, `CIII_1908_EW`, and `MGII_2796_EW`;
- inverse-variance columns for IVAR-weighted residual maps;
- fit-diagnostic and photometric proxy columns used as nuisance controls.

The QSO `DN4000_MODEL` observable was tested only as a smoke check and then excluded from the primary family: strict quality cuts leave only 90 high-redshift QSO rows, making it unsuitable for a scientific QSO population-gradient test.

### 2.3 External systematics templates

For QSO density and QSO line-property tests, we include external HEALPix templates at `nside=16`:

- Galactic extinction proxy `EBV`;
- stellar density;
- imaging depth in `g/r/z`;
- PSF size in `g/r/z`;
- sky-brightness proxies in `g/r/z`.

These are used as regression controls rather than interpreted as physical variables.

## 3. Methods

### 3.1 Sky maps and dipole estimator

Objects are pixelized into HEALPix maps. Unless otherwise stated, we use `nside=16`. For density tests, the corrected overdensity map is based on data counts and random counts. For property tests, we first construct object-level residuals and then average them into sky pixels.

For a map value `y_p` in pixel `p` and a unit vector `n_p` pointing to the pixel center, the dipole model is

```text
y_p = monopole + d · n_p + noise_p .
```

The fitted dipole amplitude is `|d|`, and the dipole axis is converted to RA/DEC. The axis should not be interpreted alone; it is meaningful only together with the null distribution and cross-checks across tracers, regions, and redshift bins.

### 3.2 Population and line residuals

For FastSpecFit observables, object-level residuals are estimated by weighted least squares. The generic residual model includes redshift controls:

```text
observable ~ 1 + z + z^2 + nuisance controls .
```

For LRG `DN4000_MODEL`, the controls are `z`, `z^2`, and `LOGMSTAR`.

For QSO line equivalent widths and line ratios, the controls include:

- `z` and `z^2`;
- `LOGMSTAR` when present;
- line-fit diagnostics `RCHI2_LINE` and `DELTA_LINECHI2`;
- photometric/luminosity proxies: `FLUX_SYNTH_G`, `FLUX_SYNTH_R`, `FLUX_SYNTH_Z`, `FLUX_G`, `FLUX_R`, `FLUX_Z`, `FLUX_W1`, `FLUX_W2`;
- external systematics templates listed above.

The residual map uses IVAR-weighted pixel means:

```text
residual_pixel = sum_i residual_i * IVAR_i / sum_i IVAR_i .
```

For line ratios of the form `ln(x/y)`, the propagated inverse variance is

```text
IVAR_ratio = 1 / (1 / (IVAR_x x^2) + 1 / (IVAR_y y^2)).
```

Only rows with finite observables and positive IVAR are retained.

### 3.3 Quality cuts

The LRG `DN4000_MODEL` production run uses:

- `ZWARN == 0`;
- `RCHI2_CONT < 2`;
- `DELTACHI2 > 25`;
- positive `DN4000_IVAR`;
- winsorization of `DN4000_MODEL` at the 1st and 99th percentiles.

For QSO line observables, we do not hard-cut on `DELTA_LINECHI2 > 25`. An audit of the high-redshift QSO sample found that this threshold would leave only 55 usable `CIV_1549_EW` rows and 68 usable `CIII_1908_EW` rows out of approximately 799,000 high-redshift QSO matches. Instead, `RCHI2_LINE` and `DELTA_LINECHI2` are included as nuisance controls, while hard cuts retain finite observables, positive IVAR, and `ZWARN == 0`.

### 3.4 Null tests

The primary null for property maps is a spatial block permutation:

1. divide the sky into coarse HEALPix blocks with `block_nside=2`;
2. permute residual patterns by block;
3. refit the dipole for each permuted map;
4. compute the p-value as the fraction of null amplitudes at least as large as the observed amplitude, using a `+1` finite-sample correction.

For the density tests, we use pixel permutation and block-null diagnostics, with the reported interpretation based on the corrected maps after random-catalog and external-template regression.

### 3.5 Look-elsewhere accounting

We report the minimum p-value over the controlled observational family and apply simple Bonferroni and Sidak corrections. These corrections are conservative sanity checks because the tests are correlated. Since all p-values are high, the exact multiple-testing correction does not affect the conclusion.

## 4. Results

### 4.1 LRG population residuals

The LRG production run tests `DN4000_MODEL` in the NGC region over `0.4 <= z < 0.6`.

| tracer | region | z range | observable | rows | amplitude | RA | DEC | block-null p |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|
| LRG | NGC | 0.4-0.6 | `DN4000_MODEL` | 342,791 | 0.075220 | 247.056 | 38.150 | 0.409182 |

This is compatible with the isotropic null. It is the smallest p-value in the controlled observational family, but it is not small in any scientific sense.

### 4.2 QSO high-redshift density

The QSO density test uses `1.5 <= z < 2.1` and `2.1 <= z < 3.5`, NGC+SGC combined, with DESI random catalogs and external-template regression.

| z range | n_data | corrected amplitude | RA | DEC | corrected block p |
|:--|--:|--:|--:|--:|--:|
| 1.5-2.1 | 432,458 | 0.003703 | 87.961 | -78.002 | 0.840432 |
| 2.1-3.5 | 366,560 | 0.003987 | 329.996 | -4.010 | 0.780044 |

Region-split tests likewise return high p-values:

| region | z range | n_data | amplitude | RA | DEC | block p |
|:--|:--|--:|--:|--:|--:|--:|
| NGC | 1.5-2.1 | 281,438 | 0.009911 | 120.968 | 80.962 | 0.653069 |
| NGC | 2.1-3.5 | 237,306 | 0.006319 | 339.286 | 44.405 | 0.876625 |
| SGC | 1.5-2.1 | 151,020 | 0.018968 | 187.966 | -54.340 | 0.765847 |
| SGC | 2.1-3.5 | 129,254 | 0.017125 | 263.096 | -0.890 | 0.787043 |

The cap-split axes are not coherent, and no density bin provides evidence for a preferred direction.

### 4.3 QSO emission-line equivalent-width residuals

The combined high-redshift QSO line residual tests use `CIV_1549_EW` and `CIII_1908_EW`.

| observable | rows | amplitude | RA | DEC | block p |
|:--|--:|--:|--:|--:|--:|
| `CIV_1549_EW` | 761,395 | 0.303061 | 85.281 | 7.464 | 0.958084 |
| `CIII_1908_EW` | 758,337 | 0.148250 | 132.129 | -42.987 | 0.904192 |

The two axes are separated by 65.953 degrees. Region and redshift-bin splits also remain null-compatible:

| region | z range | observable | rows | amplitude | block p |
|:--|:--|:--|--:|--:|--:|
| NGC | 1.5-2.1 | `CIV_1549_EW` | 266,722 | 0.831954 | 0.816367 |
| NGC | 1.5-2.1 | `CIII_1908_EW` | 264,221 | 0.129588 | 0.972056 |
| NGC | 2.1-3.5 | `CIV_1549_EW` | 228,136 | 0.452652 | 0.954092 |
| NGC | 2.1-3.5 | `CIII_1908_EW` | 228,933 | 0.207706 | 0.942116 |
| SGC | 1.5-2.1 | `CIV_1549_EW` | 142,895 | 0.625780 | 0.988024 |
| SGC | 1.5-2.1 | `CIII_1908_EW` | 141,485 | 0.087681 | 0.996008 |
| SGC | 2.1-3.5 | `CIV_1549_EW` | 123,642 | 0.541443 | 0.990020 |
| SGC | 2.1-3.5 | `CIII_1908_EW` | 123,698 | 0.267788 | 0.966068 |

### 4.4 Preregistered QSO line-ratio family

After the raw line-EW tests returned null results, we preregistered a line-ratio family before execution. The primary observables are:

- `LOG_CIV_CIII_EW = ln(CIV_1549_EW / CIII_1908_EW)`;
- `LOG_CIV_MGII2796_EW = ln(CIV_1549_EW / MGII_2796_EW)`;
- `LOG_CIII_MGII2796_EW = ln(CIII_1908_EW / MGII_2796_EW)`.

The primary family contains six tests: three ratios times two redshift bins, NGC+SGC combined.

| z range | observable | rows | amplitude | RA | DEC | block p |
|:--|:--|--:|--:|--:|--:|--:|
| 1.5-2.1 | `LOG_CIV_CIII_EW` | 396,908 | 0.005927 | 90.086 | 13.236 | 0.970060 |
| 1.5-2.1 | `LOG_CIV_MGII2796_EW` | 407,103 | 0.002747 | 149.263 | -31.179 | 0.994012 |
| 1.5-2.1 | `LOG_CIII_MGII2796_EW` | 403,644 | 0.005951 | 251.516 | 25.093 | 0.980040 |
| 2.1-3.5 | `LOG_CIV_CIII_EW` | 345,404 | 0.004468 | 171.736 | 10.948 | 0.980040 |
| 2.1-3.5 | `LOG_CIV_MGII2796_EW` | 186,086 | 0.017718 | 174.848 | -54.049 | 0.880240 |
| 2.1-3.5 | `LOG_CIII_MGII2796_EW` | 186,114 | 0.012246 | 109.881 | 27.207 | 0.980040 |

The minimum primary p-value is `p = 0.880240`. With six primary tests, the Bonferroni-adjusted value is 1.0 and the Sidak global value is 0.999997. The stability suite over NGC and SGC separately has minimum p-value `p = 0.724551`.

Line ratios therefore also return an active null result.

### 4.5 Master look-elsewhere summary

The controlled observational family, excluding the QSO `DN4000_MODEL` smoke diagnostic, contains 17 tests.

| family | tests | minimum block-null p-value | status |
|:--|--:|--:|:--|
| LRG population residual | 1 | 0.409182 | null-compatible |
| QSO density corrected combined | 2 | 0.780044 | null-compatible |
| QSO density region split | 4 | 0.653069 | null-compatible |
| QSO line residual combined | 2 | 0.904192 | null-compatible |
| QSO line residual region/bin | 8 | 0.816367 | null-compatible |

The minimum p-value is `0.4091816367`. The Bonferroni correction gives 1.0 and the Sidak global value is 0.999870.

## 5. Discussion

The tests in this paper deliberately probe several different ways a directional gradient might appear:

- as a density dipole in high-redshift QSOs;
- as a stellar-population residual in mature LRGs;
- as a raw QSO emission-line residual;
- as a QSO line-ratio residual designed to suppress broad calibration and continuum effects;
- as a region-specific or redshift-specific axis.

None produces a low p-value. No axis repeats across independent observable families in a statistically meaningful way. The result is therefore not merely a failure of one observable; it is a coherent absence of evidence across the tested DR1 families.

The most important feature of the analysis is that it avoids raw-count anisotropy claims. Density tests are corrected using random catalogs and external templates. Property tests are residualized against redshift, fit-quality diagnostics, luminosity proxies, and imaging templates before sky projection. The p-values are obtained from spatial block permutations intended to retain coarse sky correlations more realistically than independent pixel shuffles.

This does not prove exact cosmic isotropy. It shows that the particular directional-gradient signatures tested here are not detected in the current public DESI DR1 data products under the implemented corrections.

## 6. Limitations

Several limitations should be emphasized.

First, property-residual tests do not yet use official mocks calibrated for the exact FastSpecFit observable residuals. The density tests are better anchored by DESI random catalogs, but property residuals depend on the adequacy of the residual model and systematics templates.

Second, the LRG production population test currently covers a single redshift interval and NGC region. Additional LRG/ELG population tests could be valuable, but further DR1 scanning without preregistration would risk data dredging.

Third, QSO `DN4000_MODEL` is not useful at high redshift under the strict quality cuts used here. It was excluded from the primary family after a smoke run left only 90 rows.

Fourth, all multiple-testing calculations are approximate because related sky splits and observables are correlated. Since all p-values are high, the conclusion is insensitive to the exact correction.

Finally, this is a first-pass directional-gradient search, not a full cosmological parameter analysis. It does not fit a physical anisotropic cosmology, nor does it replace standard DESI BAO/RSD analyses.

## 7. Stop condition

After the preregistered QSO line-ratio family also returned an active null result, a stop condition was fixed for DESI DR1 scanning. Further searches on the same DR1 data are stopped unless one of the following conditions is met:

- a new DESI data release is used;
- official mocks become available for the exact property/residual estimator;
- a documented methodological bug invalidates an existing result;
- a new external systematic template set materially changes the correction model;
- a new physical observable family is preregistered before execution.

This stop condition is meant to prevent uncontrolled post-hoc observable searches after repeated null results.

## 8. Conclusions

We searched for directional gradients in DESI DR1 LSS and FastSpecFit observables using density maps, galaxy population residuals, QSO emission-line residuals, and preregistered QSO line ratios. The tests use random catalogs, external systematics templates, residual modeling, HEALPix dipole fitting, and spatial block-null p-values.

No robust directional gradient is detected.

The current empirical status is therefore:

> DESI DR1 LSS + FastSpecFit directional-gradient search: active null.

Future work should focus on DESI DR2/DR3 replication, official mocks for property residuals, and preregistered observable families rather than additional unregistered DR1 scanning.

## Data and Code Availability

The analysis was run locally in the `cosmo_genesis_gradient` Python project. Key generated artifacts include:

- `outputs/reports/observational_gradient_master_null_report.md`
- `outputs/tables/observational_gradient_master_null_tests.csv`
- `outputs/reports/qso_line_ratio_preregistration.md`
- `outputs/reports/qso_line_ratio_preregistered_results.md`
- `outputs/tables/qso_line_ratio_preregistered_results.csv`
- `outputs/reports/observational_stop_condition.md`

The code is structured as a reproducible Python project with command-line entry points under `cosmo-gradient`.

## References and Source Links

- DESI Data Release 1 documentation: [https://data.desi.lbl.gov/doc/releases/dr1/](https://data.desi.lbl.gov/doc/releases/dr1/)
- DESI DR1 LSS catalog location: [https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/](https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/)
- DESI data organization documentation for LSS catalogs: [https://data.desi.lbl.gov/doc/organization/](https://data.desi.lbl.gov/doc/organization/)
- DESI DR1 paper: [https://arxiv.org/abs/2503.14745](https://arxiv.org/abs/2503.14745)
- FastSpecFit documentation: [https://fastspecfit.readthedocs.io/en/2.1.2/](https://fastspecfit.readthedocs.io/en/2.1.2/)
- DESI DR1 AGN/QSO VAC documentation noting FastSpecFit Iron v2.1: [https://data.desi.lbl.gov/doc/releases/dr1/vac/agnqso/](https://data.desi.lbl.gov/doc/releases/dr1/vac/agnqso/)
