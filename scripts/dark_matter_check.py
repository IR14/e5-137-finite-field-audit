"""Conservative dark-matter viability audit for the fourth-neutrino benchmark."""

from __future__ import annotations

from pathlib import Path

from cosmo_gradient.theory import dark_matter_closure_check


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "dark_matter_closure_verification.md"


def render_report() -> str:
    check = dark_matter_closure_check()
    nu = check.neutrino
    return f"""# Dark Matter Closure Verification

## Status

Conditional WIMP-like heavy sterile-neutrino benchmark.

This report does **not** verify the particle as the dominant component of Cold
Dark Matter from mass alone. It verifies only the limited kinematic and
consistency filters that can be evaluated without specifying couplings,
production history, annihilation/decay channels, or active-sterile mixing.

## Input Mass

| quantity | value |
|---|---:|
| m_nu_tau_prime [eV] | {nu.mass_ev:.6f} |
| m_nu_tau_prime [MeV] | {nu.mass_mev:.9f} |
| m_nu_tau_prime [GeV] | {nu.mass_gev:.12f} |

## Test A: Relic Density

Observed reference:

```text
Omega_c h^2 ~= {check.observed_omega_c_h2:.3f}
```

For a thermal WIMP-like relic, the required annihilation scale is usually of
order

```text
<sigma v> ~= {check.canonical_thermal_cross_section_cm3_s:.2e} cm^3/s
```

The e-5-137 mass formula does not specify an annihilation cross-section,
freeze-out history, freeze-in coupling, gravitational production amplitude, or
entropy dilution factor. Therefore the relic abundance is not calculable from
the mass formula alone.

| check | result |
|---|---:|
| relic density calculable from mass only | {check.relic_density_calculable_from_mass_only} |
| conditional viability if fully sterile/nonthermal | {check.conditionally_viable_if_fully_sterile} |

## Test B: Structure Formation and Free Streaming

The mass is GeV-scale, so a nonrelativistically produced population would behave
as cold dark matter for large-scale-structure purposes. This is a kinematic
classification only. A precise free-streaming scale still depends on the
momentum distribution at production.

| check | result |
|---|---:|
| GeV-scale cold candidate | {check.geV_scale_cold_candidate} |
| active-neutrino Z-width constraint avoided automatically | {not check.active_neutrino_excluded_by_z_width} |

Because the mass is below 45 GeV, it cannot be an ordinary active fourth
neutrino with Standard-Model-strength Z coupling. It must be sterile or
otherwise decoupled.

## Test C: Decay and Gamma-Ray Bounds

For a heavy sterile neutrino, radiative and cascade decay constraints depend on
the mixing angle and decay operators. The statement `theta^2 -> 0` is an
additional model assumption, not a consequence of the mass value.

| quantity | value |
|---|---:|
| age of Universe [s] | {check.universe_age_seconds:.6e} |
| target lifetime for age x 1e12 [s] | {check.target_stability_lifetime_seconds:.6e} |
| gamma lifetime calculable from mass only | {check.gamma_lifetime_calculable_from_mass_only} |

If exact sterility is imposed, the particle can be stable by assumption. If any
active mixing or visible decay channel is present, Fermi-LAT and related
X-ray/gamma-ray constraints require a dedicated model-dependent lifetime
calculation.

## Verdict

The benchmark mass

```text
m_nu_tau_prime = {nu.mass_gev:.12f} GeV
```

is compatible with a heavy cold sterile-particle mass scale. It is **not** a
standalone verification of `Omega_c h^2 = 0.120`, nor does it prove gamma-ray
stability. The defensible final status is:

```text
Conditionally viable only as a fully sterile/nonthermal GeV-scale dark-matter
candidate; not verified as a complete cosmological model.
```

## References

- Planck 2018 cosmological parameters: `Omega_c h^2 ~= 0.120`.
- Fermi-LAT gamma-ray line and diffuse-photon searches constrain visible
  annihilation or decay channels.
- Sterile-neutrino dark-matter reviews emphasize that mass alone is not enough;
  production and mixing determine viability.
"""


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(), encoding="utf-8")
    check = dark_matter_closure_check()
    print("Dark matter closure verification")
    print(f"m_nu_tau_prime = {check.neutrino.mass_gev:.12f} GeV")
    print(f"geV_scale_cold_candidate = {check.geV_scale_cold_candidate}")
    print(f"relic_density_calculable_from_mass_only = {check.relic_density_calculable_from_mass_only}")
    print(f"gamma_lifetime_calculable_from_mass_only = {check.gamma_lifetime_calculable_from_mass_only}")
    print(f"conditionally_viable_if_fully_sterile = {check.conditionally_viable_if_fully_sterile}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
