"""Gamma-resonance audit for the GeV sterile-neutrino benchmark."""

from __future__ import annotations

from pathlib import Path

from cosmo_gradient.theory import gce_resonance_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "fermi_lat_gce_resonance.md"


def render_report() -> str:
    audit = gce_resonance_audit()
    return f"""# Fermi-LAT GCE Resonance Audit

## Status

Gamma appendix for the fourth sterile-neutrino benchmark.

This report does not claim a Fermi-LAT detection or a catalog-level p-value.
It checks whether the mass and phase-modulated annihilation scale are compatible
with broad Galactic Center GeV Excess (GCE) expectations.

## Input Benchmark

| quantity | value |
|---|---:|
| m_nu_tau_prime [GeV] | {audit.neutrino.mass_gev:.12f} |
| q | {audit.q:.15f} |
| delta_phi | {audit.delta_phi:.15f} |
| q * delta_phi | {audit.phase_modulation:.15e} |

## Test 1: Annihilation Cross-Section

The reference GCE/WIMP-like scale is taken as

```text
<sigma v>_ref = {audit.reference_cross_section_cm3_s:.2e} cm^3/s
```

Two simple phase-modulated estimates were evaluated:

| model | formula | value [cm^3/s] | fraction of reference |
|---|---|---:|---:|
| linear rate modulation | sigma_ref * (q delta_phi) | {audit.sigma_v_linear_cm3_s:.6e} | {audit.linear_fraction_of_reference:.6e} |
| quadratic amplitude modulation | sigma_ref * (q delta_phi)^2 | {audit.sigma_v_quadratic_cm3_s:.6e} | {audit.quadratic_fraction_of_reference:.6e} |

Both estimates are far below the canonical `~2e-26 cm^3/s` GCE scale. The
phase operator alone therefore does not generate enough annihilation rate to
explain the GCE.

## Test 2: Decoupled vs Induced Coupling

The fully sterile limit gives no visible gamma-ray annihilation or decay signal.
A toy induced-mixing proxy was computed as

```text
theta2_toy = (q * delta_phi / D)^2 = {audit.induced_theta2_toy:.6e}
```

This is only a dimensional toy proxy. It is not a physical mixing calculation,
because the model does not specify a mediator, coupling, density-dependent
operator, or Galactic Center baryonic-environment response. Sagittarius A* is
therefore not enough, by itself, to activate a calculable mixing angle.

## Test 3: Spectral/Kinematic Compatibility

| check | result |
|---|---:|
| gamma line energy if annihilation/decay produces a monochromatic line [GeV] | {audit.gamma_line_energy_gev:.12f} |
| line lies in 1--4 GeV photon band | {audit.in_gce_photon_energy_band} |
| DM mass lies in common GCE annihilation mass window, roughly 7--200 GeV | {audit.in_common_gce_dm_mass_window} |
| Fermi-LAT catalog/likelihood p-value available | {audit.p_value_available} |

The line energy is in the broad GeV photon band, but the particle mass is below
the common GCE dark-matter mass windows reported for continuum annihilation
fits. In particular, the usual `b bbar` interpretation is inaccessible at this
mass, and even tau/leptonic interpretations generally use heavier dark matter
than 1.56 GeV.

## Verdict

The benchmark does **not** pass the GCE resonance audit as a standalone
explanation:

```text
mass: GeV-scale, but below common continuum-fit DM mass windows
cross-section: phase-modulated estimates are too small
p-value: not computed because no Fermi-LAT likelihood data/model channel was supplied
```

The defensible status is:

```text
No Fermi-LAT/GCE confirmation. The state remains a conditional sterile
dark-matter benchmark, not an explanation of the Galactic Center GeV Excess.
```

## References

- NASA/Fermi public summaries report GCE-compatible dark matter masses around
  31--40 GeV in early analyses.
- Reviews of the GCE typically quote masses of tens of GeV and annihilation
  cross sections near `1e-26 cm^3/s`, depending on channel.
- Fermi-LAT gamma-ray line and diffuse-photon searches constrain visible
  annihilation or decay channels; a p-value requires the actual likelihood and
  spectral model.
"""


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(), encoding="utf-8")
    audit = gce_resonance_audit()
    print("Fermi-LAT GCE resonance audit")
    print(f"m_nu_tau_prime = {audit.neutrino.mass_gev:.12f} GeV")
    print(f"q_delta_phi = {audit.phase_modulation:.15e}")
    print(f"sigma_v_linear = {audit.sigma_v_linear_cm3_s:.6e} cm^3/s")
    print(f"sigma_v_quadratic = {audit.sigma_v_quadratic_cm3_s:.6e} cm^3/s")
    print(f"in_common_gce_dm_mass_window = {audit.in_common_gce_dm_mass_window}")
    print(f"p_value_available = {audit.p_value_available}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
