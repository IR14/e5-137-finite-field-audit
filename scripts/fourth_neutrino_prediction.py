"""Generate the Phase 5 fourth-neutrino closure report."""

from __future__ import annotations

from pathlib import Path

from cosmo_gradient.theory import (
    LEP_Z_WIDTH_REFERENCE_GEV,
    fourth_neutrino_prediction,
    tau_prime_prediction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "phase5_fourth_neutrino_final.md"


def render_report() -> str:
    neutrino = fourth_neutrino_prediction()
    tau_prime = tau_prime_prediction()
    mass_ratio_to_tau_prime = neutrino.mass_gev / tau_prime.mass_gev
    return f"""# Phase 5 Fourth-Neutrino Final Closure

## Formula

The proposed sterile fourth-generation neutrino benchmark is

```text
m_nu_tau_prime = m_e * (F26 / alpha^-1)
                 * [1 + (DN / pi) * delta_phi * (1 - s)]

delta_phi = cos(1/N) - cos(2/N)
s = alpha / (2*pi)
N = {neutrino.n}
D = {neutrino.d}
F26 = {neutrino.f26}
```

## Numerical Result

| quantity | value |
|---|---:|
| s | {neutrino.s:.15f} |
| delta_phi | {neutrino.delta_phi:.15f} |
| bracket | {neutrino.bracket:.15f} |
| m_nu_tau_prime [eV] | {neutrino.mass_ev:.6f} |
| m_nu_tau_prime [keV] | {neutrino.mass_kev:.6f} |
| m_nu_tau_prime [MeV] | {neutrino.mass_mev:.9f} |
| m_nu_tau_prime [GeV] | {neutrino.mass_gev:.12f} |
| m_nu_tau_prime / m_tau_prime | {mass_ratio_to_tau_prime:.12f} |

## Dark-Matter Plausibility Check

The benchmark lies in the broad GeV-scale sterile-particle mass range:

```text
m_nu_tau_prime = {neutrino.mass_gev:.12f} GeV
```

This is compatible with a cold-matter mass scale in the purely kinematic sense.
It does not, by itself, establish a viable dark-matter model. Relic abundance,
free-streaming, direct/indirect detection, active-sterile mixing, and lifetime
constraints depend on couplings and production history, which are not specified
by this mass formula.

| check | threshold/range | passes |
|---|---:|---:|
| broad sterile keV-to-TeV benchmark window | 1 keV to 1 TeV | {neutrino.in_sterile_dm_mass_window} |
| Z-width-scale active-neutrino threshold | {LEP_Z_WIDTH_REFERENCE_GEV:.1f} GeV | {neutrino.above_z_width_reference} |

The mass is below the Z-width-scale 45 GeV reference. Therefore it cannot be
treated as an ordinary active fourth neutrino with Standard-Model-strength
coupling to the Z boson. If retained, the state must be sterile or otherwise
decoupled from the usual Z-width constraint.

Reference links used for the guardrail:

- PDG Live, stable neutral heavy lepton mass limits:
  https://pdglive.lbl.gov/view/S077MNS
- PDG Live, heavy neutral lepton mass limits:
  https://pdglive.lbl.gov/view/S077MN
- keV sterile-neutrino dark-matter review:
  https://arxiv.org/abs/1602.04816

Status: HARD SHUTDOWN for this extension unless a new pre-registered
experimental or cosmological test is defined.
"""


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = render_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    prediction = fourth_neutrino_prediction()
    print("Phase 5 fourth-neutrino closure")
    print(f"m_nu_tau_prime = {prediction.mass_ev:.6f} eV")
    print(f"m_nu_tau_prime = {prediction.mass_gev:.12f} GeV")
    print(f"in_broad_sterile_dm_mass_window = {prediction.in_sterile_dm_mass_window}")
    print(f"above_45GeV_Z_width_reference = {prediction.above_z_width_reference}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
