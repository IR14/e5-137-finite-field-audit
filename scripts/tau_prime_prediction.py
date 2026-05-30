"""Generate the Phase 5 tau-prime prediction report."""

from __future__ import annotations

from pathlib import Path

from cosmo_gradient.theory import (
    LEP_Z_WIDTH_REFERENCE_GEV,
    PDG_SEQUENTIAL_CHARGED_LEPTON_LIMIT_GEV,
    tau_prime_prediction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "phase5_tau_prime_final.md"


def render_report() -> str:
    prediction = tau_prime_prediction()
    return f"""# Phase 5 Tau-Prime Final Appendix

## Formula

The proposed fourth-generation benchmark is

```text
m_tau_prime / m_e = alpha^-1 * [D^2 + q * (D * N)]
q = N(N - 2) / (e^4 pi^3)
N = {prediction.n}
D = {prediction.d}
```

The previously empirical digit moment is rewritten as

```text
I5 = 2 * (D - N) = {prediction.i5_topological}
```

and the fourth-generation coefficient is

```text
D * N = {prediction.generation_coefficient}
```

## Numerical Result

| quantity | value |
|---|---:|
| q | {prediction.q:.15f} |
| bracket D^2 + qDN | {prediction.bracket:.12f} |
| m_tau_prime / m_e | {prediction.mass_ratio_to_electron:.12f} |
| m_tau_prime [MeV] | {prediction.mass_mev:.9f} |
| m_tau_prime [GeV] | {prediction.mass_gev:.12f} |

## Threshold Check

| reference threshold | threshold [GeV] | passes |
|---|---:|---:|
| LEP Z-width-scale reference used in this project | {LEP_Z_WIDTH_REFERENCE_GEV:.1f} | {prediction.above_lep_z_width_reference} |
| PDG sequential charged heavy lepton mass limit | {PDG_SEQUENTIAL_CHARGED_LEPTON_LIMIT_GEV:.1f} | {prediction.above_sequential_charged_lepton_limit} |

## Interpretation Guardrail

The value is above the project-level 45 GeV threshold, but it is below the
current PDG sequential charged-heavy-lepton limit of about 100.8 GeV. Therefore
this benchmark should not be described as an allowed ordinary charged fourth
generation lepton. If retained, it must be framed as a sterile, weakly coupled,
or otherwise non-standard benchmark requiring a dedicated collider and
cosmology reinterpretation.

Reference links used for the threshold guardrail:

- PDG Live, sequential charged heavy lepton mass limits:
  https://pdglive.lbl.gov/view/S025MS
- PDG Live, stable neutral heavy lepton mass limits:
  https://pdglive.lbl.gov/view/S077MNS

Status: ARCHIVED/SHUTDOWN for this speculative extension unless a new
pre-registered experimental test is defined.
"""


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = render_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    prediction = tau_prime_prediction()
    print("Phase 5 tau-prime final")
    print(f"m_tau_prime = {prediction.mass_gev:.12f} GeV")
    print(f"passes_45GeV = {prediction.above_lep_z_width_reference}")
    print(
        "passes_PDG_sequential_charged_lepton_100p8GeV = "
        f"{prediction.above_sequential_charged_lepton_limit}"
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
