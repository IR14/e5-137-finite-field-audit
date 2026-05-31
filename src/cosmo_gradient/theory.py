"""Small deterministic theory-side calculations used by exploratory notes.

These helpers are intentionally separate from the DESI observational pipeline.
They provide reproducible numerical bookkeeping for speculative formula audits
without mixing those assumptions into the survey-analysis code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

CODATA_ALPHA_INV = 137.035999084
ELECTRON_MASS_MEV = 0.51099895069
ELECTRON_MASS_EV = ELECTRON_MASS_MEV * 1_000_000.0
PROTON_MASS_MEV = 938.27208816
NEUTRON_MASS_MEV = 939.56542052
ATOMIC_MASS_UNIT_MEV = 931.49410242
PI0_MASS_MEV = 134.9768
N_TOPOLOGICAL = 5
D_STRING = 26
F26 = 121_393
LEP_Z_WIDTH_REFERENCE_GEV = 45.0
PDG_SEQUENTIAL_CHARGED_LEPTON_LIMIT_GEV = 100.8
PLANCK_OMEGA_C_H2 = 0.120
CANONICAL_THERMAL_RELIC_CROSS_SECTION_CM3_S = 2.2e-26
GCE_REFERENCE_CROSS_SECTION_CM3_S = 2.0e-26
UNIVERSE_AGE_SECONDS = 13.787e9 * 365.25 * 24 * 3600
STANDARD_LCDM_NEFF = 3.046
PLANCK_ACT_NEFF_MEAN = 2.99
PLANCK_ACT_NEFF_SIGMA = 0.17
LSS_SAFE_CUTOFF_KPC = 100.0
NUCLEAR_CHEMISTRY_PPM_THRESHOLD = 10.0
LANDAUER_ROOM_TEMPERATURE_K = 300.0
BOLTZMANN_J_PER_K = 1.380649e-23
NUCLEAR_ISOTOPE_ATOMIC_MASS_U = {
    "H-1": 1.00782503223,
    "He-4": 4.00260325413,
    "C-12": 12.0,
    "Fe-56": 55.93493633,
}
GLOBAL_AXIS_LOOKUP_NUM_DEN = (
    (-26, -9880, 3),
    (-25, -8825, 3),
    (-24, -2616, 1),
    (-23, -6946, 3),
    (-22, -6116, 3),
    (-21, -1785, 1),
    (-20, -4660, 3),
    (-19, -4028, 3),
    (-18, -1152, 1),
    (-17, -2941, 3),
    (-16, -2480, 3),
    (-15, -690, 1),
    (-14, -1708, 3),
    (-13, -1391, 3),
    (-12, -372, 1),
    (-11, -880, 3),
    (-10, -680, 3),
    (-9, -171, 1),
    (-8, -376, 3),
    (-7, -266, 3),
    (-6, -60, 1),
    (-5, -115, 3),
    (-4, -68, 3),
    (-3, -12, 1),
    (-2, -16, 3),
    (-1, -5, 3),
    (0, 0, 1),
    (1, 2, 3),
    (2, 4, 3),
    (3, 3, 1),
    (4, 20, 3),
    (5, 40, 3),
    (6, 24, 1),
    (7, 119, 3),
    (8, 184, 3),
    (9, 90, 1),
    (10, 380, 3),
    (11, 517, 3),
    (12, 228, 1),
    (13, 884, 3),
    (14, 1120, 3),
    (15, 465, 1),
    (16, 1712, 3),
    (17, 2074, 3),
    (18, 828, 1),
    (19, 2945, 3),
    (20, 3460, 3),
    (21, 1344, 1),
    (22, 4664, 3),
    (23, 5359, 3),
    (24, 2040, 1),
    (25, 6950, 3),
    (26, 7852, 3),
)
REQUESTED_GLOBAL_AXIS_NODES = (
    (-6, -120, 1),
    (-3, -12, 1),
    (1, 2, 3),
    (6, 24, 1),
    (7, 42, 1),
    (13, 286, 1),
)


@dataclass(frozen=True)
class TauPrimePrediction:
    n: int
    d: int
    alpha_inv: float
    electron_mass_mev: float
    q: float
    i5_topological: int
    generation_coefficient: int
    bracket: float
    mass_ratio_to_electron: float
    mass_mev: float
    mass_gev: float
    above_lep_z_width_reference: bool
    above_sequential_charged_lepton_limit: bool


@dataclass(frozen=True)
class FourthNeutrinoPrediction:
    n: int
    d: int
    f26: int
    alpha_inv: float
    electron_mass_ev: float
    s: float
    delta_phi: float
    bracket: float
    mass_ev: float
    mass_kev: float
    mass_mev: float
    mass_gev: float
    in_sterile_dm_mass_window: bool
    above_z_width_reference: bool


@dataclass(frozen=True)
class DarkMatterClosureCheck:
    neutrino: FourthNeutrinoPrediction
    observed_omega_c_h2: float
    canonical_thermal_cross_section_cm3_s: float
    universe_age_seconds: float
    target_stability_lifetime_seconds: float
    geV_scale_cold_candidate: bool
    relic_density_calculable_from_mass_only: bool
    gamma_lifetime_calculable_from_mass_only: bool
    active_neutrino_excluded_by_z_width: bool
    conditionally_viable_if_fully_sterile: bool


@dataclass(frozen=True)
class GceResonanceAudit:
    neutrino: FourthNeutrinoPrediction
    q: float
    delta_phi: float
    phase_modulation: float
    reference_cross_section_cm3_s: float
    sigma_v_linear_cm3_s: float
    sigma_v_quadratic_cm3_s: float
    linear_fraction_of_reference: float
    quadratic_fraction_of_reference: float
    induced_theta2_toy: float
    gamma_line_energy_gev: float
    in_gce_photon_energy_band: bool
    in_common_gce_dm_mass_window: bool
    p_value: float | None
    p_value_available: bool


@dataclass(frozen=True)
class CmbNeffValidation:
    neutrino: FourthNeutrinoPrediction
    q: float
    delta_phi: float
    delta_neff: float
    neff_total: float
    planck_act_mean: float
    planck_act_sigma: float
    planck_act_z_score: float
    within_one_sigma: bool
    within_two_sigma: bool
    mass_keV: float
    thermal_equivalent_cutoff_mpc: float
    thermal_equivalent_cutoff_kpc: float
    cutoff_limit_kpc: float
    below_lss_cutoff_limit: bool
    cutoff_wavenumber_inv_mpc: float
    cold_nonthermal_assumption_required: bool
    lss_bao_indistinguishable_from_cdm: bool


@dataclass(frozen=True)
class Phase9HadronUpgradeAudit:
    q: float
    invariant_117: int
    pdg_pi0_mass_mev: float
    baseline_pi0_mass_mev: float
    baseline_pi0_ppm: float
    literal_117_over_137_mass_mev: float
    literal_117_over_137_ppm: float
    modulated_q_mass_mev: float
    modulated_q_ppm: float
    literal_replacement_improves: bool
    modulated_q_improves: bool


@dataclass(frozen=True)
class Phase9ProtonUpgradeAudit:
    experimental_ratio: float
    baseline_ratio: float
    baseline_ppm: float
    upgraded_ratio: float
    upgraded_ppm: float
    correction: float
    phi_gap: int
    complexity_score: int
    passes_threshold: bool


@dataclass(frozen=True)
class NuclearMassAudit:
    isotope: str
    symbol: str
    z: int
    a: int
    atomic_mass_u: float
    reference_nuclear_mass_mev: float
    unbound_nucleon_mass_mev: float
    reference_mass_defect_mev: float
    running_step_ev: float
    predicted_mass_defect_mev: float
    predicted_nuclear_mass_mev: float
    mass_residual_mev: float
    mass_residual_ppm: float
    defect_fraction_of_reference: float
    within_10ppm: bool


@dataclass(frozen=True)
class NuclearChemistryValidation:
    d: int
    invariant_117: int
    mirror_step: int
    base_binding_step_ev: float
    nuclear_running_scale_factor: float
    nuclear_running_step_ev: float
    resonance_factor: float
    ppm_threshold: float
    fe_z_equals_d_string: bool
    rows: tuple[NuclearMassAudit, ...]
    max_abs_mass_residual_ppm: float
    all_within_10ppm: bool
    fe_mass_residual_ppm: float
    fe_reference_mass_defect_mev: float
    fe_predicted_mass_defect_mev: float
    fe_defect_fraction_of_reference: float
    release_gate_passed: bool


@dataclass(frozen=True)
class MinimumComputationalActionAudit:
    raw_symbols: int
    rs_symbols: int
    repetition_symbols: int
    temperature_k: float
    fp32_storage_bits: int
    gf137_raw_storage_bits: int
    rs_storage_bits: int
    repetition_storage_bits: int
    landauer_fp32_storage_j: float
    landauer_gf137_raw_storage_j: float
    landauer_rs_storage_j: float
    gf137_raw_vs_fp32_ratio: float
    rs_vs_fp32_ratio: float
    rs_storage_reduction_fraction: float
    s: float
    one_minus_s: float
    mac137_barrett_mu: int
    inv137_fermat_exponent: int
    square_gemm_complexity: str
    tested_kernel_complexity: str
    physical_minimum_proven: bool


@dataclass(frozen=True)
class CalabiYauRoutingValidation:
    n: int
    compact_dimension: int
    projection_operator: float
    integral_operator_symbolic: str
    integral_operator_value: float
    phase_invariant_symbolic: str
    phase_invariant: complex
    phase_invariant_real: float
    phase_invariant_imag: float
    expected_phase_invariant: complex
    phase_matches_two_thirds_i: bool
    mirror_antistate_n: int
    mirror_integral_operator_value: float
    mirror_inverse_regularizer: float
    zeta_negative_one_regularization: float
    mirror_matches_borcherds_denominator: bool
    field_modulus: int
    residue_axis_count: int
    tick_complexity: str
    routing_proxy_only: bool
    regularization_boundary_only: bool
    superconductivity_model_proven: bool
    physical_current_model_proven: bool

    def D(self, n: int) -> float:
        """Evaluate D(n) = (n^3 - 3 n^2 + 6 n) / 6 for this compact dimension."""

        return self.projection_operator * (n**3 - 3 * n**2 + 6 * n)


@dataclass(frozen=True)
class GlobalAxisNode:
    n: int
    numerator: int
    denominator: int
    value: Fraction


@dataclass(frozen=True)
class GlobalAxisMismatch:
    n: int
    requested_value: Fraction
    formula_value: Fraction


@dataclass(frozen=True)
class GlobalAxisValidation:
    axis_min: int
    axis_max: int
    axis_count: int
    nonzero_axis_count: int
    d_string_boundary: int
    formula_symbolic: str
    lookup: tuple[GlobalAxisNode, ...]
    requested_key_values: tuple[GlobalAxisNode, ...]
    requested_key_mismatches: tuple[GlobalAxisMismatch, ...]
    formula_values_verified: bool
    requested_nodes_all_match: bool

    def D(self, n: int) -> Fraction:
        """Return the exact lookup value for D(n) in the stored axis interval."""

        for node in self.lookup:
            if node.n == n:
                return node.value
        raise KeyError(n)


def vacuum_compression_operator(n: int = N_TOPOLOGICAL) -> float:
    """Return q = N(N-2)/(e^4 pi^3)."""

    return n * (n - 2) / (math.e**4 * math.pi**3)


def schwinger_s(alpha_inv: float = CODATA_ALPHA_INV) -> float:
    """Return s = alpha/(2 pi)."""

    return (1.0 / alpha_inv) / (2.0 * math.pi)


def minimum_computational_action_audit(
    *,
    raw_symbols: int = 2113,
    rs_symbols: int = 3458,
    repetition_symbols: int = 4256,
    temperature_k: float = LANDAUER_ROOM_TEMPERATURE_K,
    alpha_inv: float = CODATA_ALPHA_INV,
) -> MinimumComputationalActionAudit:
    """Return a bounded energy-cost proxy for GF(137) checkpoint storage.

    The audit uses Landauer's lower bound, `k_B T ln(2)` per erased bit, to
    compare FP32 checkpoint storage with byte-valued GF(137) storage and the
    `RS(26,16)` repaired checkpoint used by HYP-007.  This is a storage-bit
    accounting proxy only; it is not a measured hardware power result and it
    does not prove that physical systems minimize energy by using GF(137).
    """

    fp32_bits = raw_symbols * 32
    gf137_bits = raw_symbols * 8
    rs_bits = rs_symbols * 8
    repetition_bits = repetition_symbols * 8
    landauer_per_bit = BOLTZMANN_J_PER_K * temperature_k * math.log(2.0)
    fp32_j = fp32_bits * landauer_per_bit
    gf137_j = gf137_bits * landauer_per_bit
    rs_j = rs_bits * landauer_per_bit
    s_value = schwinger_s(alpha_inv)
    return MinimumComputationalActionAudit(
        raw_symbols=raw_symbols,
        rs_symbols=rs_symbols,
        repetition_symbols=repetition_symbols,
        temperature_k=temperature_k,
        fp32_storage_bits=fp32_bits,
        gf137_raw_storage_bits=gf137_bits,
        rs_storage_bits=rs_bits,
        repetition_storage_bits=repetition_bits,
        landauer_fp32_storage_j=fp32_j,
        landauer_gf137_raw_storage_j=gf137_j,
        landauer_rs_storage_j=rs_j,
        gf137_raw_vs_fp32_ratio=gf137_j / fp32_j,
        rs_vs_fp32_ratio=rs_j / fp32_j,
        rs_storage_reduction_fraction=1.0 - rs_j / fp32_j,
        s=s_value,
        one_minus_s=1.0 - s_value,
        mac137_barrett_mu=(1 << 32) // 137,
        inv137_fermat_exponent=135,
        square_gemm_complexity="O(N^3) for generic square dense GEMM",
        tested_kernel_complexity="O(R K H + R H) for the two-layer audit kernel",
        physical_minimum_proven=False,
    )


def calabi_yau_routing_validation(
    *,
    n: int = N_TOPOLOGICAL,
    compact_dimension: int = 6,
    field_modulus: int = 137,
    residue_axis_count: int = D_STRING,
) -> CalabiYauRoutingValidation:
    """Return the symbolic sixth-level routing invariant audit.

    This helper records two algebraic identities used by the Phase-14 notes:

        D(N) = (N^3 - 3N^2 + 6N) / 6

    and

        (1/6) (sqrt(3) - i) i (i + sqrt(3)) = (2/3) i.

    It also stores the mirror boundary D(-3) = -12, whose inverse is -1/12,
    matching the zeta-regularized value zeta(-1).  This is kept as an
    algebraic boundary condition, not as a derivation of string dynamics.

    The returned flags intentionally keep the electrical-current and
    superconductivity language at the level of a routing proxy.  No condensed
    matter model, measured zero-resistance state, or microscopic teleportation
    mechanism is inferred from these identities alone.
    """

    projection_operator = 1.0 / compact_dimension
    integral_operator_value = projection_operator * (n**3 - 3 * n**2 + 6 * n)
    mirror_antistate_n = -3
    mirror_integral_operator_value = projection_operator * (
        mirror_antistate_n**3 - 3 * mirror_antistate_n**2 + 6 * mirror_antistate_n
    )
    mirror_inverse_regularizer = 1.0 / mirror_integral_operator_value
    zeta_negative_one_regularization = -1.0 / 12.0
    sqrt3 = math.sqrt(3.0)
    phase_invariant = projection_operator * (sqrt3 - 1j) * 1j * (1j + sqrt3)
    expected_phase = (2.0 / 3.0) * 1j
    return CalabiYauRoutingValidation(
        n=n,
        compact_dimension=compact_dimension,
        projection_operator=projection_operator,
        integral_operator_symbolic="D(N) = (N^3 - 3 N^2 + 6 N) / 6",
        integral_operator_value=integral_operator_value,
        phase_invariant_symbolic="(1/6)(sqrt(3)-i)i(i+sqrt(3))",
        phase_invariant=phase_invariant,
        phase_invariant_real=phase_invariant.real,
        phase_invariant_imag=phase_invariant.imag,
        expected_phase_invariant=expected_phase,
        phase_matches_two_thirds_i=abs(phase_invariant - expected_phase) < 1.0e-15,
        mirror_antistate_n=mirror_antistate_n,
        mirror_integral_operator_value=mirror_integral_operator_value,
        mirror_inverse_regularizer=mirror_inverse_regularizer,
        zeta_negative_one_regularization=zeta_negative_one_regularization,
        mirror_matches_borcherds_denominator=(
            mirror_integral_operator_value == -12.0
            and mirror_inverse_regularizer == zeta_negative_one_regularization
        ),
        field_modulus=field_modulus,
        residue_axis_count=residue_axis_count,
        tick_complexity="O(1) per symbolic residue-routing tick",
        routing_proxy_only=True,
        regularization_boundary_only=True,
        superconductivity_model_proven=False,
        physical_current_model_proven=False,
    )


def _global_axis_formula(n: int) -> Fraction:
    return Fraction(n**3 - 3 * n**2 + 6 * n, 6)


def global_axis_validation(
    *,
    axis_min: int = -D_STRING,
    axis_max: int = D_STRING,
) -> GlobalAxisValidation:
    """Return the exact D(N) lookup table over the string-axis interval.

    The table is stored as exact numerator/denominator triples for every
    integer in [-26, 26].  Requested release nodes are audited against the same
    formula rather than overwritten.  This keeps contradictory labels visible:
    D(-6), D(7), and D(13) do not equal the requested values under the stored
    D(N) polynomial.
    """

    lookup = tuple(
        GlobalAxisNode(n=n, numerator=num, denominator=den, value=Fraction(num, den))
        for n, num, den in GLOBAL_AXIS_LOOKUP_NUM_DEN
    )
    requested = tuple(
        GlobalAxisNode(n=n, numerator=num, denominator=den, value=Fraction(num, den))
        for n, num, den in REQUESTED_GLOBAL_AXIS_NODES
    )
    lookup_by_n = {node.n: node.value for node in lookup}
    mismatches = tuple(
        GlobalAxisMismatch(
            n=node.n,
            requested_value=node.value,
            formula_value=lookup_by_n[node.n],
        )
        for node in requested
        if lookup_by_n[node.n] != node.value
    )
    return GlobalAxisValidation(
        axis_min=axis_min,
        axis_max=axis_max,
        axis_count=len(lookup),
        nonzero_axis_count=sum(1 for node in lookup if node.value != 0),
        d_string_boundary=D_STRING,
        formula_symbolic="D(N) = (N^3 - 3 N^2 + 6 N) / 6",
        lookup=lookup,
        requested_key_values=requested,
        requested_key_mismatches=mismatches,
        formula_values_verified=all(node.value == _global_axis_formula(node.n) for node in lookup),
        requested_nodes_all_match=not mismatches,
    )


def delta_phi(n: int = N_TOPOLOGICAL) -> float:
    """Return cos(1/N) - cos(2/N)."""

    return math.cos(1.0 / n) - math.cos(2.0 / n)


def tau_prime_prediction(
    *,
    alpha_inv: float = CODATA_ALPHA_INV,
    electron_mass_mev: float = ELECTRON_MASS_MEV,
    n: int = N_TOPOLOGICAL,
    d: int = D_STRING,
    lep_z_width_reference_gev: float = LEP_Z_WIDTH_REFERENCE_GEV,
    charged_lepton_limit_gev: float = PDG_SEQUENTIAL_CHARGED_LEPTON_LIMIT_GEV,
) -> TauPrimePrediction:
    """Compute the proposed fourth-generation heavy-lepton mass benchmark.

    The expression is

        m_tau_prime / m_e = alpha^-1 * [D^2 + q * (D N)].

    The returned experimental flags are threshold comparisons only; they are not
    collider reinterpretations.
    """

    q = vacuum_compression_operator(n)
    i5_topological = 2 * (d - n)
    generation_coefficient = d * n
    bracket = d**2 + q * generation_coefficient
    mass_ratio = alpha_inv * bracket
    mass_mev = electron_mass_mev * mass_ratio
    mass_gev = mass_mev / 1000.0
    return TauPrimePrediction(
        n=n,
        d=d,
        alpha_inv=alpha_inv,
        electron_mass_mev=electron_mass_mev,
        q=q,
        i5_topological=i5_topological,
        generation_coefficient=generation_coefficient,
        bracket=bracket,
        mass_ratio_to_electron=mass_ratio,
        mass_mev=mass_mev,
        mass_gev=mass_gev,
        above_lep_z_width_reference=mass_gev > lep_z_width_reference_gev,
        above_sequential_charged_lepton_limit=mass_gev > charged_lepton_limit_gev,
    )


def fourth_neutrino_prediction(
    *,
    alpha_inv: float = CODATA_ALPHA_INV,
    electron_mass_ev: float = ELECTRON_MASS_EV,
    n: int = N_TOPOLOGICAL,
    d: int = D_STRING,
    f26: int = F26,
    z_width_reference_gev: float = LEP_Z_WIDTH_REFERENCE_GEV,
) -> FourthNeutrinoPrediction:
    """Compute the proposed sterile fourth-neutrino mass benchmark.

    The expression is

        m_nu_tau_prime = m_e * (F26 / alpha^-1)
                         * [1 + (DN/pi) delta_phi (1 - s)].

    The mass-window flag only checks that the value lies in a broad keV-to-GeV
    sterile-dark-matter benchmark interval. It does not validate relic density,
    lifetime, thermal history, or active-sterile mixing constraints.
    """

    s_value = schwinger_s(alpha_inv)
    delta = delta_phi(n)
    bracket = 1.0 + ((d * n) / math.pi) * delta * (1.0 - s_value)
    mass_ev = electron_mass_ev * (f26 / alpha_inv) * bracket
    mass_gev = mass_ev / 1_000_000_000.0
    return FourthNeutrinoPrediction(
        n=n,
        d=d,
        f26=f26,
        alpha_inv=alpha_inv,
        electron_mass_ev=electron_mass_ev,
        s=s_value,
        delta_phi=delta,
        bracket=bracket,
        mass_ev=mass_ev,
        mass_kev=mass_ev / 1_000.0,
        mass_mev=mass_ev / 1_000_000.0,
        mass_gev=mass_gev,
        in_sterile_dm_mass_window=1_000.0 <= mass_ev <= 1_000_000_000_000.0,
        above_z_width_reference=mass_gev > z_width_reference_gev,
    )


def dark_matter_closure_check(
    *,
    observed_omega_c_h2: float = PLANCK_OMEGA_C_H2,
    canonical_cross_section_cm3_s: float = CANONICAL_THERMAL_RELIC_CROSS_SECTION_CM3_S,
    universe_age_seconds: float = UNIVERSE_AGE_SECONDS,
    stability_margin: float = 1.0e12,
) -> DarkMatterClosureCheck:
    """Return a conservative viability audit for the fourth-neutrino benchmark.

    The check intentionally distinguishes mass-scale compatibility from model
    completion. A GeV sterile particle can be kinematically cold, but relic
    abundance and radiative lifetime require couplings, branching ratios, and a
    production history.
    """

    neutrino = fourth_neutrino_prediction()
    return DarkMatterClosureCheck(
        neutrino=neutrino,
        observed_omega_c_h2=observed_omega_c_h2,
        canonical_thermal_cross_section_cm3_s=canonical_cross_section_cm3_s,
        universe_age_seconds=universe_age_seconds,
        target_stability_lifetime_seconds=universe_age_seconds * stability_margin,
        geV_scale_cold_candidate=neutrino.mass_gev >= 1.0,
        relic_density_calculable_from_mass_only=False,
        gamma_lifetime_calculable_from_mass_only=False,
        active_neutrino_excluded_by_z_width=not neutrino.above_z_width_reference,
        conditionally_viable_if_fully_sterile=True,
    )


def gce_resonance_audit(
    *,
    reference_cross_section_cm3_s: float = GCE_REFERENCE_CROSS_SECTION_CM3_S,
) -> GceResonanceAudit:
    """Audit the fourth-neutrino benchmark against broad GCE expectations.

    Two phase-modulated cross-section estimates are reported:

    - linear rate modulation: sigma_v = sigma_ref * (q delta_phi)
    - quadratic amplitude modulation: sigma_v = sigma_ref * (q delta_phi)^2

    Neither replaces a gamma-ray likelihood fit. The p-value is intentionally
    absent unless external Fermi-LAT spectral/likelihood data are supplied.
    """

    neutrino = fourth_neutrino_prediction()
    q = vacuum_compression_operator()
    delta = delta_phi()
    phase = q * delta
    sigma_linear = reference_cross_section_cm3_s * phase
    sigma_quadratic = reference_cross_section_cm3_s * phase**2
    return GceResonanceAudit(
        neutrino=neutrino,
        q=q,
        delta_phi=delta,
        phase_modulation=phase,
        reference_cross_section_cm3_s=reference_cross_section_cm3_s,
        sigma_v_linear_cm3_s=sigma_linear,
        sigma_v_quadratic_cm3_s=sigma_quadratic,
        linear_fraction_of_reference=sigma_linear / reference_cross_section_cm3_s,
        quadratic_fraction_of_reference=sigma_quadratic / reference_cross_section_cm3_s,
        induced_theta2_toy=(phase / D_STRING) ** 2,
        gamma_line_energy_gev=neutrino.mass_gev,
        in_gce_photon_energy_band=1.0 <= neutrino.mass_gev <= 4.0,
        in_common_gce_dm_mass_window=7.0 <= neutrino.mass_gev <= 200.0,
        p_value=None,
        p_value_available=False,
    )


def cmb_neff_validation(
    *,
    standard_neff: float = STANDARD_LCDM_NEFF,
    planck_act_mean: float = PLANCK_ACT_NEFF_MEAN,
    planck_act_sigma: float = PLANCK_ACT_NEFF_SIGMA,
    cutoff_limit_kpc: float = LSS_SAFE_CUTOFF_KPC,
) -> CmbNeffValidation:
    """Compute the CMB-facing N_eff and small-scale cutoff audit.

    The release ansatz for the sterile fourth-neutrino radiation trace is

        Delta N_eff = (D / 137) * delta_phi * (1 - q).

    The matter-power cutoff is reported as a conservative thermal-relic
    equivalent scaling,

        lambda_cut ~= 0.11 (m / keV)^(-4/3) Mpc,

    which is deliberately only a scale audit. A GeV sterile state is CDM-like
    on DESI/BAO scales only if its production is cold or sufficiently
    nonthermal; the mass alone does not calculate the abundance or momentum
    distribution.
    """

    neutrino = fourth_neutrino_prediction()
    q = vacuum_compression_operator()
    delta = delta_phi()
    delta_neff = (D_STRING / 137.0) * delta * (1.0 - q)
    neff_total = standard_neff + delta_neff
    z_score = (neff_total - planck_act_mean) / planck_act_sigma
    mass_kev = neutrino.mass_kev
    cutoff_mpc = 0.11 * mass_kev ** (-4.0 / 3.0)
    cutoff_kpc = cutoff_mpc * 1_000.0
    cutoff_wavenumber = 2.0 * math.pi / cutoff_mpc
    return CmbNeffValidation(
        neutrino=neutrino,
        q=q,
        delta_phi=delta,
        delta_neff=delta_neff,
        neff_total=neff_total,
        planck_act_mean=planck_act_mean,
        planck_act_sigma=planck_act_sigma,
        planck_act_z_score=z_score,
        within_one_sigma=abs(z_score) <= 1.0,
        within_two_sigma=abs(z_score) <= 2.0,
        mass_keV=mass_kev,
        thermal_equivalent_cutoff_mpc=cutoff_mpc,
        thermal_equivalent_cutoff_kpc=cutoff_kpc,
        cutoff_limit_kpc=cutoff_limit_kpc,
        below_lss_cutoff_limit=cutoff_kpc < cutoff_limit_kpc,
        cutoff_wavenumber_inv_mpc=cutoff_wavenumber,
        cold_nonthermal_assumption_required=True,
        lss_bao_indistinguishable_from_cdm=cutoff_kpc < cutoff_limit_kpc,
    )


def nuclear_chemistry_validation(
    *,
    ppm_threshold: float = NUCLEAR_CHEMISTRY_PPM_THRESHOLD,
    isotope_atomic_masses_u: dict[str, float] | None = None,
) -> NuclearChemistryValidation:
    """Audit the proposed O(1) nuclear mass-defect encoder.

    The tested release formula is

        Delta M(Z,A) = Z^2 * E_step(Z)/c^2
                       * [1 + 42/(137D) * (1 - 13/117)].

    with E_step(1) = 5e eV and
    E_step(Z>1) = 5e * (137 * 42) * (1 - D/(137N)) eV.

    Values are compared to reference nuclear masses estimated from neutral
    isotope masses by subtracting `Z` electron masses. Total electronic binding
    energies are below the ppm-level conclusion for these rows and are not used
    as tunable corrections.
    """

    atomic_masses = isotope_atomic_masses_u or NUCLEAR_ISOTOPE_ATOMIC_MASS_U
    isotopes = [
        ("H-1", "H", 1, 1),
        ("He-4", "He", 2, 4),
        ("C-12", "C", 6, 12),
        ("Fe-56", "Fe", 26, 56),
    ]
    base_step_ev = 5.0 * math.e
    nuclear_scale = 137.0 * 42.0 * (1.0 - D_STRING / (137.0 * N_TOPOLOGICAL))
    nuclear_step_ev = base_step_ev * nuclear_scale
    resonance_factor = 1.0 + (42.0 / (137.0 * D_STRING)) * (1.0 - 13.0 / 117.0)
    rows: list[NuclearMassAudit] = []
    for isotope, symbol, z, a in isotopes:
        atomic_mass_u = atomic_masses[isotope]
        reference_nuclear_mass = atomic_mass_u * ATOMIC_MASS_UNIT_MEV - z * ELECTRON_MASS_MEV
        unbound_mass = z * PROTON_MASS_MEV + (a - z) * NEUTRON_MASS_MEV
        reference_defect = unbound_mass - reference_nuclear_mass
        running_step_ev = base_step_ev if z == 1 else nuclear_step_ev
        predicted_defect = z * z * running_step_ev * resonance_factor / 1_000_000.0
        predicted_mass = unbound_mass - predicted_defect
        residual = predicted_mass - reference_nuclear_mass
        residual_ppm = _ppm(predicted_mass, reference_nuclear_mass)
        defect_fraction = predicted_defect / reference_defect if reference_defect else math.nan
        rows.append(
            NuclearMassAudit(
                isotope=isotope,
                symbol=symbol,
                z=z,
                a=a,
                atomic_mass_u=atomic_mass_u,
                reference_nuclear_mass_mev=reference_nuclear_mass,
                unbound_nucleon_mass_mev=unbound_mass,
                reference_mass_defect_mev=reference_defect,
                running_step_ev=running_step_ev,
                predicted_mass_defect_mev=predicted_defect,
                predicted_nuclear_mass_mev=predicted_mass,
                mass_residual_mev=residual,
                mass_residual_ppm=residual_ppm,
                defect_fraction_of_reference=defect_fraction,
                within_10ppm=abs(residual_ppm) < ppm_threshold,
            )
        )

    fe = next(row for row in rows if row.isotope == "Fe-56")
    max_abs_ppm = max(abs(row.mass_residual_ppm) for row in rows)
    all_within = all(row.within_10ppm for row in rows)
    return NuclearChemistryValidation(
        d=D_STRING,
        invariant_117=D_STRING * N_TOPOLOGICAL - 13,
        mirror_step=42,
        base_binding_step_ev=base_step_ev,
        nuclear_running_scale_factor=nuclear_scale,
        nuclear_running_step_ev=nuclear_step_ev,
        resonance_factor=resonance_factor,
        ppm_threshold=ppm_threshold,
        fe_z_equals_d_string=fe.z == D_STRING,
        rows=tuple(rows),
        max_abs_mass_residual_ppm=max_abs_ppm,
        all_within_10ppm=all_within,
        fe_mass_residual_ppm=fe.mass_residual_ppm,
        fe_reference_mass_defect_mev=fe.reference_mass_defect_mev,
        fe_predicted_mass_defect_mev=fe.predicted_mass_defect_mev,
        fe_defect_fraction_of_reference=fe.defect_fraction_of_reference,
        release_gate_passed=all_within,
    )


def _ppm(model: float, target: float) -> float:
    return (model - target) / target * 1_000_000.0


def phase9_hadron_upgrade_audit(
    *,
    pi0_mass_mev: float = PI0_MASS_MEV,
    alpha_inv: float = CODATA_ALPHA_INV,
    electron_mass_mev: float = ELECTRON_MASS_MEV,
    n: int = N_TOPOLOGICAL,
    d: int = D_STRING,
) -> Phase9HadronUpgradeAudit:
    """Audit the 117-invariant modification of the neutral-pion ansatz.

    The literal replacement requested in Phase 9, `q -> 117/137`, is evaluated
    explicitly and retained as a failed branch when it worsens the residual.
    A low-complexity modulation of the original compression operator is also
    recorded:

        q_117 = q * (1 - 2^2 / (117 * 137)).

    This keeps the original scale of `q` while allowing the discovered
    `117 = DN - 13` invariant to act as a small correction.
    """

    q = vacuum_compression_operator(n)
    invariant_117 = d * n - 13
    delta = delta_phi(n)
    bracket_tail = 1.0 + 1.0 / n + 1.0 / math.pi
    baseline_ratio = alpha_inv * (2.0 - delta - q * bracket_tail)
    literal_ratio = alpha_inv * (2.0 - delta - (invariant_117 / 137.0) * bracket_tail)
    modulated_q = q * (1.0 - 4.0 / (invariant_117 * 137.0))
    modulated_ratio = alpha_inv * (2.0 - delta - modulated_q * bracket_tail)
    baseline_mass = electron_mass_mev * baseline_ratio
    literal_mass = electron_mass_mev * literal_ratio
    modulated_mass = electron_mass_mev * modulated_ratio
    baseline_ppm = _ppm(baseline_mass, pi0_mass_mev)
    literal_ppm = _ppm(literal_mass, pi0_mass_mev)
    modulated_ppm = _ppm(modulated_mass, pi0_mass_mev)
    return Phase9HadronUpgradeAudit(
        q=q,
        invariant_117=invariant_117,
        pdg_pi0_mass_mev=pi0_mass_mev,
        baseline_pi0_mass_mev=baseline_mass,
        baseline_pi0_ppm=baseline_ppm,
        literal_117_over_137_mass_mev=literal_mass,
        literal_117_over_137_ppm=literal_ppm,
        modulated_q_mass_mev=modulated_mass,
        modulated_q_ppm=modulated_ppm,
        literal_replacement_improves=abs(literal_ppm) < abs(baseline_ppm),
        modulated_q_improves=abs(modulated_ppm) < abs(baseline_ppm),
    )


def phase9_proton_upgrade_audit(
    *,
    proton_mass_mev: float = PROTON_MASS_MEV,
    electron_mass_mev: float = ELECTRON_MASS_MEV,
    alpha_inv: float = CODATA_ALPHA_INV,
    n: int = N_TOPOLOGICAL,
    d: int = D_STRING,
    f26: int = F26,
) -> Phase9ProtonUpgradeAudit:
    """Audit a compact GF(137)-style proton-ratio correction.

    Baseline:

        K0 = [2 F26/(DN)] cos(1/N) [1 + e s].

    Phase 9 correction uses Euler's phi decomposition
    `phi(137)=136=DN+6`, so the gap `6` enters as

        6 / [137^2 (D + N + 2^2)].

    The expression is intentionally reported as an empirical low-complexity
    candidate, not as a derivation from QCD.
    """

    experimental_ratio = proton_mass_mev / electron_mass_mev
    s_value = schwinger_s(alpha_inv)
    base = 2.0 * f26 / (d * n) * math.cos(1.0 / n)
    baseline_ratio = base * (1.0 + math.e * s_value)
    phi_gap = (137 - 1) - d * n
    correction = math.e * s_value + phi_gap / (137.0**2 * (d + n + 4.0))
    upgraded_ratio = base * (1.0 + correction)
    baseline_ppm = _ppm(baseline_ratio, experimental_ratio)
    upgraded_ppm = _ppm(upgraded_ratio, experimental_ratio)
    return Phase9ProtonUpgradeAudit(
        experimental_ratio=experimental_ratio,
        baseline_ratio=baseline_ratio,
        baseline_ppm=baseline_ppm,
        upgraded_ratio=upgraded_ratio,
        upgraded_ppm=upgraded_ppm,
        correction=correction,
        phi_gap=phi_gap,
        complexity_score=11,
        passes_threshold=abs(upgraded_ppm) < 0.5,
    )
