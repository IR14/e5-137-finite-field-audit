# Phase 14 Quantum Routing Manifest

## Scope

This note records a passive symbolic extension of the e-5-137 finite-field
audit.  It is a bookkeeping layer for algebraic identities and routing
language.  It is not a condensed-matter derivation, not a superconductivity
model, and not evidence for microscopic teleportation of charge carriers.

## Calabi-Yau Projection Operator

The sixth-level projection is represented by the factor

```text
P6 = 1 / 6.
```

The integral geometric operator is stored as

```text
D(N) = (1 / 6) * (N^3 - 3 N^2 + 6 N).
```

For the repository default `N = 5`,

```text
D(5) = (125 - 75 + 30) / 6 = 40 / 3.
```

This value is used only as a symbolic compactification coordinate in the
audit.  No topological manifold is reconstructed from it.

## Mirror Boundary and Zeta Regularization

The mirror anti-state boundary is evaluated at `N = -3`:

```text
D(-3) = ((-3)^3 - 3(-3)^2 + 6(-3)) / 6
      = (-27 - 27 - 18) / 6
      = -12.
```

Its multiplicative inverse is therefore

```text
1 / D(-3) = -1 / 12.
```

The value `-1/12` is the zeta-regularized value conventionally written as
`zeta(-1)`.  In this repository it is stored as a boundary condition for the
symbolic cascade operator.  The report does not infer a Borcherds denominator
formula, a critical string dimension, or a cancellation of physical vacuum
divergences from this identity alone.

## Full String-Axis Lookup

The release stores the exact lookup table for every integer `N` in `[-26, 26]`
as numerator/denominator triples.  The interval contains 53 integer nodes, or
52 nonzero nodes when `N = 0` is excluded.

```text
N    D(N)
-26  -9880/3
-25  -8825/3
-24  -2616
-23  -6946/3
-22  -6116/3
-21  -1785
-20  -4660/3
-19  -4028/3
-18  -1152
-17  -2941/3
-16  -2480/3
-15  -690
-14  -1708/3
-13  -1391/3
-12  -372
-11  -880/3
-10  -680/3
-9   -171
-8   -376/3
-7   -266/3
-6   -60
-5   -115/3
-4   -68/3
-3   -12
-2   -16/3
-1   -5/3
0    0
1    2/3
2    4/3
3    3
4    20/3
5    40/3
6    24
7    119/3
8    184/3
9    90
10   380/3
11   517/3
12   228
13   884/3
14   1120/3
15   465
16   1712/3
17   2074/3
18   828
19   2945/3
20   3460/3
21   1344
22   4664/3
23   5359/3
24   2040
25   6950/3
26   7852/3
```

The requested release labels are audited against the same polynomial rather
than substituted into the table.  Three labels match the stored formula:

```text
D(-3) = -12
D(1)  = 2/3
D(6)  = 24
```

Three labels do not match the stored formula:

```text
requested D(-6) = -120, formula D(-6) = -60
requested D(7)  = 42,   formula D(7)  = 119/3
requested D(13) = 286,  formula D(13) = 884/3
```

These mismatches are kept as audit output.  They are not patched by changing
the lookup table, because doing so would make the table inconsistent with the
declared function `D(N) = (N^3 - 3N^2 + 6N) / 6`.

## Complex Phase Invariant

The complex invariant is evaluated directly:

```text
D_phase = (1 / 6) * (sqrt(3) - i) * i * (i + sqrt(3)).
```

Because

```text
(sqrt(3) - i) * (i + sqrt(3)) = 4,
```

the expression reduces to

```text
D_phase = (2 / 3) i.
```

The implementation stores the real and imaginary parts separately:

```text
Re(D_phase) = 0
Im(D_phase) = 0.6666666666666666.
```

The passive pytest assertion checks agreement with `(2/3)i` at floating-point
precision.

## Quantum Routing Proxy

The electrical-current language is treated as a routing proxy.  In this
representation, a conductor is described as a residue-synchronized channel over
`GF(137)`, with `D = 26` bookkeeping axes and one symbolic tick per local
routing update:

```text
T_tick = O(1).
```

The intended analogy is:

- lattice coherence corresponds to phase alignment of residue classes;
- scattering corresponds to loss of alignment between local residue channels;
- a superconducting limit corresponds to an idealized zero-dissipation branch
  of the proxy, not to a measured material state;
- Landauer losses are not physically removed by the model.  They are excluded
  only in the ideal reversible-routing limit used for accounting.

This layer therefore preserves the existing release discipline: the algebraic
identity is asserted, while the physical-current interpretation remains
unproven.

## Stored Invariants

```text
compact_dimension = 6
projection_operator = 1 / 6
field_modulus = 137
residue_axis_count = 26
integral_operator_value_at_N5 = 40 / 3
mirror_antistate_n = -3
mirror_integral_operator_value = -12
mirror_inverse_regularizer = -1 / 12
zeta_negative_one_regularization = -1 / 12
axis_min = -26
axis_max = 26
axis_count = 53
nonzero_axis_count = 52
formula_values_verified = true
requested_nodes_all_match = false
phase_invariant = (2 / 3) i
routing_proxy_only = true
regularization_boundary_only = true
superconductivity_model_proven = false
physical_current_model_proven = false
```
