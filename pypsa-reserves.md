# Modelling reserve requirements in PyPSA

Implementation notes for adding energy and reserve co-optimisation to an existing
PyPSA network via `extra_functionality`.

Scope decisions baked into these notes:

- Only **dispatchable generators** supply upward reserve. VRE is excluded.
- **Storage units participate**, subject to both power headroom and energy adequacy.
- The reserve requirement is a **weighted percentage of demand plus available
  solar and wind**, computed before the solve.

## Theory in one page

Take the single-period co-optimisation with elastic demand:

```
max_{p,d,r}  ∫₀ᵈ MB(x)dx − Σ_g ∫₀^{p_g} MC_g(x)dx

(λ)    d − Σ_g p_g = 0            power balance
(μ)    Σ_g r_g ≥ R                reserve requirement
(ν_g)  p_g + r_g ≤ P_g            joint capacity
(ξ_g)  r_g ≤ R_g                  reserve capability
       p_g, r_g, d ≥ 0
```

Stationarity at an interior optimum gives:

```
λ = MB(d)
λ = MC_g(p_g) + ν_g
μ = ν_g + ξ_g
```

Three consequences the implementation has to reproduce:

1. **Reserve is priced off foregone energy margin.** For a unit whose capacity
   binds but whose capability does not, `μ = λ − MC_g`. Nothing bids for reserve
   directly.
2. **Reserve goes to expensive units first.** Procurement cost is `Σ_g ν_g r_g`
   with `ν_g = max(λ − MC_g, 0)`, so the requirement fills in reverse merit
   order. Units above the marginal unit have `ν_g = 0` and supply reserve free.
3. **Products only price apart when capability binds.** `ν_g` carries no product
   index, so `μ_s − μ_s' = ξ_{g,s} − ξ_{g,s'}`. If every product clears at the
   same price, capability never binds and the product dimension buys you nothing.

`R` is exogenous in this formulation, so `μ_s` is a clean scarcity price: the
cost of one more MW of reserve obligation in that hour. It reads directly as an
integration cost when you multiply it through the VRE-driven part of the
requirement after the solve.

## Step 1: Data the user attaches to the network

Four additions to a base network, none of which require PyPSA to know about them.

```python
import pandas as pd

DISPATCHABLE = ["CCGT", "OCGT", "coal", "lignite", "nuclear", "biomass", "oil"]
VRE = ["solar", "onwind", "offwind", "solar rooftop"]

# a) product definitions, including the requirement coefficients
n.reserve_products = pd.DataFrame(
    {
        "response_minutes": [0.5, 5.0, 15.0],
        "direction": ["up", "up", "up"],
        "demand_frac": [0.01, 0.02, 0.03],   # alpha_s
        "vre_frac":    [0.00, 0.05, 0.05],   # beta_s
    },
    index=["FCR", "aFRR", "mFRR"],
)

# b) who may supply reserve
n.generators["reserve_eligible"] = n.generators.carrier.isin(DISPATCHABLE)
n.storage_units["reserve_eligible"] = True

# c) who drives the requirement
n.generators["reserve_driver"] = n.generators.carrier.isin(VRE)
```

Dispatchable hydro is a judgement call. Reservoir hydro modelled as a
`StorageUnit` is handled by the storage path below. Run-of-river modelled as a
`Generator` with a fixed `p_max_pu` profile should be a driver, not a supplier.

Ramp capability comes from `ramp_limit_up` / `ramp_limit_down`, which PyPSA
carries as a fraction of `p_nom` per hour. NaN means unlimited, so `fillna(1.0)`
before use or you will silently drop a unit's capability cap. Keep
`ramp_limit_down` separate: ramp rates are asymmetric and downward reserve
depends on the down rate.

## Step 2: Variables

```python
from pypsa.descriptors import get_switchable_as_dense as as_dense


def add_reserves(n, sns):
    m = n.model
    prods = n.reserve_products.index

    gens = n.generators.index[n.generators.reserve_eligible]
    sus = n.storage_units.index[n.storage_units.reserve_eligible]

    r_g = m.add_variables(
        lower=0,
        coords={"snapshot": sns, "Generator": gens, "product": prods},
        name="Generator-r",
    )
    r_s = m.add_variables(
        lower=0,
        coords={"snapshot": sns, "StorageUnit": sus, "product": prods},
        name="StorageUnit-r",
    )
```

One variable per component class with a product dimension, not one variable per
product. The shared component dimension is what lets a single capacity
constraint couple the products together.

## Step 3: Supply-side constraints

### Generator joint capacity

The `.sum("product")` is the entire coupling mechanism. Fixed and extendable
capacity are handled separately because `p_nom` is a parameter in one case and a
variable in the other.

```python
    p = m.variables["Generator-p"].sel(Generator=gens)
    pmax = as_dense(n, "Generator", "p_max_pu", sns)[gens]

    ext = gens.intersection(n.generators.index[n.generators.p_nom_extendable])
    fix = gens.difference(ext)

    if len(fix):
        m.add_constraints(
            p.sel(Generator=fix) + r_g.sel(Generator=fix).sum("product")
            <= (pmax[fix] * n.generators.p_nom[fix]),
            name="Generator-reserve-joint-fix",
        )

    if len(ext):
        p_nom = m.variables["Generator-p_nom"].sel(Generator=ext)
        m.add_constraints(
            p.sel(Generator=ext)
            + r_g.sel(Generator=ext).sum("product")
            - pmax[ext].to_xarray() * p_nom
            <= 0,
            name="Generator-reserve-joint-ext",
        )
```

`p_max_pu` stays in there so derating and outage profiles carry through to
headroom.

### Generator capability

`ramp_limit_up` is per hour, so scale by the response window in hours.

```python
    window = (n.reserve_products.response_minutes / 60).to_xarray()
    ramp = n.generators.ramp_limit_up[gens].fillna(1.0).to_xarray()

    m.add_constraints(
        r_g <= ramp * window * n.generators.p_nom[gens].to_xarray(),
        name="Generator-reserve-capability",
    )
```

For extendable units move `p_nom` to the left hand side:
`r_g − ramp * window * p_nom <= 0`.

### Storage power headroom

A storage unit provides upward reserve two ways: by discharging harder, or by
charging less. Both show up in net output, so write headroom against net output
rather than against `p_dispatch` alone.

```python
    p_dis = m.variables["StorageUnit-p_dispatch"].sel(StorageUnit=sus)
    p_sto = m.variables["StorageUnit-p_store"].sel(StorageUnit=sus)
    su_pmax = as_dense(n, "StorageUnit", "p_max_pu", sns)[sus]

    m.add_constraints(
        p_dis - p_sto + r_s.sum("product")
        <= (su_pmax * n.storage_units.p_nom[sus]),
        name="StorageUnit-reserve-joint",
    )
```

If storage `p_nom` is extendable, split fixed and extendable exactly as for
generators.

### Storage energy adequacy

Power headroom alone lets a nearly empty battery promise delivery it cannot
sustain. Require enough stored energy to hold the reserve for the response
window, converted through discharge efficiency:

```python
    soc = m.variables["StorageUnit-state_of_charge"].sel(StorageUnit=sus)
    eff = n.storage_units.efficiency_dispatch[sus].to_xarray()

    m.add_constraints(
        (r_s * window).sum("product") - eff * soc <= 0,
        name="StorageUnit-reserve-energy",
    )
```

`state_of_charge` in PyPSA is end of period, which is the conservative choice
here. For short products the binding constraint is almost always power, not
energy. For mFRR at 15 minutes a one-hour battery can still hit the energy limit
when it is close to empty.

If `max_hours` is short relative to the response window, this constraint is what
stops the model from over-crediting storage. Do not omit it.

### Downward reserve

Add a second variable set and a footroom constraint. This is a separate
inequality, not a rearrangement of the upward one:

```
generator:  p_g − r_dn_g ≥ p_min_pu_g · p_nom_g · x_g
storage:    p_dis − p_sto − r_dn_s ≥ −p_nom_s · p_max_pu   (room to charge)
            (max_hours · p_nom) − soc ≥ Σ_s r_dn_s · Δt_s / eff_store
```

The generator version needs commitment status `x_g` to mean anything. Without a
minimum stable level the LP backs a unit down to zero and downward reserve is
nearly free.

Downward reserve prices as `MC_g − λ`, the mirror of the upward case: cheap on
inframarginal units running flat out, expensive on units that would have to be
dispatched out of merit to create footroom. Storage provides both directions
symmetrically, which is a large part of why storage value rises once reserve is
modelled at all.

## Step 4: The requirement

The requirement is a weighted percentage of demand and of available solar and
wind:

```
R_{s,t} = α_s · D_t + β_s · Σ_{v∈VRE} ( p_nom_v · p_max_pu_{v,t} )
```

Both terms are known before the solve, so `R_{s,t}` is a plain parameter and the
constraint has a numeric right hand side. The VRE term uses **available**
output, `p_nom · p_max_pu`, not dispatched output. That is the right quantity on
its own merits: forecast error scales with what the resource could have produced,
not with what the optimiser chose to take. It also keeps the requirement
independent of the dispatch decision, which is what you want.

```python
    vre = n.generators.index[n.generators.reserve_driver]
    vre_avail = (
        as_dense(n, "Generator", "p_max_pu", sns)[vre] * n.generators.p_nom[vre]
    ).sum(axis=1)                                  # MW per snapshot

    load = n.loads_t.p_set.loc[sns].sum(axis=1)    # MW per snapshot

    alpha = n.reserve_products.demand_frac
    beta = n.reserve_products.vre_frac

    R = (
        pd.DataFrame(index=sns, columns=n.reserve_products.index, dtype=float)
        .apply(lambda col: alpha[col.name] * load + beta[col.name] * vre_avail)
    )

    m.add_constraints(
        r_g.sum("Generator") + r_s.sum("StorageUnit") >= R.to_xarray(),
        name="Reserve-requirement",
    )
```

Compute `R` once outside `extra_functionality` and attach it as
`n.reserve_requirement` if you want to inspect or plot it before solving. The
logic is that hours with a lot of wind and solar on the system carry more
forecast error and more sudden-loss risk, so they need more headroom held back.

### One exception: capacity expansion

If VRE `p_nom` is extendable then `p_nom_v · p_max_pu` contains a decision
variable, and the VRE term has to move to the left hand side:

```python
    vre_ext = vre.intersection(n.generators.index[n.generators.p_nom_extendable])
    p_nom_vre = m.variables["Generator-p_nom"].sel(Generator=vre_ext)
    avail_pu = as_dense(n, "Generator", "p_max_pu", sns)[vre_ext].to_xarray()

    m.add_constraints(
        r_g.sum("Generator")
        + r_s.sum("StorageUnit")
        - beta.to_xarray() * (avail_pu * p_nom_vre).sum("Generator")
        >= alpha.to_xarray() * load.to_xarray(),
        name="Reserve-requirement",
    )
```

This is still linear, and it is linear in **investment**, not in dispatch. Build
more solar and you commit to holding more reserve every hour of the year. That
feedback is a real cost of VRE expansion and belongs in the model. What it does
not do is give the optimiser a reason to curtail, since dispatch no longer
appears in the constraint.

The shadow cost per MW of VRE capacity is `Σ_t Σ_s μ_{s,t} · β_s ·
p_max_pu_{v,t}`, which is a clean post-solve integration cost you can report
alongside LCOE.

## Step 5: Reserve shortfall slack

With VRE excluded from supply and driving the requirement, high-VRE low-load
hours can be infeasible. Add a penalised slack rather than debugging an
`INFEASIBLE` status with no information:

```python
    shortfall = m.add_variables(
        lower=0, coords={"snapshot": sns, "product": prods},
        name="Reserve-shortfall",
    )
    # add `shortfall` to the LHS of Reserve-requirement
    m.objective += (shortfall * 5000).sum()   # EUR/MW, above any plausible mu
```

Price it above any credible `μ`, then check post-solve that it is zero. Where it
is not, you have found either a genuine adequacy result or a data problem, and
the pattern in time tells you which.

## Step 6: Solve and extract prices

```python
n.optimize(extra_functionality=add_reserves)

mu = n.model.constraints["Reserve-requirement"].dual    # EUR/MW, snapshot x product
r_gen = n.model.variables["Generator-r"].solution
r_sto = n.model.variables["StorageUnit-r"].solution
slack = n.model.variables["Reserve-shortfall"].solution
```

Sign convention: linopy minimises, so the dual on a `>=` requirement comes out
non-negative and reads directly as the reserve price.

## Step 7: Validation

Build small test networks by hand before running anything real.

| Case | Setup | Expected |
|---|---|---|
| Free reserve | Cheap 20/100MW, peaker 80/50MW, load 100, R = 30 | Reserve all on peaker, `μ = 0`, `λ = 80` |
| Binding reserve | Same, R = 80 | Cheap unit backed off, `μ = 60 = λ − MC` |
| Storage energy | Battery 50MW / 0.25h, mFRR window 15min | Reserve capped by energy, not power |
| Requirement | Add 100MW solar at `p_max_pu = 0.8`, `β = 0.05` | Requirement rises by exactly 4MW in that hour, and does not move if solar is curtailed |

Then check `μ_s` actually differs across products. If they all clear
identically, either capability never binds or the `response_minutes` values are
not differentiating the fleet, and you may as well collapse to one product.

## Gotchas

**Commitment kills the duals.** With `committable=True` the problem is a MILP and
`mu` is undefined. You need fix-and-resolve: solve the MILP, fix the binaries,
re-solve as an LP. PyPSA does not do this for you.

**Eligibility is doing real work now.** Restricting supply to dispatchable units
means an LP can no longer take free spinning reserve from an offline peaker at
`p = 0`. That is the intended behaviour, but it is still not physical: an LP
peaker at zero output is offline and cannot spin. Either accept the
understatement, or add commitment and require
`r_g ≤ p_max_pu · p_nom · x_g − p_g`.

**High VRE systems bind downward, not upward.** At midday with solar saturating
the stack, upward reserve is abundant because thermal is backed off, while
downward reserve is scarce because so little is running. A single upward
requirement systematically understates the flexibility need in solar heavy
hours. Since forecast error cuts both ways, make the VRE-driven requirement two
sided.

**In capacity expansion, reserve pulls investment.** Intended, but check what it
builds. With no commitment and a dispatchable-only eligibility rule, the
cheapest way to satisfy a reserve requirement is a peaker that never runs. If
that shows up in results, the capacity is real but its reserve contribution is
an artefact of the missing online constraint.

**API drift.** `get_switchable_as_dense` has moved between `pypsa.descriptors`
and `pypsa.common` across versions, and linopy's `coords` handling has changed
too. Pin your versions and expect the import line to be the first thing that
breaks.
