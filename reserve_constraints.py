"""Energy and reserve co-optimisation via `extra_functionality`.

Implements the design in pypsa-reserves.md:

- Only dispatchable generators and storage units supply upward reserve; VRE
  is excluded from supply.
- Storage participates subject to both power headroom and energy adequacy.
- The requirement is a weighted percentage of demand plus available VRE
  output, computed before the solve.
- A penalised shortfall variable keeps high-VRE/low-load hours feasible
  instead of raising INFEASIBLE with no diagnostic.

One deviation from the doc: the requirement and the requirement-side
`Reserve-requirement` constraint are enforced **per bus**, not pooled
system-wide. The doc's own formulation has no bus dimension at all (it is
implicitly single-node), which understates the need for reserve on a
multi-region network like this one -- pooling lets reserve sitting on one
bus cover a deficit on another regardless of the transmission link between
them. Every product is treated as regional here (including FCR, which in
most real synchronous grids is pooled system-wide); revisit if that turns
out to be too conservative.

Usage:

    import reserve_constraints as rc

    rc.attach_reserve_data(n)
    n.optimize(extra_functionality=rc.add_reserve_constraints)

Note on dimension names: pypsa's own per-component variables (`Generator-p`,
`StorageUnit-p_dispatch`, ...) carry the component index under the dim name
`name`, not under a dim named after the component class. The reserve
variables defined here follow that same convention so they line up with the
built-in variables under `.sel(name=...)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from pypsa.descriptors import get_switchable_as_dense as as_dense

DISPATCHABLE_CARRIERS = [
    "nuclear",
    "coal-supercritical",
    "coal-subcritical",
    "gas-conventional",
    "gas-combined-cycle",
    "gas-advanced-combined-cycle",
    "gas-more-advanced-combined-cycle",
    "oil-unspecified",
]

VRE_CARRIERS = [
    "photovoltaic-unspecified",
    "wind-onshore",
    "wind-offshore-unspecified",
    "hydro-reservoir-and-run-of-river",
    "biomass",
    "geothermal-unspecified",
]

RESERVE_PRODUCTS = pd.DataFrame(
    {
        "response_minutes": [0.5, 5.0, 15.0],
        "direction": ["up", "up", "up"],
        "demand_frac": [0.01, 0.05, 0.10],  # alpha_s
        "vre_frac": [0.00, 0.05, 0.05],  # beta_s
    },
    index=pd.Index(["FCR", "aFRR", "mFRR"], name="product"),
)

SHORTFALL_PENALTY = 5000  # EUR/MW, priced above any plausible reserve dual


def attach_reserve_data(n: pypsa.Network, products: pd.DataFrame = RESERVE_PRODUCTS) -> None:
    """Attach reserve product definitions and eligibility/driver flags.

    Call once before solving with `add_reserve_constraints` passed as
    `extra_functionality`.
    """
    n.reserve_products = products.rename_axis("product")
    n.generators["reserve_eligible"] = n.generators.carrier.isin(DISPATCHABLE_CARRIERS)
    n.generators["reserve_driver"] = n.generators.carrier.isin(VRE_CARRIERS)
    n.storage_units["reserve_eligible"] = True


def _snapshot_coords(sns: pd.Index) -> xr.Coordinates:
    if isinstance(sns, pd.MultiIndex):
        return xr.Coordinates.from_pandas_multiindex(sns, "snapshot")
    return xr.Coordinates({"snapshot": sns})


def _group_by_bus(component_values: pd.DataFrame, bus_of: pd.Series, buses: pd.Index) -> pd.DataFrame:
    """Sum a (snapshot x component) frame into (snapshot x bus), zero-filled for bus with none."""
    grouped = component_values.T.groupby(bus_of.reindex(component_values.columns)).sum().T
    return grouped.reindex(columns=buses, fill_value=0.0)


def _bus_labels(names: pd.Index, bus_of: pd.Series) -> xr.DataArray:
    """DataArray of bus labels along dim `name`, for `.groupby(...)` reductions on linopy objects."""
    return xr.DataArray(
        bus_of.reindex(names).to_numpy(),
        coords={"name": names},
        dims=["name"],
        name="bus",
    )


def _reserve_requirement(n: pypsa.Network, sns: pd.Index) -> tuple[xr.DataArray, pd.Index]:
    """R_{s,t,bus} from fixed-capacity load and VRE availability, per bus.

    Returns extendable VRE separately since its availability contains a
    decision variable and has to move to the constraint's left hand side.
    """
    buses = n.buses.index.rename("bus")

    vre = n.generators.index[n.generators.reserve_driver]
    vre_ext = vre.intersection(n.generators.index[n.generators.p_nom_extendable])
    vre_fix = vre.difference(vre_ext)

    avail_pu = as_dense(n, "Generator", "p_max_pu", sns)
    vre_avail_fix = avail_pu[vre_fix] * n.generators.p_nom[vre_fix]
    vre_avail_fix_by_bus = _group_by_bus(vre_avail_fix, n.generators.bus, buses)

    load = as_dense(n, "Load", "p_set", sns)
    load_by_bus = _group_by_bus(load, n.loads.bus, buses)

    alpha = n.reserve_products.demand_frac
    beta = n.reserve_products.vre_frac

    R = np.stack(
        [
            alpha[s] * load_by_bus.to_numpy() + beta[s] * vre_avail_fix_by_bus.to_numpy()
            for s in n.reserve_products.index
        ],
        axis=1,
    )  # shape (snapshot, product, bus)

    R_da = xr.DataArray(
        R,
        coords={**_snapshot_coords(sns), "product": n.reserve_products.index, "bus": buses},
        dims=["snapshot", "product", "bus"],
    )

    return R_da, vre_ext


def add_reserve_constraints(n: pypsa.Network, sns: pd.Index) -> None:
    """extra_functionality entry point for energy/reserve co-optimisation.

    Requires `attach_reserve_data(n)` to have been called first.
    """
    if not hasattr(n, "reserve_products"):
        raise RuntimeError("call attach_reserve_data(n) before solving")

    m = n.model
    prods = n.reserve_products.index
    buses = n.buses.index.rename("bus")

    gens = n.generators.index[n.generators.reserve_eligible]
    sus = n.storage_units.index[n.storage_units.reserve_eligible]

    r_g = m.add_variables(
        lower=0,
        coords={"snapshot": sns, "name": gens, "product": prods},
        name="Generator-r",
    )
    r_s = m.add_variables(
        lower=0,
        coords={"snapshot": sns, "name": sus, "product": prods},
        name="StorageUnit-r",
    )
    shortfall = m.add_variables(
        lower=0,
        coords={"snapshot": sns, "product": prods, "bus": buses},
        name="Reserve-shortfall",
    )

    # --- Generator joint capacity: p + sum_products(r) <= available capacity ---
    p = m.variables["Generator-p"].sel(name=gens)
    pmax = as_dense(n, "Generator", "p_max_pu", sns)[gens]

    ext = gens.intersection(n.generators.index[n.generators.p_nom_extendable])
    fix = gens.difference(ext)

    if len(fix):
        m.add_constraints(
            p.sel(name=fix) + r_g.sel(name=fix).sum("product")
            <= (pmax[fix] * n.generators.p_nom[fix]),
            name="Generator-reserve-joint-fix",
        )

    if len(ext):
        p_nom = m.variables["Generator-p_nom"].sel(name=ext)
        m.add_constraints(
            p.sel(name=ext)
            + r_g.sel(name=ext).sum("product")
            - pmax[ext].to_xarray() * p_nom
            <= 0,
            name="Generator-reserve-joint-ext",
        )

    # --- Generator capability: r <= ramp_up * response window * capacity ---
    window = (n.reserve_products.response_minutes / 60).to_xarray()
    ramp = n.generators.ramp_limit_up[gens].fillna(1.0).to_xarray()

    if len(fix):
        m.add_constraints(
            r_g.sel(name=fix)
            <= ramp.sel(name=fix) * window * n.generators.p_nom[fix].to_xarray(),
            name="Generator-reserve-capability-fix",
        )

    if len(ext):
        p_nom = m.variables["Generator-p_nom"].sel(name=ext)
        m.add_constraints(
            r_g.sel(name=ext) - ramp.sel(name=ext) * window * p_nom <= 0,
            name="Generator-reserve-capability-ext",
        )

    if len(sus):
        # --- Storage power headroom: net output + sum_products(r) <= discharge capacity ---
        p_dis = m.variables["StorageUnit-p_dispatch"].sel(name=sus)
        p_sto = m.variables["StorageUnit-p_store"].sel(name=sus)
        su_pmax = as_dense(n, "StorageUnit", "p_max_pu", sns)[sus]

        su_ext = sus.intersection(n.storage_units.index[n.storage_units.p_nom_extendable])
        su_fix = sus.difference(su_ext)

        if len(su_fix):
            m.add_constraints(
                p_dis.sel(name=su_fix)
                - p_sto.sel(name=su_fix)
                + r_s.sel(name=su_fix).sum("product")
                <= (su_pmax[su_fix] * n.storage_units.p_nom[su_fix]),
                name="StorageUnit-reserve-joint-fix",
            )

        if len(su_ext):
            su_p_nom = m.variables["StorageUnit-p_nom"].sel(name=su_ext)
            m.add_constraints(
                p_dis.sel(name=su_ext)
                - p_sto.sel(name=su_ext)
                + r_s.sel(name=su_ext).sum("product")
                - su_pmax[su_ext].to_xarray() * su_p_nom
                <= 0,
                name="StorageUnit-reserve-joint-ext",
            )

        # --- Storage energy adequacy: hold the reserve for the response window ---
        soc = m.variables["StorageUnit-state_of_charge"].sel(name=sus)
        eff = n.storage_units.efficiency_dispatch[sus].to_xarray()

        m.add_constraints(
            (r_s * window).sum("product") - eff * soc <= 0,
            name="StorageUnit-reserve-energy",
        )

    # --- Requirement, per bus: R_{s,t,bus} = alpha_s*demand + beta_s*available VRE, plus shortfall slack ---
    R, vre_ext = _reserve_requirement(n, sns)

    gen_bus = _bus_labels(gens, n.generators.bus)
    su_bus = _bus_labels(sus, n.storage_units.bus)

    r_g_by_bus = r_g.groupby(gen_bus).sum().reindex(bus=buses, fill_value=0)
    r_s_by_bus = r_s.groupby(su_bus).sum().reindex(bus=buses, fill_value=0)

    lhs = r_g_by_bus + r_s_by_bus + shortfall

    if len(vre_ext):
        avail_pu = as_dense(n, "Generator", "p_max_pu", sns)[vre_ext].to_xarray()
        p_nom_vre = m.variables["Generator-p_nom"].sel(name=vre_ext)
        beta = n.reserve_products.vre_frac.to_xarray()
        vre_ext_bus = _bus_labels(vre_ext, n.generators.bus)
        vre_ext_term = (beta * avail_pu * p_nom_vre).groupby(vre_ext_bus).sum().reindex(bus=buses, fill_value=0)
        lhs = lhs - vre_ext_term

    m.add_constraints(lhs >= R, name="Reserve-requirement")

    m.objective += (shortfall * SHORTFALL_PENALTY).sum()


def reserve_results(n: pypsa.Network) -> dict[str, object]:
    """Post-solve extraction: shadow prices and cleared reserve quantities."""
    return {
        "price": n.model.constraints["Reserve-requirement"].dual,
        "generator_reserve": n.model.variables["Generator-r"].solution,
        "storage_reserve": n.model.variables["StorageUnit-r"].solution,
        "shortfall": n.model.variables["Reserve-shortfall"].solution,
    }
