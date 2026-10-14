# SiP Refactoring Plan

Plan for splitting the `SiP` class (`siesta_python/siesta_python.py`, 4,711 lines, 131 methods, 167 attributes) into smaller, maintainable pieces while keeping functionality and usage patterns identical.

> Revision 2 — incorporates review corrections: reuse-mode dataflow in `__init__`, shared-names module to avoid circular imports, step-1 sequencing, the `dQ` landmine, faithful `__init__` ordering, and smoke-test fixtures.

## 1. Strategy: mixin composition, facade preserved

`SiP` stays exactly where it is — defined in `siesta_python/siesta_python.py`, same constructor signature, same method names. Method bodies move into mixin classes in new modules; `SiP` inherits them:

```python
class SiP(SiPCore, SiPGeometryMixin, SiPFdfMixin, SiPRunnerMixin,
          SiPDFTBPlusMixin, SiPElectrodeMixin, SiPReaderMixin,
          SiPTransportMixin, SiPSislIOMixin):
```

Why this approach:

- **Zero API change.** External code (`Zandpack`, `GrainBoundaryModule`, etc.) does `from siesta_python.siesta_python import SiP` — that import path and every method name keep working untouched.
- **`funcs.py` needs zero changes.** Its functions take the whole object (`write_fdf(object, ...)`) and read `object.lat`, `object.dir`, etc. Since `self` is still one object with all attributes, this keeps working as-is. Verified: `funcs.py`, `dftbplus_interface.py`, and `config.py` import nothing from `siesta_python.py`, so no hidden circularity exists on that side.
- **Pickle/deepcopy keep working.** `pickle()` and `clone()` rely on `SiP`'s module path (`siesta_python.siesta_python.SiP`) and `deepcopy(self)`. Keeping the class definition in the same file preserves both. Old `.SiP` pickle files also remain loadable **provided the facade re-exports moved classes** — pickle resolves a class via `getattr` on the module named in the pickle, so a pickled `SiP` containing `self.dic['dftb_charge']` only unpickles if `siesta_python.siesta_python.dftb_charge` still resolves. See §4 (facade) and §7 (re-export timing).
- **Cross-domain calls keep working.** Methods calling `self.write_more_fdf()` from another domain resolve through the MRO automatically.

## 2. Naming conventions

| Thing | Convention | Example |
|---|---|---|
| Mixin modules | `sip_<domain>.py`, lowercase, `sip_` prefix (sorts together, no stdlib clashes) | `sip_geometry.py` |
| Shared low-level module | `sip_shared.py` — constants, aliases, `error()`; imported by all mixins and the facade; **never imports from the facade or other mixins** | `sip_shared.py` |
| Mixin classes | `SiP<Domain>Mixin` (greppable, self-documenting) | `SiPGeometryMixin` |
| Core base class | `SiPCore` (not a mixin — owns directory + state init) | `SiPCore` |
| Per-domain init helpers | `_init_<domain>(self, ...)` — private, one per mixin, called from `SiP.__init__` | `_init_transport(...)` |
| Extracted pure helpers | module-level snake_case functions in the same `sip_` module; method becomes a thin wrapper | `build_bandpath(sym_dic, two_dim, zdir)` |
| Existing methods/attributes | **No renaming.** Renaming 167 attributes would break `funcs.py` and external callers — that's a separate, breaking change, out of scope | — |

Attribute ownership rule: **the mixin whose methods primarily read/write an attribute initializes it** in its `_init_<domain>` method. Attributes set in `__init__` today get distributed this way; attributes first assigned inside a method belong to that method's mixin.

## 3. File layout

```
siesta_python/siesta_python/
├── siesta_python.py      # facade: SiP class (init + inheritance), module-level
│                         # helpers (rotation_matrix, load_calculation), re-exports
│                         # of everything moved (shared names + moved classes)
├── sip_shared.py         # error(), no_rem, rem, ld, terminal, wd, exec defaults
│                         # (sp, tb_trans_exec, B, pdos_exec, gulp_path, gen_basis,
│                         #  mix_ps, M, PP_path, joblib_backend)
├── sip_core.py           # SiPCore
├── sip_geometry.py       # SiPGeometryMixin
├── sip_fdf.py            # SiPFdfMixin
├── sip_runners.py        # SiPRunnerMixin
├── sip_dftb.py           # SiPDFTBPlusMixin + dftb_charge class
├── sip_electrodes.py     # SiPElectrodeMixin
├── sip_readers.py        # SiPReaderMixin
├── sip_transport.py      # SiPTransportMixin (+ conditional Gf_Module import)
├── sip_sisl_io.py        # SiPSislIOMixin + sisl_replica_for_projection class
├── funcs.py              # unchanged
├── dftbplus_interface.py # unchanged
└── ...
```

Import mechanics:

- Each `sip_*.py` module begins with `sys.path.append(os.path.dirname(os.path.abspath(__file__)))` before `from funcs import ...`. **Do not copy the `__file__[:-16]` hack** from `siesta_python.py` — it is filename-length-specific and computes the wrong path for `sip_*.py` files.
- `sip_shared.py` holds everything mixin code needs from module level (`error`, `no_rem`, `rem`, `ld`, `terminal`, `wd`, exec defaults). This is what makes the dependency graph acyclic: `sip_shared` ← all mixins ← facade. The facade imports from `sip_shared` and re-exports, so old import paths keep working.
- `wd = os.getcwd()` is evaluated at import time, same as today (it moves to `sip_shared.py`, still evaluated once at import).

## 4. Class-by-class breakdown

### `SiPCore` — `sip_core.py`
Directory management, state initialization, object utilities.

| | |
|---|---|
| Methods | `clone`, `pickle`, `save_file`, `set`, `sleep`, `scream`, `hear`, `_init_core`, `_init_directory` |
| Owns attributes | `dir`, `wd`, `sm`, `sl`, `initialised_attributes`, `denchar_files`, `non_ortho`, `dQ` (init value `None`), plus SCF config: `xc`, `xc_authors`, `basis`, `mesh_cutoff`, `max_scf_it`, `min_scf_it`, `dm_tol`, `dm_mixing_weight`, `number_pulay`, `mixer`, `n_hist`, `mix_what`, `energy_shift`, `electronic_temperature_mev`, `spin_pol`, `solution_method`, `calc_bands`, `write_matrices`, `reuse_dm`, `set_pdos`, `save_potential_grids`, `print_mulliken`, `lua_script`, `dic`, `kp`, `k_shift`, `kgridcutoff` |

`scream`/`hear` are lua_comm file IPC — they only need `self.dir`, so core is fine.

### `SiPGeometryMixin` — `sip_geometry.py` (31 methods)

| | |
|---|---|
| Methods | `set_struct`, `reararange`, `standardise_cell`, `set_buffer_atoms`, `set_device_atoms`, `make_bandpath`, `Check_Distances`, `cell_offset`, `cell_deviation`, `center_cell`, `make_pretty`, `Wrap_unit_cell`, `x_to_z`, `move`, `move_atom`, `add_atom`, `add_buffer`, `pretty_cell`, `toASE`, `read_basis_and_geom`, `ase_visualise`, `Visualise`, `figures`, `find_polygons`, `nnr`, `Find_edges`, `Passivate`, `Passivate_with_molecule`, `tile_dm`, `get_labelled_indices`, `sisl_sub` |
| Owns attributes | `lat`, `pos`, `pos_real_space`, `s`, `sname`, `cell_spg`, `get_path`, `sym_dic`, `k_path`, `sym`, `standard_cell`, `UC_idx`, `buffer_atoms`, `Device_atoms`, `Standardize_cell`, `skip_spg_sp`, `TwoDim` |

`set_struct` is the keystone — called by `__init__`, mutates ~10 attributes everything reads. It reads `self.skip_spg_sp` and `self.Standardize_cell`, so `_init_geometry_attrs` must run before it. Move it whole, unchanged.

### `SiPFdfMixin` — `sip_fdf.py` (29 methods)

| | |
|---|---|
| Methods | `fdf`, `fdf_relax`, `barebone_fdf`, `write_more_fdf`, `delete_fdf`, `fatband_fdf`, `custom_bandlines`, `manual_k_points`, `make_meshed_k_grid`, `make_concentrated_k_points`, `set_parallel_k`, `get_pseudo_paths`, `create_mixed_pseudo_dic`, `add_mixed_pseudo_dic`, `ps_mixer`, `dump_pseudo_list_to_struct`, `set_synthetic_mixes`, `E_field`, `set_methfessel_paxton`, `Dipole_Correction_Vacuum`, `Add_Bounded_Plane`, `Add_Charged_Bounded_Plane`, `Add_Charged_Box`, `Add_Charged_Sphere`, `Put_Variable_Charged_Sphere_On_All`, `add_stuff`, `create_added_stuff_dic`, `write_tb_trans_kp`, `gen_basis`, `ideal_cell_charge` |
| Owns attributes | `pseudo_paths`, `mixed_pseudo_list`, `_mixing_dic`, `Added_stuff`, `_manual_kgrid`, `_manual_weights`, `_manual_kgrid_tri`, `Mulliken_C`, `Mulliken_idx` |

Note: `gen_basis` uses `ld`/`rem`/`no_rem`/`error` — import them from `sip_shared`.

### `SiPRunnerMixin` — `sip_runners.py` (11 methods)

| | |
|---|---|
| Methods | `run_siesta_in_dir`, `run_siesta_electrode_in_dir`, `run_analyze_in_dir`, `run_tbtrans_in_dir`, `run_tbt_analyze_in_dir`, `run_gulp_in_dir`, `gin` (gulp input writer — kept next to its runner), `run_fatbands`, `plot_calculated_bands`, `is_siesta_done`, `write_analysis_results` |
| Owns attributes | `gulp_results` + exec paths: `siesta_exec`, `tb_trans_exec`, `mpi`, `pdos_exec`, `gulp_exec`, `bands_plot`, `mix_ps`, `gen_basis_exec`, `pp_path`, `print_no_print` |

### `SiPDFTBPlusMixin` — `sip_dftb.py` (9 methods + 1 class)

| | |
|---|---|
| Methods | `dftb_hsd`, `write_and_run_dftb_in_dir`, `run_dftb_in_dir`, `read_dftb_hamiltonian`, `dftb2sisl`, `fast_dftb_hk`, `dftb_find_equiv`, `dftb_mullikencharge_interface`, `get_potential_energy_dftb` |
| Also moves | class `dftb_charge` |
| Owns attributes | `dic` entries: `dftb_angmom`, `dftb_equivH/S`, `dftb_charge` |

Most self-contained cluster (only 7 data attrs) — extract first. `run_dftb_in_dir` belongs here, **not** in runners.

### `SiPElectrodeMixin` — `sip_electrodes.py` (10 methods)

| | |
|---|---|
| Methods | `add_elecs`, `find_elec_inds`, `make_device_coords`, `copy_DM_from`, `copy_TSHS_from`, `reset_minimal_hamiltonian`, `undo_RSSE`, `get_RS_pos`, `is_RSSE`, `which_type_SE` |
| Owns attributes | `elecs`, `elec_inds`, `elec_RSSE`, `elec_SurRSSE`, `elec_2E_RSSE`, `elec_3D_RSSE`, `elec_bloch`, `semi_inf`, `_rearange_indices`, `old_sl`, `Deci`, `RSSEdata` |

### `SiPReaderMixin` — `sip_readers.py` (9 methods)

| | |
|---|---|
| Methods | `read_relax`, `get_potential_energy`, `get_fermi_level`, `relaxed_system_energy`, `dQ`, `read_tbt`, `read_fermi_level_from_out`, `_has_tbt_nc_files`, `mulliken` |
| Owns attributes | `relaxed_pos`, `relaxed_lat` |

Note: `dQ` the method re-sets `self.dQ` — see landmine §6.1.

### `SiPTransportMixin` — `sip_transport.py` (15 methods)

| | |
|---|---|
| Methods | `Real_space_SE`, `Real_space_SI`, `Real_space_SI_3D`, `self_energy_from_tbtrans`, `tbtrans_H_S_btd_pivot`, `solve_qp_equation`, `projection`, `Hubbard_electrode`, `calculate_hubbard_transport`, `mimic_tbtrans_spin`, `calculate_2E_RSSE`, `calculate_bulksurf_RSSE`, `from_custom`, `get_contour_from_failed_RSSE`, `run_TS_RSSE_calc` (RSSE workflow driver — physics, not a plain runner) |
| Owns attributes | `trans_emin`, `trans_emax`, `trans_delta`, `custom_tbtrans_contour`, `Chem_Pot`, `Voltage`, `NEGF_calc`, `save_SE`, `save_SE_only`, `kp_tbtrans`, `contour_settings`, `RSSE_Energy_from_to`, `SurRSSE_dict`, `_RSSI_SE`, `_RSSI_SRSSE`, `_ax_decimation`, `_ax_integrate`, `_dk`, `_supercell`, `manual_tbtrans_kpoint`, `tbtrans_params_dic`, `Hubbard_electrode` (attribute) |

Most entangled cluster — extract last. The conditional `from Gf_Module.Gf import read_SE_from_tbtrans` (guarded by `GfModule_extension` from `config`) moves here with `self_energy_from_tbtrans`. `Real_space_SE`/`Real_space_SI` call `error()` — import from `sip_shared`.

### `SiPSislIOMixin` — `sip_sisl_io.py` (8 methods + 1 class)

| | |
|---|---|
| Methods | `to_sisl`, `read_TSHS`, `read_minimal_TSHS`, `read_rssi_surface_TSHS`, `to_TB_model`, `write_TB_model`, `fois_gras`, `manual_H` |
| Also moves | class `sisl_replica_for_projection` (unused elsewhere — safe to move) |

### Facade — `siesta_python.py` keeps

`SiP` class (with `__init__` + mixin inheritance), `rotation_matrix`, `load_calculation`, and re-exports:

```python
from sip_shared import error, no_rem, rem, ld, terminal, wd
from sip_shared import sp, tb_trans_exec, B, pdos_exec, gulp_path, gen_basis, mix_ps, M, PP_path, joblib_backend
from sip_dftb import dftb_charge
from sip_sisl_io import sisl_replica_for_projection
```

The re-exports are load-bearing for backward compatibility in two ways: (a) old import paths (`from siesta_python.siesta_python import dftb_charge`, etc.) keep working; (b) **pickle compat** — pickle resolves classes via `getattr` on the module named in the pickle, so old `.SiP` pickle files containing `dftb_charge` instances only load if the facade re-export exists. Land the `dftb_charge` re-export **with step 2** and the `sisl_replica_for_projection` re-export **with step 5** — not as a final touch-up.

## 5. Splitting `__init__` (269 lines)

`SiP.__init__` keeps its **full, unchanged signature** in the facade and preserves the original statement order. The corrected delegation:

```python
def __init__(self, lat, pos, s, directory_name='noname', sm='siesta', ...):  # unchanged
    assert pos.shape[0] == len(s)
    # SiPCore — directory loop, mkdir/reuse; reuse mode REPLACES lat/pos/s from
    # STRUCT.fdf, so this must return the (possibly updated) geometry:
    lat, pos, s = self._init_directory(directory_name, overwrite, wd)
    self._init_core(sm, sl, spin_pol, xc, xc_authors, basis, ...)      # SiPCore
    self._init_geometry_attrs(Standardize_cell, skip_spg_sp, TwoDim)   # SiPGeometryMixin — must precede set_struct
    self.set_struct(lat, pos, s)                                       # SiPGeometryMixin
    # ... remaining attribute assignments stay INLINE in the facade until each
    #     domain's mixin is extracted (steps 2-9); each step moves its block
    #     into that mixin's _init_<domain>(...) call at this point ...
    if not skip_distance_check:
        self.Check_Distances()                                         # SiPGeometryMixin
    self.denchar_files = denchar_files                                 # SiPCore
    self.initialised_attributes = list(self.__dict__.keys())           # SiPCore — SAME position as today (line 340)
    # non_ortho block → self.non_ortho                                  # SiPCore
    # 2-D space-group print; elec_bloch == [1,1,1] → None               # end of init, unchanged
```

Rules for the split:

- **Reuse dataflow**: `_init_directory` returns `lat, pos, s` because `overwrite='reuse'` replaces all three from `STRUCT.fdf` before `set_struct` runs. The facade must pass the returned values to `set_struct`, never the constructor arguments.
- **Same assignment point for `initialised_attributes`**: it is assigned after `Check_Distances`/`denchar_files` and before the `non_ortho` block, exactly as today. Moving it changes its contents (adds `non_ortho`).
- **Step-wise creation of `_init_*`**: each `_init_<domain>` is created in the same step as its mixin. Step 1 introduces only `_init_directory` and `_init_core`; the facade keeps the remaining assignments inline until each domain's step. Do not write the full delegation on day one — it references mixins that don't exist yet.
- **Call-order dependencies** among `_init_*` blocks: `_init_electrodes` before `_init_transport` (`Chem_Pot` defaults to `[0.0]*len(elecs)`); `Voltage` and the `NEGF_calc` auto-set are computed inside `_init_transport` from `Chem_Pot`; `if elec_SurRSSE: self.elec_RSSE = True` happens inside `_init_electrodes`; `self.mpi = mpi + ' '` and `print_no_print` derivation happen inside `_init_runners`. Alternatively, pass raw constructor params to each `_init_*` and let each compute its own derived values — then order matters less.
- No `super().__init__` chains — explicit calls are easier to verify and don't depend on MRO order.

## 6. Landmines — preserve, do not fix

1. **`dQ` method/attribute collision** (`:312` vs `:1206`) — worse than `Hubbard_electrode`: `__init__` sets `self.dQ = None`, so the instance attribute shadows the `dQ()` method **from construction** — `sip.dQ()` raises `TypeError: 'NoneType' object is not callable`. The method only survives as `SiP.dQ(sip)`. Move both as-is; do not "fix".
2. **`Hubbard_electrode` name collision** (`:3382` vs `:3400`): it's both a method and an instance attribute (`self.Hubbard_electrode = 'Yes'` shadows the method after first call; `read_TSHS:2154` reads the attribute). Before the first call, the class method conveniently serves the `== 'Yes'` check. Move both as-is; do not "fix".
3. **`save_file` bug** (`:1937`): uses `object.dir` (the builtin) instead of `self.dir` — raises `AttributeError` if ever called. Preserve the bug; flag for a separate fix.
4. **`initialised_attributes`** (`:340`): written, never read anywhere in this repo or externally (verified). Keep the assignment at the same point in `__init__` — see §5.
5. **`clone` uses `deepcopy(self)`** — mixins are deepcopy-safe, but verify with a roundtrip test after the refactor.
6. **`run_dftb_in_dir`** belongs in `SiPDFTBPlusMixin` only, not in runners.

## 7. Migration order (one commit per step, verified each time)

| Step | Action | Why this order |
|---|---|---|
| 0 | Write smoke test + API-surface snapshot script, **with fixtures** (see §8). Commit. | Safety net — without it, don't start |
| 1 | Create `sip_shared.py` (shared names) and `sip_core.py` (`SiPCore`: misc methods, `_init_core`, `_init_directory`). Facade `__init__` delegates only the directory + core blocks; all other assignments stay inline. Verify. | Establishes the `_init_*` pattern and the acyclic import graph |
| 2 | Extract `sip_dftb.py` + `dftb_charge`; add the facade re-export **in this step** (pickle compat). Verify. | Most self-contained (7 attrs) |
| 3 | Extract `sip_runners.py` + its `_init_runners`. Verify. | Mostly exec paths |
| 4 | Extract `sip_readers.py`. Verify. | Small, read-only |
| 5 | Extract `sip_sisl_io.py` + `sisl_replica_for_projection`; add the facade re-export **in this step**. Verify. | Self-contained I/O |
| 6 | Extract `sip_electrodes.py` + `_init_electrodes`. Verify. | Medium coupling |
| 7 | Extract `sip_fdf.py` + `_init_fdf`. Verify. | Big but mostly self-contained |
| 8 | Extract `sip_geometry.py` + `_init_geometry_attrs` (contains the `set_struct` keystone). Verify. | `set_struct` must land with its attribute init |
| 9 | Extract `sip_transport.py` + `_init_transport` (incl. conditional `Gf_Module` import). Verify. | Last — most entangled |
| 10 | Final pass: facade is ~150 lines; full verification + real end-to-end siesta run if available | |

## 8. Verification at every step

1. **API-surface diff**: script that snapshots `sorted(dir(SiP))` + signatures of all 131 methods before the refactor; after each step, diff must be empty.
2. **Smoke test** (with fixtures): run from a temp working directory. Create a fake pseudopotential directory (default `../pp`, or pass `pp_path=...`) containing dummy `<Sym>.<xc>.psf` files for the species used — `get_pseudo_paths` does `ld(pp_path)` and symlinks matches, so `fdf()` fails without it. `SiP.__init__` also does `mkdir`, so the temp dir is required. Then: construct `SiP` on a small cell → `fdf()` → assert `RUN.fdf` written with expected content → `make_bandpath()` → `to_sisl()` → `pickle()`/unpickle roundtrip → `clone()`. Output must be identical before/after.
3. **Import check**: `from siesta_python.siesta_python import SiP, dftb_charge, sisl_replica_for_projection, rotation_matrix, load_calculation, error, no_rem` all still work.
4. **External caller check**: grep Zandpack/GrainBoundaryModule/etc. for `sip.<method>` calls; confirm every called method still exists on `SiP`.
5. **End-to-end** (final step): run one real siesta + tbtrans calculation through the refactored class.

## 9. Explicitly out of scope (future work, separate PRs)

- Renaming attributes/methods to consistent snake_case (breaking change).
- Pulling the 167 attributes into a `SiPState` dataclass (would require touching `funcs.py` and external code).
- Fixing the `save_file` bug, the `dQ` shadowing, and the `Hubbard_electrode` shadowing.
- Moving `funcs.py` free functions into the mixins as methods.
