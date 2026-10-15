# SiP Refactoring Plan

Revision 3 — recentred on **copy-paste method extraction into free functions** (`self` → `sip`),
plus a separate, optional refactor of the `SiP.__init__` input surface.

Baseline measured for this revision: `siesta_python/siesta_python.py` = **4,376 lines / 137 methods**.
Line numbers below are from that baseline and will drift — treat names/context as authoritative.

> Rev 2 was a mixin-composition plan. Rev 3 keeps verified facts from it (dependency graph, import
> mechanics, verification) but changes the *method* of extraction: instead of moving bodies into
> `SiP*Mixin` classes, move them into module-level functions that take the object as their first
> argument. See §1.4 for why, and when the mixin approach is still the better end state.
>
> **Scope.** In scope: structural extraction of `SiP` method bodies into free functions, and the
> `__init__` input refactor. Out of scope: behaviour changes, `funcs.py`, and the `.SiP` pickle
> format (§9). The extraction itself must be behaviour-neutral.

---

## 0. State of the code right now (do not re-litigate)

Already done, and reflected throughout this plan:

- `dQ` method/attribute collision **fixed** — `dQ` is now a normal method returning the value.
- `Hubbard_electrode` collision **fixed** — flag is now `self.is_hubbard_electrode` (bool), set in
  `Hubbard_electrode` and read in `read_TSHS`; the method name is unchanged.
- `save_file` `object.dir` bug **fixed** — now uses `self.dir`.
- Methods `set`, `sleep`, `scream`, `hear` and the module-level `load_calculation` **removed**.
- `from datetime import datetime` removed (its only user, `scream`, is gone).
- New helper `ossystem(string)` (thin wrapper over `os.system`).
- Most `np.savez` → `np.savez_compressed`; dead comments/whitespace trimmed.

These are behavioural changes to the public surface and belong in their own commits with a
before/after check (see §8), not folded into the structural extraction.

External-surface facts to respect (verified by grep over the parent tree, not just this repo):

- `em.Hubbard_electrode(...)` is called from `Zandpack/Zandpack/examples/Hubbard/1Hamiltonian.py`.
  → the **method name is public API**; keep it.
- ~15 sites read `*.SiP` with a plain `pickle.load(open(..., 'rb'))` (`Zandpack/Zandpack/wrapper.py`
  `load_object` + `:1306`/`:1416`, `Zandpack/.../AI_fit_base.py`, and the `examples/*/…py` scripts
  that do `from pickle import load`). → **do not change the on-disk pickled format** without a
  sniffing loader (§9).
- `Zandpack/Zandpack/documentation*.py` hold `drop_names_SiP` string lists naming `'scream'`,
  `'set'`, etc. — harmless strings, no code dependency.

---

## 1. Strategy: extract method bodies into free functions

### 1.1 The idea

`SiP` stays exactly where it is — same class, same name, same constructor, same method names.
For each method to be extracted:

1. **Copy** the method body into a module-level function in a domain module.
2. Add the object as the first parameter (`sip`) and **replace `self` with `sip`**.
3. Replace the method body in the class with a **one-line wrapper** that calls the free function.

This is a mechanical, copy-paste transform. It preserves the public API 1:1 (the method still
exists with the same signature and behaviour), so no caller anywhere has to change.

### 1.2 Why this is the primary strategy

- **It is copy-paste, not surgery.** The only edits inside a moved body are textual: `self` → `sip`.
  No MRO reasoning, no `super()` chains, no inheritance-ordering puzzles.
- **The codebase already works this way.** `funcs.py` has 11 functions that take the object as
  `object` and read `object.lat`, `object.dir`, etc. `barebone_fdf` already delegates to
  `barebones_RUN(self)`. `fdf`-family methods already call `write_fdf(self)`. `ossystem` is a fresh
  example of "helper is a thin shim". Extraction formalises an existing pattern.
- **Fully incremental and independently verifiable.** Each extracted function is a pure move; the
  diff is readable (one added module function, one shrunk method).
- **`funcs.py` needs zero changes.** It keeps receiving the same object with the same attributes.

### 1.3 What stays identical

- `class SiP` definition location and name (`siesta_python.siesta_python.SiP`) — pickle and imports.
- Every method name and signature (wrappers).
- Every attribute name and flat-attribute layout (`sip.lat`, `sip.elecs`, …).
- Re-exports of moved module-level names and classes from the facade (pickle + import compat).

### 1.4 Tradeoffs vs the Rev-2 mixin approach

| | Free functions (this plan) | Mixins (Rev 2) |
|---|---|---|
| Transform difficulty | copy-paste, `self`→`sip` | move + inheritance wiring |
| API risk | very low (wrappers) | low, but MRO / `__init__` ordering matters |
| Testable in isolation | yes — call the function with a stub/new object | yes, via the class |
| Attribute discoverability | lower — attributes are set by free functions | same as today |
| IDE "find usages" of a method | wrapper is trivial; body is a plain function | single class |
| Cross-mixin name collisions | n/a | must be checked by hand |

Pick free functions if the goal is **minimum-risk, easy-to-review progress**. Mixins remain a
reasonable end state if you later want the methods to *belong* to a class; the wrappers produced
here convert to mixins trivially (paste the free function back as a method). You are not locked in.

### 1.5 End state

`siesta_python.py` shrinks to a facade: `class SiP` whose methods are one-line wrappers, plus
`__init__`, plus `rotation_matrix`, plus re-exports. All logic lives in `sip_*.py` as plain
functions taking `sip`. `dir(SiP)` and every signature are unchanged, so callers and pickles are
unaffected. The win is reviewability and testability of the logic, not a smaller public surface.

---

## 2. The mechanical transform (do this per method)

### 2.1 Recipe

```python
# BEFORE (in class SiP)
def name(self, a, b=1):
    """Docstring."""
    x = self.lat               # attribute read
    self.write_more_fdf([a])   # call to another method
    return helper(b)
```

```python
# AFTER — new function in sip_<domain>.py
from sip_shared import ld, error, ...      # module-level names the body needs
def name(sip, a, b=1):
    """Docstring."""
    x = sip.lat                        # self. -> sip.
    sip.write_more_fdf([a])            # still resolves: the wrapper method exists
    return helper(b)
```

```python
# AFTER — wrapper in class SiP
def name(self, a, b=1):
    return sip_domain.name(self, a, b)
```

### 2.2 Copy-paste checklist (run through for every method)

1. Copy signature verbatim, delete the leading `self`, re-add `sip` as the first parameter.
2. Replace every `self` token with `sip`. Watch for **bare `self` passed as a value**
   (`write_fdf(self, …)`, `deepcopy(self)`, `self.add_stuff(dic)`), not just `self.x`.
3. Keep the docstring unchanged.
4. If the body references a module-level name (`error`, `no_rem`, `rem`, `ld`, `wd`, `PP_path`,
   `sp`, `tb_trans_exec`, `B`, `pdos_exec`, `gulp_path`, `gen_basis`, `mix_ps`, `joblib_backend`,
   `M`), import it from `sip_shared` (§3).
5. If the body references the **class name `SiP`**, replace with a lazy import (§2.3).
6. If the body contains `global X`, keep it — it now refers to the new module's global (§2.3).
7. Leave `os.chdir`, `ossystem`, `Popen`, prints, and numeric code untouched.
8. In the class, replace the body with `return <module>.<name>(self, …)` (drop `return` if the
   method returned `None`).
9. Confirm the method signature is byte-identical to before (defaults included).
10. Re-run the API snapshot + pyflakes + smoke test (§8).

### 2.3 Gotchas (all present in this codebase)

| Gotcha | Where | Handling |
|---|---|---|
| `isinstance(x, SiP)` | `copy_DM_from`, `add_atom` | lazy `from siesta_python.siesta_python import SiP` inside the function, or a duck-type check |
| `SiP(...)` construction | `projection` | same lazy import (circular-import safe: runs at call time) |
| `global SE_func` | `Real_space_SE`, `Real_space_SI` | keep the block intact; `SE_func` becomes a global of the new module and `del SE_func` still deletes it. It is a shared module global, so the `parallel_E` path is not thread-safe (pre-existing) — verify it still runs after the move (§8.7) |
| `self.__dict__[...]` / `initialised_attributes` | `__init__` (`:333`) | leave in `__init__`/core; do not move |
| parameter literally named `object` | `copy_DM_from(self, object, …)` | rename to `src` **while moving** (removes a builtin shadow) |
| mutable default args | `angmom_dic={}`, `manual_pp=[]`, `Custom=[]` | leave for behaviour fidelity; convert in a separate step if desired |
| `sys.path.append(__file__[:-16])` | module head | replace with one path fix in `__init__.py` / relative imports (§3) |

### 2.4 Worked example (`dQ`)

```python
# sip_readers.py
def dQ(sip):
    """Read excess charge from transiesta calculation."""
    dq = None
    with open(sip.dir + '/' + 'RUN.out', 'r') as f:
        for l in f:
            if 'Excess charge' in l:
                dq = float(l[37:])
                break
    return dq
```

```python
# siesta_python.py (class SiP)
def dQ(self):
    return sip_readers.dQ(self)
```

That is the entire pattern. Everything else is repetition.

---

## 3. Module layout

```
siesta_python/siesta_python/
├── siesta_python.py      # facade: class SiP (thin wrappers), rotation_matrix,
│                         # re-exports of moved names/classes
├── sip_shared.py         # error(), no_rem, rem, ld, wd, PP_path, sp, tb_trans_exec,
│                         # B, pdos_exec, gulp_path, gen_basis, mix_ps, joblib_backend, M
├── sip_core.py           # directory/lifecycle: clone, pickle, save_file, _init_directory, ...
├── sip_readers.py        # read_relax, get_potential_energy, get_fermi_level, dQ, read_tbt, ...
├── sip_runners.py        # run_*_in_dir, gin, is_siesta_done, write_analysis_results, ...
├── sip_dftb.py           # dftb_* methods + dftb_charge class
├── sip_fdf.py            # fdf, fdf_relax, write_more_fdf, *_k_points, Add_Charged_*, ...
├── sip_geometry.py       # set_struct, Wrap_unit_cell, Passivate, add_atom, Visualise, ...
├── sip_electrodes.py     # find_elec_inds, copy_DM_from, copy_TSHS_from, is_RSSE, ...
├── sip_sisl_io.py        # to_sisl, read_TSHS, to_TB_model, write_TB_model, ...
├── sip_transport.py      # Real_space_*, projection, calculate_*_RSSE, Hubbard_electrode, ...
├── funcs.py              # unchanged
└── dftbplus_interface.py # unchanged
```

Import mechanics:

- `sip_shared.py` holds the module-level names above, imported by the domain modules and by the
  facade (which re-exports them). Dependency graph stays acyclic:
  `sip_shared` ← `sip_*.py` ← facade. `sip_shared` never imports the facade or a domain module.
- Replace the filename-length hack `sys.path.append(__file__[:-16])` with a single, correct path
  setup (e.g. in `siesta_python/siesta_python/__init__.py`, or switch internal imports to relative
  `from .funcs import …`). Do **not** copy the `[:-16]` trick into new modules — it is tied to
  `siesta_python.py` being exactly 16 characters and computes the wrong path elsewhere.
  Caution: relative imports only work when the package is imported as a package
  (`from siesta_python.siesta_python import SiP`, as the README does). If anything still runs the
  file as a script (`python siesta_python.py`) or relies on the path hack (e.g. `basistest.py`),
  keep a single explicit `sys.path` insert instead of relative imports.
- `wd = os.getcwd()` still evaluates once at import; it moves to `sip_shared` unchanged.
- `if GfModule_extension: from Gf_Module.Gf import read_SE_from_tbtrans` moves with its only user
  (`self_energy_from_tbtrans` / `projection`) into `sip_transport.py`.
- The facade re-exports moved names (`error`, `no_rem`, exec defaults) and moved classes
  (`dftb_charge`, `sisl_replica_for_projection`) so old import paths and old pickles keep working.

---

## 4. Refactoring the `SiP.__init__` input

`__init__` currently takes ~70 keyword arguments with a ~270-line body and uses mutable defaults
(`kp=[1,1,1]`, `elecs=[]`, `buffer_atoms=[]`, `Device_atoms=[]`, `kp_tbtrans=[1,1,1]`).

### 4.1 Target: a config object with a thin kwargs shim

Keep the flat-kwarg call syntax working (external code and this repo call
`SiP(lat, pos, s, basis=…, elecs=…)`), but move the *storage and defaults* into a dataclass, so
there is one source of truth and typo safety:

```python
from dataclasses import dataclass, field, fields

@dataclass
class SiPConfig:
    # --- SCF ---
    xc: str = 'GGA'
    xc_authors: str = 'PBE'
    basis: str = 'SZP'
    mesh_cutoff: float = 150
    # --- runner ---
    siesta_exec: str = sp
    mpi: str = M
    # --- transport ---
    trans_emin: float = -1
    # ... one field per current kwarg ...
    # --- collections: factory, never a bare [] ---
    kp: list = field(default_factory=lambda: [1, 1, 1])
    elecs: list = field(default_factory=list)
    buffer_atoms: list = field(default_factory=list)
    Device_atoms: list = field(default_factory=list)

class SiP:
    def __init__(self, lat, pos, s, directory_name='noname', **overrides):
        unknown = set(overrides) - {f.name for f in fields(SiPConfig)}
        if unknown:
            raise TypeError(f'SiP: unexpected keyword(s) {sorted(unknown)}')
        cfg = SiPConfig(**overrides)
        # flat attributes kept for funcs.py and all existing callers
        for f in fields(cfg):
            setattr(self, f.name, getattr(cfg, f.name))
        self.dir = directory_name
        ...
```

Benefits: one place for defaults, `field(default_factory=…)` kills the shared-mutable-default
footgun, and unknown keywords raise instead of being silently accepted.

Tip: generate the `SiPConfig` field list mechanically from the current signature
(`inspect.signature(SiP.__init__)`), then hand-edit the few that need `default_factory` or a
derived value. Do not retype ~70 fields by hand.

Cost: the constructor no longer advertises its parameters in `help(SiP)` / IDE completion. If that
matters more than grouping, keep the explicit signature (§4.4 option B).

### 4.2 Derived values → `__post_init__` (or explicit helpers)

Compute these in one place instead of scattering them through the 270-line body. Compute them
**from `cfg` fields** (not from `self`), store them back on `cfg`, and let the `setattr` loop copy
them out — that keeps `__post_init__` independent of attribute-assignment order:

- `solution_method` (defaults from `elecs`).
- `Voltage` (from `Chem_Pot`).
- `NEGF_calc` (auto-set when `Chem_Pot` nonzero — note the current `np.array(Chem_Pot) != 0.0` idiom
  forces `True` when `Chem_Pot is None`; fix deliberately, in its own commit).
- `print_no_print` (from `print_console`).
- `mpi + ' '`.

### 4.3 Staged migration (each step independently verifiable)

1. **Extract the body, signature unchanged.** Keep the explicit kwargs; move the block into
   `_init_directory`, `_init_core`, `_init_geometry_attrs` (called in the current order). This
   alone shrinks `__init__` to ~30 lines with **zero** API change.
2. **Introduce `SiPConfig`** and have `__init__` build it from the explicit kwargs, then set flat
   attributes. Signature still explicit — no caller change.
3. **Switch the signature to `**overrides`** and rely on `SiPConfig` for defaults + validation.
   This is the only step that changes `help(SiP)`; land it last so it is easy to revert.

Do **not** skip step 1: it is the safe, no-API-change win and it makes steps 2–3 mechanical.

### 4.4 Alternative (lowest risk, if grouping is not wanted)

Keep the explicit signature and only extract the body into `_init_*` helpers. No config object, no
constructor change. This satisfies "shorter, more readable `__init__`" without touching the call
surface. The config dataclass is optional value-add, not a requirement.

### 4.5 Compatibility invariants (all options)

- Accept the exact same keyword names (no renames).
- Set the exact same flat attributes (`funcs.py` reads `object.lat`, `object.spin_pol`, …).
- `self.initialised_attributes = list(self.__dict__.keys())` must remain present and at the same
  point in the sequence. Under §4.1 its *contents* will change (dataclass field order, not the
  current assignment order) — acceptable, because nothing reads it; just don't delete the line.
- `self.dir` and the directory create/reuse logic stay in `__init__`/core.

---

## 5. Domain breakdown (methods → modules)

Group by where the method's `self.*` reads/writes cluster, not by name. Move the whole method; do
not split a method across modules.

| Module | Representative methods | Attributes it reads/writes |
|---|---|---|
| `sip_core.py` | `clone`, `pickle`, `save_file`, `_init_directory`, `_init_core` | `dir`, `wd`, `sm`, `sl`, `initialised_attributes`, `denchar_files`, `non_ortho` |
| `sip_readers.py` | `read_relax`, `get_potential_energy`, `get_fermi_level`, `relaxed_system_energy`, `dQ`, `read_tbt`, `read_fermi_level_from_out`, `_has_tbt_nc_files`, `mulliken` | `relaxed_pos`, `relaxed_lat`, `Mulliken_C`, `Mulliken_idx` |
| `sip_runners.py` | `run_siesta_in_dir`, `run_siesta_electrode_in_dir`, `run_analyze_in_dir`, `run_tbtrans_in_dir`, `run_tbt_analyze_in_dir`, `run_gulp_in_dir`, `gin`, `run_fatbands`, `plot_calculated_bands`, `is_siesta_done`, `write_analysis_results` | `gulp_results`, exec paths |
| `sip_dftb.py` | `dftb_hsd`, `write_and_run_dftb_in_dir`, `run_dftb_in_dir`, `read_dftb_hamiltonian`, `dftb2sisl`, `fast_dftb_hk`, `dftb_find_equiv`, `dftb_mullikencharge_interface`, `get_potential_energy_dftb` + class `dftb_charge` | `dic['dftb_*']` |
| `sip_fdf.py` | `fdf`, `fdf_relax`, `barebone_fdf`, `write_more_fdf`, `delete_fdf`, `fatband_fdf`, `custom_bandlines`, `manual_k_points`, `make_meshed_k_grid`, `make_concentrated_k_points`, `set_parallel_k`, `get_pseudo_paths`, `*_mixed_pseudo*`, `ps_mixer`, `set_synthetic_mixes`, `E_field`, `set_methfessel_paxton`, `Add_*`, `add_stuff`, `write_tb_trans_kp`, `gen_basis`, `ideal_cell_charge` | `pseudo_paths`, `mixed_pseudo_list`, `_mixing_dic`, `Added_stuff`, `_manual_*` |
| `sip_geometry.py` | `set_struct`, `reararange`, `standardise_cell`, `set_buffer_atoms`, `set_device_atoms`, `make_bandpath`, `Check_Distances`, `cell_*`, `center_cell`, `make_pretty`, `Wrap_unit_cell`, `x_to_z`, `move`, `move_atom`, `add_atom`, `add_buffer`, `pretty_cell`, `toASE`, `read_basis_and_geom`, `ase_visualise`, `Visualise`, `figures`, `find_polygons`, `nnr`, `Find_edges`, `Passivate*`, `tile_dm`, `get_labelled_indices`, `sisl_sub` | `lat`, `pos`, `pos_real_space`, `s`, `sym*`, `k_path`, `UC_idx`, `buffer_atoms`, `Device_atoms`, `TwoDim` |
| `sip_electrodes.py` | `add_elecs`, `find_elec_inds`, `make_device_coords`, `copy_DM_from`, `copy_TSHS_from`, `reset_minimal_hamiltonian`, `undo_RSSE`, `get_RS_pos`, `is_RSSE`, `which_type_SE` | `elecs`, `elec_inds`, `elec_*RSSE`, `semi_inf`, `_rearange_indices`, `RSSEdata` |
| `sip_sisl_io.py` | `to_sisl`, `read_TSHS`, `read_minimal_TSHS`, `read_rssi_surface_TSHS`, `to_TB_model`, `write_TB_model`, `fois_gras`, `manual_H` + class `sisl_replica_for_projection` | — |
| `sip_transport.py` | `Real_space_SE`, `Real_space_SI`, `Real_space_SI_3D`, `self_energy_from_tbtrans`, `tbtrans_H_S_btd_pivot`, `solve_qp_equation`, `projection`, `Hubbard_electrode`, `calculate_hubbard_transport`, `mimic_tbtrans_spin`, `calculate_2E_RSSE`, `calculate_bulksurf_RSSE`, `from_custom`, `get_contour_from_failed_RSSE`, `run_TS_RSSE_calc` | `trans_*`, `Chem_Pot`, `Voltage`, `NEGF_calc`, `save_SE*`, `kp_tbtrans`, `contour_settings`, `RSSE_*`, `SurRSSE_dict`, `tbtrans_params_dic`, `is_hubbard_electrode` |

There is no class to inherit from, so "attribute ownership" is a convention: the module whose
functions primarily read/write an attribute documents it. Keep the attribute split in the table as
the source of truth so two modules don't quietly disagree about a field.

Note: the `dQ` and `Hubbard_electrode` collisions are gone (§0), so they are no longer special
cases — `dQ` moves to `sip_readers.py` and `Hubbard_electrode` to `sip_transport.py` like any other
method.

---

## 6. Landmines / do-not-touch

1. **`initialised_attributes`** (`__init__`) — written, never read. Keep the assignment at the same
   point in the sequence; moving it changes the snapshot contents. Don't "clean it up" mid-move.
2. **`clone` uses `deepcopy(self)`** — extraction doesn't change this, but verify the roundtrip
   after each step.
3. **Pickle** (`pickle`) — the format and the picklable attribute set must not change. Old `.SiP`
   files exist externally (§0) and are read with plain `pickle.load`. If you ever add compression,
   pair it with a magic-byte-sniffing loader and update the ~15 readers (out of scope here).
4. **`isinstance(x, SiP)` / `SiP(...)` inside moved bodies** — resolve with a lazy import; never a
   module-level `from siesta_python.siesta_python import SiP` in a domain module (circular import).
5. **`global SE_func`** in `Real_space_SE` / `Real_space_SI` — keep the block intact; it becomes a
   module global of `sip_transport.py`.
6. **`overwrite=True` default is destructive** (deletes the target directory's contents; `False`
   blocks on `input()`). Pre-existing; flag in a dedicated commit, do not change during the split.
7. **Behaviour changes stay separate.** The `dQ`/`Hubbard_electrode` API changes and the
   `set`/`sleep`/`scream`/`hear`/`load_calculation` removals are already landed — record them as
   intentional in a commit note; the structural extraction must be behaviour-neutral.
8. **`funcs.py` is read-only** for this effort. If a moved body needs a helper, move the helper too;
   do not fork logic.

---

## 7. Migration order (one commit per module)

Step 0 is mandatory before anything moves.

| Step | Action | Why |
|---|---|---|
| 0 | API snapshot script (`sorted(dir(SiP))` + all signatures) + a smoke test with fixtures; commit | Without it you cannot tell a move from a change |
| 1 | Create `sip_shared.py`; fix the `sys.path` hack; facade re-exports shared names | Establishes the acyclic import graph and the stop-the-hack fix |
| 2 | `sip_core.py` (`clone`, `pickle`, `save_file`, `_init_directory`, `_init_core`) | Small, mostly `self.dir`; proves the pattern end to end |
| 3 | `sip_readers.py` | Small, read-only, few cross-calls |
| 4 | `sip_runners.py` | Mostly exec strings |
| 5 | `sip_sisl_io.py` (+ `sisl_replica_for_projection` re-export) | Self-contained I/O |
| 6 | `sip_dftb.py` (+ `dftb_charge` re-export) | Self-contained cluster |
| 7 | `sip_fdf.py` | Large but mostly self-contained |
| 8 | `sip_geometry.py` | Contains `set_struct`, the keystone everything reads |
| 9 | `sip_electrodes.py` | Medium coupling |
| 10 | `sip_transport.py` | Most entangled; `global SE_func`, lazy `SiP` import, conditional `Gf_Module` import |
| 11 | `__init__` refactor: body extraction (§4.3 step 1), verify | No-API-change win |
| 12 | `__init__` refactor: `SiPConfig` (§4.3 steps 2–3), verify | Optional, revertable |

### 7.1 Copy-paste order within each module (easiest first)

Start with methods that make **no** `self.<method>()` calls and touch few attributes — these convert
with a pure `self`→`sip` pass and no reasoning. Candidate openers:

`dQ`, `delete_fdf`, `is_siesta_done`, `get_fermi_level`, `get_potential_energy`, `mulliken`,
`get_RS_pos`, `is_RSSE`, `which_type_SE`, `set_parallel_k`, `E_field`, `set_methfessel_paxton`,
`fatband_fdf`, `fois_gras`, `manual_H`, `read_minimal_TSHS`, `read_rssi_surface_TSHS`, `add_stuff`,
`create_added_stuff_dic`, `create_mixed_pseudo_dic`, `add_mixed_pseudo_dic`, `set_synthetic_mixes`,
`add_elecs`, `plot_calculated_bands`, `gen_basis`, `Check_Distances`.

Leave methods that call many other methods (`fdf`, `run_TS_RSSE_calc`, `projection`,
`calculate_bulksurf_RSSE`, `Real_space_*`) for last within their module. For each method: apply
§2.2, then confirm the §8 checks are green before moving on.

---

## 8. Verification at every step

1. **API-surface diff**: snapshot `sorted(dir(SiP))` and the `inspect.signature` of all 137 methods
   before the refactor; after each step the diff must be empty.
2. **Import check**: `from siesta_python.siesta_python import SiP, dftb_charge,
   sisl_replica_for_projection, rotation_matrix, error, no_rem` all still import.
   (`load_calculation` was removed in §0 — drop it from the old check list.)
3. **Structural lint**: `python3 -m pyflakes siesta_python/siesta_python.py
   siesta_python/siesta_python/sip_*.py` — must show no *undefined names* (unused-locals/imports
   noise is pre-existing; a new one is a signal).
4. **Smoke test** (temp cwd): create a dummy `../pp` with `<Sym>.<xc>.psf` for the species used
   (`get_pseudo_paths` symlinks from it); construct `SiP` on a small cell → `fdf()` → assert
   `RUN.fdf` contents → `make_bandpath()` → `to_sisl()` → `pickle()`/unpickle roundtrip → `clone()`.
   Output must be identical before/after.
5. **External caller check**: grep Zandpack/GrainBoundary for `sip.<method>`; confirm every called
   method still exists (names unchanged by design, but verify).
6. **Constructor call check** (before §4.3 step 3): grep for `SiP(` calls passing more than 3
   positional arguments. In-tree callers all use exactly `(lat, pos, s)` + keywords (verified), but
   confirm external code before switching to `**overrides`.
7. **Parallel-path check** (after moving `Real_space_SE`/`Real_space_SI`): run one `parallel_E=True`
   case — the `global SE_func` joblib path depends on the function's module-qualified name.
8. **End-to-end** (final): one real siesta + tbtrans run through the refactored class.

---

## 9. Explicitly out of scope (future work, separate PRs)

- Renaming attributes/methods to consistent snake_case (breaking change).
- A `SiPState` dataclass for the 167-style attribute set (would touch `funcs.py` and external code).
- Compressing the `.SiP` pickle format (needs the ~15 readers updated; §6.3).
- Changing the destructive `overwrite=True` default.
- Folding `funcs.py`'s free functions into the domain modules (they already take the object; low
  value, do last if ever).
- Removing the wrapper methods so the free functions become the public API (breaking change).
- `wd = os.getcwd()` captured at import time (cwd-dependent global state).
- `error()` uses `assert 1==0`, which is a no-op under `python -O`.
