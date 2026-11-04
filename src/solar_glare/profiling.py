"""Per-stage timing + provenance registry for the glare pipeline.

Every function on the ``run_analysis`` path is registered here with

* ``does``          - one line saying what the function actually computes,
* ``outputs``       - the ``(name, unit, stored_in)`` triples worth recording
  numerically.  ``stored_in`` names the file that ALREADY holds that number;
  when it is non-empty the value is suppressed, because the log exists to
  profile the run, not to restate results.
* ``persisted_in``  - free-text note for stages with nothing left to record
  (why the log shows timing only).

``StageTimer`` measures ``wall_s`` (inclusive) and ``self_s`` (exclusive of
nested stages) per stage.  Accumulators are thread-local, so the sub-panel
loop in ``analysis.analyze_pair`` (a ``ThreadPoolExecutor``) is measured
without a lock and without perturbing thread scheduling.  Nothing here feeds
the physics: profiling is opt-in and bit-identical either way.
"""

import functools
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Phase order used for the stacked-bar panel and the console table.
PHASES = ["config", "site", "solar", "analysis", "io", "figures"]

# Root stage of each phase -> its `wall_s` is the phase's inclusive time.
PHASE_ROOT = {
    "config": "load_config",
    "site": "build_site",
    "solar": "compute_ephemeris",
    "analysis": "run_analysis",
    "io": "write_csvs",
    "figures": "draw_figures",
}

STAGE_SPECS = {
    # ------------------------------------------------------------------ config
    "load_config": {
        "phase": "config",
        "does": "Parse config/site.yaml into plain dicts (site, arrays, "
                "receptors, environment).",
        "outputs": (("n_cases", "cases", ""),),
        "persisted_in": "",
    },
    # -------------------------------------------------------------------- site
    "build_site": {
        "phase": "site",
        "does": "Instantiate PVArray/Receptor objects and place each in the "
                "site ENU frame.",
        "outputs": (("n_arrays", "arrays", ""), ("n_receptors", "receptors", ""),
                    ("n_groups", "sub-panels/array", "")),
        "persisted_in": "",
    },
    "PVArray.setup": {
        "phase": "site",
        "does": "Per array: ENU centre, unit surface normal, characteristic "
                "size, area and hazard_area (the full contiguous sheet).",
        "outputs": (),
        "persisted_in": "config/site.yaml (width_m x height_m)",
    },
    "PVArray._split_groups": {
        "phase": "site",
        "does": "Split the panel height into `groups` equal sub-panels along "
                "the in-plane height axis s = cross(n, e).",
        "outputs": (("subpanel_height", "m", ""), ("subpanel_area", "m2", "")),
        "persisted_in": "",
    },
    "Receptor.setup": {
        "phase": "site",
        "does": "Per receptor: convert lat / lon / AGL height to site ENU.",
        "outputs": (),
        "persisted_in": "config/site.yaml (lat, lon, height_agl_m)",
    },
    # ------------------------------------------------------------------- solar
    "compute_ephemeris": {
        "phase": "solar",
        "does": "Build the year's time base and return sun position + DNI at "
                "the configured resolution.",
        "outputs": (("n_timesteps", "steps", ""),),
        "persisted_in": "",
    },
    "build_times": {
        "phase": "solar",
        "does": "DatetimeIndex for the whole year at site.time_freq "
                "('min' -> 525,600 steps, 'h' -> 8,760).",
        "outputs": (("n_timesteps", "steps", ""), ("time_step", "s", "")),
        "persisted_in": "",
    },
    "solar_position": {
        "phase": "solar",
        "does": "pvlib NREL SPA: apparent solar azimuth / elevation per step "
                "(+/-0.0003 deg).",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv (glare minutes only)",
    },
    "clearsky_dni": {
        "phase": "solar",
        "does": "pvlib Ineichen clear-sky direct normal irradiance per step.",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv (dni_wm2, glare minutes only)",
    },
    # ---------------------------------------------------------------- analysis
    "run_analysis": {
        "phase": "analysis",
        "does": "Drive the full-year analysis over every array x receptor pair "
                "and assemble the summary frames.",
        "outputs": (("n_pairs", "pairs", ""),),
        "persisted_in": "hazard counts -> summary.csv + glare_events.csv",
    },
    "analyze_pair": {
        "phase": "analysis",
        "does": "One (array, receptor) pair: run every sub-panel, then merge "
                "worst-hazard-wins per timestep. THE PARALLEL REGION.",
        "outputs": (("parallel_tasks", "tasks", ""),
                    ("parallel_threads", "threads", ""),
                    ("peak_concurrency", "threads", ""),
                    ("n_task_spans", "spans", ""),
                    ("speedup_vs_serial", "x", "")),
        "persisted_in": "",
    },
    "_analyze_single": {
        "phase": "analysis",
        "does": "One sub-panel vs one receptor over all timesteps: detect the "
                "glare minutes, then E_c, E_r, G/Y/R and IEC 62471.",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv (one row per glare minute)",
    },
    "sun_vector": {
        "phase": "analysis",
        "does": "Unit sun direction in ENU from azimuth + apparent elevation.",
        "outputs": (),
        "persisted_in": "intermediate; no separate output",
    },
    "reflect": {
        "phase": "analysis",
        "does": "Mirror-law reflection r = 2(s.n)n - s (light-travel direction).",
        "outputs": (),
        "persisted_in": "intermediate; no separate output",
    },
    "angle_between": {
        "phase": "analysis",
        "does": "Element-wise angle (rad) between two ENU vector arrays.",
        "outputs": (),
        "persisted_in": "intermediate; no separate output",
    },
    "beam_spread_angle": {
        "phase": "analysis",
        "does": "beta = theta_sun + 12*sigma, the reflected beam's full angular "
                "spread (SGHAT v6 Sec 4.2).",
        "outputs": (),
        "persisted_in": "intermediate; no separate output",
    },
    "panel_corners_enu": {
        "phase": "analysis",
        "does": "The four ENU corners of a rectangular sub-panel.",
        "outputs": (),
        "persisted_in": "intermediate; no separate output",
    },
    "glare_polygon_hit": {
        "phase": "analysis",
        "does": "True-rectangle polyhedral-emanation test: is the sun inside "
                "the spherical polygon of reflected corner directions?",
        "outputs": (),
        "persisted_in": "glare_events.csv + glare_hours_*.csv",
    },
    "effective_subtended_angle": {
        "phase": "analysis",
        "does": "Subtended source angle x = min(beta, theta) from hazard_area "
                "(SGHAT v6 Sec 5.5).",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv (subtended_rad)",
    },
    "solar_weighted_reflectivity": {
        "phase": "analysis",
        "does": "ASTM E903-20-style solar-weighted spectral Fresnel "
                "reflectance of the cover glass at the per-minute AOI.",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv (reflectivity_used, aoi_deg)",
    },
    "atmospheric_transmittance": {
        "phase": "analysis",
        "does": "Beer-Lambert transmittance tau = exp(-sigma*d) with the "
                "Kruse-Kim extinction coefficient for the visibility.",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv (atmos_transmittance)",
    },
    "corneal_irradiance": {
        "phase": "analysis",
        "does": "Corneal irradiance E_c = DNI * rho / 1e4 (W/cm2), then "
                "scaled by tau.",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv (corneal_wcm2)",
    },
    "retinal_irradiance": {
        "phase": "analysis",
        "does": "Retinal irradiance E_r = E_c * k_f / x^2 (W/cm2).",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv (retinal_wcm2)",
    },
    "classify": {
        "phase": "analysis",
        "does": "Sandia / SGHAT ocular band: Green / Yellow / Red against the "
                "flash + burn thresholds.",
        "outputs": (),
        "persisted_in": "glare_events.csv + glare_hours_*.csv (hazard)",
    },
    "evaluate_glare": {
        "phase": "analysis",
        "does": "IEC 62471 / BS EN 62471 assessment from the same E_c / E_r: "
                "alpha, L_R, L_B, t_max, risk group.",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv (iec_*) + iec62471_summary.csv",
    },
    "_rank_codes": {
        "phase": "analysis",
        "does": "Map the N/G/Y/R letters to integer ranks for the max merge.",
        "outputs": (),
        "persisted_in": "intermediate; no separate output",
    },
    "_codes_from_rank": {
        "phase": "analysis",
        "does": "Inverse of _rank_codes: integer ranks back to N/G/Y/R.",
        "outputs": (),
        "persisted_in": "intermediate; no separate output",
    },
    "_dedupe_detail": {
        "phase": "analysis",
        "does": "Collapse the sub-panel detail frames to one row per glare "
                "minute (worst hazard, tie-broken by max retinal).",
        "outputs": (),
        "persisted_in": "glare_hours_*.csv",
    },
    # ---------------------------------------------------------------------- io
    "write_csvs": {
        "phase": "io",
        "does": "Write summary.csv, glare_events.csv, the per-pair "
                "glare_hours_*.csv and iec62471_summary.csv.",
        "outputs": (("n_files_written", "files", ""),),
        "persisted_in": "the row values live in the CSVs themselves",
    },
    # ------------------------------------------------------------------ figures
    "draw_figures": {
        "phase": "figures",
        "does": "Render every report figure for this case.",
        "outputs": (("n_figures_written", "files", ""),),
        "persisted_in": "",
    },
    "plot_site_map": {
        "phase": "figures",
        "does": "Reference site map: array, receptor routes, terminals, "
                "observation points (white + transparent variants).",
        "outputs": (("n_files_written", "files", ""),),
        "persisted_in": "geometry is in the PNGs",
    },
    "plot_site_map_hazard": {
        "phase": "figures",
        "does": "Site map annotated with each receptor's annual glare minutes.",
        "outputs": (),
        "persisted_in": "the annotation is in the PNG",
    },
    "plot_group_glare_map": {
        "phase": "figures",
        "does": "Per-receptor map of which sub-panels actually produce the "
                "glare, read from the worst-wins `group` column.",
        "outputs": (("n_figures_written", "files", ""),),
        "persisted_in": "the attribution is in the PNGs",
    },
    "plot_site_map_folium": {
        "phase": "figures",
        "does": "Interactive folium map (needs internet for the tile feed).",
        "outputs": (),
        "persisted_in": "the map is the HTML",
    },
    "plot_sun_path": {
        "phase": "figures",
        "does": "Annual sun-path polar chart for the site.",
        "outputs": (),
        "persisted_in": "the path is in the PNG",
    },
    "plot_glare_heatmap": {
        "phase": "figures",
        "does": "Per-pair 30-minute glare-occurrence heatmap, coloured by IEC "
                "62471 risk group.",
        "outputs": (("n_figures_written", "files", ""),),
        "persisted_in": "the occurrence data is in the PNGs",
    },
    "plot_daily_duration": {
        "phase": "figures",
        "does": "Per-pair daily glare duration (min/day) stacked by risk group.",
        "outputs": (),
        "persisted_in": "the durations are in the PNGs",
    },
    "plot_summary": {
        "phase": "figures",
        "does": "Annual glare hours per receptor, stacked by IEC risk group.",
        "outputs": (),
        "persisted_in": "the hours are in the PNG",
    },
    "plot_ocular_chart": {
        "phase": "figures",
        "does": "SGHAT ocular chart: every plotted minute inside its G/Y/R "
                "band, E_c against subtended angle x.",
        "outputs": (),
        "persisted_in": "the scatter is in the PNG",
    },
    "plot_sun_chart": {
        "phase": "figures",
        "does": "Glare minutes by sun position, coloured by risk group.",
        "outputs": (),
        "persisted_in": "the scatter is in the PNG",
    },
    "plot_iec62471_zones": {
        "phase": "figures",
        "does": "IEC 62471 emissive-surface radius / zone chart.",
        "outputs": (),
        "persisted_in": "the zones are in the PNG",
    },
    "plot_iec62471_tmax": {
        "phase": "figures",
        "does": "Allowed exposure duration t_max, one chart per hazard clause "
                "(retinal thermal and blue light).",
        "outputs": (("n_figures_written", "files", ""),),
        "persisted_in": "the curves are in the PNGs",
    },
    "plot_iec62471_weights": {
        "phase": "figures",
        "does": "IEC 62471 Figure 4.2 spectral weighting functions "
                "B(lam) and R(lam).",
        "outputs": (),
        "persisted_in": "the curves are in the PNG",
    },
    "plot_iec62471_spectrum_weighted": {
        "phase": "figures",
        "does": "IEC 60904-3:2019 direct spectrum before and after the "
                "B(lam) / R(lam) weighting.",
        "outputs": (),
        "persisted_in": "the spectra are in the PNG",
    },
    "plot_3d_geometry": {
        "phase": "figures",
        "does": "3D view of the worst-case glare geometry for each receptor "
                "that has glare.",
        "outputs": (("n_figures_written", "files", ""),),
        "persisted_in": "the geometry is in the PNGs",
    },
}

_FALLBACK_SPEC = {"phase": "analysis", "does": "", "outputs": (),
                  "persisted_in": "unregistered stage"}


def _spec(stage):
    return STAGE_SPECS.get(stage, _FALLBACK_SPEC)


class _NullHandle:
    """Stand-in handle yielded by ``_NULL``: absorbs ``key()`` / ``parallel()``
    so instrumented call sites need no `if profiling_enabled` branches."""

    __slots__ = ()

    def key(self, **kw):
        return self

    def parallel(self, tasks=None, workers=None, peak=None):
        return self


_NULL_HANDLE = _NullHandle()


class _NullCtx:
    """Zero-cost stand-in used when profiling is disabled."""

    __slots__ = ()

    def __enter__(self):
        return _NULL_HANDLE

    def __exit__(self, *exc):
        return False


_NULL = _NullCtx()

# ``stage`` keeps the live handle on a thread-local slot so ``__exit__`` can read
# the outputs declared inside the ``with`` body.
_HANDLE_SLOT = "_last_handle"


@dataclass
class _Handle:
    """Per-call handle yielded by ``StageTimer.stage``."""

    timer: "StageTimer"
    stage: str
    keys: dict = field(default_factory=dict)
    parallel_tasks: object = None
    parallel_workers: object = None
    peak_concurrency: object = None

    def key(self, **kw):
        """Declare a numeric output. The first non-None value per name wins, so
        a stage called N times reports its first call's value."""
        for k, v in kw.items():
            if v is not None and k not in self.keys:
                self.keys[k] = v
        return self

    def parallel(self, tasks=None, workers=None, peak=None):
        self.parallel_tasks = tasks
        self.parallel_workers = workers
        self.peak_concurrency = peak
        return self


class _Frame:
    """Per-thread nesting frame: accumulates the wall time of child stages so
    the parent's `self_s` can be made exclusive.

    ``foreign_s`` is the subset of `child_s` spent in children belonging to a
    DIFFERENT phase.  It is what lets `phase_frame` report non-overlapping
    phase times: the pipeline reuses `compute_ephemeris` both standalone (to
    draw the sun path) and inside `run_analysis`, so the solar phase root is
    for part of its life a child of the analysis phase root, and adding the two
    root wall times would count those seconds twice.

    ``sid`` is this frame's id in the call timeline, allocated on entry so a
    child can point at its parent without a second pass.  ``None`` when span
    recording is off.
    """

    __slots__ = ("child_s", "foreign_s", "stage", "sid")

    def __init__(self, stage=None, sid=None):
        self.child_s = 0.0
        self.foreign_s = 0.0
        self.stage = stage
        self.sid = sid


class _StageCtx:
    """Context manager returned by ``StageTimer.stage``. An explicit class
    rather than @contextmanager, so the disabled path can hand back a shared
    no-op and the enabled path stays a plain (fast) enter/exit pair."""

    __slots__ = ("_timer", "_stage", "_rec", "_handle", "_stack", "_slot",
                 "_prev", "_t0", "_frame", "_sid", "_pid", "_depth", "_tid")
    def __init__(self, timer, stage):
        self._timer = timer
        self._stage = stage
        self._rec = timer._rec(stage)
        self._handle = _Handle(timer, stage)
        self._stack = None
        self._slot = None
        self._prev = None
        self._t0 = 0.0
        self._frame = None
        self._sid = -1
        self._pid = -1
        self._depth = 0
        self._tid = 0

    def __enter__(self):
        tl = self._timer._tl
        stack = getattr(tl, "stack", None)
        if stack is None:
            stack = tl.stack = []
        slot = getattr(tl, _HANDLE_SLOT, None)
        if slot is None:
            slot = tl._last_handle = {}
        self._stack = stack
        self._slot = slot
        self._prev = slot.get(self._stage)
        slot[self._stage] = self._handle
        self._t0 = time.perf_counter()
        if self._timer._rec_spans:
            # Timeline bookkeeping: depth is the stack depth BEFORE this frame
            # is pushed, and the parent is whoever is on top of it.  The token
            # is cached per thread, so the steady-state cost is one getattr.
            self._depth = len(stack)
            self._pid = stack[-1].sid if stack else -1
            tok = getattr(tl, "tok", None)
            if tok is None:
                tok = tl.tok = self._timer._token()
            self._tid = tok
            self._sid = self._timer._next_sid()
            self._frame = _Frame(stage=self._stage, sid=self._sid)
        else:
            self._frame = _Frame(stage=self._stage)
        stack.append(self._frame)
        return self._handle

    def __exit__(self, *exc):
        dt = time.perf_counter() - self._t0
        stack = self._stack
        popped = stack.pop()                    # drop this frame
        if stack:
            parent = stack[-1]
            parent.child_s += dt
            # time hosted by a parent of a different phase, so that phase_frame
            # can subtract it and keep the phase totals disjoint
            if parent.stage is not None and \
                    _spec(parent.stage)["phase"] != _spec(self._stage)["phase"]:
                parent.foreign_s += dt
        rec = self._rec
        rec["calls"] += 1
        rec["wall_s"] += dt
        rec["self_s"] += dt - popped.child_s
        rec["foreign_s"] += popped.foreign_s
        for k, v in self._handle.keys.items():
            if k not in rec["keys"]:
                rec["keys"][k] = v
        # parallel metadata: first non-None wins, except peak concurrency, where
        # the maximum over all calls is the informative figure (the first pair
        # may not saturate the pool).
        for attr in ("parallel_tasks", "parallel_workers"):
            val = getattr(self._handle, attr)
            if val is not None and rec[attr] is None:
                rec[attr] = val
        peak = self._handle.peak_concurrency
        if peak is not None:
            prev = rec["peak_concurrency"]
            rec["peak_concurrency"] = peak if prev is None else max(prev, peak)
        if self._prev is None:
            self._slot.pop(self._stage, None)
        else:
            self._slot[self._stage] = self._prev
        if self._timer._rec_spans:
            # Times are stored relative to the timer origin, so the exported
            # frame is directly plottable and never leaks a perf_counter value.
            self._timer._call_spans.append(
                (self._sid, self._pid, self._stage,
                 self._t0 - self._timer.t0, self._t0 - self._timer.t0 + dt,
                 dt - popped.child_s, self._tid, self._depth))
        return False


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):d}"
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) >= 1e-4:
            return f"{v:.4g}"
        return f"{v:.3e}"
    return str(v)


class StageTimer:
    """Accumulates per-stage wall / self time, call counts, declared outputs,
    the sub-panel task spans of the one parallel region, and (optionally) a
    full per-call span timeline.

    ``enabled=False`` makes ``stage()`` return a shared null context, so the
    instrumented call sites cost ~0.2 us each when profiling is off.  The
    validators, which assert bit-identical results, run that way.

    Two span families are kept, deliberately separate:

    ``_spans``       task spans of the parallel region only (thread, t0, t1,
                     sub-panel index) - what the static lane chart needs.
    ``_call_spans``  every stage entry/exit, with parent link, nesting depth
                     and thread - the event timeline behind
                     ``output/run_timeline.html``.  ~14,000 per case.
    """

    def __init__(self, case_label="", enabled=True, record_spans=True):
        self.enabled = bool(enabled)
        self.case_label = case_label
        self.record_spans = bool(record_spans)
        self.t0 = time.perf_counter()
        self._t_stop = None
        self._recs = {}
        self._order = []
        self._tl = threading.local()
        self._spans = []
        # Per-call timeline.  ``_rec_spans`` is precomputed so the hot
        # enter/exit pair tests one attribute instead of three conditions.
        self._rec_spans = bool(self.enabled and self.record_spans)
        self._call_spans = []
        self._sid = 0
        # Stable per-thread token.  `threading.get_ident()` is NOT safe here: a
        # run creates one ThreadPoolExecutor per (array, receptor) pair, so
        # ~16 threads per pair live and die, and CPython recycles the ident of
        # a dead thread - which silently merged two different workers' spans
        # into one lane.  Holding a strong reference to the Thread object makes
        # `id()` unforgeable for the lifetime of the run.
        self._thread_refs = {}
        self.main_tid = self._token()

    def _token(self):
        """Identity of the calling thread that cannot be recycled."""
        th = threading.current_thread()
        key = id(th)
        self._thread_refs[key] = th        # strong ref: keeps `key` unique
        return key

    def _next_sid(self):
        self._sid += 1
        return self._sid - 1

    # ------------------------------------------------------------------- core
    def _rec(self, stage):
        rec = self._recs.get(stage)
        if rec is None:
            rec = {"calls": 0, "self_s": 0.0, "wall_s": 0.0, "keys": {},
                   "parallel_tasks": None, "parallel_workers": None,
                   "peak_concurrency": None, "foreign_s": 0.0}
            self._recs[stage] = rec
            self._order.append(stage)
        return rec

    def stage(self, name):
        """Context manager around one pipeline stage; yields a ``_Handle``."""
        if not self.enabled:
            return _NULL
        return _StageCtx(self, name)

    def task_span(self, thread, t0, t1, index):
        """Record one unit of parallel work (a sub-panel run) for the lane
        chart. ~1,200 tiny structs per case."""
        if not (self.enabled and self.record_spans):
            return
        self._spans.append((thread, float(t0), float(t1), int(index)))

    # --------------------------------------------------------------- reporting
    @property
    def total_wall_s(self):
        if self._t_stop is not None:
            return self._t_stop - self.t0
        return time.perf_counter() - self.t0

    def stop(self):
        """Freeze ``total_wall_s`` at the current instant.

        Used for stages that are paid once for the whole sweep (config parsing):
        without this the timer would keep counting for the rest of the process
        and report the whole script's wall time as its own."""
        if self._t_stop is None:
            self._t_stop = time.perf_counter()
        return self._t_stop - self.t0

    def span_frame(self):
        return pd.DataFrame(self._spans,
                            columns=["thread", "t0", "t1", "index"])

    # The call timeline is stored as a flat list of tuples, one per stage
    # entry/exit, in COMPLETION order (a parallel run interleaves).  Times are
    # already relative to `t0`, so the frame needs no post-hoc shifting.
    CALL_SPAN_COLUMNS = ["span_id", "parent_id", "stage", "t0", "t1",
                         "self_s", "thread", "depth"]

    def call_span_frame(self):
        """One row per instrumented stage call: the event timeline."""
        return pd.DataFrame.from_records(
            self._call_spans, columns=self.CALL_SPAN_COLUMNS)

    def call_span_count(self):
        return len(self._call_spans)

    def thread_labels(self):
        """Map thread tokens to dense lane labels, one label per CONCURRENT
        worker rather than per thread object.

        The pipeline builds a fresh ``ThreadPoolExecutor`` per (array,
        receptor) pair, so a case sees ~65 generations of worker threads that
        are created and joined one pair at a time.  Lanes are therefore
        allocated from a free list keyed on liveness: a thread that is seen
        while another is still inside its span window takes a free lane, and a
        lane returns to the pool once every thread holding it has finished.
        Generations then reuse the same 1..N rows, which is what makes a
        thread-lane timeline readable - and it is also the honest picture, the
        run really does have at most ``n_workers`` workers in flight.
        """
        spans = {}
        for span in self._call_spans:
            tok = span[6]
            win = spans.get(tok)
            if win is None:
                spans[tok] = [span[3], span[4]]
            else:
                if span[3] < win[0]:
                    win[0] = span[3]
                if span[4] > win[1]:
                    win[1] = span[4]
        order = sorted(spans, key=lambda t: spans[t][0])
        main = self.main_tid
        # `lane_of` is permanent (every thread keeps its label); `live` holds
        # the tokens whose window is still open, and `free` the lane indices
        # they released.  Retiring a thread must NOT drop it from `lane_of` -
        # the export looks every token up again.
        lane_of, live, free, n_lanes = {}, [], [], 0
        for tok in order:
            t0 = spans[tok][0]
            if tok == main:
                lane_of[tok] = -1                  # the driver is its own row
                continue
            if live:
                still = []
                for other in live:
                    if spans[other][1] < t0:
                        free.append(lane_of[other])
                    else:
                        still.append(other)
                live = still
            if free:
                lane = min(free)
                free.remove(lane)
            else:
                lane = n_lanes
                n_lanes += 1
            lane_of[tok] = lane
            live.append(tok)
        labels = {}
        for tok, lane in lane_of.items():
            labels[tok] = "main" if lane < 0 else "W%d" % (lane + 1)
        return labels

    def n_thread_lanes(self):
        return len({v for v in self.thread_labels().values() if v != "main"})

    def _key_text(self, rec, stage):
        """Render the outputs that are not already stored anywhere."""
        spec = _spec(stage)
        parts = []
        for name, unit, stored_in in spec["outputs"]:
            if not stored_in and name in rec["keys"]:
                parts.append(f"{name}={_fmt(rec['keys'][name])} {unit}")
        return "; ".join(parts)

    def to_frame(self):
        total_self = sum(r["self_s"] for r in self._recs.values())
        rows = []
        for stage in self._order:
            rec = self._recs[stage]
            spec = _spec(stage)
            rows.append({
                "phase": spec["phase"],
                "stage": stage,
                "does": spec["does"],
                "calls": int(rec["calls"]),
                "self_s": round(rec["self_s"], 6),
                "wall_s": round(rec["wall_s"], 6),
                "mean_ms": round(1e3 * rec["self_s"] / rec["calls"], 4)
                          if rec["calls"] else 0.0,
                "pct_self": round(100.0 * rec["self_s"] / total_self, 3)
                            if total_self else 0.0,
                "parallel_tasks": rec["parallel_tasks"],
                "parallel_workers": rec["parallel_workers"],
                "peak_concurrency": rec["peak_concurrency"],
                "key_outputs": self._key_text(rec, stage),
                "persisted_in": spec["persisted_in"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            order = {p: i for i, p in enumerate(PHASES)}
            df["_o"] = df["phase"].map(lambda p: order.get(p, len(order)))
            df = (df.sort_values(["_o", "self_s"], ascending=[True, False])
                    .drop(columns="_o").reset_index(drop=True))
        return df

    def phase_frame(self):
        """Disjoint wall time per phase, taken from the phase's root stage.

        The root's wall time has the time it spent hosting a *different* phase
        subtracted, because a phase root can legitimately sit inside another
        phase root (compute_ephemeris runs both standalone and inside
        run_analysis).  The result is a true partition: the phases sum to at
        most the instrumented wall time, with no double counting.
        """
        total = 0.0
        rows = []
        for phase in PHASES:
            root = PHASE_ROOT.get(phase)
            stages = [s for s in self._order if _spec(s)["phase"] == phase]
            if not stages:
                continue
            if root in self._recs:
                r = self._recs[root]
                wall = r["wall_s"] - r["foreign_s"]
                root_name = root
            else:
                wall = sum(self._recs[s]["self_s"] for s in stages)
                root_name = "(sum of stages)"
            total += wall
            rows.append({"phase": phase, "root_stage": root_name,
                         "n_stages": len(stages), "wall_s": round(wall, 6)})
        df = pd.DataFrame(rows)
        if not df.empty and total > 0:
            df["pct"] = (100.0 * df["wall_s"] / total).round(2)
        return df

    def format_table(self, key_width=52):
        """Compact console table (one row per stage, exclusive self time)."""
        df = self.to_frame()
        if df.empty:
            return "(no stages recorded)"
        head = (f"{'phase':<9} {'stage':<24} {'calls':>7} {'self_s':>9} "
                f"{'pct':>6}  key output (value unit)")
        lines = [head, "-" * len(head)]
        for r in df.itertuples():
            key = r.key_outputs
            if not key and r.persisted_in:
                key = f"[already stored: {r.persisted_in}]"
            if len(key) > key_width:
                key = key[:key_width - 3] + "..."
            lines.append(f"{r.phase:<9} {r.stage:<24} {r.calls:>7} "
                         f"{r.self_s:>9.3f} {r.pct_self:>5.1f}%  {key}")
        return "\n".join(lines)


class _StageTimerProxy:
    """Module-level handle used by the instrumented call sites. Defaults to
    disabled, so importing the pipeline never starts profiling."""

    def __init__(self):
        self._timer = None

    def set(self, timer):
        self._timer = timer

    def get(self):
        return self._timer

    @property
    def enabled(self):
        return self._timer is not None and self._timer.enabled

    def stage(self, name):
        if self._timer is None:
            return _NULL
        return self._timer.stage(name)

    def task_span(self, thread, t0, t1, index):
        if self._timer is not None:
            self._timer.task_span(thread, t0, t1, index)


TIMER = _StageTimerProxy()


def set_timer(timer):
    """Install (or clear, with ``None``) the process-wide stage timer."""
    TIMER.set(timer)
    return timer


def stage_fn(name):
    """Decorator: run the wrapped function inside the named stage.

    The ``_Handle`` is not exposed, so decorate only stages that declare no
    numeric output.  Orchestration stages that report values (build_site,
    compute_ephemeris, analyze_pair, run_analysis) use ``with TIMER.stage(...)``
    directly so they can attach keys to the handle.
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with TIMER.stage(name):
                return fn(*args, **kwargs)
        return wrapper
    return decorate
