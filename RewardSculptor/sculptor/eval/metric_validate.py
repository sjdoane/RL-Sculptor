"""Validation gates for AUTO-GENERATED objective metrics (§Ship 35).

An LLM-authored metric is untrusted code that will score rollouts. Before
it may be used at all (even observe-only), it must clear the MUST-HAVE
gates here; before it may STEER a run it must additionally pass
calibration (see metric_calibration). These gates implement the red-team's
non-negotiable list:

  1. AST safety         — no imports except numpy; no exec/eval/open/dunder
  2. Array-contract     — references only persisted physical arrays
  3. Determinism        — identical output on 3 repeated runs
  4. Bounded [0,1]      — finite, in-range, never raises on diverse inputs
  5. Non-degeneracy     — discriminates: a dead-still / fallen policy must
                          score BELOW an active-upright one, with spread

These are a SMELL TEST, not a proof of task-validity — that comes from
calibration against the hand-authored ground-truth metrics. The metric is
constrained to physical quantities (the whole point: an objective signal,
not LLM judgment).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from sculptor.eval.generated_metric import (
    ALLOWED_ARRAYS,
    GENERATED_FN_NAME,
    inject_joint_roles,
    load_generated_module,
    read_required_roles,
)
from sculptor.eval.joint_resolver import resolve_joint_roles
from sculptor.world.channels import (
    ChannelCatalog,
    catalog_fixture_arrays,
    resolve_channel_catalog,
)

#: Modules a generated metric may import (numpy only — it is a pure
#: physical-quantity function).
_ALLOWED_IMPORTS = {"numpy"}
#: numpy SUBMODULES a metric may import (the rest — ctypeslib/f2py/distutils/testing —
#: are native-code / RCE surfaces; the root-only import gate used to let them through).
_SAFE_NUMPY_SUBMODULES = frozenset({"linalg", "random"})
#: Names that must never appear (code-exec / IO / introspection vectors).
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "Path",
    "importlib", "pickle", "marshal", "ctypes", "memoryview",
    # §round-24: a metric that `raise SystemExit()` (or KeyboardInterrupt/GeneratorExit, all
    # BaseException — NOT Exception) crashes the never-raises calibration/runtime scorers that
    # only `except Exception`. None is legitimate in a physical-quantity metric (which uses
    # `except Exception`). exit/quit are the same vector as bare builtins.
    "SystemExit", "KeyboardInterrupt", "GeneratorExit", "BaseException", "exit", "quit",
    # §round-32 SECURITY (CRITICAL): the interactive site builtins are import/exec FRONT-ENDS.
    # `help(name)` → `_sitebuiltins.Helper` → `pydoc.render_doc` → `importlib.import_module(name)`,
    # which RUNS the named module's top-level code (arbitrary import + RCE). The module name can be
    # assembled with `chr(...)` so the round-19 `__`-in-string scan never sees it, and `help` is a
    # bare builtin Name with no gi_/cr_/ag_/f_/tb_/func_ prefix, no dunder, no `.format` — it slipped
    # every prior gate. copyright/license/credits are the same `_sitebuiltins._Printer` family; dir/
    # hasattr are reflection primitives (siblings of the already-blocked getattr/vars). None appears
    # in a legitimate physical-quantity metric. (Durable close: the designed subprocess sandbox.)
    "help", "copyright", "license", "credits", "dir", "hasattr",
}
#: §round-9 SECURITY: numpy is itself a full IO / native-code / reflection surface, so the
#: numpy-only import gate is NOT a sandbox on its own. These attribute names (numpy fns,
#: ndarray methods, and numpy submodules reachable as attributes) are arbitrary file
#: read/write, native-code, or pickle-RCE primitives and are DENIED as attribute access
#: AND as `from numpy import <name>` bindings. None appears in a legitimate physical-
#: quantity metric. (Full containment still wants a real process sandbox — see the
#: module docstring follow-on — but this closes every escape round-9 reproduced.)
_FORBIDDEN_ATTRS = frozenset({
    # file / buffer IO + persistence (np.save(dtype=object)+np.load(allow_pickle) = RCE)
    "save", "savez", "savez_compressed", "savetxt", "load", "loadtxt", "genfromtxt",
    "fromfile", "tofile", "memmap", "fromregex", "datasource", "DataSource",
    "fromstring", "frombuffer", "getbuffer", "newbuffer",
    # §round-19 SECURITY: ndarray.dump(path) PICKLES the array to an arbitrary filesystem
    # path (arbitrary write/overwrite), and ndarray.dumps()/tobytes()/tostring() serialize
    # in-process (pickle/raw-bytes exfil surfaces). The round-9 gate blocked tofile/savetxt
    # but the literal-sibling dump/dumps were uncovered → a metric escaped the IO gate during
    # validation. None appears in a legitimate physical-quantity metric.
    "dump", "dumps", "tobytes", "tostring",
    # native-code / reflection submodules reachable as `numpy.<x>` attributes
    "ctypeslib", "f2py", "distutils", "testing", "lib", "core", "_core", "ctypes",
    # §round-24 SECURITY (CRITICAL): numpy RE-EXPORTS whole stdlib modules under PUBLIC,
    # non-underscore names through its submodules — e.g. np.ma.extras.ma IS numpy.ma.core,
    # which re-exports inspect/io/builtins/operator. The round-23 single-underscore gate was
    # bypassed by np.ma.extras.ma.inspect.dis.io.FileIO (arbitrary write → RCE),
    # np.ma…inspect.linecache.getline (read/exfil), np.ma…builtins.exit (SystemExit). A
    # leaf-name denylist over numpy's full alias surface cannot be PROVEN complete (the
    # durable fix is a restricted subprocess with curated __builtins__ — see the module
    # docstring), but these names are checked on EVERY attribute regardless of how the chain
    # is rooted (so variable-laundering `m = np.ma; m.extras…` cannot evade the NAME check),
    # which closes the entry submodules + the stdlib modules + the dangerous leaves. None
    # appears in a legitimate physical-quantity metric.
    #   numpy stdlib-re-exporting / object (non-physical) submodules (entry points). NOT
    #   fft/polynomial (legit compute a physical metric may use — any re-export through them
    #   still has to NAME a blocked stdlib module below to do harm):
    "ma", "char", "rec", "records", "chararray", "mrecords", "matlib", "emath",
    "extras", "typing", "dtypes", "exceptions", "strings", "version", "matrixlib",
    "polyutils", "compat", "ufunclike",
    #   ROBUST CORE — stdlib module names: any re-export chain must NAME a stdlib module to
    #   reach FS/process/reflection, and these are checked on EVERY attribute regardless of
    #   root, so the chain dies at the stdlib hop even through an unlisted numpy submodule:
    "inspect", "dis", "io", "builtins", "linecache", "operator", "enum", "re",
    "functools", "itertools", "collections", "tempfile", "runpy", "platform", "gc",
    "ast", "types", "warnings", "traceback", "code", "codeop", "pdb", "bdb",
    "shelve", "sqlite3", "threading", "multiprocessing", "asyncio", "signal",
    "atexit", "weakref", "copyreg", "gettext", "locale", "glob", "fnmatch",
    "posix", "nt", "posixpath", "ntpath", "genericpath", "fileinput", "webbrowser",
    "pty", "ssl", "http", "urllib", "ftplib", "smtplib", "json", "base64",
    "binascii", "zlib", "gzip", "bz2", "lzma", "tarfile", "zipfile", "struct",
    "mmap", "fcntl", "termios", "stat", "errno", "tokenize", "keyword",
    # NOTE: 'select' is deliberately NOT blocked — np.select is a benign public numpy
    # piecewise function; the stdlib select module needs an import (already gated).
    #   dangerous terminal callables reachable off a module object:
    "FileIO", "getline", "getlines", "exit", "quit", "system", "popen", "spawn",
    "fork", "execv", "execve", "execvp", "fdopen", "getattr_static", "find_module",
    "import_module", "reload", "getsource", "getsourcefile", "getfile",
    # §round-32 SECURITY (CRITICAL): numpy's PUBLIC introspection helper np.info(<str>,
    # toplevel=<name>) routes the caller-controlled name into numpy.lib._utils_impl._makenamedict,
    # which calls `__import__(name)` → RUNS the named module's top-level code (RCE), proven through
    # the live load_generated_module gate with _ast_safety()==[] (no blocked NAME ever appears — the
    # import target is a plain runtime str the AST can't see). np.test runs the pytest suite (code
    # exec). source/lookfor/who/deprecate/safe_eval are the historical numpy introspection/eval
    # helpers (absent in numpy 2.4 but denied defensively). None appears in a legitimate
    # physical-quantity metric. (The leaf-name denylist over numpy's public surface cannot be proven
    # complete — fresh empirical support for the designed subprocess sandbox.)
    "info", "test", "source", "lookfor", "who", "deprecate", "safe_eval",
})
#: §round-28 SECURITY (CRITICAL): frame / generator / coroutine / traceback INTROSPECTION
#: attribute prefixes. These reach a live execution frame's namespaces WITHOUT any dunder or
#: single-underscore token, so the round-23/24 gates never saw them:
#:   (i for i in ()).gi_frame.f_builtins["<assembled '__import__'>"]("os").system(...)
#: `gi_frame` exposes the generator's frame; `f_builtins`/`f_globals`/`f_locals` ARE the live
#: builtins/module dicts (every builtin incl. __import__/open/eval reachable as a dict subscript,
#: which no AST string/name check can see); `f_back` walks to the CALLER's frame (the scorer's
#: own globals). Generators (`gi_*`), coroutines (`cr_*`), async-gens (`ag_*`), frames (`f_*`),
#: and tracebacks (`tb_*`) are the full set of objects carrying these. NONE is ever part of a
#: physical-quantity numpy metric (numpy public attrs are `flags`/`flat`/`shape`/… — none match
#: a `<prefix>_` form), so a PREFIX deny closes the whole family rather than enumerating leaves
#: (which round-25 showed cannot be proven complete). The durable containment is still the
#: restricted subprocess (curated `__builtins__`) — see the generated_metric module docstring;
#: this attribute-graph denylist, like the numpy one, cannot be PROVEN complete.
#: §round-29 SECURITY (CRITICAL): numpy's CYTHON callables (e.g. np.random.seed) expose the
#: Python-2-era function aliases `func_globals` / `func_code` / `func_defaults` / `func_closure`
#: — `func_globals` IS the module-globals namespace (its `__builtins__` reaches `__import__`), and
#: `func_code` + `types.FunctionType`/`code.replace` rebuilds a callable that resolves the import at
#: the C level with NO python-visible dunder attribute. `func_` does NOT start with `f_` (the 2nd
#: char is `u`, not `_`), so the round-28 prefix list missed it. No public numpy attribute begins
#: with `func_`, so the prefix deny is non-colliding. (A plain python function uses the dunder
#: `__code__`/`__globals__`, already blocked; only cython callables carry the bare `func_` aliases.)
_FRAME_INTROSPECTION_PREFIXES = ("gi_", "cr_", "ag_", "f_", "tb_", "func_")
#: Benign dunder ATTRIBUTES a metric may read. `type(e).__name__` — naming an
#: exception class in a diagnostic — is the exact idiom the never-raise rule (a
#: try/except wrapper) encourages, and the model emits it constantly; a name STRING
#: carries no escape vector. The class-traversal dunders (__class__, __bases__,
#: __subclasses__, __globals__, __dict__, __getattribute__, __code__, …) stay
#: blocked, and `getattr` is forbidden, so `__name__` cannot be chained to anything
#: dangerous. (Pre-fix: `type(e).__name__` false-rejected ~most candidates.)
_ALLOWED_DUNDER_ATTRS = {"__name__"}

T, E, J = 120, 4, 12
_NAMES_12 = [
    "left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_shoulder_pitch", "right_shoulder_pitch",
    "left_elbow", "right_elbow", "torso", "neck",
]

#: §Ship 41: behavior family → the hand-authored ground-truth metric a
#: generated metric of that family calibrates against (metric_calibration).
#: Used by the launch-time auto-calibration path; exported for reuse.
FAMILY_TO_BUILTIN = {
    "kick": "g1_kick",
    "floss": "g1_floss",
    "jump": "g1_jump",
    "locomotion": "go1_trot",
    "cartpole": "cartpole_balance",
}


def resolve_behavior_family(
    behavior_goal: Optional[str], robot_hint: Optional[str] = None,
) -> Optional[str]:
    """§Ship 41: map a natural-language behavior goal to a behavior family
    (`kick`/`floss`/`jump`/`locomotion`/`cartpole`) so the non-degeneracy gate
    can anchor a non-locomotion metric against a behavior-APPROPRIATE positive
    archetype instead of a hard-coded forward-walker (the false-rejection bug).

    WORD-level (not substring) keyword match: real goals are paraphrased (the
    on-disk kick goal does not equal the benchmark string), but every benchmark
    goal contains a clean family word. `None` → no family matched. The family
    selects the CALIBRATION ground truth (and the returned label); it does NOT
    narrow the non-degeneracy gate."""
    g = (behavior_goal or "").lower()
    tokens = set(re.findall(r"[a-z]+", g))

    def has(*words: str) -> bool:
        return any(w in tokens for w in words)

    # §Ship 41 review: WORD matching — substring "hop" matched "Hopper" (a
    # locomotion example) and "strike" matched the idiom "strike a balance",
    # false-rejecting good metrics. "bound"/"strike" dropped ("bound" is a
    # quadruped GAIT; "strike" is too idiomatic — "kick" is the clear token).
    # §compound-traversal fix: a parkour / traversal objective (climb onto
    # supports, jump OFF / OVER / ACROSS an obstacle, vault, traverse a course)
    # is a LOCOMOTIVE task whose success REQUIRES base travel. It must never
    # resolve to a STATIONARY family (kick/floss/jump) — that fires the
    # stationarity axiom and the walker ceiling against the very forward motion
    # the goal demands (the live "climb two boxes … then jump off" false
    # rejection: the metric scored the parkour positive high, but family "jump"
    # then vetoed forward travel, and no metric could ever be accepted). Resolve
    # to None (novel/compound) — the `active_parkour` + prompt-derived positive
    # archetypes anchor non-degeneracy, NO stationary-skill gate fires, and
    # task-VALIDITY defers to `calibrate_task_derived` behind the firewall (the
    # same path every family-None novel goal already takes). Checked BEFORE the
    # kick/floss/jump words so a compound "climb … jump off" is not captured by
    # the bare "jump" token. A plain in-place "jump as high as you can" has no
    # traversal cue and still resolves to the stationary jump family below.
    _traversal_word = has(
        "climb", "climbs", "climbing", "parkour", "traverse", "traverses",
        "traversing", "vault", "vaults", "vaulting", "clamber", "clambers",
        "clambering")
    _traversal_phrase = any(
        p in g for p in (
            "jump off", "jumps off", "jumping off", "jump over", "jumps over",
            "jumping over", "jump across", "jumps across", "jump onto",
            "jumps onto", "leap off", "leaps off", "leap over", "leaps over",
            "hop off", "hop over", "obstacle course"))
    if _traversal_word or _traversal_phrase:
        return None
    if has("kick", "kicks", "kicking"):
        return "kick"
    if has("floss", "flossing", "opposition", "antiphase") or "anti-phase" in g:
        return "floss"
    if has("jump", "jumps", "jumping", "hop", "hops", "hopping",
           "leap", "leaps", "leaping"):
        return "jump"
    # §<resolution fix>: locomotion needs a real GAIT verb, OR a bare directional/
    # velocity cue (forward/ahead/velocity) when NO posture/gesture verb is present —
    # so "forward_velocity" (Hopper) resolves to locomotion but "bend FORWARD into a
    # bow" / "lean FORWARD and bow" does NOT (the stray "forward" used to mis-resolve a
    # FOLD goal to locomotion, then its locomotion positive scored ~0 and the metric was
    # false-rejected). "in place" / "in-place" NEVER resolves to locomotion (an in-place
    # march/run is a stationary skill, not forward travel — its positive `active` is ~0).
    _posture_gesture = has(
        "bend", "bends", "bending", "bow", "bows", "bowing", "squat", "squats",
        "squatting", "crouch", "crouches", "crouching", "stoop", "stooping", "kneel",
        "kneeling", "touch", "touches", "touching", "reach", "reaches", "reaching",
        "lean", "leans", "leaning", "wave", "waves", "waving", "raise", "raises",
        "raising", "twist", "twists", "twisting", "curl", "curls", "arch", "fold",
        "folds", "dip", "dips", "nod", "nods", "shake", "shakes", "sit", "sits",
        "lie", "roll", "rolls", "rolling")
    _in_place = "in place" in g or "in-place" in g
    _gait = has("trot", "trotting", "walk", "walking", "gait", "locomote",
                "locomotion", "run", "running", "march", "marching", "stride",
                "striding", "jog", "jogging", "sprint", "sprinting", "gallop",
                "skip", "skipping")
    _directional = has("forward", "forwards", "ahead", "velocity")
    # §<round-5/6 fix>: BALANCE/cartpole is checked before locomotion (so a pure balance
    # goal that mentions "velocity" — in `_directional` — resolves to cartpole, not the
    # wrong go1_trot anchor), but ONLY when there is NO real GAIT verb: a gait verb is a
    # far stronger locomotion signal than an incidental "balance" token, so "walk/trot/run
    # while balancing …" must stay locomotion (round-6: the unconditional reorder
    # introduced the symmetric mis-resolution). "cartpole" is unconditional (unambiguous).
    if has("cartpole") or (has("balance", "balancing") and not _gait):
        return "cartpole"
    if (_gait or (_directional and not _posture_gesture)) and not _in_place:
        return "locomotion"
    # Robot-family fallback: a quadruped goal with no behavior word is almost
    # always locomotion.
    rh = (robot_hint or "").lower()
    if any(w in rh for w in ("go1", "go2", "quadruped")):
        return "locomotion"
    return None


def resolve_torso_target(behavior_goal: Optional[str]) -> str:
    """§Metric-quality laws (LAW 13): the body orientation a COMPETENT execution
    holds, so the uprightness-monotonicity axiom (which penalises tilting toward
    horizontal) is applied ONLY to upright skills. A flip / dive / roll / crawl
    is competently HORIZONTAL mid-move; a handstand / cartwheel is inverted/any.

    Returns `"upright"` (default — the common case), `"horizontal"`, or `"any"`.
    WORD-level match like `resolve_behavior_family`. The default is SAFE: a
    metric for a truly novel orientation the keyword set misses self-scopes via
    its ~0 score on the upright synthetic battery (so M1 passes vacuously rather
    than false-rejecting), and most real goals genuinely are upright."""
    g = (behavior_goal or "").lower()
    tokens = set(re.findall(r"[a-z]+", g))

    def has(*words: str) -> bool:
        return any(w in tokens for w in words)

    if (has("backflip", "frontflip", "flip", "flips", "flipping", "somersault",
            "somersaults", "dive", "dives", "diving", "roll", "rolls", "rolling",
            "crawl", "crawling", "prone")
            or "back flip" in g or "front flip" in g):
        return "horizontal"
    if has("handstand", "handstands", "headstand", "cartwheel", "cartwheels",
           "inverted", "invert", "upside"):
        return "any"
    return "upright"


def resolve_goal_frame(
    behavior_goal: Optional[str], robot_hint: Optional[str] = None,
) -> dict[str, Optional[str]]:
    """§Metric-quality laws (LAW 0): the task-declared FRAME a metric's
    directional / postural / support gates must read BEFORE they fire. Any field
    that is unresolved (`None`) means the matching gate ABSTAINS — it adds
    neither a penalty nor a pass — so a NOVEL task (a rearward mule-kick, a
    single-support flamingo, a backflip) is never false-rejected by a gate that
    silently assumes forward + upright + double-support.

      goal_axis    — "+x" (forward), "-x" (rearward), or None (non-directional).
      support_mode — "double", "single", "flight", or None.
      torso_target — "upright" | "horizontal" | "any" (see resolve_torso_target).

    WORD-level match like resolve_behavior_family / resolve_torso_target. The
    defaults are SAFE (a plain directional skill is forward + double — the
    benchmark case); any miss self-scopes via the metric's ~0 score on the
    synthetic battery rather than a false rejection."""
    g = (behavior_goal or "").lower()
    tokens = set(re.findall(r"[a-z]+", g))

    def has(*words: str) -> bool:
        return any(w in tokens for w in words)

    torso = resolve_torso_target(behavior_goal)

    # Support mode. NOTE the single-support patterns are deliberately specific
    # ("single leg" / "one foot" / "balance on one" / flamingo / handstand) — a
    # bare "with one leg" describes a normal kick (still DOUBLE-support stance),
    # so it must NOT resolve to single.
    if has("jump", "jumps", "jumping", "hop", "hops", "hopping", "leap",
           "leaps", "leaping", "flip", "flips", "somersault", "dive", "dives"):
        support: Optional[str] = "flight"
    elif (has("flamingo", "handstand", "headstand", "stork")
          or "one foot" in g or "single leg" in g or "single-leg" in g
          or "one-legged" in g or "balance on one" in g):
        support = "single"
    elif torso != "upright":
        support = None              # a flip/roll/crawl has mixed/unclear support
    else:
        support = "double"

    # Goal axis: rearward when the goal explicitly says so; else forward for a
    # directional skill; else None (in-place / non-directional, e.g. a spin).
    if (has("backward", "backwards", "behind", "rearward", "reverse")
            or "mule kick" in g or "mule-kick" in g):
        goal_axis: Optional[str] = "-x"
    elif has("kick", "kicks", "kicking", "punch", "punches", "punching",
             "strike", "strikes", "reach", "reaches", "throw", "throws",
             "forward", "ahead", "front", "trot", "walk", "walking", "run",
             "running", "march", "stride", "sprint", "jog", "gait", "locomote"):
        goal_axis = "+x"
    else:
        goal_axis = None

    return {"goal_axis": goal_axis, "support_mode": support, "torso_target": torso}


def _is_allowed_module(name: str) -> bool:
    """A generated metric may import only `numpy` and a vetted numpy submodule
    (numpy.linalg/random). The FULL dotted path is gated — the prior root-only check
    (`name.split('.')[0]`) let `import numpy.ctypeslib` (native-code RCE) through."""
    if not name:
        return False
    if name in _ALLOWED_IMPORTS:
        return True
    if name.startswith("numpy."):
        return name[len("numpy."):].split(".")[0] in _SAFE_NUMPY_SUBMODULES
    return False


def _ast_safety(source: str) -> list[str]:
    """Return a list of safety violations (empty = safe).

    §round-9 SECURITY hardening: a generated metric is UNTRUSTED code, and this static
    gate is the first containment layer (the runtime exec has full builtins + numpy, both
    IO/RCE surfaces). Beyond the original import/forbidden-name/dunder-attr checks it now
    closes the reproduced escapes: numpy submodule imports (full dotted gate), numpy IO /
    pickle / native attrs (`_FORBIDDEN_ATTRS`), `allow_pickle=True`, dunder FUNCTION/CLASS
    definitions (`def __reduce__`), and dunder tokens inside STRING literals (the
    `'{f.__globals__[__builtins__]}'.format(...)` reflection trick the AST walker can't
    see into). Reflection is thus blocked through every syntactic path."""
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    # Catalog names intentionally use a double-underscore namespace. Permit
    # that token only in the literal node directly used as an arrays key; the
    # same text anywhere else remains blocked by the reflection defense.
    safe_array_literals: set[int] = set()
    for parent in ast.walk(tree):
        if (isinstance(parent, ast.Subscript)
                and isinstance(parent.value, ast.Name)
                and parent.value.id == "arrays"
                and isinstance(parent.slice, ast.Constant)
                and isinstance(parent.slice.value, str)):
            safe_array_literals.add(id(parent.slice))
        elif (isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Attribute)
                and parent.func.attr == "get"
                and isinstance(parent.func.value, ast.Name)
                and parent.func.value.id == "arrays"
                and parent.args
                and isinstance(parent.args[0], ast.Constant)
                and isinstance(parent.args[0].value, str)):
            safe_array_literals.add(id(parent.args[0]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not _is_allowed_module(a.name):
                    problems.append(f"forbidden import: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if any(al.name == "*" for al in node.names):   # `from numpy import *` rebinds IO fns as bare names
                problems.append("forbidden star import")
            if not _is_allowed_module(node.module or ""):
                problems.append(f"forbidden import-from: {node.module}")
            for al in node.names:                          # block `from numpy import save/load/...`
                if (al.name in _FORBIDDEN_ATTRS or al.name in _FORBIDDEN_NAMES
                        or al.name.startswith("__")):
                    problems.append(f"forbidden imported name: {al.name}")
        elif isinstance(node, ast.Name) and (
                node.id in _FORBIDDEN_NAMES or node.id.startswith("__")):
            # §Ship 35 review: also reject ANY dunder NAME (e.g.
            # __builtins__ is in every module namespace without an import
            # and reaches eval/exec).
            problems.append(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr not in _ALLOWED_DUNDER_ATTRS:
                problems.append(f"dunder attribute access: {node.attr}")
            elif node.attr.startswith("_") and not node.attr.startswith("__"):
                # §round-23 SECURITY (CRITICAL): a single-underscore "private/internal" attribute
                # is never part of the numpy public API a physical-quantity metric uses, but numpy
                # RE-EXPORTS os / builtins / importlib through internal modules reachable as
                # NON-dunder attribute chains the public-name and forbidden-attr checks never see:
                #   np._pytesttester.os.system("…")      → arbitrary shell (RCE)
                #   np._globals.enum.bltns.open(path,"w") → arbitrary file write
                # Deny ALL single-underscore private attribute access — it closes the whole class.
                problems.append(f"forbidden private attribute access: {node.attr}")
            elif node.attr.startswith(_FRAME_INTROSPECTION_PREFIXES):
                # §round-28 SECURITY (CRITICAL): a generator/frame introspection attribute
                # (gi_frame.f_builtins) reaches the LIVE builtins dict, from which __import__ /
                # open / eval are retrieved as a dict SUBSCRIPT keyed by a chr()-assembled string
                # the AST can't see — no dunder, no single-underscore, no `format`. Deny the whole
                # frame/generator/coroutine/traceback prefix family (never a physical metric attr).
                problems.append(f"forbidden frame/generator introspection attribute: {node.attr}")
            elif node.attr in _FORBIDDEN_ATTRS or node.attr in _FORBIDDEN_NAMES:
                # §round-23: also reject a forbidden NAME reached as an ATTRIBUTE (x.os / x.open /
                # x.system) — _FORBIDDEN_NAMES was previously matched only on a bare ast.Name/import.
                problems.append(f"forbidden attribute (IO/native/reflection): {node.attr}")
            elif node.attr in ("format", "format_map"):
                # §round-19 SECURITY: str.format/format_map is the reflection PRIMITIVE
                # ('{0.__globals__[__builtins__]}'.format(fn) reads module globals/builtins
                # via field access). The string-literal '__' scan below only sees ast.Constant
                # nodes, so a dunder string assembled at runtime (chr(95)*2 + 'globals' + ...)
                # bypassed it. Rejecting the primitive itself closes every assembly path — a
                # physical-quantity metric never needs .format (f-strings are ast.JoinedStr
                # and are unaffected).
                problems.append(f"forbidden reflection primitive: str.{node.attr}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("__"):                 # `def __reduce__` → pickle-RCE gadget
                problems.append(f"forbidden dunder definition: {node.name}")
        elif isinstance(node, ast.keyword) and node.arg == "allow_pickle":
            problems.append("forbidden kwarg: allow_pickle")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            text = (node.value if isinstance(node.value, str)
                    else node.value.decode("latin-1", "ignore"))
            if "__" in text and id(node) not in safe_array_literals:
                problems.append("forbidden dunder token in string literal")
    return problems


def _referenced_array_keys(source: str) -> set[str]:
    """Extract the string keys used to index the `arrays` mapping —
    `arrays["k"]` / `arrays.get("k")`. Constrains the metric to the
    persisted-array contract."""
    keys: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return keys
    for node in ast.walk(tree):
        # arrays["k"]
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "arrays"):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                keys.add(sl.value)
        # arrays.get("k")
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "arrays"
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
    return keys


def _catalog_array_access_violations(source: str) -> list[str]:
    """Reject non-literal access to a catalog-backed arrays mapping.

    Exact allowlisting and observable partitioning require every generated
    metric access to be statically attributable to one declared name.  Aliases,
    iteration, and computed keys would make that proof impossible even though
    the runtime still drops undeclared NPZ members.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    parents: dict[ast.AST, ast.AST] = {
        child: parent for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id != "arrays":
            continue
        parent = parents.get(node)
        if (isinstance(parent, ast.Subscript) and parent.value is node
                and isinstance(parent.slice, ast.Constant)
                and isinstance(parent.slice.value, str)):
            continue
        if (isinstance(parent, ast.Attribute) and parent.value is node
                and parent.attr == "get"):
            call = parents.get(parent)
            if (isinstance(call, ast.Call) and call.func is parent
                    and call.args and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)):
                continue
        violations.append(
            f"line {getattr(node, 'lineno', '?')}: arrays must be accessed "
            "only with a literal declared key")
    return violations


def _is_int_const(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return True
    # `-1` parses as UnaryOp(USub, Constant) — count it too.
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and _is_int_const(node.operand))


#: The two (T, E, J) joint arrays — the ONLY arrays for which a literal third-axis
#: index is a hard-coded JOINT column. root_link_pos_w / projected_gravity_b /
#: *_foot_pos_b are (T, E, 3) — their third axis is x/y/z, NOT a joint.
_JOINT_ARRAY_KEYS = ("joint_pos", "joint_vel")


def _is_joint_array_expr(node: ast.AST, joint_vars: frozenset[str]) -> bool:
    """True if `node` evaluates to a joint array (joint_pos/joint_vel): the direct
    `arrays["joint_pos"]` / `arrays.get("joint_pos")` form, or a Name tracked as
    holding one (`_collect_joint_vars`)."""
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "arrays"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in _JOINT_ARRAY_KEYS):
        return True                                   # arrays["joint_pos"]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "arrays"
            and node.args and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in _JOINT_ARRAY_KEYS):
        return True                                   # arrays.get("joint_pos")
    return isinstance(node, ast.Name) and node.id in joint_vars


def _collect_joint_vars(tree: ast.AST) -> frozenset[str]:
    """Variables assigned DIRECTLY from a joint array — `jp = arrays['joint_pos']`,
    `jv = arrays.get('joint_vel')`. A conservative one-hop tracker; deeper aliasing
    is the runtime permutation-robustness gate's job."""
    joint_vars: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and _is_joint_array_expr(node.value, frozenset())):
            joint_vars.add(node.targets[0].id)
    return frozenset(joint_vars)


def _raw_joint_index_violations(source: str) -> list[str]:
    """§Ship 49: flag a HARD-CODED integer index into a JOINT axis — the
    `joint_vel[:, :, 0]` form — which silently reads the wrong joint the moment the
    robot or joint order changes (the §3A failure). Metrics must select joints via
    name-resolved indices (`meta['joint_roles']`).

    §<3-vector fix>: ONLY flag `[:, :, N]` when the base is a JOINT array
    (joint_pos/joint_vel — direct or one-hop-tracked). A 3-vector axis read in the
    explicit-slice form — `root_link_pos_w[:, :, 2]` (height),
    `projected_gravity_b[:, :, 2]` (uprightness) — is LEGITIMATE and must NOT be
    flagged: a model that writes `[:, :, 2]` instead of the ellipsis `[..., 2]` was
    being false-rejected (observed in real toe-touch/bow generations). The Ellipsis
    form `[..., N]` is still never statically flagged; the runtime
    permutation-robustness gate is the semantic backstop for any index-hardcoding
    this conservative static check misses (e.g. via aliasing)."""
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return problems
    joint_vars = _collect_joint_vars(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        sl = node.slice
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 3 \
                and isinstance(sl.elts[0], ast.Slice) \
                and isinstance(sl.elts[1], ast.Slice) \
                and _is_int_const(sl.elts[2]) \
                and _is_joint_array_expr(node.value, joint_vars):
            problems.append(
                "hard-coded integer joint index `joint_*[:, :, N]` — select joints "
                "by name via meta['joint_roles'], not a literal column")
    return problems


# §Ship 49: a non-trivial relabelling of the 12-joint archetype axis. The
# permutation-robustness gate applies it CONSISTENTLY to joint_names AND the
# joint_pos/joint_vel columns — a name/role-resolving metric is invariant
# (it follows the names to the same physical joints), an index-hardcoding one
# swings (column 0 is now a different joint).
def _permute_joint_arrays(
    arrays: dict, perm: list[int],
) -> dict:
    out = {}
    for k, v in arrays.items():
        if k in ("joint_pos", "joint_vel") and getattr(v, "ndim", 0) >= 3 \
                and v.shape[2] == len(perm):
            out[k] = v[:, :, perm]
        else:
            out[k] = v
    return out


# ── synthetic archetype rollouts (non-degeneracy gate) ───────────────


def _upright_g() -> np.ndarray:
    g = np.zeros((T, E, 3), dtype=np.float64)
    g[..., 2] = -1.0
    return g


_ABSTRACT_PHASES = frozenset({
    "climb", "dwell", "move_forward", "move_backward", "move_left",
    "move_right", "jump", "jump_off", "land", "crouch", "tilt",
    "recover", "oscillate", "reach", "kick",
})


def _abstract_objective_program(
    behavior_goal: Optional[str], declared: Any = None,
) -> list[str]:
    """Compile free text (or a generated declarative companion) to task-space phases.

    The phase vocabulary is intentionally embodiment-neutral: it describes root motion,
    dwell, posture, and generic articulation rather than joint indices, robot names, or
    simulator tasks.  A generated metric may declare ``ABSTRACT_OBJECTIVE =
    {"phases": [...]}`` beside ``compute_spec``; old metrics get the deterministic text
    inference below.  The declaration only selects from this closed vocabulary and can
    never provide executable validator code.
    """
    phases: list[str] = []
    if isinstance(declared, Mapping):
        raw = declared.get("phases")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            phases = [str(p).strip().lower() for p in raw]
            phases = [p for p in phases if p in _ABSTRACT_PHASES][:8]
    if phases:
        return phases

    g = (behavior_goal or "").lower()
    tokens = set(re.findall(r"[a-z]+", g))

    def has(*words: str) -> bool:
        return any(word in tokens for word in words)

    # Keep this traversal vocabulary ALIGNED with resolve_behavior_family's
    # `_traversal_word`: a goal that function sends to family None (vault,
    # traverse, clamber, …) MUST also produce a climb/jump_off probe here, or
    # `abstract_is_traversal` stays False, the parkour positives are never
    # re-added, and the metric is false-rejected (the two detectors disagreeing).
    staged_climb = has(
        "climb", "climbs", "climbing", "ascend", "ascending", "stairs",
        "steps", "boxes", "platforms", "parkour", "vault", "vaults", "vaulting",
        "traverse", "traverses", "traversing", "clamber", "clambers",
        "clambering",
    )
    wants_dwell = has(
        "pause", "pauses", "pausing", "wait", "waiting", "hold", "holding",
        "stop", "stopping", "dwell", "dwelling",
    )
    if staged_climb:
        # Two levels express progression without assuming a particular scene count.
        for _ in range(2):
            phases.append("climb")
            if wants_dwell:
                phases.append("dwell")

    if has("forward", "forwards", "ahead") and not staged_climb:
        phases.append("move_forward")
    elif has("backward", "backwards", "reverse"):
        phases.append("move_backward")
    elif has("left"):
        phases.append("move_left")
    elif has("right"):
        phases.append("move_right")

    wants_jump = has(
        "jump", "jumps", "jumping", "hop", "hops", "hopping", "leap",
        "leaps", "leaping",
    )
    # "jump OFF/OVER/ACROSS/ONTO" is a forward LEAP (jump_off — travel + arc +
    # flight); a bare "jump" is the stationary skill. The over/across/onto words
    # mirror resolve_behavior_family's `_traversal_phrase` so a "jump over the
    # hurdle" goal (routed to family None there) carries a jump_off probe here.
    if wants_jump:
        leap = staged_climb or has("off", "over", "across", "onto")
        phases.append("jump_off" if leap else "jump")
    if has("land", "lands", "landing") and (not phases or phases[-1] != "jump_off"):
        phases.append("land")
    if has("crouch", "crouches", "crouching", "squat", "squats", "squatting"):
        phases.append("crouch")
    if has("bend", "bends", "bending", "bow", "bows", "lean", "leans", "tilt"):
        phases.append("tilt")
    if has("recover", "recovers", "return", "returns", "upright", "stand", "standing"):
        phases.append("recover")
    if has("wave", "waves", "waving", "reach", "reaches", "reaching", "raise"):
        phases.append("reach")
    if has("kick", "kicks", "kicking"):
        phases.append("kick")
    if has("oscillate", "oscillates", "oscillating", "floss", "flossing", "shake"):
        phases.append("oscillate")
    return phases[:8]


def _abstract_objective_probe(phases: Sequence[str]) -> Optional[dict[str, np.ndarray]]:
    """Retarget an abstract phase program onto the validator's universal channels.

    This is a kinematic *validator exemplar*, not a stored robot trajectory.  Root and
    gravity live in task space; generic articulation is projected onto the synthetic
    named body and later subjected to the existing joint-permutation gate.  Therefore
    the same prompt-derived oracle works for quadrupeds, bipeds, and arms whenever the
    authored metric uses their persisted universal/task channels.
    """
    clean = [str(p) for p in phases if str(p) in _ABSTRACT_PHASES][:8]
    if not clean:
        return None

    root = np.zeros((T, E, 3), dtype=np.float64)
    root[..., 2] = 0.55
    gravity = _upright_g()
    joints = np.zeros((T, E, J), dtype=np.float64)
    # Physically-consistent end-effector channels: a nominal grounded stance whose
    # support schedule LIFTS both feet during flight phases and whose left foot
    # SWINGS forward during a kick. The former exemplar fabricated both feet as
    # permanently in-contact at the origin — strictly WORSE than omitting them: the
    # metric prompt tells authors to read the foot-contact SUPPORT SCHEDULE to
    # detect a jump/leap flight phase, so an always-in-contact probe scored EVERY
    # such (correct) metric 0 (the live "climb … jump off" non-degeneracy all-zero).
    # An ABSENT channel abstains to a neutral 1.0; a present-but-wrong one FAILS.
    lfoot = np.zeros((T, E, 3), dtype=np.float64)
    rfoot = np.zeros((T, E, 3), dtype=np.float64)
    lfoot[..., 1] = 0.10   # nominal pelvis-frame stance (foot below + to the side)
    rfoot[..., 1] = -0.10
    lfoot[..., 2] = -0.55
    rfoot[..., 2] = -0.55
    contact_l = np.ones((T, E), dtype=np.float64)
    contact_r = np.ones((T, E), dtype=np.float64)
    start = max(5, int(0.10 * T))
    bounds = np.linspace(start, T, len(clean) + 1, dtype=int)
    x = y = 0.0
    z = 0.55
    tilt = 0.0

    for phase, a, b in zip(clean, bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        n = b - a
        u = np.linspace(0.0, 1.0, n)
        root[a:b, :, 0] = x
        root[a:b, :, 1] = y
        root[a:b, :, 2] = z
        gravity[a:b, :, 0] = tilt
        gravity[a:b, :, 2] = -np.sqrt(max(0.0, 1.0 - tilt * tilt))

        if phase == "climb":
            root[a:b, :, 0] = (x + 0.35 * u)[:, None]
            root[a:b, :, 2] = (z + 0.30 * u)[:, None]
            x += 0.35
            z += 0.30
        elif phase == "move_forward":
            root[a:b, :, 0] = (x + 0.90 * u)[:, None]
            x += 0.90
        elif phase == "move_backward":
            root[a:b, :, 0] = (x - 0.90 * u)[:, None]
            x -= 0.90
        elif phase == "move_left":
            root[a:b, :, 1] = (y + 0.70 * u)[:, None]
            y += 0.70
        elif phase == "move_right":
            root[a:b, :, 1] = (y - 0.70 * u)[:, None]
            y -= 0.70
        elif phase == "jump":
            root[a:b, :, 0] = (x + 0.70 * u)[:, None]
            root[a:b, :, 2] = (z + 0.45 * np.sin(np.pi * u))[:, None]
            x += 0.70
        elif phase == "jump_off":
            root[a:b, :, 0] = (x + 1.20 * u)[:, None]
            arc = (1.0 - u) * z + u * 0.55 + 0.25 * np.sin(np.pi * u)
            root[a:b, :, 2] = arc[:, None]
            x += 1.20
            z = 0.55
        elif phase == "land":
            root[a:b, :, 2] = ((1.0 - u) * z + u * 0.55)[:, None]
            z = 0.55
        elif phase == "crouch":
            root[a:b, :, 2] = (z - 0.25 * u)[:, None]
            joints[a:b] = (0.9 * u)[:, None, None]
            z -= 0.25
        elif phase == "tilt":
            gravity[a:b, :, 0] = (0.85 * u)[:, None]
            gravity[a:b, :, 2] = (-np.sqrt(1.0 - (0.85 * u) ** 2))[:, None]
            tilt = 0.85
        elif phase == "recover":
            gravity[a:b, :, 0] = (tilt * (1.0 - u))[:, None]
            gravity[a:b, :, 2] = (-np.sqrt(
                1.0 - (tilt * (1.0 - u)) ** 2))[:, None]
            root[a:b, :, 2] = (z + (0.55 - z) * u)[:, None]
            tilt = 0.0
            z = 0.55
        elif phase in {"oscillate", "reach", "kick"}:
            wave = np.sin(6.0 * np.pi * u)
            cols = (6, 7, 8, 9) if phase == "reach" else (0, 1, 2, 3)
            amplitude = 1.4 if phase == "reach" else 0.8
            joints[a:b, :, list(cols)] = (amplitude * wave)[:, None, None]

        if phase in {"climb", "jump", "jump_off"}:
            burst = np.sin(np.pi * np.clip(u * 3.0, 0.0, 1.0))
            joints[a:b, :, 0:4] = (0.8 * burst)[:, None, None]

        # End-effector support schedule + kick swing (see the channel note above):
        # flight phases LIFT both feet so a support-schedule metric sees a real
        # airborne window; a kick swings the left foot forward (signed anterior +x)
        # so a foot-direction metric reads a correctly-directed strike. Every window
        # ENDS grounded (touchdown), so the final frame certifies a landed stance.
        if phase == "jump":
            air = np.sin(np.pi * u) > 0.25
        elif phase == "jump_off":
            air = (u > 0.12) & (u < 0.92)
        elif phase == "land":
            air = u < 0.85
        else:
            air = None
        if air is not None:
            grounded = np.where(air, 0.0, 1.0)[:, None]
            contact_l[a:b] = grounded
            contact_r[a:b] = grounded
        elif phase == "kick":
            swing = 0.30 * np.sin(np.pi * u)
            lfoot[a:b, :, 0] = swing[:, None]
            contact_l[a:b] = (swing < 0.15).astype(np.float64)[:, None]

        if b < T:
            root[b:, :, 0] = x
            root[b:, :, 1] = y
            root[b:, :, 2] = z
            gravity[b:, :, 0] = tilt
            gravity[b:, :, 2] = -np.sqrt(max(0.0, 1.0 - tilt * tilt))
            joints[b:] = joints[b - 1]

    return {
        "joint_pos": joints,
        "joint_vel": np.gradient(joints, axis=0) / 0.02,
        "projected_gravity_b": gravity,
        "root_link_pos_w": root,
        "left_foot_pos_b": lfoot,
        "right_foot_pos_b": rfoot,
        "left_foot_contact": contact_l,
        "right_foot_contact": contact_r,
    }


def _archetypes() -> dict[str, dict]:
    """Synthetic archetype rollouts spanning a competence axis. Negatives
    (`still`/`fallen`/`chaotic`/`upright_flail`) plus a POSITIVE per behavior
    family (`active` locomotion, `active_kick`, `active_floss`, `active_jump`) plus
    a generic compound traversal (`active_parkour`: climb, dwell, climb, dwell,
    then jump away and land).
    A valid task metric scores its family positive strictly above the negatives
    with non-trivial spread (the gate picks the family in
    `validate_generated_metric`)."""
    rng = np.random.default_rng(0)
    t = np.arange(T)

    def arrays(joint_pos, joint_vel, gravity, root, lfoot=None, rfoot=None):
        d = {"joint_pos": joint_pos, "joint_vel": joint_vel,
             "projected_gravity_b": gravity, "root_link_pos_w": root}
        if lfoot is not None:
            d["left_foot_pos_b"] = lfoot
        if rfoot is not None:
            d["right_foot_pos_b"] = rfoot
        return d

    # §Metric-quality laws: a left-foot anterior (pelvis-frame x) swing for the
    # kick archetypes — forward (+1) for a real kick, rearward (−1) for the
    # kick-behind hack; the right (stance) foot stays put. The signed x is what
    # lets a direction-aware metric (LAW 4) tell a forward kick from a rear one.
    def foot_swing(direction):
        lf = np.zeros((T, E, 3)); rf = np.zeros((T, E, 3))
        for start in range(20, T, 40):
            for k in range(10):
                if start + k < T:
                    lf[start + k, :, 0] = direction * 0.30 * np.sin(np.pi * k / 10)
        return lf, rf

    # dead-still upright
    jp0 = np.zeros((T, E, J)); jv0 = rng.normal(0, 0.01, (T, E, J))
    root0 = np.zeros((T, E, 3)); root0[..., 2] = 0.5
    still = arrays(jp0, jv0, _upright_g(), root0)

    # fallen (gravity sideways), thrashing
    g_fall = np.zeros((T, E, 3)); g_fall[..., 0] = 1.0
    jvf = rng.normal(0, 3.0, (T, E, J))
    rootf = np.zeros((T, E, 3)); rootf[..., 2] = 0.1
    fallen = arrays(rng.normal(0, 1, (T, E, J)), jvf, g_fall, rootf)

    # chaotic upright (random large motion, no structure, no travel)
    jpc = rng.normal(0, 1.5, (T, E, J)); jvc = rng.normal(0, 5.0, (T, E, J))
    rootc = np.zeros((T, E, 3)); rootc[..., 2] = 0.5
    chaotic = arrays(jpc, jvc, _upright_g(), rootc)

    # active: upright, smooth periodic joints, steady forward travel
    jpa = np.zeros((T, E, J))
    for jj in range(J):
        jpa[:, :, jj] = (0.4 * np.sin(2 * np.pi * t / 25 + jj))[:, None]
    jva = np.gradient(jpa, axis=0)
    roota = np.zeros((T, E, 3)); roota[..., 2] = 0.5
    roota[..., 0] = (t * 0.04)[:, None]   # forward
    active = arrays(jpa, jva, _upright_g(), roota)

    # §Ship 36: upright_flail — large, fast limb oscillation while standing
    # still with ZERO travel: the "stand still and flail" reward-hack Sam's
    # G1 kick run fell into. A valid task metric must score this BELOW
    # `active`; a metric that merely rewards motion magnitude will not, and
    # the non-degeneracy gate now rejects it.
    jpf = np.zeros((T, E, J))
    for jj in range(J):
        jpf[:, :, jj] = (1.2 * np.sin(2 * np.pi * t / 6.0 + jj))[:, None]
    jvf2 = np.gradient(jpf, axis=0)
    rootf2 = np.zeros((T, E, 3)); rootf2[..., 2] = 0.5   # upright, no travel
    upright_flail = arrays(jpf, jvf2, _upright_g(), rootf2)

    # §Ship 41: behavior-family POSITIVE archetypes — a COMPETENT example of
    # each non-locomotion behavior, so a kick/floss/jump metric is measured
    # against its own behavior rather than the forward-walker `active`. All are
    # upright, stationary, at STANDING height (z≈0.7 — a kick metric's height
    # gate needs ≥0.65; z=0.5 would leave a good metric below the spread floor).
    def _standing_root() -> np.ndarray:
        r = np.zeros((T, E, 3)); r[..., 2] = 0.7
        return r

    # active_kick: discrete leg-velocity bursts (left hip-pitch/knee/ankle =
    # indices 0/2/4), stance leg quiet — a clean, repeated, stationary kick.
    jvk = np.zeros((T, E, J))
    for start in range(20, T, 40):            # 3 discrete kicks
        for jdx in (0, 2, 4):
            jvk[start:start + 5, :, jdx] = 8.0
    jpk = np.cumsum(jvk, axis=0) * 0.02       # consistent integrated position
    lf_fwd, rf_fwd = foot_swing(+1.0)         # foot swings FORWARD (a real kick)
    active_kick = arrays(jpk, jvk, _upright_g(), _standing_root(), lf_fwd, rf_fwd)

    # active_floss: anti-phase hip↔arm oscillation. SLOW (period 25, like the
    # g1_floss ladder) so a motion-MAGNITUDE metric cannot mistake it for the
    # fast `upright_flail` negative — flossing is structure, not speed.
    jpfl = np.zeros((T, E, J))
    hip = 0.4 * np.sin(2 * np.pi * t / 25)
    arm = 0.4 * np.sin(2 * np.pi * t / 25 + np.pi)
    for jdx in (0, 1):                        # hips
        jpfl[:, :, jdx] = hip[:, None]
    for jdx in (6, 7, 8, 9):                  # shoulders + elbows
        jpfl[:, :, jdx] = arm[:, None]
    jvfl = np.gradient(jpfl, axis=0)
    active_floss = arrays(jpfl, jvfl, _upright_g(), _standing_root())

    # active_jump: repeated vertical hops (crouch→launch→apex→land) with knee
    # extension bursts, upright, ZERO horizontal travel. Has real leg motion so
    # a stillness-rewarder cannot mistake it for a quiet stance.
    zj = np.full(T, 0.55)                     # crouched baseline
    jvj = np.zeros((T, E, J))
    for start in range(15, T, 35):            # ~3 hops
        for k in range(20):
            if start + k < T:
                zj[start + k] = 0.55 + 0.45 * np.sin(np.pi * k / 20)  # apex ≈1.0
                if k < 6:                     # launch: knees extend
                    for jdx in (2, 3):
                        jvj[start + k, :, jdx] = 6.0
    jpj = np.cumsum(jvj, axis=0) * 0.02
    rootj = np.zeros((T, E, 3)); rootj[..., 2] = zj[:, None]
    active_jump = arrays(jpj, jvj, _upright_g(), rootj)

    # Compound traversal: approach/climb onto two progressively higher supports,
    # remain still on each one, then jump horizontally away while descending to
    # the starting height.  This is deliberately robot- and task-name agnostic:
    # it represents the observable physics of a staged parkour objective, not a
    # particular quadruped, scene, box count, or simulator.  The former jump-only
    # positive had zero horizontal travel and no elevated dwell, making every
    # honest climb→pause→jump completion metric look constant-zero and therefore
    # impossible to generate regardless of retries.
    i0 = max(4, int(0.20 * T))
    i1 = max(i0 + 4, int(0.30 * T))
    i2 = max(i1 + 6, int(0.46 * T))
    i3 = max(i2 + 4, int(0.56 * T))
    i4 = max(i3 + 6, int(0.72 * T))
    i5 = max(i4 + 6, int(0.88 * T))

    zp = np.full(T, 0.55)
    xp = np.zeros(T)
    zp[i0:i1] = np.linspace(0.55, 0.85, i1 - i0, endpoint=False)
    zp[i1:i2] = 0.85
    zp[i2:i3] = np.linspace(0.85, 1.15, i3 - i2, endpoint=False)
    zp[i3:i4] = 1.15
    zp[i4:i5] = np.linspace(1.15, 0.55, i5 - i4)
    xp[i0:i1] = np.linspace(0.0, 0.45, i1 - i0, endpoint=False)
    xp[i1:i2] = 0.45
    xp[i2:i3] = np.linspace(0.45, 0.90, i3 - i2, endpoint=False)
    xp[i3:i4] = 0.90
    xp[i4:i5] = np.linspace(0.90, 1.80, i5 - i4)
    xp[i5:] = 1.80

    jvp = np.zeros((T, E, J))
    for start in (i0, i2, i4):
        for jdx in (0, 1, 2, 3):
            jvp[start:min(start + 5, T), :, jdx] = 6.0
    jpp = np.cumsum(jvp, axis=0) * 0.02
    rootp = np.zeros((T, E, 3))
    rootp[..., 0] = xp[:, None]
    rootp[..., 2] = zp[:, None]
    active_parkour = arrays(jpp, jvp, _upright_g(), rootp)

    # §Ship 47: a realistic forward WALKER — upright, at STANDING height
    # (z≈0.70), travelling forward with FAST alternating hip/knee gait
    # swings (peak ≈6 rad/s, well above any kick threshold). This is the
    # exact Goodhart confound that stalled g1-kick-v3: a non-kicking gait
    # that a naive kick metric scores high (the on-disk gen_005 metric
    # scores it ~0.50). The existing `active` archetype could NOT catch
    # this — its joint velocities are ~0.1 rad/s (smooth, small-amplitude)
    # AND it sits at z=0.5, so a height/threshold-gated kick metric scores
    # it ~0. A valid STATIONARY-skill metric (kick/floss/jump) must score
    # the walker LOW; the family-scoped ceiling in validate_generated_metric
    # enforces it. NOT a negative for locomotion (there a walker is the
    # target) — see _STATIONARY_FAMILIES.
    jpw = np.zeros((T, E, J))
    phase = 2 * np.pi * 1.5 * t * 0.02       # 1.5 Hz gait
    jpw[:, :, 0] = (0.64 * np.sin(phase))[:, None]            # left hip pitch
    jpw[:, :, 2] = (0.51 * np.sin(phase))[:, None]            # left knee
    jpw[:, :, 1] = (0.64 * np.sin(phase + np.pi))[:, None]    # right hip pitch
    jpw[:, :, 3] = (0.51 * np.sin(phase + np.pi))[:, None]    # right knee
    jvw = np.gradient(jpw, axis=0) / 0.02
    rootw = np.zeros((T, E, 3)); rootw[..., 2] = 0.70
    rootw[..., 0] = (t * 0.04)[:, None]      # forward travel
    walker = arrays(jpw, jvw, _upright_g(), rootw)

    # §Metric-quality laws: the documented g1-kick-v5 hacks as KICK-FAMILY
    # negatives (scoped in validate_generated_metric so a single-support or
    # rearward NOVEL task is never false-rejected — LAW 0).
    # active_kick_behind: the SAME leg bursts as active_kick but the foot swings
    # REARWARD — a direction-blind metric scores it == active_kick (gameable); a
    # signed-direction metric (LAW 4) scores it ~0. This is the "kicks behind it"
    # failure Sam observed on g1-kick-v5.
    lf_back, rf_back = foot_swing(-1.0)
    active_kick_behind = arrays(jpk.copy(), jvk.copy(), _upright_g(),
                                _standing_root(), lf_back, rf_back)
    # one_leg_balance: a raised, HELD single leg with only sub-threshold wiggle
    # from a stationary upright stance — no real launch (LAW 1 completion gate)
    # and no swing (LAW 4). The "balance on one leg" half of the v5 hack.
    jvol = rng.normal(0, 0.3, (T, E, J))      # sub-threshold; never crosses the burst floor
    jpol = np.cumsum(jvol, axis=0) * 0.02
    lf_hold = np.zeros((T, E, 3)); lf_hold[..., 0] = 0.20   # foot held forward, static
    one_leg_balance = arrays(jpol, jvol, _upright_g(), _standing_root(),
                             lf_hold, np.zeros((T, E, 3)))
    # partial_kick: a SINGLE brief, small-amplitude forward flick (a partial,
    # non-repeated half-motion) — correctly-directed but must score BELOW a full
    # repeated kick (LAW 1 completion + LAW 2 amplitude).
    jvp = np.zeros((T, E, J))
    for jdx in (0, 2, 4):
        jvp[20:23, :, jdx] = 3.0              # one short sub-floor flick
    jpp = np.cumsum(jvp, axis=0) * 0.02
    lf_p, rf_p = foot_swing(+1.0)
    lf_p = lf_p * 0.2                          # tiny forward nudge (correct direction)
    partial_kick = arrays(jpp, jvp, _upright_g(), _standing_root(), lf_p, rf_p)

    return {"still": still, "fallen": fallen, "chaotic": chaotic,
            "active": active, "upright_flail": upright_flail,
            "active_kick": active_kick, "active_floss": active_floss,
            "active_jump": active_jump, "active_parkour": active_parkour,
            "walker": walker,
            "active_kick_behind": active_kick_behind,
            "one_leg_balance": one_leg_balance, "partial_kick": partial_kick}


def _score(fn, arrays, meta) -> float:
    out = fn(arrays, {"max_episode_steps": T, "rollout_num_envs": E,
                      "step_dt": 0.02}, meta)
    return float(out.get("spec_score", float("nan")))


# §Ship 47: skills performed from a roughly stationary base. For these a
# forward WALKER is a Goodhart distractor that must score LOW; for the
# locomotion family a walker IS the target, and for an unresolved family
# (None) we don't gate (the calibration firewall is the task-validity check),
# so the smell-test never false-rejects an ambiguous goal.
_STATIONARY_FAMILIES = frozenset({"kick", "floss", "jump"})

#: A fixed-battery score at or below this counts as "no signal from that
#: archetype". When EVERY archetype is this low AND the goal resolves to no
#: family, the battery cannot represent the goal — the selectivity probe decides.
_BATTERY_NEAR_ZERO = 1e-3


#: The probe's physical timestep (matches `_score`'s behavior dict). joint_vel is
#: supplied in PHYSICAL rad/s = d(joint_pos)/dt — the runtime + calibration-ladder
#: convention (the real adapters emit rad/s, and the kick/jump ladders set raw rad/s
#: bursts). A bare `np.gradient(joint_pos)` is dt× (50×) too small, so a metric that
#: THRESHOLDS a joint velocity (a wave's arm oscillation, a burst floor, a kick
#: amplitude) scored ~0 on every probe rollout and was false-rejected; scaling by dt
#: removes that whole class of false reject. (Pre-fix: a real best-of-4 toe-touch +
#: arm-wave generation had all 4 candidates rejected this way.)
_PROBE_DT = 0.02


def _physical_vel(jp: np.ndarray) -> np.ndarray:
    """joint_vel in PHYSICAL rad/s = d(joint_pos)/dt — the runtime/calibration-ladder
    convention. THE single source of truth for the synthetic-rollout velocity unit,
    shared by `_selectivity_probe` (the validator) and the `_graded_*_rung` builders
    (the best-of-N selector) so the two never diverge (a bare np.gradient is dt× too
    small and silently mis-scales any velocity-thresholding channel)."""
    return np.gradient(jp, axis=0) / _PROBE_DT


def _selectivity_probe(fn, meta) -> dict[str, float]:
    """Goal-AGNOSTIC selectivity probe: score `fn` on a deterministic, offline
    SET of hand-rolled competent-vs-degenerate rollouts to answer "is this metric
    SELECTIVE at all" — distinct from "does it match the goal" (the task-derived
    calibration firewall's job). Used only to rescue a novel-task metric the fixed
    `_archetypes()` battery cannot represent (e.g. a toe-touch/squat metric gated
    on a pelvis DIP-AND-RETURN, which no fixed archetype performs → all score ~0,
    yet the metric is perfectly selective).

    Hand-rolled numpy (NOT render_rung — its `base_height_m` is a monotone ramp and
    cannot express a dip-and-return), so this stays deterministic + offline (no LLM,
    no API), preserving the validate hot path. The competent set spans posture axes
    AND a gesture + a SEQUENCED compound, so an UPRIGHT posture skill (squat /
    sit-to-stand), a forward-TILT skill (toe-touch / bow / deep bend), a NON-UPRIGHT
    skill (roll / crawl), a GESTURE skill (wave / reach / raise-arms), AND a
    multi-phase compound ("do A then B" — bend-then-wave, whose completion gate
    needs a distinct second phase after the first returns) each light up at least one
    probe; the degenerate set (still + fallen) is what a degenerate metric (all-zero
    / still-rewarding / fall-rewarding) cannot be separated from. The optional foot
    channels (foot_pos_b / foot_contact, planted) are supplied so a metric that READS
    them is exercised, not auto-zeroed.

    SAFETY: a metric only reaches this branch when it scored the WHOLE fixed battery
    — chaotic / upright_flail / active included — ~0, so it is provably NOT a generic
    motion-magnitude rewarder; the still/fallen anchor is the remaining anti-degeneracy
    teeth. Returns `{competent, degenerate, spread}` (MAX over each set). Never raises
    (a probe crash scores 0.0 = no signal)."""
    t = np.arange(T)
    fold = (1.0 - np.cos(2.0 * np.pi * t / T)) / 2.0     # 0→1→0 over the rollout (returns)
    fold_c = fold[:, None, None]                          # broadcast over (E, J)
    ones = np.ones((T, E, J))

    def _g_upright() -> np.ndarray:
        g = np.zeros((T, E, 3)); g[..., 2] = -1.0
        return g

    _vel = _physical_vel                                  # PHYSICAL rad/s (shared module helper)

    def _feet(d: dict) -> dict:
        # Supply the optional end-effector channels (planted, stationary feet) so a
        # metric that reads foot_pos_b / foot_contact WITHOUT a None-guard is exercised
        # rather than auto-zeroed. Anterior foot displacement ~0 (a fold/wave/bow does
        # not kick); contact 1.0 (feet on the ground). A metric that abstains on absence
        # (the contract) is unaffected; one that hard-reads them no longer false-rejects.
        z = np.zeros((T, E, 3)); c = np.ones((T, E))
        d["left_foot_pos_b"] = z; d["right_foot_pos_b"] = z.copy()
        d["left_foot_contact"] = c; d["right_foot_contact"] = c.copy()
        return d

    # C1 — UPRIGHT pelvis dip-and-return + full-ROM joint fold (squat / sit-to-stand).
    # joints sweep 0→1.2→0 (ROM 1.2); pelvis dips 0.35 m and returns; torso stays upright.
    jp1 = 1.2 * fold_c * ones
    root1 = np.zeros((T, E, 3)); root1[..., 2] = 0.7 - 0.35 * fold[:, None]
    c1 = _feet({"joint_pos": jp1, "joint_vel": _vel(jp1),
                "projected_gravity_b": _g_upright(), "root_link_pos_w": root1})

    # C2 — forward TORSO-TILT arc (gravity tilts away and back) + ROM, stationary.
    # Lights up a toe-touch / bow / deep bend (torso pitches forward then recovers) AND
    # a skill whose competent execution leaves upright (roll, crawl, lie-and-rise).
    jp2 = 1.0 * fold_c * ones
    g2 = np.zeros((T, E, 3))
    g2[..., 2] = -1.0 + fold[:, None]                     # gz: -1→0→-1 (tilt + recover)
    g2[..., 0] = fold[:, None]                            # gx: 0→1→0
    root2 = np.zeros((T, E, 3)); root2[..., 2] = 0.6
    c2 = _feet({"joint_pos": jp2, "joint_vel": _vel(jp2),
                "projected_gravity_b": g2, "root_link_pos_w": root2})

    # C3 — GESTURE: arms RAISE to a sustained OVERHEAD posture and OSCILLATE side-to-side
    # (wave / reach / raise-arms), upright, stationary. The symmetric single-arc C1/C2
    # sweep has no sustained raise + repeated swing, so a gesture metric (overhead hold +
    # swept amplitude + repeated direction changes) needs this one. Held at ~2.2 rad
    # (clearly "arms up") + oscillating ±0.7 at a MODERATE in-band frequency (period 24
    # frames ≈ 2 Hz at dt=0.02 — inside the typical 0.3-4 Hz wave band; the prior period-12
    # ≈ 4.2 Hz fell just OUTSIDE it and a spectral-band gesture metric scored 0).
    jp3 = np.zeros((T, E, J))
    raise_env = 2.8 * np.clip(t / (0.08 * T), 0.0, 1.0)   # arms ramp UP overhead (first ~8%),
    swing = 0.7 * np.sin(2.0 * np.pi * t / 24.0)          # held so |angle| stays > 2 rad through
    for j in (6, 7, 8, 9):                                # the swing (overhead window ≫ 1.5 s);
        jp3[:, :, j] = (raise_env + swing)[:, None]       # oscillate ~2 Hz, ≥4 cycles
    root3 = np.zeros((T, E, 3)); root3[..., 2] = 0.7
    c3 = _feet({"joint_pos": jp3, "joint_vel": _vel(jp3),
                "projected_gravity_b": _g_upright(), "root_link_pos_w": root3})

    # C4 — SEQUENCED posture-then-gesture (bend / toe-touch, RECOVER to upright, THEN
    # wave): a compound "do A then B" goal whose completion gate requires a distinct
    # second phase AFTER the first returns — which no single-phase probe satisfies.
    n_bend = max(4, int(0.55 * T))                        # the bend completes in the first 55%
    barc = np.zeros(T)
    barc[:n_bend] = (1.0 - np.cos(2.0 * np.pi * np.arange(n_bend) / n_bend)) / 2.0
    g4 = np.zeros((T, E, 3))
    g4[..., 0] = barc[:, None]                            # torso tilts forward and recovers
    g4[..., 2] = -np.sqrt(np.clip(1.0 - barc ** 2, 0.0, 1.0))[:, None]
    jp4 = np.zeros((T, E, J))
    for j in (0, 1, 2, 3):                                # hips + knees flex during the bend
        jp4[:, :, j] = (1.2 * barc)[:, None]
    n0 = int(0.6 * T)                                     # the gesture starts AFTER the bend returns
    gest_raise = np.clip((t - n0) / (0.1 * T), 0.0, 1.0)
    gest = (gest_raise + 0.7 * np.sin(2.0 * np.pi * (t - n0) / 10.0)) * (t >= n0)
    for j in (6, 7, 8, 9):
        jp4[:, :, j] = gest[:, None]
    root4 = np.zeros((T, E, 3)); root4[..., 2] = 0.7 - 0.30 * barc[:, None]
    c4 = _feet({"joint_pos": jp4, "joint_vel": _vel(jp4),
                "projected_gravity_b": g4, "root_link_pos_w": root4})

    # C5 — REPEATED BIDIRECTIONAL torso/waist oscillation (twist L/R, rotate, shake),
    # upright, stationary. A repeated-motion goal needs left-AND-right excursions with
    # several reversals — which the one-sided 0→1.2→0 arc of C1/C2 (and the arm-only C3)
    # never produces on the torso joints. Oscillates the torso + neck columns ±0.6 at
    # ~1.7 Hz (period 30, ~4 cycles), structured + moderate-speed. A pure MOTION-MAGNITUDE
    # rewarder is gated by the folded `chaotic` negative (rng joint_vel σ=5 rad/s, peak
    # ~19 — comfortably above this 2-joint ±0.6 oscillation) AND the independent L1
    # `no_reward_for_chaos` axiom; NOTE the folded `upright_flail`/`active_floss` carry
    # rad/FRAME joint_vel (the fixed battery's convention — left as-is so the chaos axiom,
    # which relies on that scale, keeps working), so they guard AMPLITUDE/STRUCTURE, not
    # raw speed — `chaotic` is the speed backstop.
    jp5 = np.zeros((T, E, J))
    twist = 0.6 * np.sin(2.0 * np.pi * t / 30.0)          # bidirectional ±0.6, ≥4 reversals
    for j in (10, 11):                                    # torso + neck (synthetic body)
        jp5[:, :, j] = twist[:, None]
    root5 = np.zeros((T, E, 3)); root5[..., 2] = 0.7
    c5 = _feet({"joint_pos": jp5, "joint_vel": _vel(jp5),
                "projected_gravity_b": _g_upright(), "root_link_pos_w": root5})

    # D1 — STILL: alive-but-frozen upright stance. A still / low-motion-rewarding
    # metric scores this high → no competent-vs-degenerate separation → reject.
    jp0 = np.zeros((T, E, J)); jv0 = np.zeros((T, E, J))
    root_up = np.zeros((T, E, 3)); root_up[..., 2] = 0.7
    d1 = _feet({"joint_pos": jp0, "joint_vel": jv0,
                "projected_gravity_b": _g_upright(), "root_link_pos_w": root_up})

    # D2 — FALLEN: toppled + collapsed. A fall-rewarding metric scores this high.
    gf = np.zeros((T, E, 3)); gf[..., 0] = 1.0           # gravity sideways = toppled
    rootf = np.zeros((T, E, 3)); rootf[..., 2] = 0.3
    d2 = _feet({"joint_pos": jp0, "joint_vel": jv0,
                "projected_gravity_b": gf, "root_link_pos_w": rootf})

    def _s(a) -> float:
        try:
            v = _score(fn, a, meta)
            return float(v) if np.isfinite(v) else 0.0
        except Exception:  # noqa: BLE001 — a probe crash is "no signal", never raises
            return 0.0

    comp = max(_s(c1), _s(c2), _s(c3), _s(c4), _s(c5))
    degen = max(_s(d1), _s(d2))
    return {"competent": comp, "degenerate": degen, "spread": comp - degen}


# ── §best-of-N graded discriminator (candidate SELECTION, not validity) ─────


def _graded_fold_rung(depth: float, rom: float) -> dict:
    """An UPRIGHT fold-and-return rung (the `_selectivity_probe` C1 math, parameterized
    for grading): the pelvis dips `depth` m and returns, every joint sweeps 0→`rom`→0
    in phase, upright + stationary. depth/rom up ⇒ more competent."""
    t = np.arange(T)
    fold = (1.0 - np.cos(2.0 * np.pi * t / T)) / 2.0
    jp = rom * fold[:, None, None] * np.ones((T, E, J))
    root = np.zeros((T, E, 3)); root[..., 2] = 0.7 - depth * fold[:, None]
    return {"joint_pos": jp, "joint_vel": _physical_vel(jp),
            "projected_gravity_b": _upright_g(), "root_link_pos_w": root}


def _graded_posture_rung(tilt: float, rom: float) -> dict:
    """A NON-UPRIGHT posture-arc rung (the `_selectivity_probe` C2 math, parameterized):
    body-frame gravity tilts `tilt` away and back while joints sweep 0→`rom`→0,
    stationary. Lights up a skill whose competent execution leaves upright (roll, deep
    bow, crawl). tilt/rom up ⇒ more competent."""
    t = np.arange(T)
    fold = (1.0 - np.cos(2.0 * np.pi * t / T)) / 2.0
    jp = rom * fold[:, None, None] * np.ones((T, E, J))
    g = np.zeros((T, E, 3))
    g[..., 2] = -1.0 + tilt * fold[:, None]
    g[..., 0] = tilt * fold[:, None]
    root = np.zeros((T, E, 3)); root[..., 2] = 0.6
    return {"joint_pos": jp, "joint_vel": _physical_vel(jp),
            "projected_gravity_b": g, "root_link_pos_w": root}


def graded_discrimination(
    fn, meta, *, channel_catalog: ChannelCatalog | None = None,
) -> dict[str, Any]:
    """Deterministic, OFFLINE discrimination score for best-of-N candidate selection
    (§best-of-N): how SHARPLY + MONOTONICALLY does `fn` grade competence on a GRADED
    competence ladder (degenerate → partial → good → ideal)? Extends the goal-agnostic
    `_selectivity_probe` (competent-vs-degenerate) into a 3-rung graded ladder along
    BOTH axes it covers — an upright fold-and-return and a non-upright posture arc —
    and returns, per axis, `separation` (ideal − the degenerate floor) + `monotonicity`
    (fraction of adjacent graded steps that strictly increase). `score` is the SHARPER
    axis, so a metric that grades EITHER posture family well selects above a coarse one.

    This is NOT task-validity (that is calibration's job) — it is a goal-agnostic
    TIE-BREAK: when the probe cannot excite the metric (e.g. a kick/locomotion metric,
    blind to a smooth fold), every candidate scores ~0 and the caller keeps the first
    valid one (today's behavior). Rung scores are clamped to the spec_score [0,1]
    contract, so `separation` is bounded [−1,1] and co-scale with `monotonicity` — a
    high-amplitude coarse metric can't swamp the monotonicity term. Pure-numpy, no RNG:
    identical across calls (the selector must add no nondeterminism). Never raises (a
    crash on a rung scores 0)."""
    jp0 = np.zeros((T, E, J))
    root_up = np.zeros((T, E, 3)); root_up[..., 2] = 0.7
    still = {"joint_pos": jp0, "joint_vel": jp0,
             "projected_gravity_b": _upright_g(), "root_link_pos_w": root_up}
    gf = np.zeros((T, E, 3)); gf[..., 0] = 1.0
    rootf = np.zeros((T, E, 3)); rootf[..., 2] = 0.3
    fallen = {"joint_pos": jp0, "joint_vel": jp0,
              "projected_gravity_b": gf, "root_link_pos_w": rootf}

    def _with_case(a: dict, case: str) -> dict:
        if channel_catalog is None:
            return a
        merged = dict(a)
        first = next(iter(a.values()))
        merged.update(catalog_fixture_arrays(
            channel_catalog, time_steps=int(first.shape[0]),
            num_envs=int(first.shape[1]), case=case))
        return merged

    def _s(a, case: str = "competent") -> float:
        try:
            v = _score(fn, _with_case(a, case), meta)
            # CLAMP to the spec_score [0,1] contract: separation is then bounded
            # [−1,1] and co-scale with monotonicity, so a high-AMPLITUDE coarse metric
            # (a binary gate that returns, say, 5.0) saturates to 1.0 and cannot
            # out-rank a smooth grader on raw output scale — amplitude is an author
            # artifact, not discrimination.
            return float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else 0.0
        except Exception:  # noqa: BLE001 — a rung crash is "no signal", never raises
            return 0.0

    degen_case = "far_idle" if channel_catalog is not None else "competent"
    degen = max(_s(still, degen_case), _s(fallen, degen_case))

    def _axis(rungs: list) -> dict:
        scores = [_s(r) for r in rungs]                 # partial → good → ideal
        separation = scores[-1] - degen
        steps = sum(1 for a, b in zip(scores, scores[1:]) if b > a + 1e-4)
        monotonicity = steps / float(max(1, len(scores) - 1))
        return {"scores": [round(s, 4) for s in scores],
                "separation": round(separation, 4),
                "monotonicity": round(monotonicity, 4),
                "disc": round(separation + monotonicity, 4)}

    fold_axis = _axis([_graded_fold_rung(0.22, 0.80),
                       _graded_fold_rung(0.30, 1.05),
                       _graded_fold_rung(0.35, 1.20)])
    posture_axis = _axis([_graded_posture_rung(0.50, 0.70),
                          _graded_posture_rung(0.75, 0.90),
                          _graded_posture_rung(1.00, 1.10)])
    catalog_axis = None
    if channel_catalog is not None:
        catalog_scores = [
            _s(still, "far_idle"),
            _s(still, "edge_camping"),
            _s(still, "contact_flicker"),
            _s(still, "competent"),
        ]
        separation = catalog_scores[-1] - max(catalog_scores[:-1])
        steps = sum(
            1 for a, b in zip(catalog_scores, catalog_scores[1:])
            if b > a + 1e-4)
        catalog_axis = {
            "scores": [round(s, 4) for s in catalog_scores],
            "separation": round(separation, 4),
            "monotonicity": round(steps / 3.0, 4),
            "disc": round(separation + steps / 3.0, 4),
        }
    score = max(
        fold_axis["disc"], posture_axis["disc"],
        catalog_axis["disc"] if catalog_axis is not None else float("-inf"))
    return {"score": round(float(score), 4), "degenerate": round(degen, 4),
            "fold_axis": fold_axis, "posture_axis": posture_axis,
            "catalog_axis": catalog_axis}


def discrimination_of_metric(
    module_path: Path | str, required_roles: Optional[Sequence[str]] = None,
    *, channel_catalog: ChannelCatalog | Mapping[str, Any] | Path | str | None = None,
) -> dict[str, Any]:
    """Load a generated metric module and score it on the offline `graded_discrimination`
    ladder — the best-of-N candidate selector. Builds the same synthetic 12-joint meta +
    LENIENT role injection the non-degeneracy gate uses, so a role-based metric reads the
    right columns. Never raises: a load/score failure scores 0.0 (the candidate simply
    loses the deterministic tie-break, never crashes the generator)."""
    try:
        mod = load_generated_module(module_path)
        fn = getattr(mod, GENERATED_FN_NAME, None)
        if not callable(fn):
            return {"score": 0.0, "error": f"no callable {GENERATED_FN_NAME}"}
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "error": f"{type(e).__name__}: {e}"}
    meta = {"joint_names": list(_NAMES_12)}
    inject_joint_roles(meta, list(required_roles or []), lenient=True)
    try:
        catalog = resolve_channel_catalog(channel_catalog)
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "error": f"invalid catalog: {type(e).__name__}: {e}"}
    return graded_discrimination(fn, meta, channel_catalog=catalog)


# ── §REFERENCE_TRAJECTORY_PLAN §5: reference-anchored validation ────────


def _score_reference_entry(
    fn, clip: dict, required_roles: list[str],
) -> tuple[float, dict[str, Any]]:
    """`clip_to_arrays` + `inject_joint_roles` (lenient, on the reference's
    OWN joint_names) + `_score` for one clip. Never raises: a crash scores
    `nan` (the caller treats that as "no signal", mirroring `_score`'s
    existing archetype error path — no bypass of the bounded/determinism
    machinery)."""
    from sculptor.refs.convert import clip_to_arrays

    try:
        arrays, meta = clip_to_arrays(clip)
        inject_joint_roles(meta, required_roles, lenient=True)
        return _score(fn, arrays, meta), meta
    except Exception:  # noqa: BLE001 — "no signal", never raises out
        return float("nan"), {}


def _reference_components(fn, clip: dict, required_roles: list[str]) -> dict:
    """The metric's FULL output dict on a clip (not just spec_score) — the
    named sub-channels a generated metric returns are the only diagnostic
    window into WHY a reference scored the way it did. Fed into gate-failure
    reasons so the authoring retry loop sees e.g. `completion_gate: 0.0,
    progress_score: 0.93` instead of a bare `full 0.000` (a real fable-5
    metric died 3 straight retries on an invisible np.convolve boundary
    artifact before this existed — D12). Numeric values only, rounded;
    never raises."""
    from sculptor.refs.convert import clip_to_arrays

    try:
        arrays, meta = clip_to_arrays(clip)
        inject_joint_roles(meta, required_roles, lenient=True)
        T = next(iter(arrays.values())).shape[0]
        E = next(iter(arrays.values())).shape[1]
        out = fn(arrays, {"max_episode_steps": T, "rollout_num_envs": E,
                          "step_dt": 0.02}, meta)
        comp: dict[str, Any] = {}
        for k, v in (out or {}).items():
            try:
                fv = float(v)
                comp[k] = round(fv, 4) if np.isfinite(fv) else str(fv)
            except (TypeError, ValueError):
                comp[k] = str(v)[:60]
        return comp
    except Exception:  # noqa: BLE001 — diagnostics only, never raises out
        return {}


def _validate_references(
    fn, references: list[tuple[str, dict]], required_roles: list[str],
    *, spread_min: float, degenerate_anchor: float,
    eval_reset: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Score every reference (full clip + its `perturbation_suite`) and
    apply the three §5 gates per reference:

      1. `reference_nondegeneracy` — the full reference must score >= the
         fixed battery's degenerate anchors (still/fallen/chaotic/
         upright_flail) + `spread_min`. Replaces the vacuous-probe fallback
         as the nondegeneracy signal when a reference is attached (wired by
         the caller, which only reaches here for a family-None goal without
         a battery positive OR always when references are given —
         `validate_generated_metric` decides which; this function just
         SCORES and gates).
      2. `reference_monotonicity` — NON-INVERSION with plateau tolerance:
         trunc_25 <= trunc_50 <= trunc_75 <= full (each within 1e-6), AND
         discrimination: full >= trunc_25 + spread_min. Strict growth at
         every quarter was WRONG for real mocap (found live 2026-07-09,
         D8 in REFERENCE_BUILD_LOG): fall-and-get-up segments spend their
         first half lying still, so an HONEST righting metric scores
         trunc_25 == trunc_50 == 0.0 — a tie, not an inversion. Plateaus
         are legitimate pacing; inversions (an earlier prefix scoring
         HIGHER) and full==earliest (no discrimination) remain rejected.
      3. `reference_negatives` — none of reversal/freeze_start/freeze_end/
         shuffle may score above `degenerate_anchor + spread_min`.
      4. `reference_complete_then_hold` — a REQUIRED-HIGH positive gate
         (§D24 F3, closes the D23 diagnosis): `perturb.complete_then_hold`
         (the full clip plus an appended hold of its LAST frame — a
         completed motion whose terminal state is then sustained, NOT added
         to `perturbation_suite` since it is a positive exemplar, not a
         negative/ladder rung) must score >= `max(0.5, 0.8 * full_score)`.
         This is the exact shape of a live rollout that reaches the goal
         EARLY and holds it — a metric that only recognizes completion
         within a fixed absolute window, or that punishes post-completion
         stillness, fails here even though it passes the other three gates.
         `freeze_end` (never transitions) and `complete_then_hold` (transitions
         then holds) must coexist: the former stays a rejected negative, the
         latter is a required positive, on the same honest metric.

    `speed_slow`/`speed_fast` are SCORED AND RECORDED but NOT gated: a
    kinematic metric may legitimately score a time-scaled completion high
    (it's still the same completed motion, just faster/slower), so gating
    it would false-reject an honest metric. Revisit under the R5 adversarial
    suite if a time-scaling exploit is ever observed in practice.

      5. `reference_settled_start` (§D24 F2) — a POSITIVE gate, present
         ONLY when the caller supplies `eval_reset` (the stage's
         certified eval-reset preview — `mission_metrics._compute_
         eval_reset_preview`, threaded from `validate_generated_metric`):
         a variant built by `sculptor.refs.perturb.settled_start_hold`
         (~0.5s prepend of frame 0, root height — and orientation when
         cheaply constructible — replaced by the SETTLED eval-reset
         scalars, velocities zero in the prepend) must score
         `>= max(0.5, 0.8 * full_score)`, the same threshold family as
         `reference_complete_then_hold`. This kills the H2 class (D23,
         excluded there but observed live elsewhere on g1-standing
         feet_under_crouch): a start-window gate implicitly calibrated to
         the clip's own frame 0 that silently disagrees with the ACTUAL
         certified rollout start. When `eval_reset` is `None` (no settle
         preview available — e.g. a standing-start goal that never
         derives one, or settling was unavailable at cert time) the gate
         is OMITTED from `ref_gates` entirely (never scored as a pass OR
         a fail) and the entry instead carries
         `"settled_start_abstained": True` — an explicit, never-silent
         abstain distinguishable from a real pass.

    `speed_slow`/`speed_fast` are SCORED AND RECORDED but NOT gated: a
    kinematic metric may legitimately score a time-scaled completion high
    (it's still the same completed motion, just faster/slower), so gating
    it would false-reject an honest metric. Revisit under the R5 adversarial
    suite if a time-scaling exploit is ever observed in practice.

    Returns `(per_reference_results, all_scores)` where `all_scores` keys
    are `"reference:<id>"` / `"reference:<id>:<perturbation>"` for the
    caller's `archetype_scores`."""
    from sculptor.refs.perturb import (
        complete_then_hold,
        ends_settled,
        fast_completion,
        perturbation_suite,
    )

    results: list[dict[str, Any]] = []
    all_scores: dict[str, float] = {}
    for clip_id, clip in references:
        full_score, _ = _score_reference_entry(fn, clip, required_roles)
        all_scores[f"reference:{clip_id}"] = full_score

        try:
            suite = perturbation_suite(clip)
        except Exception:  # noqa: BLE001 — build per-entry instead: ONE
            # failing perturbation must not blank all nine (a quat edge
            # case in speed() once nan'd every truncation, making
            # monotonicity unpassable for the whole clip — D14).
            from sculptor.refs import perturb as _pt
            suite = {}
            for _name, _fn in (
                ("reversal", lambda c: _pt.time_reverse(c)),
                ("freeze_start", lambda c: _pt.freeze_start(c)),
                ("freeze_end", lambda c: _pt.freeze_end(c)),
                ("shuffle", lambda c: _pt.segment_shuffle(c)),
                # root_only only when posture channels exist (else it
                # equals the original clip — false-convicts everything).
                ("root_only", lambda c: _pt.root_motion_only(c)
                    if ("joint_pos" in c or "root_quat_wxyz" in c)
                    else (_ for _ in ()).throw(ValueError("no posture"))),
                ("speed_slow", lambda c: _pt.speed(c, 0.25)),
                ("speed_fast", lambda c: _pt.speed(c, 4.0)),
                ("trunc_25", lambda c: _pt.truncate(c, 0.25)),
                ("trunc_50", lambda c: _pt.truncate(c, 0.5)),
                ("trunc_75", lambda c: _pt.truncate(c, 0.75)),
            ):
                try:
                    suite[_name] = _fn(clip)
                except Exception:  # noqa: BLE001 — that entry scores nan
                    continue
        pert_scores: dict[str, float] = {}
        for name, pclip in suite.items():
            s, _ = _score_reference_entry(fn, pclip, required_roles)
            pert_scores[name] = s
            all_scores[f"reference:{clip_id}:{name}"] = s

        # §D24 F3: `complete_then_hold` is a POSITIVE exemplar (the reference
        # completed, terminal state then held), NOT a member of
        # `perturbation_suite` (which is negatives/ladder rungs only) — score
        # it separately here. Recorded in `archetype_scores` regardless of
        # pass/fail (a crash scores nan — "no signal", the same convention
        # every other reference score uses).
        # Scored at TWO hold ratios: x1 (hold as long as the motion) and x24.
        # The second ratio is chosen from the OPERATING ENVELOPE, not
        # arbitrarily: max episode (~500 frames) / fastest plausible
        # completion (~25 frames, the D23 live sit-up) ≈ 20, so x24 exceeds
        # every realistic hold ratio — a metric that clears it is
        # hold-invariant over the whole regime. Two verification passes
        # earned this: a single-ratio gate was narrowly passable by
        # fraction-of-episode-length window reads (forbidden by the
        # RELATIVE-TIME rule), and a x4 second ratio was still evadable by
        # denominators k in [5, 20) that collapse at the realistic 19x hold.
        # Fraction windows with k > 25 technically evade x24 too but are
        # benign at real episode shapes (their window is already smaller
        # than the motion itself) — envelope-bounded, not an arms race.
        try:
            cth_clip = complete_then_hold(clip)
            cth_score, _ = _score_reference_entry(fn, cth_clip, required_roles)
        except Exception:  # noqa: BLE001 — never let one clip's perturbation crash validation
            cth_score = float("nan")
        try:
            cth_x24_clip = complete_then_hold(clip, hold_frac=24.0)
            cth_x24_score, _ = _score_reference_entry(
                fn, cth_x24_clip, required_roles)
        except Exception:  # noqa: BLE001 — same convention as above
            cth_x24_score = float("nan")
        pert_scores["complete_then_hold"] = cth_score
        all_scores[f"reference:{clip_id}:complete_then_hold"] = cth_score
        pert_scores["complete_then_hold_x24"] = cth_x24_score
        all_scores[f"reference:{clip_id}:complete_then_hold_x24"] = cth_x24_score

        # §D24 live finding #3: `fast_completion` — the reference sped 16x
        # then held (motion ~5% of the trajectory). A real policy completed
        # the 8.1 s torso-righting span in ~0.5 s and a freshly certified
        # metric zeroed it: its start-state read was a 0.5 s WINDOW MEAN,
        # so the completed state leaked into the "start" window. Scored for
        # every reference; GATED only when the reference itself is
        # reach-and-hold shaped (`ends_settled`) — pace-sensitive motions
        # (gait/rhythm) stay record-only, preserving D3's false-reject
        # protection in the one place it actually binds.
        try:
            fc_clip = fast_completion(clip)
            fc_score, _ = _score_reference_entry(fn, fc_clip, required_roles)
        except Exception:  # noqa: BLE001 — same convention as above
            fc_score = float("nan")
        try:
            fc_gated = bool(ends_settled(clip))
        except Exception:  # noqa: BLE001 — an unclassifiable clip abstains
            fc_gated = False
        pert_scores["fast_completion"] = fc_score
        all_scores[f"reference:{clip_id}:fast_completion"] = fc_score

        ref_gates: dict[str, bool] = {}
        ref_reasons: list[str] = []

        finite_full = np.isfinite(full_score)
        nondegen_ok = bool(
            finite_full and full_score >= degenerate_anchor + spread_min)
        ref_gates["reference_nondegeneracy"] = nondegen_ok
        if not nondegen_ok:
            ref_reasons.append(
                f"[reference:{clip_id}] nondegeneracy: full-reference score "
                f"{full_score:.3f} does not beat the battery degenerate "
                f"anchor {degenerate_anchor:.3f} + spread_min {spread_min} "
                f"— the metric does not clearly reward the reference motion")

        t75 = pert_scores.get("trunc_75", float("nan"))
        t50 = pert_scores.get("trunc_50", float("nan"))
        t25 = pert_scores.get("trunc_25", float("nan"))
        mono_terms = [full_score, t75, t50, t25]
        # Non-inversion (plateaus allowed): a later prefix may TIE an
        # earlier one (real mocap paces unevenly — a get-up segment can
        # lie still for its whole first half, honestly scoring 0.0 at 25%
        # AND 50%), but must never score LOWER. Discrimination: the full
        # clip must clearly beat the earliest prefix, or the metric can't
        # tell completion from onset (a constant metric dies here).
        _eps = 1e-6
        mono_ok = all(np.isfinite(v) for v in mono_terms) and (
            t25 <= t50 + _eps
            and t50 <= t75 + _eps
            and t75 <= full_score + _eps
            and full_score >= t25 + spread_min)
        ref_gates["reference_monotonicity"] = mono_ok
        if not mono_ok:
            ref_reasons.append(
                f"[reference:{clip_id}] monotonicity: expected no inversion "
                f"(trunc_25 {t25:.3f} <= trunc_50 {t50:.3f} <= trunc_75 "
                f"{t75:.3f} <= full {full_score:.3f}, plateaus allowed) AND "
                f"full >= trunc_25 + spread_min ({spread_min}) — the metric "
                f"either inverts partial-completion order or cannot "
                f"discriminate completion from onset")

        # root_only (D18, audit-proven): the clip's root trajectory with
        # posture frozen at frame 0 — a displacement-only metric scores
        # it like the real motion and dies here; the honest metric sees
        # no posture change and scores it degenerate.
        neg_names = ("reversal", "freeze_start", "freeze_end", "shuffle",
                     "root_only")
        neg_ceiling = degenerate_anchor + spread_min
        offenders = [
            n for n in neg_names
            if np.isfinite(pert_scores.get(n, float("nan")))
            and pert_scores[n] > neg_ceiling
        ]
        ref_gates["reference_negatives"] = not offenders
        if offenders:
            ref_reasons.append(
                f"[reference:{clip_id}] negatives: {offenders} scored above "
                f"the degenerate ceiling {neg_ceiling:.3f} "
                f"({ {n: round(pert_scores[n], 3) for n in offenders} }) "
                f"— gameable by a reversed/frozen/shuffled near-miss of the "
                f"reference")

        # §D24 F3: `reference_complete_then_hold` — a REQUIRED-HIGH positive
        # gate (the opposite polarity of `reference_negatives` above): the
        # reference completed then held its terminal state must score >=
        # max(0.5, 0.8 * full_score). Named numbers ride the failure reason
        # (D12 idiom) so the authoring retry loop sees exactly what fell
        # short instead of a bare fail.
        cth_threshold = max(0.5, 0.8 * full_score) if finite_full else float("inf")
        cth_ok = bool(
            finite_full and np.isfinite(cth_score)
            and np.isfinite(cth_x24_score)
            and cth_score >= cth_threshold - 1e-9
            and cth_x24_score >= cth_threshold - 1e-9)
        ref_gates["reference_complete_then_hold"] = cth_ok
        if not cth_ok:
            ref_reasons.append(
                f"[reference:{clip_id}] complete_then_hold: scores x1 "
                f"{cth_score:.3f} / x24 {cth_x24_score:.3f} must BOTH reach "
                f"{cth_threshold:.3f} (max(0.5, 0.8 * full {full_score:.3f})) "
                f"— the metric does not score a reach-then-hold completion "
                f"of the reference as the positive exemplar it is (D23: this "
                f"is the exact shape of a live rollout that reaches the goal "
                f"early and holds it, which scored spec 0.0; x24 exceeds the "
                f"realistic episode/motion hold ratio, so window reads scaled "
                f"to episode length cannot survive it)")

        # §D24 live finding #3: `reference_fast_completion` — REQUIRED-HIGH
        # positive, but ONLY for reach-and-hold-shaped references
        # (`ends_settled`); otherwise recorded with an explicit abstain
        # marker (D3: convicting a pace-sensitive metric on a 16x-sped
        # gait clip would be a false reject). Threshold family identical
        # to complete_then_hold.
        if fc_gated:
            fc_ok = bool(
                finite_full and np.isfinite(fc_score)
                and fc_score >= cth_threshold - 1e-9)
            ref_gates["reference_fast_completion"] = fc_ok
            if not fc_ok:
                ref_reasons.append(
                    f"[reference:{clip_id}] fast_completion: score "
                    f"{fc_score:.3f} must reach {cth_threshold:.3f} "
                    f"(max(0.5, 0.8 * full {full_score:.3f})) — the policy "
                    f"may complete the transition MUCH faster than the "
                    f"reference (a live rollout finished an 8.1 s righting "
                    f"span in ~0.5 s and was zeroed): start-state reads "
                    f"must use the EARLIEST frames or the EVAL START STATE "
                    f"numbers, never a wide start-window mean that a fast "
                    f"completion finishes inside")
        else:
            pert_scores["fast_completion_abstained"] = True

        # §D24 F2: `reference_settled_start` — see this function's
        # docstring for the full rationale. `eval_reset` carries the
        # SETTLED (or best-effort unsettled) scalars from `mission_
        # metrics._compute_eval_reset_preview`; absence means abstain,
        # never a silent pass/fail.
        settled_scalars = (eval_reset or {}).get("scalars") if eval_reset else None
        settled_start_abstained = not settled_scalars
        settled_start_orientation_adjusted: Optional[bool] = None
        if settled_scalars:
            ss_build_error: Optional[str] = None
            try:
                from sculptor.reference import G1_CLASS_STAND_M
                from sculptor.refs.perturb import settled_start_hold

                settled_z = G1_CLASS_STAND_M + float(
                    settled_scalars.get("reset_height_offset_m", 0.0))
                ss_clip, settled_start_orientation_adjusted = settled_start_hold(
                    clip, z=settled_z,
                    pitch=settled_scalars.get("reset_pitch_offset_rad"),
                    roll=settled_scalars.get("reset_roll_offset_rad"))
                ss_score, _ = _score_reference_entry(
                    fn, ss_clip, required_roles)
            except Exception as e:  # noqa: BLE001 — never crash validation
                ss_score = float("nan")
                settled_start_orientation_adjusted = False
                ss_build_error = f"{type(e).__name__}: {e}"

            ss_threshold = (
                max(0.5, 0.8 * full_score) if finite_full else float("inf"))
            ss_ok = bool(
                finite_full and np.isfinite(ss_score)
                and ss_score >= ss_threshold - 1e-9)
            ref_gates["reference_settled_start"] = ss_ok
            pert_scores["settled_start"] = ss_score
            all_scores[f"reference:{clip_id}:settled_start"] = ss_score
            if not ss_ok:
                extra = f" (build error: {ss_build_error})" if ss_build_error else ""
                ref_reasons.append(
                    f"[reference:{clip_id}] settled_start: score "
                    f"{ss_score:.3f} must reach {ss_threshold:.3f} "
                    f"(max(0.5, 0.8 * full {full_score:.3f})){extra} — the "
                    f"metric does not score the reference motion starting "
                    f"from the stage's ACTUAL certified eval-reset state as "
                    f"highly as the reference itself (H2/D23: a start-"
                    f"window gate calibrated to the clip's own frame 0 can "
                    f"silently mis-score a rollout that actually begins "
                    f"from the settled reset)")

        entry: dict[str, Any] = {
            "clip_id": clip_id,
            "gates": ref_gates,
            "reasons": ref_reasons,
            "scores": {"full": full_score, **pert_scores},
        }
        if settled_start_abstained:
            entry["settled_start_abstained"] = True
        else:
            entry["settled_start_orientation_adjusted"] = (
                settled_start_orientation_adjusted)
        if not all(ref_gates.values()):
            # Diagnostic window for the authoring retry loop (D12): the
            # metric's OWN named sub-channels on the full reference tell
            # the author WHY it failed (e.g. completion_gate 0.0 while
            # progress 0.93 = a gating/boundary bug, not a wrong signal).
            comp = _reference_components(fn, clip, required_roles)
            if comp:
                entry["full_components"] = comp
                ref_reasons.append(
                    f"[reference:{clip_id}] full-reference sub-components "
                    f"(the metric's own outputs on the real exemplar — "
                    f"diagnose which channel/gate zeroed it): {comp}")
        results.append(entry)
    return results, all_scores


def validate_generated_metric(
    source: str,
    module_path: Path | str,
    *,
    spread_min: float = 0.1,
    distractor_ceiling: float = 0.3,
    behavior_goal: Optional[str] = None,
    robot_hint: Optional[str] = None,
    robot_joint_names: Optional[Sequence[str]] = None,
    references: Optional[list[tuple[str, dict]]] = None,
    eval_reset: Optional[dict[str, Any]] = None,
    channel_catalog: ChannelCatalog | Mapping[str, Any] | Path | str | None = None,
) -> dict[str, Any]:
    """Run all MUST-HAVE gates on a generated metric. `source` is the
    module text (for static gates); `module_path` is where it's been
    written (for the runtime gates). Returns
    `{ok, gates, reasons, archetype_scores, family, required_roles}`.
    Never raises — a crashing metric is a failed gate, not an exception.

    §Ship 41: `behavior_goal`/`robot_hint` (optional) resolve a behavior
    FAMILY so the non-degeneracy gate anchors a non-locomotion metric
    (kick/jump/floss) against a behavior-appropriate positive archetype
    instead of a hard-coded forward-walker. Both default `None` → today's
    behavior (any of the four positives may anchor the metric).

    §Ship 49: `robot_joint_names` (optional — the ACTUAL robot's joint_names,
    sourced from the manifest at launch) gates the metric's declared
    `REQUIRED_JOINT_ROLES`: every role must resolve to exactly one joint on
    THIS robot, or the metric is rejected pre-project. Plus two new gates that
    run regardless: a static ban on hard-coded integer joint indices and a
    permutation-robustness check (a metric that reads joints by index rather
    than name swings when the joint axis is relabelled).

    §REFERENCE_TRAJECTORY_PLAN §5: `references` (optional — a list of
    `(clip_id, loaded clip dict)` pairs) attaches reference-anchored
    validation, ADDITIVE to everything above (nothing changes when
    `references` is omitted). Each reference contributes FOUR new gates
    (`reference_nondegeneracy`, `reference_monotonicity`,
    `reference_negatives`, `reference_complete_then_hold` — see their
    per-function docstrings below) scored with `sculptor.refs.convert
    .clip_to_arrays` + `sculptor.refs.perturb.perturbation_suite`
    (`complete_then_hold` scored separately — it is a positive exemplar, not
    a perturbation-suite member), using the reference's OWN meta (the fixed battery
    keeps its own synthetic meta). When any reference is attached, it
    REPLACES the goal-agnostic selectivity-probe fallback as the
    nondegeneracy signal for a family-`None` goal (a real exemplar beats a
    synthetic probe); the probe still runs when no reference is given. The
    result gains a `"references"` key: a list of
    `{"clip_id", "gates", "scores"}` per reference. Overall `ok` requires
    every per-reference gate to pass too.

    §D24 F2: `eval_reset` (optional — the stage's certified eval-reset
    preview, `mission_metrics._compute_eval_reset_preview`'s return
    value, shape `{"scalars": {...}, "settled": bool, "reason": ...}`)
    ADDS a fifth per-reference gate, `reference_settled_start` (see
    `_validate_references`'s docstring). `None` (the default) → each
    reference's entry instead carries `"settled_start_abstained": True`
    and NO `reference_settled_start` key lands in `gates` — never a
    silent pass or fail, and byte-identical to pre-F2 behavior when this
    argument is omitted entirely (aside from that one advisory key)."""
    gates: dict[str, bool] = {}
    reasons: list[str] = []
    try:
        catalog = resolve_channel_catalog(channel_catalog)
    except Exception as e:  # noqa: BLE001 - invalid artifact is a hard contract fail
        return {
            "ok": False,
            "gates": {"channel_catalog": False},
            "reasons": [f"[channel-catalog] {type(e).__name__}: {e}"],
            "archetype_scores": {},
            "family": resolve_behavior_family(behavior_goal, robot_hint),
            "required_roles": [],
            "references": [],
            "channel_catalog_hash": None,
        }
    if catalog is not None:
        gates["channel_catalog"] = True
    family = resolve_behavior_family(behavior_goal, robot_hint)
    # §LAW 0: the goal frame scopes the directional / support gates so a novel
    # rearward / single-support / non-upright task is never false-rejected.
    frame = resolve_goal_frame(behavior_goal, robot_hint)
    required_roles: list[str] = []

    # 1. AST safety
    safety = _ast_safety(source)
    gates["ast_safety"] = not safety
    reasons += [f"[safety] {s}" for s in safety]

    # contract: must define compute_spec
    gates["defines_compute_spec"] = (GENERATED_FN_NAME in source)
    if GENERATED_FN_NAME not in source:
        reasons.append(f"[contract] missing def {GENERATED_FN_NAME}(arrays, behavior, meta)")

    # 2. Array-contract
    referenced_keys = _referenced_array_keys(source)
    allowed_arrays = set(ALLOWED_ARRAYS)
    if catalog is not None:
        allowed_arrays.update(catalog.names())
    bad_keys = referenced_keys - allowed_arrays
    gates["array_contract"] = not bad_keys
    if bad_keys:
        reasons.append(f"[contract] references unavailable arrays: {sorted(bad_keys)} "
                       f"(allowed: {sorted(allowed_arrays)})")

    # Authored task metrics must consume the compiler's hold-qualified success
    # channel when one exists.  Distance/inside alone admit the two canonical
    # hacks: camping on the region boundary and transient predicate flicker.
    if catalog is not None:
        success_names = {
            channel.name for channel in catalog.channels
            if channel.producer == "success_hold"
        }
        completion_ok = not success_names or bool(referenced_keys & success_names)
        gates["catalog_completion_channel"] = completion_ok
        if not completion_ok:
            reasons.append(
                "[channel-catalog] metric does not reference a hold-qualified "
                f"success channel; expected one of {sorted(success_names)}")
        dynamic_access = _catalog_array_access_violations(source)
        gates["catalog_literal_array_access"] = not dynamic_access
        reasons.extend(
            f"[channel-catalog] {problem}" for problem in dynamic_access)

    # §Ship 49: static ban on hard-coded integer joint indices.
    raw_idx = _raw_joint_index_violations(source)
    gates["no_raw_joint_index"] = not raw_idx
    reasons += [f"[joint-index] {r}" for r in raw_idx]

    # Static gates must pass before we exec the module.
    if not (gates["ast_safety"] and gates["defines_compute_spec"]):
        return {"ok": False, "gates": gates, "reasons": reasons,
                "archetype_scores": {}, "family": family,
                "required_roles": required_roles, "references": [],
                "channel_catalog_hash": (
                    catalog.catalog_hash if catalog is not None else None)}

    try:
        mod = load_generated_module(module_path)
        fn = getattr(mod, GENERATED_FN_NAME, None)
        if not callable(fn):
            raise ValueError(f"no callable {GENERATED_FN_NAME}()")
    except Exception as e:  # noqa: BLE001
        gates["loads"] = False
        reasons.append(f"[load] {type(e).__name__}: {e}")
        return {"ok": False, "gates": gates, "reasons": reasons,
                "archetype_scores": {}, "family": family,
                "required_roles": required_roles, "references": [],
                "channel_catalog_hash": (
                    catalog.catalog_hash if catalog is not None else None)}
    gates["loads"] = True
    required_roles = read_required_roles(mod)

    # §Ship 49: required-roles gate. When the ACTUAL robot's joint_names are
    # known (manifest at launch), every declared role must resolve to exactly
    # one joint on this robot — else reject pre-project ("metric needs
    # swing_hip_pitch; robot exposes no matching joint"). When the robot is
    # unknown we cannot reject (the runtime resolution + permutation gate are
    # the backstop), so the gate passes informationally.
    if required_roles and robot_joint_names:
        rr = resolve_joint_roles(list(robot_joint_names), required_roles)
        gates["joint_roles_resolve"] = rr.ok
        if not rr.ok:
            reasons += [f"[joint-roles] {p}" for p in rr.problems()]

    # Archetypes run on the synthetic 12-joint biped. Inject the metric's
    # declared roles LENIENTLY (the synthetic body has no roll/yaw columns,
    # so an anatomically valid roll/yaw role still maps to its segment) so a
    # role-based metric can score the battery.
    meta = {"joint_names": list(_NAMES_12)}
    inject_joint_roles(meta, required_roles, lenient=True)
    arche = _archetypes()
    abstract_program = _abstract_objective_program(
        behavior_goal, getattr(mod, "ABSTRACT_OBJECTIVE", None))
    abstract_probe = _abstract_objective_probe(abstract_program)
    # The prompt-derived probe anchors non-degeneracy ONLY for a genuine TRAVERSAL /
    # parkour goal, where its ROOT-trajectory signature (climb height + a forward
    # jump-off) is hard to game. For an in-place gesture or a recognized family, the
    # FIXED family positive is the authoritative, hard-to-game anchor; the probe there
    # is mere joint oscillation that a flail/energy rewarder scores high — so, added as
    # a universal positive, it MASKS a still/flail negative (the live 4f1dfef/a6e2eec
    # regression that broke the walker-fold and flail-under-kick gates) instead of
    # discriminating. Those goals keep the fixed battery + vacuous/selectivity path.
    abstract_is_traversal = (
        abstract_probe is not None
        and ("climb" in abstract_program or "jump_off" in abstract_program))
    if abstract_is_traversal:
        arche["prompt_competent"] = abstract_probe
    catalog_cases: dict[str, str] = {}
    if catalog is not None:
        catalog_cases = {
            "catalog_far_idle": "far_idle",
            "catalog_edge_camping": "edge_camping",
            "catalog_contact_flicker": "contact_flicker",
            "catalog_forbidden_contact": "forbidden_contact",
            "catalog_competent": "competent",
        }
        # Base archetypes keep their physical diversity while carrying a
        # deterministic task state.  Positives represent a completed authored
        # task; negatives represent no progress.  The named catalog fixtures
        # below isolate the authored-channel failure modes directly.
        for name, arrays in arche.items():
            case = ("competent" if name in {
                "active", "active_kick", "active_floss", "active_jump",
                "active_parkour", "prompt_competent",
            } else "far_idle")
            arrays.update(catalog_fixture_arrays(
                catalog, time_steps=T, num_envs=E, case=case))
        for name, case in catalog_cases.items():
            fixture_base = {
                key: value.copy() for key, value in arche["still"].items()
                if key in ALLOWED_ARRAYS
            }
            fixture_base.update(catalog_fixture_arrays(
                catalog, time_steps=T, num_envs=E, case=case))
            arche[name] = fixture_base
    scores: dict[str, float] = {}

    # 3 + 4: determinism + bounded/finite, over every archetype.
    determ = True
    bounded = True
    for name, arrays in arche.items():
        try:
            s1 = _score(fn, arrays, meta)
            s2 = _score(fn, arrays, meta)
            s3 = _score(fn, arrays, meta)
        except Exception as e:  # noqa: BLE001
            bounded = False
            reasons.append(f"[run] raised on '{name}': {type(e).__name__}: {e}")
            scores[name] = float("nan")
            continue
        scores[name] = s1
        if not (s1 == s2 == s3):
            determ = False
            reasons.append(f"[determinism] '{name}' varied across runs: {s1},{s2},{s3}")
        if not (np.isfinite(s1) and 0.0 <= s1 <= 1.0):
            bounded = False
            reasons.append(f"[bounds] '{name}' out of [0,1] or non-finite: {s1}")
    gates["determinism"] = determ
    gates["bounded"] = bounded

    # §Ship 49: permutation-robustness — relabel the joint axis (names AND the
    # joint_pos/joint_vel columns, CONSISTENTLY) and re-score. A metric that
    # reads joints by NAME/role follows the relabelling to the same physical
    # joints → invariant; a metric that hard-codes a column reads a different
    # joint → its score swings. This is exactly the §3A shuffle experiment
    # turned into a gate (an index-sensitive metric silently mis-scores the
    # moment the real robot's joint order differs from the synthetic battery).
    perm = list(range(J - 1, -1, -1))           # reverse: a non-trivial relabel
    pmeta = {"joint_names": [_NAMES_12[perm[i]] for i in range(J)]}
    inject_joint_roles(pmeta, required_roles, lenient=True)
    robust = True
    for name, arrays in arche.items():
        base = scores.get(name, float("nan"))
        if not np.isfinite(base):
            continue
        try:
            ps = _score(fn, _permute_joint_arrays(arrays, perm), pmeta)
        except Exception as e:  # noqa: BLE001 — a crash under relabel = not robust
            robust = False
            reasons.append(f"[robustness] raised under joint relabel on "
                           f"'{name}': {type(e).__name__}: {e}")
            break
        if not np.isfinite(ps) or abs(ps - base) > 1e-6:
            robust = False
            reasons.append(
                f"[robustness] index-sensitive joint access: '{name}' scored "
                f"{base:.4f} but {ps:.4f} after a consistent joint relabel — "
                f"the metric reads joints by column, not by name "
                f"(use meta['joint_roles'])")
            break
    gates["joint_index_robust"] = robust

    # 5: non-degeneracy — the metric must score SOME competent behavior above
    # EVERY degenerate one, with spread. §Ship 41: positives span all behavior
    # families (locomotion `active` + kick/floss/jump) so a non-locomotion
    # metric isn't measured against a forward-walker. §Ship 41 review: the
    # resolved `family` does NOT NARROW this smell-test — narrowing falsely
    # rejected good metrics whose goal mis-resolved (e.g. "Hopper"→jump, or a
    # compound "walk forward and kick"). The metric passes if ANY positive
    # beats the negatives; `family` only selects the calibration ground truth
    # downstream (the firewall enforces task-validity). NEGATIVES that must
    # lose now include `chaotic` (upright random thrashing — the HIGHEST peak
    # joint speed of any archetype), so a peak-speed reward-hack (which scores
    # chaotic above the real positives) is rejected — closing the
    # stand-and-thrash bypass the review found.
    positive_keys = (
        "active", "active_kick", "active_floss", "active_jump",
    )
    if abstract_is_traversal:
        # Compound-traversal positives (the generic `active_parkour` exemplar + the
        # prompt-derived `prompt_competent`) anchor a parkour metric. BOTH travel
        # forward, so they are scoped to a real traversal goal — as universal
        # positives they rescued an in-place metric that merely rewards walking
        # (the walker-fold bypass) and masked the flail/still negatives.
        positive_keys += ("active_parkour", "prompt_competent")
    negative_keys = ("still", "fallen", "upright_flail", "chaotic")

    nondegen = True
    vacuous = False
    selectivity: Optional[dict[str, float]] = None
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    # §<novel-task fix>: when the goal resolves to NO family AND no fixed POSITIVE
    # archetype represents it, the fixed battery cannot REPRESENT this goal — a
    # SELECTIVE novel metric (e.g. toe-touch, gated on a pelvis dip-and-return no
    # archetype performs) is otherwise false-rejected as "near-constant", and the
    # run continues blind. Defer to a goal-agnostic selectivity probe: pass IFF the
    # metric clearly separates a competent behavior from the degenerate ones.
    # Task-VALIDITY (does it match the goal) is enforced downstream by task-derived
    # calibration; the firewall keeps an uncalibrated metric observe-only, so a
    # vacuous pass never grants steering on its own.
    #
    # §<vacuous-entry fix>: enter the goal-agnostic probe ONLY when the goal resolves
    # to NO family AND every POSITIVE archetype scored ~0 (the fixed battery genuinely
    # cannot represent the goal). Scoping to family is None keeps the family ground
    # truth authoritative for a RECOGNIZED family — a kick/jump/floss/locomotion metric
    # that scores its family positive ~0 is degenerate FOR ITS FAMILY and is rejected on
    # the normal path (incl the kick-hack / walker ceilings), NOT vacuously rescued by a
    # goal-agnostic gesture probe (the round-3-review regression: an arms-overhead metric
    # must NOT pass for a kick goal). Mis-resolution that previously forced a FOLD goal
    # into a family ("bend forward"→locomotion) is fixed at the source in
    # resolve_behavior_family, so a true novel fold/gesture goal resolves to None and
    # reaches the probe here. Keying on POSITIVES (not the whole battery) still lets a
    # novel metric that gave a fixed NEGATIVE a little credit enter the probe (the
    # "negative ≥ best-positive 0≥0 tie" false-reject). The fixed NEGATIVES are FOLDED
    # INTO the probe's degenerate anchor (a flail/chaos/fall rewarder is still rejected);
    # the forward-WALKER too, UNLESS the goal is forward-directional (goal_axis "+x").
    pos_finite = [finite[k] for k in positive_keys if k in finite]
    # The labelled degenerate anchors folded into the vacuous check (name → score), so
    # a reject can NAME the offending degenerate (actionable feedback) instead of a
    # blanket "near-constant".
    neg_anchors: dict[str, float] = {k: finite[k] for k in negative_keys if k in finite}
    if frame.get("goal_axis") != "+x" and "walker" in finite:
        neg_anchors["walker"] = finite["walker"]
    best_pos_battery = max(pos_finite) if pos_finite else 0.0
    battery_spread = (
        (max(finite.values()) - min(finite.values())) if len(finite) >= 3
        else 0.0)
    # §REFERENCE_TRAJECTORY_PLAN §5.1: an attached reference REPLACES the
    # goal-agnostic probe/fixed-battery fallback as the nondegeneracy signal
    # for a family-None goal — a real exemplar beats both a synthetic probe
    # AND the fixed battery's own spread check (which has no positive
    # exemplar for a genuinely novel motion, e.g. get-up: every archetype
    # sits near a fixed standing/fallen height, so the battery is
    # uninformative even though no single positive is exactly ~0). Deferred
    # ENTIRELY to `_validate_references`'s three gates below; the fixed
    # battery's own nondegeneracy check is skipped in this branch (both
    # spread-based paths below), NOT run in addition to it.
    # NOT gated on `family is None` (live D28 finding, g1-standing prone
    # mission): family inference is a keyword guess, and "drives up through
    # the legs ... keep balance" routed two GET-UP stages to family "jump",
    # whose battery scored every archetype 0.000 — uninformative — while
    # ALL SIX reference gates passed. The old `family is None` term then
    # blocked this defer and the uninformative battery rejected a
    # reference-clean metric as "near-constant". A real exemplar beats a
    # keyword-guessed family battery WHENEVER that battery carries no
    # signal; when the family battery IS informative (spread >= spread_min
    # and a real positive), nothing defers and family discipline holds.
    reference_anchored = (
        references and len(finite) >= 3
        and (best_pos_battery <= _BATTERY_NEAR_ZERO
             or battery_spread < spread_min))
    battery_uninformative = (
        not references and family is None and len(finite) >= 3
        and best_pos_battery <= _BATTERY_NEAR_ZERO)
    if reference_anchored:
        _fam_note = (
            f" (keyword-inferred family {family!r} carries no battery "
            f"signal for this goal — exemplar wins)" if family else "")
        reasons.append(
            "[nondegeneracy] deferred to attached reference(s): the fixed "
            f"archetype battery is uninformative for this novel goal "
            f"(best positive {best_pos_battery:.3f}, spread "
            f"{battery_spread:.3f} < {spread_min}){_fam_note} — "
            "nondegeneracy is decided by the reference_nondegeneracy/"
            "monotonicity/negatives gates below")
    elif battery_uninformative:
        probe = _selectivity_probe(fn, meta)
        selectivity = probe
        anchors = {"probe_degenerate": probe["degenerate"], **neg_anchors}
        anchor_name = max(anchors, key=lambda k: anchors[k])
        degen_anchor = anchors[anchor_name]
        if (probe["competent"] >= spread_min
                and (probe["competent"] - degen_anchor) >= spread_min):
            vacuous = True
            reasons.append(
                f"[nondegeneracy] vacuous pass: the fixed archetype battery is "
                f"uninformative for this novel goal (no positive > {_BATTERY_NEAR_ZERO}), "
                f"but the metric IS selective on the goal-agnostic probe (competent "
                f"{probe['competent']:.3f} vs degenerate {degen_anchor:.3f}) "
                f"— task-validity deferred to task-derived calibration")
        elif probe["competent"] < spread_min:
            nondegen = False
            reasons.append(
                f"[nondegeneracy] near-constant metric: fixed battery uninformative "
                f"AND not selective on the probe (competent {probe['competent']:.3f}, "
                f"degenerate {degen_anchor:.3f}) — no signal")
        else:
            # Selective on the probe, but a DEGENERATE archetype scores nearly as high
            # — the metric is GAMEABLE by that behavior. Name it so a regeneration can
            # add the distinguishing requirement (e.g. a fast whole-body 'upright_flail'
            # beats a twist metric that does not bound oscillation frequency / isolate
            # the joint; 'walker' beats an in-place metric that credits forward travel).
            nondegen = False
            reasons.append(
                f"[nondegeneracy] gameable: the metric scores a competent probe "
                f"({probe['competent']:.3f}) but the degenerate '{anchor_name}' "
                f"({degen_anchor:.3f}) scores nearly as high — distinguish the goal "
                f"from '{anchor_name}' (e.g. bound oscillation frequency, isolate the "
                f"goal joint(s), or veto base travel)")
    elif len(finite) < 3:
        nondegen = False
        reasons.append("[nondegeneracy] too few finite archetype scores")
    else:
        spread = max(finite.values()) - min(finite.values())
        if spread < spread_min:
            nondegen = False
            reasons.append(f"[nondegeneracy] near-constant metric "
                           f"(spread {spread:.3f} < {spread_min}) — no signal")
        pos = {k: finite[k] for k in positive_keys if k in finite}
        if pos:
            best_key = max(pos, key=lambda k: pos[k])
            best_pos = pos[best_key]
        else:
            best_key, best_pos = None, float("-inf")
            nondegen = False
            reasons.append("[nondegeneracy] no positive archetype scored finite")
        for low in negative_keys:
            if low in finite and finite[low] >= best_pos:
                nondegen = False
                reasons.append(
                    f"[nondegeneracy] '{low}' ({finite[low]:.3f}) scores >= the "
                    f"best positive '{best_key}' ({best_pos:.3f}) — rewards the "
                    f"wrong behavior")
        # §Ship 47: stationary-skill walker ceiling. For kick/floss/jump a
        # forward WALKER must score below an ABSOLUTE ceiling — a metric that
        # gives a walker substantial credit is rewarding locomotion, not the
        # skill (the g1-kick-v3 0.59 Goodhart: gen_005 scored a non-kicking
        # walker ~0.59, yet its `active_kick` 0.90 kept it above every negative
        # via the relative check above, so only an absolute ceiling catches
        # it). Scoped to stationary families so a locomotion metric (walker =
        # target) and an unresolved goal (family None) are never false-rejected.
        if (family in _STATIONARY_FAMILIES and "walker" in finite
                and finite["walker"] > distractor_ceiling):
            nondegen = False
            reasons.append(
                f"[nondegeneracy] forward-walker 'walker' ({finite['walker']:.3f}) "
                f"scores above the {distractor_ceiling} ceiling for the stationary "
                f"'{family}' skill — the metric rewards walking, not the behavior")
        # §Metric-quality laws: kick-family hack ceiling. A FORWARD-kick metric
        # must score the documented g1-kick-v5 hacks BELOW its kick positive —
        # a rear/sideways kick (wrong direction, LAW 4), a one-leg balance (no
        # launch, LAW 1) and a sub-amplitude partial (LAW 2). Scoped to the kick
        # family so a single-support or rearward-motion NOVEL task is never
        # false-rejected (LAW 0). A metric that can't separate these from a real
        # forward kick is gameable by exactly the behavior that stalled v5.
        if family == "kick":
            kick_pos = finite.get("active_kick", float("-inf"))
            # §LAW 0: the directional / support gates fire ONLY when the goal
            # FRAME calls for them. A forward-axis goal must beat the rear-kick
            # archetype; a double-support skill must beat the one-leg balance.
            # A rearward (mule) kick (goal_axis "-x") or a single-support kick
            # VARIANT abstains on the matching gate, so a novel kick is not
            # false-rejected. Completion/amplitude (the partial rep) is
            # frame-independent — a kick is always a completed, full-amplitude
            # motion, forward or not.
            hacks = ["partial_kick"]
            if frame.get("goal_axis") == "+x":
                hacks.append("active_kick_behind")
            if frame.get("support_mode") == "double":
                hacks.append("one_leg_balance")
            for hack in hacks:
                hv = finite.get(hack)
                if (hv is not None and kick_pos != float("-inf")
                        and hv >= kick_pos - 1e-9):
                    nondegen = False
                    reasons.append(
                        f"[nondegeneracy] kick hack '{hack}' ({hv:.3f}) scores "
                        f">= the kick positive 'active_kick' ({kick_pos:.3f}) — "
                        f"gameable; needs signed forward direction (foot_pos_b) "
                        f"+ a completion gate + an amplitude floor")
    gates["nondegeneracy"] = nondegen

    if catalog is not None:
        competent_score = scores.get("catalog_competent", float("nan"))
        catalog_degenerate_ok = np.isfinite(competent_score)
        for name in (
            "catalog_far_idle", "catalog_edge_camping",
            "catalog_contact_flicker", "catalog_forbidden_contact",
        ):
            degenerate_score = scores.get(name, float("nan"))
            if (not np.isfinite(degenerate_score)
                    or competent_score - degenerate_score < spread_min):
                catalog_degenerate_ok = False
                reasons.append(
                    f"[channel-catalog] {name} scored {degenerate_score:.3f} "
                    f"vs competent {competent_score:.3f}; require separation "
                    f">= {spread_min:.3f}")
        gates["catalog_degenerate_fixtures"] = catalog_degenerate_ok

    # §REFERENCE_TRAJECTORY_PLAN §5: reference-anchored validation — ADDITIVE,
    # only runs when the caller attaches reference clip(s) and the metric
    # loaded + ran cleanly (mirrors the axioms gate's guard below: nothing to
    # anchor against if every archetype crashed).
    reference_results: list[dict[str, Any]] = []
    if references and gates.get("bounded") and gates.get("loads"):
        # "the fixed battery's degenerate anchors" per §5.1 — max over
        # still/fallen/chaotic/upright_flail, whichever scored finite (a
        # metric that crashed on all four has no anchor to compare against,
        # so the reference gates fall back to 0.0 — any finite positive
        # reference score then clears the nondegeneracy bar on its own).
        degenerate_anchor = max(
            (finite[k] for k in ("still", "fallen", "chaotic", "upright_flail")
             if k in finite), default=0.0)
        reference_results, ref_scores = _validate_references(
            fn, references, required_roles,
            spread_min=spread_min, degenerate_anchor=degenerate_anchor,
            eval_reset=eval_reset)
        scores.update(ref_scores)
        for entry in reference_results:
            for gate_name, gate_ok in entry["gates"].items():
                key = f"{gate_name}:{entry['clip_id']}"
                gates[key] = gate_ok
            reasons += entry["reasons"]

    # §Ship 50: L1 task-agnostic axioms — controlled-perturbation invariants
    # (uprightness-monotone, no-reward-for-chaos, stationary-no-travel) that
    # harden EVERY metric beyond the L0 archetype comparison. Lazy import to
    # avoid a load-time cycle (metric_axioms imports this module's archetypes).
    # Only meaningful once the metric loads + runs (the bounded gate passed);
    # a metric that crashed every archetype has nothing to perturb.
    axioms: dict[str, Any] = {"ok": True, "axioms": {}, "reasons": [], "details": {}}
    if gates.get("bounded") and gates.get("loads"):
        from sculptor.eval.metric_axioms import check_metric_axioms

        axiom_fn = fn
        if catalog is not None:
            def axiom_fn(arrays, behavior, meta):  # type: ignore[no-redef]
                merged = dict(arrays)
                first = next(iter(arrays.values()))
                merged.update(catalog_fixture_arrays(
                    catalog, time_steps=int(first.shape[0]),
                    num_envs=int(first.shape[1]), case="competent"))
                return fn(merged, behavior, meta)

        axioms = check_metric_axioms(
            axiom_fn, family=family, required_roles=required_roles,
            torso_target=frame["torso_target"])
        gates["axioms"] = bool(axioms["ok"])
        reasons += axioms["reasons"]

    ok = all(gates.values())
    return {"ok": ok, "gates": gates, "reasons": reasons,
            "archetype_scores": scores, "family": family,
            "goal_frame": frame, "nondegeneracy_vacuous": vacuous,
            "selectivity_probe": selectivity,
            "abstract_objective_program": abstract_program,
            "required_roles": required_roles, "axioms": axioms,
            "references": reference_results,
            "channel_catalog_hash": (
                catalog.catalog_hash if catalog is not None else None)}
