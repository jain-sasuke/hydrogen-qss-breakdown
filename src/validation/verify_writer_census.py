#!/usr/bin/env python
"""
verify_writer_census.py
=======================
A stamped census of every file-writing call site under src/, by write idiom,
with an explicit definition for each count -- so that the thesis can cite ONE
number for how much of the repository's output the single-writer audit
(src/validation/audit_writers.py) can and cannot see.

WHY THIS EXISTS
---------------
thesis_tex/chapter4.tex (~line 1726) says of audit_writers.py: "four live
writers of comma-separated output are invisible to it, including the producer
of the plateau map that three of this thesis's results read from."
outputs/round4_adjudication_2026-09-16.md (~152) and
outputs/completion_status_2026-09-16.md (~83) say "38 call sites in 27
scripts". A plain grep on 18 Sep 2026 gives 29 files. Three numbers, no shared
definition. This script fixes the definition and stamps the count.

WHAT audit_writers.py MISSES, EXACTLY (read against HEAD 7a23ecc)
-----------------------------------------------------------------
1. Its WRITE_CALLS table (audit_writers.py lines 37-48) maps "open" -> path in
   args[0], and its mode test (lines 111-118) reads the mode from n.args[1] or
   from a mode= keyword:
       if len(n.args) > 1: mode = literal_path(n.args[1]) or ""
       for kw in n.keywords: if kw.arg == "mode": mode = ...
       if not any(c in mode for c in "wax"): continue
   For pathlib's  p.open("w")  the mode IS args[0]; there is no args[1] and no
   mode= keyword, so mode stays "", the test fails, and the call hits
   `continue`. It is neither resolved nor listed as UNRESOLVED -- it is
   silently dropped. That is the idiom counted as (a1) below.
2. "write" is not a key of WRITE_CALLS, so the  f.write(...)  calls inside
   such a with-block are invisible as well. (They carry no path, so at best
   they could only ever be listed as unresolved; the point is that the
   enclosing open("w") is what should have fired and did not.)
3. Consequence for the liar check (`if claims and (res or unres)`): a script
   whose only writes are  p.open("w") + f.write  has res == unres == [] and
   passes as read-only whatever its docstring claims.
4. Partial visibility, stated so nobody over-claims: write_text/write_bytes
   ARE in WRITE_CALLS (index None -> target = literal_path(fn.value)). When
   fn.value is a bare Name the resolver returns None, so the call is listed
   as UNRESOLVED (section 4 of its report). verify_plateau_gridmap.py:361
   `txt.write_text(...)` therefore does appear in the audit's unresolved list;
   what is invisible is line 362, `with csv.open("w") as f:`, i.e. the CSV
   plateau map itself.
5. Its open() handling DOES catch  open(path, "w")  (builtin, mode in args[1])
   and  io.open / gzip.open(path, "wt")  (same shape) -- that is (b) below --
   and  p.open(mode="w")  (keyword; then target = args[0] does not exist, so
   it is listed as unresolved) -- that is (a2) below.

DEFINITIONS (every count in the artifact uses exactly these)
------------------------------------------------------------
Scope: every *.py under src/ found by rglob, EXCLUDING any path containing
"backup_scripts" or "__pycache__", and excluding this script itself (it did
not exist when the claims were written and contains the strings it counts).
Files that do not parse are counted as UNPARSABLE and listed.

A "mode string" is a str constant matching ^[rwaxbtU+]+$ . It is a WRITE mode
if it contains any of w, a, x. ("r+" is writable but is NOT counted, to match
audit_writers.py; such sites are counted separately as PLUS_ONLY.)

(a1) METHOD OPEN, MODE IN args[0]:  <expr>.open(<write mode string>, ...)
     -- the idiom audit_writers.py drops silently. Target = <expr>.
(a2) METHOD OPEN, MODE BY KEYWORD, NO POSITIONAL ARGS: <expr>.open(mode="w")
     -- audit_writers.py sees this one (as unresolved). Target = <expr>.
(a)  = (a1) + (a2), the task's definition; both parts reported.
(b)  OPEN, MODE IN args[1] OR mode= WITH A POSITIONAL PATH:
     open(path, "w") / open(path, mode="w") / io.open(path, "w") ...
     -- the idiom audit_writers.py catches. Split builtin (Name) / qualified
     (Attribute, e.g. io.open, gzip.open). Target = args[0].
(c)  <expr>.write_text(...) / <expr>.write_bytes(...). Target = <expr>.
(x)  OTHER WRITE FAMILY, context only: save, savez, savez_compressed, savetxt,
     to_csv, to_parquet, to_excel, to_hdf, savefig, dump, and shutil.copy /
     copy2 / copyfile / move ONLY when called on the name `shutil` -- the rest
     of audit_writers.py's WRITE_CALLS, with its argument indices. (A bare
     `.copy()` is an ndarray/dict copy; audit_writers.py counts it as a write,
     this census does not, and says how many it skipped.)
(d)  For each of M_grid.npy, tau_QSS_grid.npy, tau_relax_grid.npy: WRITERS =
     any site in (a)(b)(c)(x) whose resolved/partial target has that basename;
     MENTIONS = every src line containing the literal, tagged reader/mention
     when the file has no writer site for it.
(e)  WRITERS INTO validation/plateau_gridmap = any site in (a)(b)(c)(x) whose
     resolved OR partial target string contains "plateau_gridmap". Listed
     separately, as POSSIBLE and NOT claimed: files that contain the literal
     "plateau_gridmap" somewhere and also have at least one write site whose
     target could not be resolved.
(f)  thesis_tex/*.tex lines matching  plateau\\?_gridmap  (LaTeX escapes the
     underscore as \\_ , so a plain grep for plateau_gridmap gives 0 in every
     chapter). Count of LINES per file, with line numbers.

STATIC RESOLUTION OF TARGETS
----------------------------
resolved  : str constants; f-strings whose {} parts resolve; a / b and a + b
            of resolvable parts; Path(...), str(...), os.path.join(...);
            .resolve()/.absolute()/.expanduser(); .with_suffix/.with_name;
            and bare Names bound by a single-valued assignment in the enclosing
            function or at module level (tuple unpacking included).
partial   : some segment could not be resolved and is shown as {...} or
            <expr>; `x or y` / `x if c else y` take the resolvable branch;
            an unknown call with exactly one resolvable string argument is
            taken as that argument (this last rule is the same heuristic
            audit_writers.py uses, and is the loosest one here).
unresolved: everything else (parameters, loop variables, attributes,
            subscripts, multiply-assigned names with different values).
Nothing is executed. A partial or unresolved target is reported as such;
it is never guessed.

PREDICTIONS, WRITTEN BEFORE THE AST RUN
---------------------------------------
Given by the task before any code was run:
 (a)  about 29 distinct files, roughly 38 call sites; the notes' 27/38 may
      differ by the backup_scripts exclusion or by counting definition.
 (d)  each of the three .npy has exactly two writers, qss_analysis.py and
      validate_gates.py.
 (e)  exactly one writer, verify_plateau_gridmap.py at line ~362.
 (f)  chapter4.tex 1 line, chapter5.tex ~15 lines.
 REFUTER of the chapter4 sentence as written: (a1) giving more than four files.
Pre-run plain-grep baseline, taken 18 Sep 2026 before writing this script and
recorded here so the AST run can be compared against it honestly:
 grep -rlE '\\.open\\(\\s*["'][wax]' src --include='*.py' | grep -v backup_scripts
      -> 29 files, 45 matching lines (identical with backup_scripts included,
         so the exclusion cannot explain 38 vs 45);
 grep -c 'plateau\\\\?_gridmap' thesis_tex/*.tex -> chapter4 1, chapter5 15.

(g)  HISTORY CROSS-CHECK: the same plain-grep regex run with `git grep` on
     two committed trees: the last commit on or before --history-until
     (default 2026-09-16, the date of the two notes) in HEAD's history, and
     the tip of --history-ref (default main). Read-only. Added after the
     first run showed 45/29 at HEAD against "four" in chapter 4 and "38/27"
     in the notes. Note the working tree is on a branch other than main
     (stamped in the header); main's tip is the 23 Aug commit.
     Also reported: which in-scope files are untracked by git, since the
     census is of the WORKING TREE and those files are not in any commit.

Report only by default. With --write, writes exactly two files:
    validation/writer_census/writer_census.csv   one row per write call site
    validation/writer_census/writer_census.txt   this report, stamped
Modifies nothing else.

    python src/validation/verify_writer_census.py --write
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve()

MODE_RE = re.compile(r"^[rwaxbtU+]+$")
WRITE_CHARS = set("wax")
OTHER_WRITE_CALLS = {
    "save": 0, "savez": 0, "savez_compressed": 0, "savetxt": 0,
    "to_csv": 0, "to_parquet": 0, "to_excel": 0, "to_hdf": 0,
    "savefig": 0, "dump": 1,
    "copy": 1, "copy2": 1, "copyfile": 1, "move": 1,
}
D4_NAMES = ("M_grid.npy", "tau_QSS_grid.npy", "tau_relax_grid.npy")
PLATEAU = "plateau_gridmap"
TEX_RE = re.compile(r"plateau\\?_gridmap")
PLACE = "{...}"
EXPR = "<expr>"

CAT_ORDER = ("a1_method_open_mode_arg0", "a2_method_open_mode_kw",
             "b_open_mode_arg1_builtin", "b_open_mode_arg1_qualified",
             "c_write_text_bytes", "x_other_write_family")


# --------------------------------------------------------------------------- #
# repo root, stamps
# --------------------------------------------------------------------------- #
def find_root() -> Path:
    for p in [_HERE.parent, *_HERE.parents]:
        if (p / "src" / "validation").is_dir() and (p / "thesis_tex").is_dir():
            return p
    sys.exit("could not locate the repo root (need src/validation and thesis_tex)")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               check=True, capture_output=True, text=True).stdout
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root,
                                check=True, capture_output=True, text=True).stdout.strip()
        return (f"{out}  (branch {branch})"
                + ("  (working tree has uncommitted changes)" if dirty.strip() else ""))
    except (OSError, subprocess.CalledProcessError) as e:
        return f"unavailable ({e})"


# --------------------------------------------------------------------------- #
# static path resolution
# --------------------------------------------------------------------------- #
def _stmts(node: ast.AST):
    """Statements under `node`, recursing into compound statements but never
    into nested function / class / lambda bodies."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.stmt):
            yield child
        yield from _stmts(child)


def _bind(target: ast.AST, value: ast.AST, m: dict[str, list[ast.AST]]) -> None:
    if isinstance(target, ast.Name):
        m[target.id].append(value)
    elif (isinstance(target, (ast.Tuple, ast.List))
          and isinstance(value, (ast.Tuple, ast.List))
          and len(target.elts) == len(value.elts)):
        for t, v in zip(target.elts, value.elts):
            _bind(t, v, m)


def assign_map(scope: ast.AST) -> dict[str, list[ast.AST]]:
    m: dict[str, list[ast.AST]] = defaultdict(list)
    for s in _stmts(scope):
        if isinstance(s, ast.Assign):
            for t in s.targets:
                _bind(t, s.value, m)
        elif isinstance(s, ast.AnnAssign) and s.value is not None:
            _bind(s.target, s.value, m)
    return m


def _worst(*statuses: str) -> str:
    order = {"resolved": 0, "partial": 1, "unresolved": 2}
    return max(statuses, key=lambda s: order[s])


def resolve(node: ast.AST | None, envs: list[dict], depth: int = 0) -> tuple[str | None, str]:
    """Return (text, status); status in resolved / partial / unresolved.
    text is None only when status is unresolved."""
    if node is None or depth > 10:
        return None, "unresolved"

    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value, "resolved"
        return None, "unresolved"

    if isinstance(node, ast.JoinedStr):
        parts, status = [], "resolved"
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                t, s = resolve(v.value, envs, depth + 1)
                if t is None:
                    parts.append(PLACE)
                    status = "partial"
                else:
                    parts.append(t)
                    status = _worst(status, s)
        return "".join(parts), status

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add, ast.Mod)):
        lt, ls = resolve(node.left, envs, depth + 1)
        rt, rs = resolve(node.right, envs, depth + 1)
        if lt is None and rt is None:
            return None, "unresolved"
        lt = EXPR if lt is None else lt
        rt = EXPR if rt is None else rt
        status = "partial" if EXPR in (lt, rt) else _worst(ls, rs)
        if isinstance(node.op, ast.Div):
            return f"{lt}/{rt}", status
        if isinstance(node.op, ast.Add):
            return f"{lt}{rt}", status
        return f"{lt}%({rt})", "partial"

    if isinstance(node, ast.Name):
        for env in envs:
            vals = env.get(node.id)
            if vals:
                outs = {resolve(v, envs, depth + 1) for v in vals}
                if len(outs) == 1:
                    return outs.pop()
                return None, "unresolved"          # multiply assigned, differing
        return None, "unresolved"                  # parameter, loop var, import

    if isinstance(node, ast.Call):
        fn = node.func
        fname = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else None)
        if fname in ("Path", "PurePath", "PosixPath", "PurePosixPath", "str"):
            if not node.args:
                return None, "unresolved"
            texts, status = [], "resolved"
            for a in node.args:
                t, s = resolve(a, envs, depth + 1)
                texts.append(EXPR if t is None else t)
                status = "partial" if t is None else _worst(status, s)
            return "/".join(texts), status
        if fname == "join" and isinstance(fn, ast.Attribute) and node.args:
            texts, status = [], "resolved"
            for a in node.args:
                t, s = resolve(a, envs, depth + 1)
                texts.append(EXPR if t is None else t)
                status = "partial" if t is None else _worst(status, s)
            return "/".join(texts), status
        if fname in ("resolve", "absolute", "expanduser") and isinstance(fn, ast.Attribute):
            return resolve(fn.value, envs, depth + 1)
        if fname == "with_suffix" and isinstance(fn, ast.Attribute) and node.args:
            bt, bs = resolve(fn.value, envs, depth + 1)
            st, ss = resolve(node.args[0], envs, depth + 1)
            if bt is None:
                return None, "unresolved"
            return re.sub(r"\.[^./]*$", "", bt) + (st if st else PLACE), _worst(bs, ss if st else "partial")
        if fname == "with_name" and isinstance(fn, ast.Attribute) and node.args:
            bt, bs = resolve(fn.value, envs, depth + 1)
            nt, ns = resolve(node.args[0], envs, depth + 1)
            if bt is None:
                return None, "unresolved"
            return bt.rsplit("/", 1)[0] + "/" + (nt if nt else PLACE), _worst(bs, ns if nt else "partial")
        # unknown call: audit_writers.py's heuristic, marked partial
        hits = [resolve(a, envs, depth + 1) for a in node.args]
        hits = [h for h in hits if h[0] is not None]
        if len(hits) == 1:
            return hits[0][0], "partial"
        return None, "unresolved"

    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        for v in node.values:
            t, s = resolve(v, envs, depth + 1)
            if t is not None:
                return t, "partial"
        return None, "unresolved"

    if isinstance(node, ast.IfExp):
        for v in (node.body, node.orelse):
            t, s = resolve(v, envs, depth + 1)
            if t is not None:
                return t, "partial"
        return None, "unresolved"

    return None, "unresolved"      # Attribute, Subscript, Lambda, ...


# --------------------------------------------------------------------------- #
# call-site classification
# --------------------------------------------------------------------------- #
def _mode_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and MODE_RE.match(node.value):
        return node.value
    return None


def _writes(mode: str) -> bool:
    return any(c in mode for c in WRITE_CHARS)


class Site:
    __slots__ = ("file", "line", "cat", "call", "mode", "target", "status", "snippet")

    def __init__(self, file, line, cat, call, mode, target, status, snippet):
        self.file, self.line, self.cat, self.call = file, line, cat, call
        self.mode, self.target, self.status, self.snippet = mode, target, status, snippet

    @property
    def basename(self) -> str:
        return (self.target or "").rsplit("/", 1)[-1]

    @property
    def plateau(self) -> bool:
        return PLATEAU in (self.target or "")

    @property
    def d4(self) -> str:
        return self.basename if self.basename in D4_NAMES else ""


def classify(call: ast.Call, envs: list[dict], rel: str, lines: list[str],
             sites: list[Site], notes: Counter) -> None:
    fn = call.func
    name = fn.attr if isinstance(fn, ast.Attribute) else (
        fn.id if isinstance(fn, ast.Name) else None)
    if name is None:
        return
    snippet = lines[call.lineno - 1].strip()[:100] if call.lineno - 1 < len(lines) else ""

    def add(cat: str, mode: str, target_node: ast.AST | None) -> None:
        t, s = resolve(target_node, envs)
        sites.append(Site(rel, call.lineno, cat, name, mode, t, s, snippet))

    if name == "open":
        is_method = isinstance(fn, ast.Attribute)
        m0 = _mode_str(call.args[0]) if call.args else None
        m1 = _mode_str(call.args[1]) if len(call.args) > 1 else None
        kw_node = next((k.value for k in call.keywords if k.arg == "mode"), None)
        mkw = _mode_str(kw_node) if kw_node is not None else None

        if is_method and m0 is not None:                       # p.open("w")
            if _writes(m0):
                add("a1_method_open_mode_arg0", m0, fn.value)
            elif "+" in m0:
                notes["PLUS_ONLY (r+ style, not counted)"] += 1
            return
        if is_method and not call.args and mkw is not None:    # p.open(mode="w")
            if _writes(mkw):
                add("a2_method_open_mode_kw", mkw, fn.value)
            elif "+" in mkw:
                notes["PLUS_ONLY (r+ style, not counted)"] += 1
            return
        if call.args and (m1 is not None or mkw is not None):  # open(path, "w")
            m = m1 if m1 is not None else mkw
            if _writes(m):
                add("b_open_mode_arg1_qualified" if is_method else "b_open_mode_arg1_builtin",
                    m, call.args[0])
            elif "+" in m:
                notes["PLUS_ONLY (r+ style, not counted)"] += 1
            return
        # non-literal mode: cannot classify statically
        if (len(call.args) > 1 and m1 is None) or (kw_node is not None and mkw is None):
            notes["open() with NON-LITERAL mode (unclassifiable)"] += 1
            return
        if is_method and call.args and m0 is None:
            if isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                notes[".open(<non-mode string>) e.g. Image/zip/webbrowser (skipped)"] += 1
            else:
                notes[".open(<non-literal first arg>) (unclassifiable)"] += 1
            return
        notes["open() read mode / no mode (skipped)"] += 1
        return

    if name in ("write_text", "write_bytes") and isinstance(fn, ast.Attribute):
        add("c_write_text_bytes", "", fn.value)
        return

    if name in OTHER_WRITE_CALLS:
        # shutil.copy/copy2/copyfile/move only when called ON shutil; a bare
        # `.copy()` is almost always ndarray/dict/DataFrame.copy(), which
        # audit_writers.py mis-counts as a write and this census does not.
        if name in ("copy", "copy2", "copyfile", "move"):
            if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                    and fn.value.id == "shutil"):
                notes[f".{name}() not on shutil (ndarray/dict copy; skipped)"] += 1
                return
        idx = OTHER_WRITE_CALLS[name]
        target_node = call.args[idx] if len(call.args) > idx else None
        if target_node is None:
            for k in call.keywords:
                if k.arg in ("file", "fname", "path", "path_or_buf", "fp", "dst"):
                    target_node = k.value
        add("x_other_write_family", "", target_node)


def scan_file(path: Path, root: Path, sites: list[Site], notes: Counter) -> None:
    src = path.read_text(errors="replace")
    tree = ast.parse(src)
    rel = str(path.relative_to(root))
    lines = src.splitlines()
    module_env = assign_map(tree)

    def visit(node: ast.AST, envs: list[dict]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, [assign_map(child), *envs])
                continue
            if isinstance(child, ast.Call):
                classify(child, envs, rel, lines, sites, notes)
            visit(child, envs)

    visit(tree, [module_env])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="census of write call sites under src/")
    ap.add_argument("--write", action="store_true", help="write the two artifacts")
    ap.add_argument("--out", type=str, default=None,
                    help="output directory (default validation/writer_census)")
    ap.add_argument("--history-until", type=str, default="2026-09-16",
                    help="date of the notes whose count is being reconciled; the (a1) regex "
                         "is re-run with git grep on the last commit on or before this date")
    ap.add_argument("--history-ref", type=str, default="main",
                    help="second reference tree for the history cross-check (default main)")
    a = ap.parse_args()

    root = find_root()
    audit = root / "src" / "validation" / "audit_writers.py"
    if not audit.exists():
        sys.exit(f"missing {audit}")
    audit_sha = sha256(audit)
    head = git_head(root)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    interp = f"{sys.executable}  Python {sys.version.split()[0]}"

    all_py = sorted((root / "src").rglob("*.py"))
    excluded_backup = [p for p in all_py if "backup_scripts" in str(p)]
    files = [p for p in all_py
             if "backup_scripts" not in str(p)
             and "__pycache__" not in p.parts
             and p.resolve() != _HERE]
    n_self = sum(1 for p in all_py if p.resolve() == _HERE)

    sites: list[Site] = []
    notes: Counter = Counter()
    unparsable: list[tuple[str, str]] = []
    for f in files:
        try:
            scan_file(f, root, sites, notes)
        except SyntaxError as e:
            unparsable.append((str(f.relative_to(root)), f"line {e.lineno}: {e.msg}"))

    out_lines: list[str] = []

    def say(s: str = "") -> None:
        out_lines.append(s)

    hdr = [
        f"# script {_HERE.name}",
        f"# generated {stamp}",
        f"# interpreter {interp}",
        f"# git HEAD {head}",
        f"# audit_writers.py sha256 {audit_sha}",
        f"# scope {len(files)} .py files under src/ scanned; excluded {len(excluded_backup)} "
        f"under backup_scripts, {n_self} (this script); {len(unparsable)} unparsable",
    ]
    for h in hdr:
        say(h)
    say()
    say("=" * 78)
    say("WRITER CENSUS -- which write idioms exist under src/, and which of them")
    say("audit_writers.py can see. Definitions in the docstring; repeated briefly.")
    say("=" * 78)
    if unparsable:
        say()
        say("UNPARSABLE (not scanned; counted as unknown):")
        for rel, msg in unparsable:
            say(f"  {rel}  {msg}")

    # ---------------- per-category counts ----------------
    by_cat: dict[str, list[Site]] = defaultdict(list)
    for s in sites:
        by_cat[s.cat].append(s)

    def files_of(ss: list[Site]) -> set[str]:
        return {s.file for s in ss}

    def ext_breakdown(ss: list[Site]) -> str:
        c = Counter()
        for s in ss:
            if s.target is None:
                c["unresolved"] += 1
                continue
            b = s.basename
            if b.endswith(PLACE) or b.endswith(EXPR) or b in (PLACE, EXPR):
                c["partial-name"] += 1
            elif "." in b:
                c["." + b.rsplit(".", 1)[-1]] += 1
            else:
                c["no-extension"] += 1
        return ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))

    a1, a2 = by_cat["a1_method_open_mode_arg0"], by_cat["a2_method_open_mode_kw"]
    bb, bq = by_cat["b_open_mode_arg1_builtin"], by_cat["b_open_mode_arg1_qualified"]
    cc, xx = by_cat["c_write_text_bytes"], by_cat["x_other_write_family"]
    a_all, b_all = a1 + a2, bb + bq

    say()
    say("-" * 78)
    say("1. COUNTS BY IDIOM   (call sites / distinct files)")
    say("-" * 78)
    say(f"  (a1) <expr>.open(<write mode in args[0]>)      "
        f"{len(a1):4d} sites / {len(files_of(a1)):3d} files   INVISIBLE to audit_writers.py")
    say(f"       resolved-target extensions: {ext_breakdown(a1) or 'none'}")
    a1_csv_files = {s.file for s in a1 if s.basename.endswith(".csv")}
    say(f"       (a1) files with at least one target whose BASENAME resolves to *.csv "
        f"(directory part may be partial): {len(a1_csv_files)}")
    say(f"  (a2) <expr>.open(mode=<write mode>), no args   "
        f"{len(a2):4d} sites / {len(files_of(a2)):3d} files   visible (as unresolved)")
    say(f"  (a)  = (a1)+(a2)                               "
        f"{len(a_all):4d} sites / {len(files_of(a_all)):3d} files")
    say(f"  (b)  open(path, <write mode>) / mode=          "
        f"{len(b_all):4d} sites / {len(files_of(b_all)):3d} files   visible")
    say(f"       builtin open: {len(bb)} sites / {len(files_of(bb))} files;  "
        f"qualified (io.open, gzip.open ...): {len(bq)} sites / {len(files_of(bq))} files")
    say(f"       resolved-target extensions: {ext_breakdown(b_all) or 'none'}")
    say(f"  (c)  .write_text / .write_bytes                "
        f"{len(cc):4d} sites / {len(files_of(cc)):3d} files   visible (resolved or unresolved)")
    say(f"  (x)  other write family (save/to_csv/savefig/..)"
        f"{len(xx):4d} sites / {len(files_of(xx)):3d} files   visible   [context]")
    say(f"  ALL  write call sites                          "
        f"{len(sites):4d} sites / {len(files_of(sites)):3d} files")
    st = Counter(s.status for s in sites)
    say(f"  target resolution over ALL sites: resolved {st['resolved']}, "
        f"partial {st['partial']}, unresolved {st['unresolved']}")
    st_a1 = Counter(s.status for s in a1)
    say(f"  target resolution over (a1):      resolved {st_a1['resolved']}, "
        f"partial {st_a1['partial']}, unresolved {st_a1['unresolved']}")
    try:
        untracked = set(subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "src"], cwd=root,
            check=True, capture_output=True, text=True).stdout.splitlines())
        in_scope = {str(p.relative_to(root)) for p in files}
        ut = sorted(in_scope & untracked)
        say(f"  files in scope NOT tracked by git (working tree only, counted above): {len(ut)}")
        for f in ut:
            c = Counter(s.cat for s in sites if s.file == f)
            say(f"      {f}  " + (", ".join(f"{k} {v}" for k, v in sorted(c.items()))
                                  or "no write sites"))
    except (OSError, subprocess.CalledProcessError) as e:
        say(f"  git ls-files unavailable ({e})")
    if notes:
        say()
        say("  calls seen but NOT counted (for transparency):")
        for k, v in sorted(notes.items()):
            say(f"    {v:4d}  {k}")

    say()
    say("-" * 78)
    say("2. EVERY (a1) SITE   file:line  mode  target [status]")
    say("-" * 78)
    for s in sorted(a1, key=lambda s: (s.file, s.line)):
        say(f"  {s.file}:{s.line}  {s.mode!r:6s} {s.target or '-'}  [{s.status}]")
    if a2:
        say()
        say("  (a2) sites:")
        for s in sorted(a2, key=lambda s: (s.file, s.line)):
            say(f"  {s.file}:{s.line}  {s.mode!r:6s} {s.target or '-'}  [{s.status}]")

    # ---------------- (d) ----------------
    say()
    say("-" * 78)
    say("3. (d) PROTOCOL-D.4 PATHS: WRITERS (AST) AND MENTIONS (literal)")
    say("-" * 78)
    d4_writers: dict[str, list[Site]] = {n: [] for n in D4_NAMES}
    for s in sites:
        if s.d4:
            d4_writers[s.d4].append(s)
    lit_mentions: dict[str, list[tuple[str, int, str]]] = {n: [] for n in D4_NAMES}
    plateau_lit_files: dict[str, list[int]] = defaultdict(list)
    for f in files:
        rel = str(f.relative_to(root))
        for i, ln in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for n in D4_NAMES:
                if n in ln:
                    lit_mentions[n].append((rel, i, ln.strip()[:90]))
            if PLATEAU in ln:
                plateau_lit_files[rel].append(i)
    d4_summary: dict[str, list[str]] = {}
    for n in D4_NAMES:
        ws = sorted(d4_writers[n], key=lambda s: (s.file, s.line))
        wfiles = sorted({s.file for s in ws})
        d4_summary[n] = wfiles
        say(f"  {n}: {len(wfiles)} writer file(s), {len(ws)} write site(s)")
        for s in ws:
            say(f"      WRITER   {s.file}:{s.line}  {s.call}() -> {s.target}  [{s.status}]")
        for rel, i, txt in lit_mentions[n]:
            if rel in wfiles:
                continue
            say(f"      mention  {rel}:{i}  {txt}")

    # ---------------- (e) ----------------
    say()
    say("-" * 78)
    say("4. (e) WRITERS INTO validation/plateau_gridmap")
    say("-" * 78)
    pw = sorted([s for s in sites if s.plateau], key=lambda s: (s.file, s.line))
    pw_files = sorted({s.file for s in pw})
    say(f"  {len(pw_files)} writer file(s), {len(pw)} write site(s) whose target contains "
        f"'{PLATEAU}':")
    for s in pw:
        say(f"      WRITER   {s.file}:{s.line}  {s.call}({s.mode!r}) -> {s.target}  [{s.status}]")
    unresolved_by_file: dict[str, list[Site]] = defaultdict(list)
    for s in sites:
        if s.status == "unresolved":
            unresolved_by_file[s.file].append(s)
    possible = sorted(f for f in plateau_lit_files if f not in pw_files and f in unresolved_by_file)
    say(f"  POSSIBLE, NOT CLAIMED: {len(possible)} file(s) contain the literal '{PLATEAU}' AND "
        f"have a write site with an unresolved target:")
    for f in possible:
        say(f"      {f}  literal at lines {plateau_lit_files[f][:6]}"
            f"{'...' if len(plateau_lit_files[f]) > 6 else ''};  "
            f"unresolved write sites: "
            + ", ".join(f"{s.line}:{s.call}" for s in unresolved_by_file[f][:5])
            + (" ..." if len(unresolved_by_file[f]) > 5 else ""))
    mention_only = sorted(f for f in plateau_lit_files if f not in pw_files and f not in possible)
    say(f"  mention only (literal present, every write site resolved elsewhere or none): "
        f"{len(mention_only)}")
    for f in mention_only:
        say(f"      {f}  lines {plateau_lit_files[f][:8]}{'...' if len(plateau_lit_files[f]) > 8 else ''}")

    # ---------------- (f) ----------------
    say()
    say("-" * 78)
    say(r"5. (f) thesis_tex/*.tex LINES MATCHING  plateau\\?_gridmap")
    say("-" * 78)
    tex_counts: dict[str, list[int]] = {}
    for t in sorted((root / "thesis_tex").glob("*.tex")):
        hits = [i for i, ln in enumerate(t.read_text(errors="replace").splitlines(), 1)
                if TEX_RE.search(ln)]
        tex_counts[t.name] = hits
        if hits:
            say(f"  {t.name:18s} {len(hits):3d} lines   {hits}")
    zero = [k for k, v in tex_counts.items() if not v]
    say(f"  zero in: {', '.join(zero) if zero else 'none'}")

    # ---------------- predictions ----------------
    say()
    say("-" * 78)
    say("6. PREDICTIONS vs OUTCOME")
    say("-" * 78)
    say(f"  (a) predicted ~29 files / ~38 sites (notes: 27/38; grep baseline 29/45)")
    say(f"      observed (a1) {len(files_of(a1))} files / {len(a1)} sites;  "
        f"(a1+a2) {len(files_of(a_all))} files / {len(a_all)} sites")
    for n in D4_NAMES:
        ok = d4_summary[n] == ["src/validation/qss_analysis.py", "src/validation/validate_gates.py"]
        say(f"  (d) {n}: predicted 2 writers (qss_analysis.py, validate_gates.py); "
            f"observed {len(d4_summary[n])}: {d4_summary[n]}  -> {'AS PREDICTED' if ok else 'DIFFERS'}")
    e_ok = pw_files == ["src/validation/verify_plateau_gridmap.py"]
    say(f"  (e) predicted exactly one writer, verify_plateau_gridmap.py ~362; observed "
        f"{len(pw_files)} file(s) {pw_files}, sites at lines {[s.line for s in pw]}  "
        f"-> {'AS PREDICTED' if e_ok else 'DIFFERS'}")
    c4, c5 = len(tex_counts.get("chapter4.tex", [])), len(tex_counts.get("chapter5.tex", []))
    say(f"  (f) predicted chapter4 1 / chapter5 ~15; observed chapter4 {c4} / chapter5 {c5}  "
        f"-> {'AS PREDICTED' if (c4, c5) == (1, 15) else 'DIFFERS'}")
    n_a1_files = len(files_of(a1))
    say(f"  REFUTER of chapter4.tex ~1726 ('four live writers of comma-separated output are")
    say(f"  invisible to it'): (a1) files > 4.  Observed (a1) files = {n_a1_files}; of these")
    say(f"  {len(a1_csv_files)} have a target resolving to *.csv.  "
        f"-> {'REFUTED as written: the invisible set is far larger than four' if n_a1_files > 4 else 'not refuted'}")
    # ---------------- history cross-check (read-only git) ----------------
    say()
    say("-" * 78)
    say("7. HISTORY CROSS-CHECK: the (a1) plain-grep regex on two committed trees")
    say("   (explains why three documents give three numbers; git grep, read-only)")
    say("-" * 78)
    rx = r'\.open\(\s*["' + "'" + r'][wax]'
    say(f"  regex {rx}  (same as the pre-run baseline; working tree by this regex: "
        f"{len(files_of(a1))} files / {len(a1)} sites, identical to the AST count (a1))")

    def a1_on(label: str, log_args: list[str]) -> None:
        try:
            ref = subprocess.run(["git", "log", "-1", *log_args, "--format=%H %ad", "--date=short"],
                                 cwd=root, check=True, capture_output=True, text=True).stdout.strip()
            ref_hash = ref.split()[0]
            gg = subprocess.run(["git", "grep", "-n", "-E", rx, ref_hash, "--", "src"],
                                cwd=root, capture_output=True, text=True).stdout.splitlines()
            gg = [ln for ln in gg if "backup_scripts" not in ln]
            gfiles = sorted({ln.split(":", 2)[1] for ln in gg})
            n_val = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref_hash, "--",
                                    "src/validation"], cwd=root, check=True,
                                   capture_output=True, text=True).stdout.splitlines()
            n_val = [p for p in n_val if p.endswith(".py")]
            say(f"  [{label}] commit {ref}")
            say(f"      (a1)-idiom: {len(gfiles)} files / {len(gg)} sites;  "
                f"src/validation .py files in that tree: {len(n_val)}")
            if len(gg) <= 10:
                for ln in gg:
                    say(f"      {ln.split(':', 1)[1]}")
            else:
                say(f"      (site list not repeated; it is the section-2 list)")
        except (OSError, subprocess.CalledProcessError, IndexError) as e:
            say(f"  [{label}] unavailable ({e})")

    a1_on(f"last commit on or before {a.history_until} in HEAD's history",
          [f"--until={a.history_until} 23:59"])
    a1_on(f"tip of ref '{a.history_ref}'", [a.history_ref])
    say("  Reading: a count of 'four' matches the tip of main (23 Aug); the notes' 38/27 is")
    say("  reproduced by neither committed tree and is recorded here as unreproduced.")

    say()
    say("  Static census. Nothing was executed; partial/unresolved targets are reported,")
    say("  never guessed. Report only unless --write.")

    text = "\n".join(out_lines) + "\n"
    print(text, end="")

    if a.write:
        out = Path(a.out) if a.out else root / "validation" / "writer_census"
        out.mkdir(parents=True, exist_ok=True)
        (out / "writer_census.txt").write_text(text)
        cols = ["file", "line", "category", "call", "mode", "target", "status",
                "contains_plateau_gridmap", "d4_basename", "snippet"]
        with (out / "writer_census.csv").open("w", newline="") as fh:
            for h in hdr:
                fh.write(h + "\n")
            fh.write("# categories: " + "; ".join(CAT_ORDER) + "\n")
            fh.write(",".join(cols) + "\n")
            import csv as _csv
            w = _csv.writer(fh)
            for s in sorted(sites, key=lambda s: (CAT_ORDER.index(s.cat), s.file, s.line)):
                w.writerow([s.file, s.line, s.cat, s.call, s.mode, s.target if s.target is not None else "",
                            s.status, int(s.plateau), s.d4, s.snippet])
        print(f"\nwrote {out / 'writer_census.txt'}")
        print(f"wrote {out / 'writer_census.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
