#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SYSTEM_FLUX_PTH_NAME = "system-flux-python.pth"
SYSTEM_FLUX_METADATA_NAME = "system-flux-python.json"


def run(
    cmd: Sequence[str],
    *,
    check: bool = True,
    capture: bool = True,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(
            cmd,
            check=check,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return subprocess.run(cmd, check=check, cwd=str(cwd) if cwd else None, env=env)


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(msg, file=sys.stderr)


def canonical_path(path: Path | str) -> Path:
    p = Path(path).expanduser()
    try:
        return p.resolve(strict=False)
    except RuntimeError:
        return p.absolute()


def find_spack() -> str:
    spack = shutil.which("spack")
    if not spack:
        die("spack not found on PATH. Run this from a shell where Spack is available and the env is activated.")
    return spack


def guess_spack_home(spack_exe: str) -> Path:
    env_root = os.environ.get("SPACK_HOME") or os.environ.get("SPACK_ROOT")
    if env_root:
        return canonical_path(env_root)

    p = canonical_path(spack_exe)
    if p.name == "spack" and p.parent.name == "bin":
        return p.parent.parent
    return p.parent


def is_under(child: Path, parent: Path) -> bool:
    child = canonical_path(child)
    parent = canonical_path(parent)
    try:
        child.relative_to(parent)
        return True
    except Exception:
        pass

    if not child.exists() or not parent.exists():
        return False

    for candidate in [child, *child.parents]:
        try:
            if candidate.samefile(parent):
                return True
        except OSError:
            continue
    return False


def find_env_dir(env_arg: Optional[str]) -> Path:
    if env_arg:
        p = canonical_path(env_arg)
        if p.is_file() and p.name == "spack.yaml":
            return p.parent
        if p.is_dir():
            return p
        die(f"-e/--env must be an env directory or a spack.yaml path, got: {env_arg}")
    return Path(".").resolve()


def spack_env_is_active(spack: str) -> bool:
    if os.environ.get("SPACK_ENV"):
        return True
    cp = run([spack, "env", "status"], check=False)
    return cp.returncode == 0 and ("active" in (cp.stdout + cp.stderr).lower())


def read_lock_json(env_dir: Path) -> dict:
    lock = env_dir / "spack.lock"
    if not lock.exists():
        die(f"spack.lock not found in env dir: {env_dir}")
    try:
        return json.loads(lock.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        die(f"Failed to parse spack.lock as JSON: {e}")


def extract_concrete_specs(lock_json: dict) -> List[dict]:
    if "concrete_specs" in lock_json:
        specs = lock_json["concrete_specs"]
        if isinstance(specs, dict):
            return list(specs.values())
        if isinstance(specs, list):
            return specs
    if "specs" in lock_json and isinstance(lock_json["specs"], dict):
        return list(lock_json["specs"].values())
    die("Could not find concrete specs in spack.lock, unsupported format.")


def spack_location(spack: str, spec_ref: str) -> Optional[Path]:
    cp = run([spack, "location", "-i", spec_ref], check=False)
    if cp.returncode != 0:
        return None
    loc = (cp.stdout or "").strip()
    if not loc:
        return None
    return canonical_path(loc)


def load_yaml_modules(spack_yaml: Path) -> List[str]:
    text = spack_yaml.read_text(encoding="utf-8").splitlines()
    modules: List[str] = []
    i = 0
    while i < len(text):
        line = text[i]
        if re.match(r"^\s*#", line) or not line.strip():
            i += 1
            continue

        m = re.match(r"^(\s*)modules\s*:\s*(.*)\s*$", line)
        if not m:
            i += 1
            continue

        indent = len(m.group(1))
        rest = m.group(2).strip()

        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            if inner:
                parts = [p.strip() for p in inner.split(",")]
                for p in parts:
                    p = p.strip().strip("'").strip('"')
                    if p and p not in modules:
                        modules.append(p)
            i += 1
            continue

        i += 1
        while i < len(text):
            l2 = text[i]
            if re.match(r"^\s*#", l2) or not l2.strip():
                i += 1
                continue
            cur_indent = len(l2) - len(l2.lstrip(" "))
            if cur_indent <= indent:
                break
            m2 = re.match(r"^\s*-\s*(.+?)\s*$", l2)
            if not m2:
                break
            mod = m2.group(1).strip().strip("'").strip('"')
            if mod and mod not in modules:
                modules.append(mod)
            i += 1
    return modules


def parse_python_external(spack_yaml: Path) -> Tuple[Optional[str], Optional[Path]]:
    lines = spack_yaml.read_text(encoding="utf-8").splitlines()

    def is_comment_or_blank(s: str) -> bool:
        return (not s.strip()) or bool(re.match(r"^\s*#", s))

    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    packages_i = None
    for i, raw in enumerate(lines):
        if is_comment_or_blank(raw):
            continue
        if re.match(r"^\s*packages\s*:\s*$", raw):
            packages_i = i
            break
    if packages_i is None:
        return (None, None)

    packages_indent = indent_of(lines[packages_i])

    python_i = None
    python_indent = None
    for i in range(packages_i + 1, len(lines)):
        raw = lines[i]
        if is_comment_or_blank(raw):
            continue
        ind = indent_of(raw)
        if ind <= packages_indent:
            break
        if re.match(r"^\s*python\s*:\s*$", raw):
            python_i = i
            python_indent = ind
            break
    if python_i is None or python_indent is None:
        return (None, None)

    externals_i = None
    externals_indent = None
    for i in range(python_i + 1, len(lines)):
        raw = lines[i]
        if is_comment_or_blank(raw):
            continue
        ind = indent_of(raw)
        if ind <= python_indent:
            break
        if re.match(r"^\s*externals\s*:\s*$", raw):
            externals_i = i
            externals_indent = ind
            break
    if externals_i is None or externals_indent is None:
        return (None, None)

    spec_re = re.compile(r"^\s*-\s*spec\s*:\s*python@([0-9]+(?:\.[0-9]+){1,2})(?:\s+.*)?$")
    in_item = False
    item_indent: Optional[int] = None
    ver: Optional[str] = None
    prefix: Optional[Path] = None

    for i in range(externals_i + 1, len(lines)):
        raw = lines[i]
        if is_comment_or_blank(raw):
            continue
        ind = indent_of(raw)
        if ind <= externals_indent:
            break

        m_spec = spec_re.match(raw)
        if m_spec:
            in_item = True
            item_indent = ind
            ver = m_spec.group(1)
            continue

        if in_item:
            if item_indent is not None and ind <= item_indent:
                break
            m_pref = re.match(r"^\s*prefix\s*:\s*(.+?)\s*$", raw)
            if m_pref:
                p = m_pref.group(1).strip().strip("'").strip('"')
                if p:
                    prefix = canonical_path(p)
                    break

    return (ver, prefix)


def lock_python_version(lock_specs: List[dict]) -> Optional[str]:
    for s in lock_specs:
        if s.get("name") != "python":
            continue
        v = s.get("version")
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]
        v2 = s.get("versions")
        if isinstance(v2, list) and v2 and isinstance(v2[0], str):
            return v2[0]
    return None


def find_spack_concrete_python_hash(lock_specs: List[dict]) -> Optional[str]:
    for s in lock_specs:
        if s.get("name") == "python":
            return s.get("hash") or s.get("full_hash") or s.get("dag_hash")
    return None


def python_bin_from_prefix(prefix: Path, want_ver: Optional[str], *, enforce: bool) -> Path:
    bindir = prefix / "bin"
    if not bindir.is_dir():
        die(f"Python prefix has no bin directory: {bindir}")

    want_mm: Optional[str] = None
    if want_ver:
        m = re.match(r"^(\d+)\.(\d+)", want_ver)
        if m:
            want_mm = f"{m.group(1)}.{m.group(2)}"

    if want_mm:
        for name in [f"python{want_mm}", f"python{want_mm}m", f"python{want_mm}dm", "python3"]:
            p = bindir / name
            if p.exists() and os.access(p, os.X_OK):
                return p
        if enforce:
            die(f"Requested python@{want_ver} but no matching interpreter found under {bindir}")

    for name in ["python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3.8", "python3", "python"]:
        p = bindir / name
        if p.exists() and os.access(p, os.X_OK):
            return p
    die(f"No python executable found under: {bindir}")


def resolve_venv_python(spack: str, spack_home: Path, env_dir: Path, spack_yaml: Path) -> Path:
    lock = read_lock_json(env_dir)
    specs = extract_concrete_specs(lock)

    yaml_ver, yaml_prefix = parse_python_external(spack_yaml)
    lock_ver = lock_python_version(specs)
    py_hash = find_spack_concrete_python_hash(specs)
    want_ver = lock_ver or yaml_ver

    if py_hash:
        prefix = spack_location(spack, f"/{py_hash}")
        if not prefix or not prefix.exists():
            die(f"Could not locate python prefix for /{py_hash}")

        spack_managed = is_under(prefix, spack_home)
        if spack_managed:
            py = python_bin_from_prefix(prefix, want_ver=None, enforce=False)
            info(f"Python prefix is under SPACK_HOME, treating as Spack-managed: {prefix}")
            info(f"Using python from spack.lock hash: {py_hash}, interpreter: {py}")
            return py

        py = python_bin_from_prefix(prefix, want_ver=want_ver, enforce=bool(want_ver))
        info(f"Python prefix is NOT under SPACK_HOME, treating as external: {prefix}")
        info(f"Python target version: {want_ver or 'unconstrained'}")
        info(f"Using python from spack.lock hash: {py_hash}, interpreter: {py}")
        return py

    if yaml_prefix is not None:
        py = python_bin_from_prefix(yaml_prefix, want_ver=yaml_ver, enforce=bool(yaml_ver))
        info(f"Python prefix from spack.yaml external: {yaml_prefix}")
        info(f"Python target version: {yaml_ver or 'unconstrained'}")
        info(f"Using external python from spack.yaml, interpreter: {py}")
        return py

    die("No python found (no python in spack.lock and no external python in spack.yaml).")


def create_venv(python_exe: Path, venv_dir: Path) -> None:
    if not python_exe.exists():
        die(f"Chosen python does not exist: {python_exe}")
    if venv_dir.exists() and any(venv_dir.iterdir()):
        die(f"Output venv directory is not empty: {venv_dir}")
    venv_dir.mkdir(parents=True, exist_ok=True)
    info(f"Creating venv at: {venv_dir} using {python_exe}")
    run([str(python_exe), "-m", "venv", str(venv_dir)], capture=True)


def ensure_python_shims(venv_dir: Path) -> None:
    bindir = venv_dir / "bin"
    py = bindir / "python"
    py3 = bindir / "python3"

    targets = [
        bindir / "python3.13",
        bindir / "python3.12",
        bindir / "python3.11",
        bindir / "python3.10",
        bindir / "python3.9",
        bindir / "python3.8",
        py3,
        py,
    ]
    target = next((t for t in targets if t.exists()), None)
    if target is None:
        warn(f"No python interpreter found in venv bin dir: {bindir}")
        return

    if not py3.exists() and target.name.startswith("python3."):
        py3.symlink_to(target.name)
    if not py.exists():
        py.symlink_to("python3" if py3.exists() else target.name)


def venv_python(venv_dir: Path) -> Path:
    py = venv_dir / "bin" / "python"
    if not py.exists():
        py = venv_dir / "bin" / "python3"
    if not py.exists():
        die("venv python not found")
    return py


def venv_sitepackages(venv_dir: Path) -> Path:
    py = venv_python(venv_dir)
    if not py.exists():
        die("venv python not found to query site-packages")
    cp = run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])
    purelib = (cp.stdout or "").strip()
    if not purelib:
        die("Failed to query venv site-packages (purelib)")
    p = canonical_path(purelib)
    if not p.exists():
        die(f"Venv site-packages path does not exist: {p}")
    return p


def venv_py_mm(venv_dir: Path) -> str:
    py = venv_python(venv_dir)
    cp = run([str(py), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"])
    mm = (cp.stdout or "").strip()
    if not re.match(r"^\d+\.\d+$", mm):
        die(f"Could not determine venv python major.minor, got: {mm!r}")
    return mm


def pick_python_related_specs(specs: List[dict]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in specs:
        name = s.get("name")
        h = s.get("hash") or s.get("full_hash") or s.get("dag_hash")
        if not name or not h:
            continue
        if name == "python" or name.startswith("py-"):
            out.append((name, h))
    seen: Set[str] = set()
    uniq: List[Tuple[str, str]] = []
    for name, h in out:
        if h in seen:
            continue
        seen.add(h)
        uniq.append((name, h))
    return uniq


def find_site_packages_under(prefix: Path) -> List[Path]:
    candidates: List[Path] = []
    for p in prefix.glob("lib/python*/site-packages"):
        if p.is_dir():
            candidates.append(p)
    for p in prefix.glob("lib64/python*/site-packages"):
        if p.is_dir():
            candidates.append(p)

    out: List[Path] = []
    seen: Set[str] = set()
    for p in candidates:
        rp = canonical_path(p)
        s = str(rp)
        if s in seen:
            continue
        seen.add(s)
        out.append(rp)
    return out


def filter_spack_sites(
    spack_sites: List[Path],
    *,
    spack_home: Path,
    restrict_mm: Optional[str],
) -> List[Path]:
    out: List[Path] = []
    seen: Set[str] = set()
    for p in spack_sites:
        rp = canonical_path(p)

        if not is_under(rp, spack_home):
            continue

        if restrict_mm:
            if f"/python{restrict_mm}/site-packages" not in str(rp):
                continue

        s = str(rp)
        if s in seen:
            continue
        seen.add(s)
        out.append(rp)
    return out


def write_pth(venv_site: Path, spack_sites: List[Path]) -> Path:
    pth = venv_site / "spack_sitepackages.pth"
    pth.write_text("\n".join(str(p) for p in spack_sites) + ("\n" if spack_sites else ""), encoding="utf-8")
    return pth


FLUX_DISCOVER_SCRIPT = r"""
import json
import pathlib
import sys

import flux
import flux.job  # noqa: F401
import flux.resource  # noqa: F401

flux_file = pathlib.Path(flux.__file__).resolve()
flux_python_path = None
for parent in flux_file.parents:
    if parent.name in {"site-packages", "dist-packages"}:
        flux_python_path = parent
        break

if flux_python_path is None and flux_file.parent.name == "flux":
    flux_python_path = flux_file.parent.parent

if flux_python_path is None:
    raise SystemExit(f"Could not derive Python import path from flux.__file__: {flux_file}")

print(
    json.dumps(
        {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "flux_file": str(flux_file),
            "flux_python_path": str(flux_python_path),
            "sys_path": sys.path,
        },
        sort_keys=True,
    )
)
"""


def parse_flux_discovery(cp: subprocess.CompletedProcess) -> Dict[str, str]:
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError as e:
        die(f"Could not parse Flux Python discovery output as JSON: {e}\nOutput was:\n{cp.stdout}")


def probe_flux_import(python_exe: Path | str, python_path: Optional[Path] = None) -> subprocess.CompletedProcess:
    env: Optional[Dict[str, str]] = None
    if python_path is not None:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(python_path) if not existing else f"{python_path}{os.pathsep}{existing}"
    return run([str(python_exe), "-c", FLUX_DISCOVER_SCRIPT], check=False, env=env)


def venv_can_import(venv_dir: Path, module: str) -> bool:
    cp = run([str(venv_python(venv_dir)), "-c", f"import {module}"], check=False)
    return cp.returncode == 0


def find_module_entry(module: str, search_paths: List[Path]) -> Optional[Path]:
    for root in search_paths:
        package = root / module
        if package.exists():
            return package
        module_file = root / f"{module}.py"
        if module_file.exists():
            return module_file
    return None


def create_system_flux_python_shim(
    venv_dir: Path,
    venv_site: Path,
    flux_python_path: Path,
    flux_python_sys_path: List[Path],
) -> Path:
    shim = venv_site / "system_flux_python"
    shim.mkdir(parents=True, exist_ok=True)

    names: Set[str] = set()
    symlink_sources: Dict[str, Path] = {}
    for source in flux_python_path.iterdir():
        source_name = source.name
        import_name = source.stem if source.is_file() else source_name
        if import_name in {"flux", "_flux"} or re.match(r"^flux[A-Za-z0-9_]*$", import_name):
            symlink_sources[source_name] = source

    for dependency in ["ply", "yaml"]:
        if venv_can_import(venv_dir, dependency):
            continue
        source = find_module_entry(dependency, flux_python_sys_path)
        if source is not None:
            symlink_sources[source.name] = source

    if not symlink_sources:
        die(f"Could not find Flux Python packages under: {flux_python_path}")

    for name, source in sorted(symlink_sources.items()):
        target = shim / name
        if target.exists() or target.is_symlink():
            die(f"Cannot create system Flux shim because target already exists: {target}")
        target.symlink_to(source, target_is_directory=source.is_dir())

    return shim


def verify_system_flux_python_path(venv_dir: Path, flux_python_path: Path) -> None:
    cp = probe_flux_import(venv_python(venv_dir), flux_python_path)
    if cp.returncode != 0:
        die(
            "The venv Python could not import Flux from the prepared system Flux Python path.\n"
            f"Flux Python path: {flux_python_path}\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}"
        )


def get_python_version(python_exe: Path) -> str:
    cp = run([str(python_exe), "-c", "import sys; print(sys.version)"])
    return cp.stdout.strip()


def discover_system_flux_python(venv_dir: Path, spack_home: Path, spack_sites: List[Path]) -> Dict[str, Any]:
    flux_exe = shutil.which("flux")
    if not flux_exe:
        die("--with-system-flux-python requires an active system Flux installation, but 'flux' was not found on PATH.")

    flux_version_cp = run([flux_exe, "version"], check=False)
    if flux_version_cp.returncode != 0:
        die(
            "Could not run 'flux version' for --with-system-flux-python.\n"
            f"stdout:\n{flux_version_cp.stdout}\n"
            f"stderr:\n{flux_version_cp.stderr}"
        )

    py = venv_python(venv_dir)
    cp = probe_flux_import(py)
    if cp.returncode != 0:
        flux_python_cp = run([flux_exe, "python", "-c", FLUX_DISCOVER_SCRIPT], check=False)
        if flux_python_cp.returncode != 0:
            die(
                "Could not import the active system Flux Python bindings with the venv Python "
                "or through 'flux python'.\n"
                "Load the system Flux environment and recreate the venv with --with-system-flux-python.\n"
                f"venv stdout:\n{cp.stdout}\n"
                f"venv stderr:\n{cp.stderr}\n"
                f"flux python stdout:\n{flux_python_cp.stdout}\n"
                f"flux python stderr:\n{flux_python_cp.stderr}"
            )
        imported = parse_flux_discovery(flux_python_cp)
    else:
        imported = parse_flux_discovery(cp)

    flux_python_sys_path = [
        str(canonical_path(path)) for path in imported.get("sys_path", []) if path
    ]

    flux_site = canonical_path(imported["flux_python_path"])
    flux_file = canonical_path(imported["flux_file"])
    spack_site_keys = {str(canonical_path(p)) for p in spack_sites}
    if str(flux_site) in spack_site_keys or is_under(flux_site, spack_home):
        die(
            "The active 'flux' Python module resolved to the AMS Spack environment, "
            f"not system Flux Python: {flux_file}"
        )

    return {
        "schema_version": 1,
        "which_flux": str(canonical_path(flux_exe)),
        "flux_version": flux_version_cp.stdout.strip(),
        "python_executable": str(py),
        "python_version": get_python_version(py),
        "flux_python_executable": imported["python_executable"],
        "flux_python_version": imported["python_version"],
        "flux_file": str(flux_file),
        "flux_site_packages": str(flux_site),
        "flux_python_sys_path": flux_python_sys_path,
    }


def write_system_flux_python_pth(venv_site: Path, flux_site_packages: Path) -> Path:
    pth = venv_site / SYSTEM_FLUX_PTH_NAME
    pth.write_text(f"{flux_site_packages}\n", encoding="utf-8")
    return pth


def write_system_flux_metadata(venv_site: Path, metadata: Dict[str, Any]) -> Path:
    metadata_path = venv_site / SYSTEM_FLUX_METADATA_NAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def capture_env_vars(keys: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v:
            out[k] = v
    return out


def write_activate_hooks(venv_dir: Path, modules: List[str], envvars: Optional[Dict[str, str]] = None) -> None:
    actived = venv_dir / "bin" / "activate.d"
    actived.mkdir(parents=True, exist_ok=True)

    modules_sh = actived / "spack_modules.sh"
    mod_lines: List[str] = []
    mod_lines.append("#!/usr/bin/env bash")
    mod_lines.append("# Auto-generated. Loads modules listed in spack.yaml (uncommented).")
    mod_lines.append("# No module purge is performed.")
    mod_lines.append("if ! command -v module >/dev/null 2>&1; then")
    mod_lines.append('  echo "spack-venv: module command not found, skipping module loads" 1>&2')
    mod_lines.append("  return 0 2>/dev/null || exit 0")
    mod_lines.append("fi")
    if modules:
        mod_lines.append("module load " + " ".join(shlex.quote(m) for m in modules))
    else:
        mod_lines.append('echo "spack-venv: no modules found in spack.yaml to load" 1>&2')
    modules_sh.write_text("\n".join(mod_lines) + "\n", encoding="utf-8")
    modules_sh.chmod(0o755)

    if envvars is not None:
        env_sh = actived / "spack_envvars.sh"
        env_lines: List[str] = []
        env_lines.append("#!/usr/bin/env bash")
        env_lines.append("# Auto-generated. Restores selected environment variables captured at venv creation time.")
        for k, v in envvars.items():
            env_lines.append(f"export {k}={shlex.quote(v)}")
        env_sh.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
        env_sh.chmod(0o755)

    activate = venv_dir / "bin" / "activate"
    act_txt = activate.read_text(encoding="utf-8")
    marker_begin = "# >>> spack-venv activate.d >>>"
    if marker_begin not in act_txt:
        snippet = "\n".join(
            [
                "",
                marker_begin,
                'if [ -d "${VIRTUAL_ENV}/bin/activate.d" ]; then',
                '  for f in "${VIRTUAL_ENV}/bin/activate.d/"*.sh; do',
                '    [ -r "$f" ] || continue',
                '    . "$f"',
                "  done",
                "fi",
                "# <<< spack-venv activate.d <<<",
                "",
            ]
        )
        activate.write_text(act_txt + snippet, encoding="utf-8")


def write_system_flux_activation_hook(venv_dir: Path, metadata_path: Path, metadata: Dict[str, Any]) -> Path:
    actived = venv_dir / "bin" / "activate.d"
    actived.mkdir(parents=True, exist_ok=True)
    hook = actived / "zz_system_flux_python.sh"
    recorded_version = str(metadata["flux_version"])
    hook.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "# Auto-generated. Warns when the active system Flux installation changed since venv creation.",
                f"_ams_system_flux_metadata={shlex.quote(str(metadata_path))}",
                f"_ams_system_flux_recorded_version={shlex.quote(recorded_version)}",
                "if ! command -v flux >/dev/null 2>&1; then",
                '  echo "spack-venv: flux command not found; load system Flux and recreate this venv" 1>&2',
                "else",
                '  _ams_system_flux_current_version="$(flux version 2>/dev/null || true)"',
                '  if [ -n "$_ams_system_flux_current_version" ] && \\',
                '     [ "$_ams_system_flux_current_version" != "$_ams_system_flux_recorded_version" ]; then',
                '    echo "spack-venv: active flux version differs from recorded system Flux metadata" 1>&2',
                '    echo "spack-venv: recreate this venv with '
                'scripts/make-spack-venv.py --with-system-flux-python" 1>&2',
                '    echo "spack-venv: metadata: ${_ams_system_flux_metadata}" 1>&2',
                "  fi",
                "fi",
                "unset _ams_system_flux_current_version",
                "unset _ams_system_flux_recorded_version",
                "unset _ams_system_flux_metadata",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    return hook


def has_any_package(specs: List[dict], names: Set[str]) -> bool:
    for s in specs:
        n = s.get("name")
        if n in names:
            return True
    return False


def write_sitecustomize_numpy_rtld_global(venv_site: Path) -> Path:
    sc = venv_site / "sitecustomize.py"
    sc.write_text(
        "\n".join(
            [
                "# Auto-generated by spack-venv.",
                "# Import NumPy with RTLD_GLOBAL to avoid MKL symbol resolution issues.",
                "import os",
                "import sys",
                "",
                "try:",
                "    _old = sys.getdlopenflags()",
                "    sys.setdlopenflags(_old | os.RTLD_GLOBAL)",
                "    import numpy  # noqa: F401",
                "finally:",
                "    try:",
                "        sys.setdlopenflags(_old)",
                "    except Exception:",
                "        pass",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return sc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--env", help="Spack environment directory or spack.yaml path (default: current directory)")
    ap.add_argument("-o", "--output", required=True, help="Output venv directory")
    ap.add_argument(
        "--capture-envvars",
        nargs="+",
        help="Env var names to capture into activation hook. Example: --capture-envvars PATH LD_LIBRARY_PATH",
    )
    ap.add_argument(
        "--no-restrict-python-mm",
        action="store_true",
        help="Do not restrict .pth entries to the venv Python major.minor (default: restrict).",
    )
    ap.add_argument(
        "--with-system-flux-python",
        action="store_true",
        help="Link the active system Flux Python package path into the venv and warn when Flux changes.",
    )
    args = ap.parse_args(argv)

    spack = find_spack()
    spack_home = guess_spack_home(spack)

    env_dir = find_env_dir(args.env)
    spack_yaml = env_dir / "spack.yaml"
    if not spack_yaml.exists():
        die(f"spack.yaml not found in env dir: {env_dir}")

    if not spack_env_is_active(spack):
        warn(
            "Spack env does not appear active (SPACK_ENV not set). "
            "Continuing, but spack queries may not match your intended env."
        )

    modules = load_yaml_modules(spack_yaml)
    info(f"Modules found in spack.yaml: {modules if modules else 'none'}")
    info(f"Detected SPACK_HOME: {spack_home}")

    python_exe = resolve_venv_python(spack, spack_home, env_dir, spack_yaml)

    venv_dir = Path(args.output).expanduser().resolve()
    create_venv(python_exe, venv_dir)
    ensure_python_shims(venv_dir)

    venv_site = venv_sitepackages(venv_dir)
    mm = None if args.no_restrict_python_mm else venv_py_mm(venv_dir)
    if mm:
        info(f"Restricting .pth entries to python{mm} site-packages")

    lock_json = read_lock_json(env_dir)
    specs = extract_concrete_specs(lock_json)

    # Install RTLD_GLOBAL NumPy preload only when MKL is in the concretized env
    # and NumPy is present.
    numpy_present = has_any_package(specs, {"py-numpy", "numpy"})
    mkl_present = has_any_package(specs, {"intel-oneapi-mkl", "intel-mkl", "mkl"})
    if mkl_present and numpy_present:
        sc = write_sitecustomize_numpy_rtld_global(venv_site)
        info(f"Detected MKL + NumPy in environment, installed sitecustomize: {sc}")

    candidates = pick_python_related_specs(specs)

    spack_sites: List[Path] = []
    for name, h in candidates:
        prefix = spack_location(spack, f"/{h}")
        if not prefix or not prefix.exists():
            continue
        if not is_under(prefix, spack_home):
            continue
        spack_sites.extend(find_site_packages_under(prefix))

    uniq_sites = filter_spack_sites(spack_sites, spack_home=spack_home, restrict_mm=mm)

    pth = write_pth(venv_site, uniq_sites)

    system_flux_pth: Optional[Path] = None
    system_flux_metadata_path: Optional[Path] = None
    system_flux_metadata: Optional[Dict[str, Any]] = None
    if args.with_system_flux_python:
        system_flux_metadata = discover_system_flux_python(venv_dir, spack_home, uniq_sites)
        system_flux_site = canonical_path(system_flux_metadata["flux_site_packages"])
        system_flux_sys_path = [canonical_path(path) for path in system_flux_metadata["flux_python_sys_path"]]
        system_flux_shim = create_system_flux_python_shim(
            venv_dir,
            venv_site,
            system_flux_site,
            system_flux_sys_path,
        )
        verify_system_flux_python_path(venv_dir, system_flux_shim)
        system_flux_metadata["flux_shim_path"] = str(system_flux_shim)
        system_flux_pth = write_system_flux_python_pth(venv_site, system_flux_shim)
        system_flux_metadata_path = write_system_flux_metadata(venv_site, system_flux_metadata)

    envvars: Optional[Dict[str, str]] = None
    if args.capture_envvars is not None:
        seen: Set[str] = set()
        keys: List[str] = []
        for k in args.capture_envvars:
            k = (k or "").strip()
            if not k or k in seen:
                continue
            seen.add(k)
            keys.append(k)
        envvars = capture_env_vars(keys)

    write_activate_hooks(venv_dir, modules, envvars)
    if system_flux_metadata_path is not None and system_flux_metadata is not None:
        system_flux_hook = write_system_flux_activation_hook(venv_dir, system_flux_metadata_path, system_flux_metadata)
        info(f"Wrote system Flux activation hook: {system_flux_hook}")

    info(f"Done. Venv: {venv_dir}")
    info(f"Wrote .pth: {pth}")
    if system_flux_pth is not None:
        info(f"Wrote system Flux .pth: {system_flux_pth}")
    if system_flux_metadata_path is not None:
        info(f"Wrote system Flux metadata: {system_flux_metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
