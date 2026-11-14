import os, re, glob, subprocess, tomllib

MODS = [
    ("detour_core", "crates/core", "base", None),
    ("detour_fallback", "crates/fallback", "fallback", 'feature = "fallback-vulkan"'),
    ("detour_intel_dialect", "crates/intel/dialect", "xe3_dialect", 'feature = "xe3"'),
    ("detour_intel_codegen", "crates/intel/codegen", "xe3_codegen", 'feature = "xe3"'),
    ("detour_intel_xe", "crates/intel/xe", "xe", 'feature = "xe3"'),
    ("detour_intel", "crates/intel/driver", "xe3", 'feature = "xe3"'),
    ("vello_cpu", "crates/vello_cpu", "vello_cpu", 'feature = "xe3"'),
    ("detour_apple_codegen", "crates/apple/codegen", "agx_codegen", 'feature = "agx-g13"'),
    ("detour_apple_asahi", "crates/apple/asahi", "asahi", 'feature = "agx-g13"'),
    ("detour_amd_codegen", "crates/amd/codegen", "rdna2_codegen", 'feature = "rdna2"'),
    ("detour_amd_amdgpu", "crates/amd/amdgpu", "amdgpu", 'feature = "rdna2"'),
]
EXPORTED = ["pass_set", "rewrites", "attr_ops", "xe3", "xe3_value"]
FEATURE = {c: (f.split('"')[1] if f else None) for c, _, _, f in MODS}
VELLO_ON = {"std", "png", "text", "u8_pipeline", "f32_pipeline"}
NAMES = sorted((c for c, _, _, _ in MODS), key=len, reverse=True)

def foreign(src, prefix):
    for c in NAMES:
        mod = next(m for cc, _, m, _ in MODS if cc == c)
        src = re.sub(r'(?<![\w:$])' + c + r'(?=\s*::)', prefix + mod, src)
    return src

def fix_macros(src):
    for m in EXPORTED:
        src = re.sub(r'(\$?crate)::\w+::' + m + r'!', r'\1::' + m + '!', src)
    return src

def rewrite_module(path, mod):
    src = open(path).read()
    src = re.sub(r'\$crate::', '$crate::' + mod + '::', src)
    src = re.sub(r'(?<![\w$])crate::', 'crate::' + mod + '::', src)
    src = foreign(src, 'crate::')
    src = fix_macros(src)
    if mod == "vello_cpu":
        src = re.sub(r'feature\s*=\s*"(\w+)"', lambda m: 'all()' if m.group(1) in VELLO_ON else 'any()', src)
    open(path, 'w').write(src)

def rewrite_external(path):
    src = open(path).read()
    new = foreign(src, 'detour::')
    new = re.sub(r'(detour::\w+)::(' + '|'.join(EXPORTED) + r')!', r'detour::\2!', new)
    if new != src:
        open(path, 'w').write(new)

targets = []
for crate, d, mod, cfg in MODS:
    for f in glob.glob(d + "/src/**/*.rs", recursive=True):
        rewrite_module(f, mod)
    lib = d + "/src/lib.rs"
    s = open(lib).read()
    s = re.sub(r'^#!\[no_std\]\n', '', s, flags=re.M)
    s = re.sub(r'^#!\[cfg_attr\(not\(test\), warn\(unused_crate_dependencies\)\)\]\n', '', s, flags=re.M)
    s = re.sub(r'^#!\[cfg_attr\(docsrs, feature\(doc_cfg\)\)\]\n', '', s, flags=re.M)
    open(lib, 'w').write(s)
    for kind in ("examples", "tests"):
        base = os.path.join(d, kind)
        if not os.path.isdir(base) or mod == "vello_cpu":
            continue
        for e in sorted(os.listdir(base)):
            p = os.path.join(base, e)
            if e.endswith(".rs"):
                name, main = e[:-3], p
            elif os.path.isfile(os.path.join(p, "main.rs")):
                name, main = e, os.path.join(p, "main.rs")
            else:
                continue
            for f in glob.glob(p + ("/**/*.rs" if os.path.isdir(p) else ""), recursive=True) or [p]:
                rewrite_external(f)
            targets.append((kind, name if kind == "examples" and mod == "xe3" else f"{mod}-{name}", main, FEATURE[crate]))
for f in glob.glob("src/**/*.rs", recursive=True) + glob.glob("benches/*/src/**/*.rs", recursive=True) + glob.glob("crates/apple/metal/**/*.rs", recursive=True):
    rewrite_external(f)

with open("target-entries.toml", "w") as out:
    for kind, name, main, feat in targets:
        out.write(f'\n[[{"example" if kind == "examples" else "test"}]]\nname = "{name}"\npath = "{main}"\n')
        if feat:
            out.write(f'required-features = ["{feat}"]\n')
print(len(targets), "targets")
