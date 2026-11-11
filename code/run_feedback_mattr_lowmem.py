"""Process-local low-memory attention projection for the custom FP32 MAttr runner.
Installed library files and official MAttr files are never edited.
"""
import argparse
import ast
import copy
import __future__
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import tempfile
import textwrap

import torch


def head_projection(z, w):
    if z.ndim != 5 or w.ndim != 5 or z.shape[-1] != 1 or tuple(w.shape[:2]) != (1, 1):
        raise RuntimeError(f"Unexpected attention projection shapes: z={tuple(z.shape)}, w={tuple(w.shape)}")
    return torch.einsum("bphd,hdm->bphm", z.squeeze(-1), w[0, 0])


def self_test():
    generator = torch.Generator().manual_seed(42)
    z = torch.randn(2, 3, 4, 5, 1, generator=generator, dtype=torch.float64, requires_grad=True)
    w = torch.randn(1, 1, 4, 5, 6, generator=generator, dtype=torch.float64, requires_grad=True)
    probe = torch.randn(2, 3, 4, 6, generator=generator, dtype=torch.float64)
    reference = (z * w).sum(-2)
    efficient = head_projection(z, w)
    torch.testing.assert_close(efficient, reference, rtol=1e-10, atol=1e-10)
    reference_grad = torch.autograd.grad((reference * probe).sum(), (z, w))
    efficient_grad = torch.autograd.grad((efficient * probe).sum(), (z, w))
    for actual, expected in zip(efficient_grad, reference_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    print("LOW-MEMORY ATTENTION: small CPU forward and gradient tests passed.", flush=True)


def install_patch():
    import transformer_lens.components.abstract_attention as module
    expected = ast.parse("(z * w).sum(-2)", mode="eval").body
    expected_dump = ast.dump(expected, include_attributes=False)
    candidates = []
    for name, cls in vars(module).items():
        if not inspect.isclass(cls) or cls.__module__ != module.__name__:
            continue
        function = cls.__dict__.get("forward")
        if not inspect.isfunction(function):
            continue
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "unhooked_result" for t in node.targets)
            and ast.dump(node.value, include_attributes=False) == expected_dump]
        if matches:
            candidates.append((cls, function, source, tree, matches))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one attention implementation to patch; found {len(candidates)}. No model loaded.")
    cls, original, source, tree, matches = candidates[0]
    if len(matches) != 1 or original.__code__.co_freevars:
        raise RuntimeError("Unexpected forward implementation; refusing an unsafe replacement.")
    before = copy.deepcopy(tree)
    matches[0].value = ast.Call(func=ast.Name(id="_feedback_lowmem_head_projection", ctx=ast.Load()),
                               args=[ast.Name(id="z", ctx=ast.Load()), ast.Name(id="w", ctx=ast.Load())],
                               keywords=[])
    function_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    function_node.decorator_list = []
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["_feedback_lowmem_head_projection"] = head_projection
    filename = inspect.getfile(original) + ":process_local_lowmem"
    exec(compile(tree, filename, "exec", flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), namespace)
    replacement = namespace[original.__name__]
    replacement.__qualname__ = original.__qualname__
    replacement.__module__ = original.__module__
    cls.forward = replacement
    manifest = {"attention_class": cls.__module__ + "." + cls.__name__,
        "installed_source": inspect.getfile(original),
        "original_forward_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "replacement": "(z*w).sum(-2) -> einsum('bphd,hdm->bphm', z.squeeze(-1), w[0,0])",
        "scope": "Current Python process only; no installed files edited.",
        "torch_version": torch.__version__,
        "validation": "Tiny CPU float64 output and gradient checks passed; full-model checks remain enabled."}
    print("LOW-MEMORY ATTENTION PATCH\n" + json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--attention-self-test-only", action="store_true")
    own, forwarded = parser.parse_known_args()
    torch.set_num_threads(2)
    self_test()
    if own.attention_self_test_only:
        return
    root = Path(own.workdir).resolve()
    target = root / "code/run_feedback_mattr_fp32.py"
    if not target.is_file():
        raise FileNotFoundError(target)
    manifest = install_patch()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", prefix="attention-lowmem-", suffix=".json",
                                     dir=logs, delete=False) as f:
        json.dump(manifest, f, indent=2)
        manifest_path = f.name
    print("Patch manifest:", manifest_path, flush=True)
    sys.argv = [str(target), "--workdir", str(root), *forwarded]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
