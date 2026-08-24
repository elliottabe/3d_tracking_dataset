"""figbuilder command line."""
from __future__ import annotations

import argparse
from pathlib import Path

from figbuilder.bundle import read_bundle
from figbuilder.export import export_figure
from figbuilder.figure import load_figure


def _cmd_export(args) -> int:
    fig_path = Path(args.figure)
    spec = load_figure(fig_path)
    bundle_path = Path(args.bundle) if args.bundle else fig_path.parent / spec.bundle
    out = Path(args.out) if args.out else fig_path.with_suffix(".svg")
    formats = tuple(f.strip() for f in args.format.split(",") if f.strip())
    res = export_figure(spec, read_bundle(bundle_path), out,
                        formats=formats, cache_dir=args.cache_dir)
    for w in res.warnings:
        print(f"warning: {w}")
    for fmt, p in res.paths.items():
        print(f"wrote {fmt}: {p}")
    return 0


def _cmd_bundle_info(args) -> int:
    b = read_bundle(args.bundle)
    print(f"meta: {b.meta}")
    for pid, pd in sorted(b.panels.items()):
        data = ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(pd.data.items()))
        assets = ", ".join(f"{k}{tuple(v.shape)}" for k, v in sorted(pd.assets.items()))
        print(f"  {pid:<18} type={pd.type}")
        if data:
            print(f"      data:   {data}")
        if assets:
            print(f"      assets: {assets}")
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    from figbuilder.server import create_app

    uvicorn.run(create_app(Path(args.figure), Path(args.bundle) if args.bundle else None),
                host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="figbuilder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="render figure.json to svg/png/pdf")
    e.add_argument("figure")
    e.add_argument("--bundle", default=None)
    e.add_argument("-o", "--out", default=None)
    e.add_argument("--format", default="svg", help="comma list: svg,png,pdf")
    e.add_argument("--cache-dir", default=None)
    e.set_defaults(func=_cmd_export)

    b = sub.add_parser("bundle-info", help="list a bundle's panels and datasets")
    b.add_argument("bundle")
    b.set_defaults(func=_cmd_bundle_info)

    s = sub.add_parser("serve", help="run the editor server")
    s.add_argument("figure")
    s.add_argument("--bundle", default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=_cmd_serve)

    args = ap.parse_args(argv)
    return args.func(args)
