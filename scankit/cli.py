"""`scan` command line: capture, depth, reconstruct and inspect scans.

    scan devices                         connected OAK and ZED cameras
    scan capture [scans/room]            capture GUI (new scan, or append to an existing one)
    scan info scans/room                 scan summary
    scan depth scans/room                depth from the stereo pairs -> depth_raftstereo/
    scan reconstruct scans/room          mesh -> scans/room/recon/<time>/
    scan view scans/room/recon/<run>/mesh.ply
    scan measure scans/room/recon/<run>/mesh.ply
    scan zed-check                       ZED Mini rectification check on live frames
    scan export-o3d scans/room out/      Open3D reconstruction_system layout

`scan <command> -h` lists the options of a command. Tab completion (commands, options, paths):
add `eval "$(register-python-argcomplete scan)"` to ~/.zshrc or ~/.bashrc. Each command imports only what it needs
(depthai and pycolmap bundle conflicting glog builds and must not share a process).
"""

from __future__ import annotations

import argparse


def _capture(a):
    from .gui import App
    App(a.scan).run()


def _info(a):
    from .scanfile import summarize
    for p in a.scans:
        print(summarize(p))


def _devices(a):
    for name in ("oak", "zed"):
        try:
            if name == "oak":
                from .cameras.oak import info
            else:
                from .cameras.zed import info
            info()
        except Exception as e:  # noqa: BLE001  report every camera, even if one fails
            print(f"{name.upper()}: {e}")


def _zed_check(a):
    from .cameras.zed import check
    check(a.resolution)


def _view(a):
    from .viewer import view
    view(a.path)


def _measure(a):
    from .viewer import measure
    measure(a.path)


# commands whose options live in their own module (add_arguments / run)
MODULE_COMMANDS = (("depth", "depth", "depth from the recorded stereo pairs"),
                   ("reconstruct", "reconstruct", "offline reconstruction"),
                   ("export-o3d", "export_o3d", "export for Open3D's reconstruction_system"))


def main(argv=None):
    import importlib
    import os
    import sys

    import argcomplete
    argv = sys.argv[1:] if argv is None else argv
    completing = "_ARGCOMPLETE" in os.environ
    if completing:          # the shell passes the command line in COMP_LINE, not in argv
        argv = os.environ.get("COMP_LINE", "").split()[1:]
    p = argparse.ArgumentParser(prog="scan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    c = sub.add_parser("capture", help="capture GUI")
    c.add_argument("scan", nargs="?", help="scan folder (appended if it exists)")
    c.set_defaults(func=_capture)
    c = sub.add_parser("info", help="scan summary")
    c.add_argument("scans", nargs="+")
    c.set_defaults(func=_info)
    sub.add_parser("devices", help="connected cameras").set_defaults(func=_devices)
    c = sub.add_parser("zed-check", help="ZED Mini rectification check")
    c.add_argument("--resolution", default="HD", choices=["2K", "FHD", "HD", "VGA"])
    c.set_defaults(func=_zed_check)
    for cmd, module, helptext in MODULE_COMMANDS:
        c = sub.add_parser(cmd, help=helptext, formatter_class=argparse.RawDescriptionHelpFormatter)
        if argv[:1] == [cmd]:       # import the (heavy) module only for the command that runs
            mod = importlib.import_module(f".{module}", __package__)
            c.description = mod.__doc__
            mod.add_arguments(c)
            c.set_defaults(func=mod.run)
    for cmd, func in (("view", _view), ("measure", _measure)):
        c = sub.add_parser(cmd, help=f"{cmd} a mesh or point cloud")
        c.add_argument("path")
        c.set_defaults(func=func)
    argcomplete.autocomplete(p)     # exits here when called by the shell for completion
    a = p.parse_args(None if completing else argv)
    a.func(a)


if __name__ == "__main__":
    main()
