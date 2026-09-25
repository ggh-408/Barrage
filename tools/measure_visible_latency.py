"""Run the existing visible window timer and append diagnostic-only metadata."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_visible_desktop():
    """Reject hidden SDL drivers and Windows desktops outside user input."""
    driver = os.environ.get("SDL_VIDEODRIVER", "").strip()
    if {value.strip().lower() for value in driver.split(",")} & {"dummy", "offscreen"}:
        raise RuntimeError(
            "Visible latency measurement rejects dummy/offscreen SDL drivers. "
            "Launch this command on the current interactive desktop with the "
            "normal window display driver."
        )
    metadata = {"sdl_videodriver_environment": driver or None, "platform": sys.platform}
    if sys.platform != "win32":
        return {**metadata, "windows_desktop_verification": "not_applicable"}

    import ctypes
    from ctypes import wintypes

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentThreadId.argtypes = []
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        user32.GetProcessWindowStation.argtypes = []
        user32.GetProcessWindowStation.restype = wintypes.HANDLE
        user32.GetThreadDesktop.argtypes = [wintypes.DWORD]
        user32.GetThreadDesktop.restype = wintypes.HANDLE
        user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        user32.OpenInputDesktop.restype = wintypes.HANDLE
        user32.CloseDesktop.argtypes = [wintypes.HANDLE]
        user32.CloseDesktop.restype = wintypes.BOOL
        user32.GetUserObjectInformationW.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ]
        user32.GetUserObjectInformationW.restype = wintypes.BOOL

        def object_name(handle):
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            required = wintypes.DWORD()
            user32.GetUserObjectInformationW(handle, 2, None, 0, ctypes.byref(required))
            if not required.value:
                raise ctypes.WinError(ctypes.get_last_error())
            buffer = ctypes.create_unicode_buffer(
                required.value // ctypes.sizeof(ctypes.c_wchar) + 1
            )
            if not user32.GetUserObjectInformationW(
                handle, 2, buffer, ctypes.sizeof(buffer), ctypes.byref(required)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not buffer.value:
                raise RuntimeError("Windows returned an empty desktop object name")
            return buffer.value

        station = object_name(user32.GetProcessWindowStation())
        desktop = object_name(user32.GetThreadDesktop(kernel32.GetCurrentThreadId()))
        input_handle = user32.OpenInputDesktop(0, False, 0x0001)  # DESKTOP_READOBJECTS
        if not input_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            input_desktop = object_name(input_handle)
        finally:
            user32.CloseDesktop(input_handle)
        if station.casefold() != "winsta0" or desktop.casefold() != input_desktop.casefold():
            raise RuntimeError(
                f"window station={station!r}, thread desktop={desktop!r}, "
                f"input desktop={input_desktop!r}"
            )
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            "Cannot verify this process is on the current interactive Windows desktop. "
            "Launch the command directly on the current user desktop; hidden sandbox "
            f"desktop measurements cannot be reported as visible latency. Detail: {error}"
        ) from error
    return {
        **metadata,
        "windows_desktop_verification": "passed",
        "window_station": station,
        "thread_desktop": desktop,
        "input_desktop": input_desktop,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    desktop_verification = verify_visible_desktop()
    checkpoint = source_root / "best.pt"
    sys.argv = [str(source_root / "Barrage.py"), "--ai",
                "--bullets", "300",
                "--targeted-probability", "0.10", "--no-music",
                "--seed", str(args.seed), "--latency-test-seconds", str(args.seconds),
                "--latency-output", str(output)]
    if args.no_ai:
        sys.argv[1] = "--no-ai"
    namespace = runpy.run_path(str(source_root / "Barrage.py"), run_name="__main__")
    game = namespace["Barrage"]
    if not args.no_ai:
        import torch
    report = json.loads(output.read_text(encoding="utf-8"))
    report.update({
        "checkpoint": None if args.no_ai else str(checkpoint),
        "ai_enabled": not args.no_ai,
        "ai_device": None if args.no_ai else str(game.AI_CONTROLLER.agent.device),
        "source_root": str(source_root),
        "damage_immunity": not game.INVINCIBLE,
        "seed": args.seed,
        "torch_threads": None if args.no_ai else torch.get_num_threads(),
        "survival_game_seconds": game.ALIVE_PHYSICS_STEPS / game.PHYSICS_FPS,
        "alive_physics_steps": game.ALIVE_PHYSICS_STEPS,
        "final_score": game.ALIVE_PHYSICS_STEPS * 10 // game.PHYSICS_FPS,
        "alive_at_end": bool(game.KEY),
        "runtime_stages": {} if args.no_ai else game.AI_CONTROLLER.runtime_stage_report(),
        "desktop_verification": desktop_verification,
        "timing_note": "Survival game time stops on death; wall duration continues. Stage timing may include reset warmup.",
    })
    from barrage_rl.artifacts import atomic_write_json
    atomic_write_json(output, report)
    print("visible_latency_report_complete " + str(output), flush=True)


if __name__ == "__main__":
    main()
