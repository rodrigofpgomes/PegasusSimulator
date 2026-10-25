"""High-resolution screenshot of the Isaac Sim viewport (scene + debug_draw lines), with the telemetry overlay composed on top.

The omni.ui overlay and the viewport HUD are not part of the renderer's output, so the scene is captured by the
renderer at `size` (optionally path traced and converged) and the overlay is redrawn with Pillow at the same scale.
Writes <path> (scene + overlay) and <path stem>_scene.png (scene only).
"""

from pathlib import Path

import carb.settings


def take_screenshot(simulation_app, path, size=(3840, 2160), pathtrace=True, spp=256, overlay=None):
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from PIL import Image

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    scene_path = path.with_name(path.stem + "_scene.png")
    settings = carb.settings.get_settings()
    vp = get_active_viewport()

    old_res = tuple(vp.resolution)
    old_mode = settings.get("/rtx/rendermode")
    old_spp = settings.get("/rtx/pathtracing/totalSpp")
    vp.resolution = tuple(size)
    if pathtrace:
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 1)
        settings.set("/rtx/pathtracing/totalSpp", int(spp))
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    n_frames = (int(spp) + 40) if pathtrace else 60  # one sample per pixel per frame: wait until it has converged
    print(f"[screenshot] rendering {size[0]}x{size[1]}{' path traced, ' + str(spp) + ' spp' if pathtrace else ''} ...", flush=True)
    for _ in range(n_frames):
        simulation_app.update()
    print(f"[screenshot] viewport {tuple(vp.resolution)}, rendermode={settings.get('/rtx/rendermode')}, "
          f"totalSpp={settings.get('/rtx/pathtracing/totalSpp')}, denoiser={settings.get('/rtx/pathtracing/optixDenoiser/enabled')}", flush=True)

    if scene_path.is_file():
        scene_path.unlink()
    capture_viewport_to_file(vp, str(scene_path))
    last_size, stable = -1, 0
    for _ in range(1200):  # the capture is asynchronous: wait for the file to appear and stop growing
        simulation_app.update()
        size_now = scene_path.stat().st_size if scene_path.is_file() else -1
        stable = stable + 1 if (size_now > 0 and size_now == last_size) else 0
        last_size = size_now
        if stable >= 10:
            break

    if scene_path.is_file():
        if overlay is not None:
            overlay.compose(Image.open(scene_path)).convert("RGB").save(path)
        else:
            Image.open(scene_path).convert("RGB").save(path)
        print(f"[screenshot] saved {path} ({'with overlay' if overlay is not None else 'NO overlay: run with --telemetry-overlay'}); "
              f"scene only: {scene_path}", flush=True)
    else:
        print(f"[screenshot] FAILED: {scene_path} was not written", flush=True)

    vp.resolution = old_res
    if pathtrace:
        if old_mode is not None:
            settings.set("/rtx/rendermode", old_mode)
        if old_spp is not None:
            settings.set("/rtx/pathtracing/totalSpp", old_spp)
