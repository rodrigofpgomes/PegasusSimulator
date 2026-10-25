"""Frame capture of a USD camera to numbered JPEG files, headless or with a viewport (play_multi --video-dir).

A Replicator render product renders the camera at the requested size (independent of the viewport, so 4K works on any
screen and without a display); each capture() renders the current scene `subframes` times with physics stopped
(World.render), so the real-time renderer converges its anti-aliasing, or path traces it with `spp` samples per pixel.
Writes <out_dir>/frame_00000.jpg, ... and <out_dir>/frames.csv (frame, step, t [s]) to synchronise overlays composed
afterwards (scripts/shuttle_glider/video/compose_mission_video.py).
"""

import csv
from pathlib import Path

import carb.settings


class VideoCapture:
    def __init__(self, out_dir, camera_path, size=(3840, 2160), quality=95, subframes=4, spp=0):
        import omni.replicator.core as rep

        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.quality, self.subframes = int(quality), max(1, int(subframes))
        if spp > 0:
            settings = carb.settings.get_settings()
            settings.set("/rtx/rendermode", "PathTracing")
            settings.set("/rtx/pathtracing/spp", int(spp))
            settings.set("/rtx/pathtracing/totalSpp", int(spp))
            settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        self._rp = rep.create.render_product(camera_path, tuple(int(v) for v in size))
        self._ann = rep.AnnotatorRegistry.get_annotator("LdrColor")
        self._ann.attach(self._rp)
        self._csv_file = open(self.out_dir / "frames.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["frame", "step", "t"])
        self.count = 0
        print(f"[video] {size[0]}x{size[1]} frames of {camera_path} -> {self.out_dir} "
              f"({'path traced, ' + str(spp) + ' spp' if spp > 0 else str(self.subframes) + ' real-time subframes'})",
              flush=True)

    def capture(self, render, step, t):
        """render: callable that renders one frame without stepping physics (World.render)."""
        from PIL import Image

        for _ in range(self.subframes):
            render()
        data = self._ann.get_data()
        if data is None or getattr(data, "size", 0) == 0:
            print(f"[video] empty frame at step {step}; skipped", flush=True)
            return
        Image.fromarray(data[..., :3]).save(self.out_dir / f"frame_{self.count:05d}.jpg", quality=self.quality)
        self._csv.writerow([self.count, step, f"{t:.6f}"])
        self.count += 1
        if self.count % 100 == 0:
            self._csv_file.flush()
            print(f"[video] {self.count} frames (t = {t:.2f} s)", flush=True)

    def close(self):
        self._csv_file.close()
        print(f"[video] {self.count} frames written to {self.out_dir}", flush=True)
