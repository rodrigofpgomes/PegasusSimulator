"""Two cameras that follow a vehicle in the Isaac Sim viewport: "chase" (behind and above) and "side".

Created as USD cameras under /World/FollowCam*; the viewport is switched between them with the C key.
The heading is the horizontal velocity direction (low-pass filtered, kept in hover), so the cameras do not
spin while the vehicle hovers or turns on the spot. Call update() once per physics step after world.step().
Group framing (fit > 0, update(..., points=...)): the cameras aim at the centre of the points' bounding box and back
off (low-pass filtered) so that the points span at most `fit` of the frame; `shift` moves the image so the aim point
sits above the centre (fraction of the frame height), e.g. to leave the bottom of a video for telemetry.
"""

import math

import carb
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom

CHASE_PATH = "/World/FollowCamChase"
SIDE_PATH = "/World/FollowCamSide"


H_APERTURE = 20.955


def _make_camera(stage, path, focal, far=5000.0, aspect=None, shift=0.0):
    """aspect: width/height of the rendered image (sets the vertical aperture, needed for `shift`); shift: image
    offset, fraction of the frame height (> 0 moves the scene down, so the aim point appears higher)."""
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(H_APERTURE)
    v_aperture = H_APERTURE / aspect if aspect else 15.2908
    cam.CreateVerticalApertureAttr(v_aperture)
    if shift:
        cam.CreateVerticalApertureOffsetAttr(-shift * v_aperture)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, far))
    return UsdGeom.Xformable(cam).AddTransformOp()


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    m = Gf.Matrix4d(1.0)
    m.SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(*up))
    return m.GetInverse()  # SetLookAt returns the view matrix; a USD camera uses its inverse


def _set_viewport_camera(path):
    try:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        try:
            vp.camera_path = path
        except Exception:
            vp.set_active_camera(path)
    except Exception as e:
        carb.log_warn(f"could not switch the viewport camera: {e!r}")


class FollowCameras:
    def __init__(self, start="chase", back=7.0, up=3.0, side=14.0, side_up=1.5, turn_rate=3.0, focal=None,
                 fit=0.0, shift=0.0, aspect=16.0 / 9.0, fit_tau=1.0):
        """back/up: chase offset behind/above the vehicle [m]; side/side_up: lateral (> 0 left, < 0 right) and vertical offset
        of the side camera [m]; focal: focal length [mm] of both cameras (default 18.1 chase, 24 side);
        fit, shift, aspect: group framing (module docstring); fit_tau: time constant of the distance filter [s]."""
        stage = omni.usd.get_context().get_stage()
        cam_kw = dict(aspect=aspect, shift=shift) if (fit or shift) else {}
        self._focal = {"chase": focal or 18.147, "side": focal or 24.0}
        self._op = {"chase": _make_camera(stage, CHASE_PATH, focal=self._focal["chase"], **cam_kw),
                    "side": _make_camera(stage, SIDE_PATH, focal=self._focal["side"], **cam_kw)}
        self._path = {"chase": CHASE_PATH, "side": SIDE_PATH}
        self.mode = start
        self.back, self.up, self.side, self.side_up, self.turn_rate = back, up, side, side_up, turn_rate
        self.fit, self.aspect, self.fit_tau = fit, aspect, fit_tau
        self._scale = 1.0  # distance factor of the group framing (>= 1)
        self._dir = np.array([1.0, 0.0])
        self._input_sub = None
        self._setup_keyboard()
        _set_viewport_camera(self._path[self.mode])
        print(f"[camera] FollowCameras: '{self.mode}' active; C toggles chase/side", flush=True)

    def _setup_keyboard(self):
        try:
            import omni.appwindow as appwindow
            inp = carb.input.acquire_input_interface()
            kb = appwindow.get_default_app_window().get_keyboard()

            def on_key(e, *_):
                if e.type == carb.input.KeyboardEventType.KEY_PRESS and e.input == carb.input.KeyboardInput.C:
                    self.toggle()
                return True

            self._input_sub = inp.subscribe_to_keyboard_events(kb, on_key)
        except Exception as e:
            carb.log_warn(f"camera keyboard shortcut unavailable: {e!r}")

    def toggle(self):
        self.mode = "side" if self.mode == "chase" else "chase"
        _set_viewport_camera(self._path[self.mode])
        print(f"[camera] {self.mode}", flush=True)

    def _fit_scale(self, points, fwd):
        """Distance factor of the side camera that keeps `points` within `fit` of the frame."""
        tan_h = 0.5 * H_APERTURE / self._focal["side"]
        tan_v = tan_h / self.aspect
        d = abs(self.side)
        along = points[:, :2] @ fwd
        need_h = (along.max() - along.min()) / (2.0 * tan_h * self.fit)
        need_v = (points[:, 2].max() - points[:, 2].min()) / (2.0 * tan_v * self.fit)
        return max(1.0, need_h / d, need_v / d)

    def update(self, pos, vel, dt, points=None):
        """pos, vel: world position [m] and velocity [m/s] of the followed vehicle (3-vectors); points: (N, 3) positions
        to frame together (group framing: the cameras aim at the centre of their bounding box)."""
        p = np.asarray(pos, dtype=float).reshape(3)
        if points is not None:
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            p = 0.5 * (points.min(axis=0) + points.max(axis=0))
        v = np.asarray(vel, dtype=float).reshape(3)[:2]
        n = np.linalg.norm(v)
        if n > 0.5:  # in hover keep the last heading
            a = 1.0 - math.exp(-self.turn_rate * dt)
            d = (1.0 - a) * self._dir + a * v / n
            self._dir = d / max(np.linalg.norm(d), 1e-6)
        fwd = self._dir
        left = np.array([-fwd[1], fwd[0]])
        eye_chase = p + np.array([-self.back * fwd[0], -self.back * fwd[1], self.up])
        if points is not None and self.fit > 0:
            target = self._fit_scale(points, fwd)
            self._scale += (target - self._scale) * (1.0 - math.exp(-dt / self.fit_tau))
        k = self._scale
        eye_side = p + np.array([k * self.side * left[0], k * self.side * left[1], k * self.side_up])
        self._op["chase"].Set(_look_at(eye_chase, p))
        self._op["side"].Set(_look_at(eye_side, p))
