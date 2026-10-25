"""Semi-transparent scrolling plots in the top-right corner of the Isaac Sim viewport.

One line plot per panel (several series may share a panel) over a sliding window of the last `window_s` seconds of
simulation time: the time axis does not compress as the run goes on, old samples scroll out on the left.
Call update() once per physics step with the layout, a list of rows, each row a list of panels; a sample is taken
every `sample_dt` seconds. The overlay is built on the first update, from the layout it receives.

A panel is a dict: key (unique), title, values (sequence, one number per series), and optionally
  fmt (format of the numbers in the label), floor (minimum half-range/top of the auto scale),
  symmetric (True: scale [-m, m], False: scale [0, m]), ymax (fixed top of the scale), color (palette index of the
  first series), autorange (True: scale [min, max] of the window, at least `floor` wide; for signals that drift),
  ref (one reference value per series, drawn dashed in white behind the signal).
"""

from collections import deque

import omni.ui as ui

# omni.ui colours are 0xAABBGGRR
PALETTE = [0xFFFFB84D, 0xFF3C96FF, 0xFF64D264, 0xFFE06CC8, 0xFF50D8F0, 0xFFB4B4B4]  # blue, orange, green, magenta, yellow, grey
PANEL_BG = 0x66000000      # black, ~40 % opaque
TEXT = 0xFFFFFFFF
REF_COLOR = 0xD0FFFFFF   # reference line, white
DASH_ON, DASH_OFF = 4, 3  # dash pattern in samples


class TelemetryOverlay:
    def __init__(self, window_s=10.0, sample_dt=0.05, width=500, plot_height=40, margin=16, top=70, dashed=True):
        self.n = int(round(window_s / sample_dt))
        self.sample_dt, self.width, self.margin, self.top = sample_dt, width, margin, top
        self.plot_height = plot_height
        self._last_t = -1e9
        self.dashed = dashed  # False: the reference is a thin solid line (if NaN gaps do not render as dashes)
        self._count = 0
        self._win = None
        self._frame = None
        self._panels = {}  # key -> dict(plots, label, bufs)
        self._layout = []  # keys of the panels, row by row

    def _build(self, rows, n_rows):
        """Build the plots as an overlay frame of the viewport window (follows it in any size, including maximized);
        fall back to a floating borderless window if the viewport window is not available."""
        height = n_rows * (self.plot_height + 26) + 12
        self._frame = None
        try:
            from omni.kit.viewport.utility import get_active_viewport_window
            self._frame = get_active_viewport_window().get_frame("telemetry_overlay")
        except Exception as e:
            print(f"[telemetry] viewport frame unavailable ({e!r}); using a floating window", flush=True)

        if self._frame is not None:
            with self._frame:
                with ui.VStack():
                    ui.Spacer(height=self.top)
                    with ui.HStack(height=height):
                        ui.Spacer()
                        with ui.ZStack(width=self.width, height=height):
                            self._content(rows)
                        ui.Spacer(width=self.margin)
                    ui.Spacer()
            return

        flags = (ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_NO_RESIZE | ui.WINDOW_FLAGS_NO_MOVE
                 | ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_COLLAPSE | ui.WINDOW_FLAGS_NO_BACKGROUND
                 | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS | ui.WINDOW_FLAGS_NO_FOCUS_ON_APPEARING)
        self._win = ui.Window("Telemetry", width=self.width, height=height, flags=flags)
        with self._win.frame:
            with ui.ZStack():
                self._content(rows)
        self._place()

    def _content(self, rows):
        ui.Rectangle(style={"background_color": PANEL_BG, "border_radius": 6})
        with ui.VStack(spacing=4):
            ui.Spacer(height=4)
            for row in rows:
                with ui.HStack(height=self.plot_height + 22, spacing=4):
                    for p in row:
                        self._panels[p["key"]] = self._cell(p)

    def _cell(self, p):
        k = len(p["values"])
        colors = [PALETTE[(p.get("color", 0) + i) % len(PALETTE)] for i in range(k)]
        plots = []
        with ui.VStack():
            with ui.HStack(height=18):
                ui.Spacer(width=8)
                ui.Label(p["title"], style={"color": colors[0], "font_size": 13}, width=0)
                ui.Spacer()
                label = ui.Label("-", style={"color": TEXT, "font_size": 12}, width=0)
                ui.Spacer(width=8)
            with ui.HStack(height=self.plot_height):
                ui.Spacer(width=6)
                with ui.ZStack():
                    ref_plots = []
                    if p.get("ref") is not None:
                        for _ in range(k):
                            ref_plots.append(ui.Plot(ui.Type.LINE, 0.0, 1.0, *([0.0] * 2),
                                                     style={"color": REF_COLOR, "background_color": 0x00000000, "border_width": 0}))
                    for c in colors:
                        plots.append(ui.Plot(ui.Type.LINE, 0.0, 1.0, *([0.0] * 2),
                                             style={"color": c, "background_color": 0x00000000, "border_width": 0}))
                ui.Spacer(width=6)
        return {"plots": plots, "ref_plots": ref_plots, "label": label, "bufs": [deque(maxlen=self.n) for _ in range(k)],
                "ref_bufs": [deque(maxlen=self.n) for _ in range(k)]}

    def _place(self):
        if self._win is None:
            return
        self._win.position_x = max(0.0, ui.Workspace.get_main_window_width() - self.width - self.margin)
        self._win.position_y = self.top

    def update(self, t, rows):
        """t: simulation time [s]; rows: list of rows, each a list of panel dicts (see module docstring)."""
        if t - self._last_t < self.sample_dt:
            return
        self._last_t = t
        panels = [p for row in rows for p in row]
        self._layout = [[p["key"] for p in row] for row in rows]
        if not self._panels:
            self._build(rows, len(rows))
        self._count += 1
        for p in panels:
            ui_p = self._panels[p["key"]]
            for bufs, vals in ((ui_p["bufs"], p["values"]), (ui_p["ref_bufs"], p.get("ref") or [])):
                for buf, v in zip(bufs, vals):
                    if not buf:  # fill the window with the first value so the plot does not start with a spike
                        buf.extend([float(v)] * self.n)
                    buf.append(float(v))
            allb = list(ui_p["bufs"]) + [b for b in ui_p["ref_bufs"] if b]
            if p.get("autorange"):
                lo_v, hi_v = min(min(b) for b in allb), max(max(b) for b in allb)
                span = max(p.get("floor", 1.0), 1.1 * (hi_v - lo_v))
                mid = 0.5 * (lo_v + hi_v)
                lo, top = mid - 0.5 * span, mid + 0.5 * span
            else:
                peak = max(max(abs(x) if p.get("symmetric") else x for x in buf) for buf in allb)
                top = p["ymax"] if p.get("ymax") is not None else max(p.get("floor", 1.0), 1.1 * peak)
                lo = -top if p.get("symmetric") else 0.0
            ui_p["meta"] = dict(title=p["title"], color=p.get("color", 0), lo=lo, top=top)
            for plot, buf in zip(ui_p["plots"], ui_p["bufs"]):
                plot.scale_min, plot.scale_max = lo, top
                plot.set_data(*buf)
            for plot, buf in zip(ui_p["ref_plots"], ui_p["ref_bufs"]):
                plot.scale_min, plot.scale_max = lo, top
                if self.dashed:
                    # dashes slide with the data: the pattern follows the absolute sample index; NaN breaks the line
                    first = self._count - len(buf)
                    plot.set_data(*[v if (first + j) % (DASH_ON + DASH_OFF) < DASH_ON else float("nan")
                                    for j, v in enumerate(buf)])
                else:
                    plot.set_data(*buf)
            fmt = p.get("fmt", "{:6.2f}")
            ui_p["label"].text = "  ".join(fmt.format(b[-1]) for b in ui_p["bufs"])
            ui_p["meta"]["label"] = ui_p["label"].text
        self._place()

    def compose(self, image, top=24):
        """Draw the overlay (as it is now) onto a PIL image, scaled to its height (1080 px = the on-screen size).
        Returns the composited RGBA image. Used for screenshots larger than the screen, where the omni.ui frame cannot go."""
        from PIL import Image, ImageDraw, ImageFont

        img = image.convert("RGBA")
        k = img.height / 1080.0
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        f_title, f_val = ImageFont.truetype(font_path, max(8, int(13 * k))), ImageFont.truetype(font_path, max(8, int(12 * k)))
        rgb = lambda c: (c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF)

        n_rows = len(self._layout)
        pw, ph = int(self.width * k), int((n_rows * (self.plot_height + 26) + 12) * k)
        layer = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(6 * k), fill=(0, 0, 0, 102))
        gap = 4 * k
        for r, keys in enumerate(self._layout):
            cw = (pw - gap * (len(keys) - 1)) / len(keys)
            y0 = (4 + r * (self.plot_height + 26)) * k
            for c, key in enumerate(keys):
                ui_p, m = self._panels[key], self._panels[key]["meta"]
                x0 = c * (cw + gap)
                colors = [PALETTE[(m["color"] + i) % len(PALETTE)] for i in range(len(ui_p["bufs"]))]
                d.text((x0 + 8 * k, y0), m["title"], font=f_title, fill=rgb(colors[0]) + (255,))
                tw = d.textlength(m["label"], font=f_val)
                d.text((x0 + cw - 8 * k - tw, y0 + 1 * k), m["label"], font=f_val, fill=(255, 255, 255, 255))
                px0, px1 = x0 + 6 * k, x0 + cw - 6 * k
                py0, py1 = y0 + 20 * k, y0 + (20 + self.plot_height) * k
                span = max(m["top"] - m["lo"], 1e-9)

                def pts(buf):
                    n = len(buf)
                    return [(px0 + (px1 - px0) * j / max(n - 1, 1), py1 - (py1 - py0) * (v - m["lo"]) / span)
                            for j, v in enumerate(buf)]

                for buf in ui_p["ref_bufs"]:       # reference: dashed white (same pattern as on screen)
                    if not buf:
                        continue
                    p_ = pts(buf)
                    first = self._count - len(buf)
                    run = []
                    for j, q in enumerate(p_):
                        if (first + j) % (DASH_ON + DASH_OFF) < DASH_ON:
                            run.append(q)
                        else:
                            if len(run) > 1:
                                d.line(run, fill=(255, 255, 255, 208), width=max(1, int(1.5 * k)))
                            run = []
                    if len(run) > 1:
                        d.line(run, fill=(255, 255, 255, 208), width=max(1, int(1.5 * k)))
                for buf, col in zip(ui_p["bufs"], colors):
                    d.line(pts(buf), fill=rgb(col) + (255,), width=max(1, int(2 * k)), joint="curve")
        img.alpha_composite(layer, (int(img.width - pw - self.margin * k), int(top * k)))
        return img
