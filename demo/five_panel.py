"""Five-panel demo visualisation.

The architecture made visible. Each panel shows one layer, and together they
explain the whole system to someone who will never read the code:

    +---------------------------+------------------+
    |                           |   AGENT'S EYE    |  what the drone sees
    |       CHASE VIEW          +------------------+
    |     (photoreal world)     |   OPTIC FLOW     |  what it computes
    |                           +------------------+
    |                           |  WIDE-FIELD BARS |  four channels
    +---------------------------+------------------+
    |         RING ATTRACTOR (heading bump)        |  where it thinks it is
    +----------------------------------------------+

THE TWO-CAMERA PRINCIPLE IS THE POINT. The chase view is rendered
beautifully; the agent's eye is 40x20 greyscale. Both drive the same flight.
Putting them side by side is the clearest possible statement that this drone
navigates on almost no information -- which is the claim the whole project
rests on.

Rendered with OpenCV only. No GUI toolkit, so it runs headless for recording.
"""

from typing import Optional

import cv2
import numpy as np

# Palette, BGR. Dark ground so the flow arrows and bars carry the eye.
BG = (24, 20, 18)
PANEL_BG = (34, 29, 26)
RULE = (70, 62, 56)
INK = (236, 240, 245)
MUTED = (140, 132, 124)
ACCENT = (40, 110, 220)      # warm orange in BGR
GOOD = (90, 180, 110)
WARN = (60, 160, 240)
DANGER = (70, 70, 220)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class FivePanelView:
    """Composes the demo frame from a PipelineState."""

    __slots__ = (
        "_width",
        "_height",
        "_upscale",
        "_chase_w",
        "_side_w",
        "_ring_h",
        "_panel_h",
    )

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        upscale: int = 8,
    ) -> None:
        self._width = width
        self._height = height
        self._upscale = upscale

        # Layout: chase view takes the left 60%, three stacked panels the
        # right 40%, and the ring attractor spans the full width beneath.
        self._ring_h = 150
        self._panel_h = (height - self._ring_h) // 3
        self._chase_w = int(width * 0.60)
        self._side_w = width - self._chase_w

    # -----------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------

    def render(
        self,
        state,
        chase_frame: Optional[np.ndarray] = None,
        retina: Optional[np.ndarray] = None,
        elapsed: float = 0.0,
    ) -> np.ndarray:
        """Compose one demo frame.

        Args:
            state: PipelineState from the current control step.
            chase_frame: the photoreal third-person view, BGR.
            retina: the 20x40 float retina the agent actually sees.
            elapsed: flight time in seconds.

        Returns:
            BGR image of (height, width, 3).
        """
        canvas = np.full((self._height, self._width, 3), BG, dtype=np.uint8)

        body_h = self._height - self._ring_h
        self._draw_chase(canvas, chase_frame, state, elapsed, body_h)

        x0 = self._chase_w
        self._draw_agent_eye(canvas, retina, x0, 0)
        self._draw_flow(canvas, state, x0, self._panel_h)
        self._draw_channels(canvas, state, x0, self._panel_h * 2)
        self._draw_ring(canvas, state, 0, body_h)

        return canvas

    # -----------------------------------------------------------------
    # Panels
    # -----------------------------------------------------------------

    def _draw_chase(self, canvas, frame, state, elapsed, height):
        """Photoreal third-person view, with the flight status overlaid."""
        region = canvas[0:height, 0:self._chase_w]
        region[:] = PANEL_BG

        if frame is not None:
            resized = self._fit(frame, self._chase_w, height)
            h, w = resized.shape[:2]
            y = (height - h) // 2
            x = (self._chase_w - w) // 2
            region[y:y + h, x:x + w] = resized

        self._label(canvas, "CHASE VIEW", 14, 26)

        if state is None:
            return

        # Status block, bottom left. These are the numbers a viewer tracks.
        lines = []
        if state.true_position is not None:
            lines.append(
                f"pos  ({state.true_position[0]:+.2f}, "
                f"{state.true_position[1]:+.2f})"
            )
        if state.battery_state is not None:
            lines.append(f"batt {state.battery_state.charge_pct:5.1f}%")
        lines.append(f"time {elapsed:6.1f}s")

        # Backing plate. MEASURED by rendering a frame and looking at it:
        # white text on the bright chase view was unreadable. A
        # pixel-difference test cannot catch legibility.
        block_h = 22 * len(lines) + 12
        y0 = height - 16 - block_h
        overlay = canvas[y0:y0 + block_h, 8:230].copy()
        cv2.rectangle(canvas, (8, y0), (230, y0 + block_h), (18, 15, 13), -1)
        cv2.addWeighted(
            canvas[y0:y0 + block_h, 8:230], 0.72, overlay, 0.28, 0,
            canvas[y0:y0 + block_h, 8:230],
        )

        y = y0 + 24
        for line in lines:
            cv2.putText(canvas, line, (18, y), FONT, 0.52, INK, 1, cv2.LINE_AA)
            y += 22

        # The override banner is the most important thing on screen when it
        # appears: it says the drone has stopped doing the mission and why.
        if state.override_reason:
            self._banner(canvas, state.override_reason, height)

    def _draw_agent_eye(self, canvas, retina, x0, y0):
        """The 40x20 greyscale view, upscaled so it is legible.

        Deliberately blocky. Smoothing it would misrepresent what the drone
        actually has to work with.
        """
        self._panel_frame(canvas, x0, y0, "AGENT'S EYE  40x20 grey")

        if retina is None:
            return

        grey = np.clip(np.asarray(retina) * 255.0, 0, 255).astype(np.uint8)
        # INTER_NEAREST keeps the pixels square and visible as pixels.
        big = cv2.resize(
            grey,
            (self._side_w - 24, self._panel_h - 46),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas[y0 + 38:y0 + 38 + big.shape[0], x0 + 12:x0 + 12 + big.shape[1]] = (
            cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        )

    def _draw_flow(self, canvas, state, x0, y0):
        """The optic flow field as arrows over a magnitude heatmap."""
        self._panel_frame(canvas, x0, y0, "OPTIC FLOW")

        if state is None or state.flow_field is None:
            return

        flow = state.flow_field.flow
        magnitude = state.flow_field.magnitude

        panel_w = self._side_w - 24
        panel_h = self._panel_h - 46

        peak = float(np.max(magnitude))
        if peak > 1e-9:
            heat = np.clip(magnitude / peak * 255.0, 0, 255).astype(np.uint8)
        else:
            heat = np.zeros_like(magnitude, dtype=np.uint8)

        heat = cv2.resize(heat, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
        coloured = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        canvas[y0 + 38:y0 + 38 + panel_h, x0 + 12:x0 + 12 + panel_w] = coloured

        # Arrows, subsampled so they stay readable at this size.
        rows, cols = magnitude.shape
        step_r = max(1, rows // 6)
        step_c = max(1, cols // 12)
        scale_x = panel_w / cols
        scale_y = panel_h / rows

        if peak > 1e-9:
            for r in range(0, rows, step_r):
                for c in range(0, cols, step_c):
                    u = float(flow[r, c, 0])
                    v = float(flow[r, c, 1])
                    if abs(u) + abs(v) < peak * 0.08:
                        continue
                    cx = int(x0 + 12 + (c + 0.5) * scale_x)
                    cy = int(y0 + 38 + (r + 0.5) * scale_y)
                    dx = int(np.clip(u / peak * 16.0, -16, 16))
                    dy = int(np.clip(v / peak * 16.0, -16, 16))
                    cv2.arrowedLine(
                        canvas, (cx, cy), (cx + dx, cy + dy),
                        INK, 1, cv2.LINE_AA, tipLength=0.35,
                    )

        if not state.flow_field.valid:
            cv2.putText(
                canvas, "no history", (x0 + 18, y0 + 58),
                FONT, 0.45, MUTED, 1, cv2.LINE_AA,
            )

    def _draw_channels(self, canvas, state, x0, y0):
        """The four wide-field channels as live bars.

        This is the panel that makes the architecture's central claim
        visible: thousands of local detectors reduced to four numbers.
        """
        self._panel_frame(canvas, x0, y0, "WIDE-FIELD  4 channels")

        if state is None or state.channels is None:
            return

        ch = state.channels
        entries = [
            ("L exp", ch.left_expansion, ACCENT),
            ("R exp", ch.right_expansion, ACCENT),
            ("vert", ch.vertical_expansion, MUTED),
            ("rot", ch.rotation, GOOD),
        ]

        # Fixed scale, so bar length means the same thing frame to frame.
        # An autoscaling bar chart looks lively and says nothing.
        span = 1.0
        bar_x = x0 + 74
        bar_w = self._side_w - 164
        centre = bar_x + bar_w // 2
        y = y0 + 52

        for name, value, colour in entries:
            cv2.putText(canvas, name, (x0 + 14, y + 5), FONT, 0.42, MUTED, 1,
                        cv2.LINE_AA)
            cv2.line(canvas, (bar_x, y), (bar_x + bar_w, y), RULE, 1)
            cv2.line(canvas, (centre, y - 9), (centre, y + 9), RULE, 1)

            extent = int(np.clip(value / span, -1.0, 1.0) * (bar_w // 2))
            if extent != 0:
                cv2.rectangle(
                    canvas,
                    (centre, y - 7),
                    (centre + extent, y + 7),
                    colour, -1,
                )
            cv2.putText(
                canvas, f"{value:+.3f}", (bar_x + bar_w + 6, y + 4),
                FONT, 0.38, INK, 1, cv2.LINE_AA,
            )
            y += 26

        # Urgency and the arbitration blend: the moment an obstacle takes
        # over is the single most legible event in the whole demo.
        if state.avoidance is not None and state.arbitration is not None:
            urgency = state.avoidance.urgency
            weight = state.arbitration.avoidance_weight
            colour = DANGER if weight > 0.5 else (WARN if weight > 0.2 else GOOD)
            cv2.putText(
                canvas,
                f"urgency {urgency:.2f}   avoid {weight:.2f}",
                (x0 + 14, y + 6), FONT, 0.46, colour, 1, cv2.LINE_AA,
            )

    def _draw_ring(self, canvas, state, x0, y0):
        """The ring attractor bump, drawn as an actual ring.

        A bar chart would show the values; a ring shows that it IS a ring,
        with a single bump that rotates and wraps at the seam.
        """
        h = self._ring_h
        cv2.rectangle(canvas, (x0, y0), (self._width, y0 + h), PANEL_BG, -1)
        cv2.line(canvas, (x0, y0), (self._width, y0), RULE, 1)
        self._label(canvas, "CENTRAL COMPLEX  ring attractor", x0 + 14, y0 + 24)

        if state is None or state.heading_state is None:
            return

        bump = state.heading_state.bump
        n = len(bump)
        peak = float(np.max(bump)) if n else 0.0

        cx = x0 + 110
        cy = y0 + h // 2 + 8
        radius = 42

        cv2.circle(canvas, (cx, cy), radius, RULE, 1, cv2.LINE_AA)

        for i in range(n):
            angle = 2.0 * np.pi * i / n
            strength = (bump[i] / peak) if peak > 1e-12 else 0.0
            length = int(6 + strength * 26)
            x1 = int(cx + radius * np.cos(angle))
            y1 = int(cy - radius * np.sin(angle))
            x2 = int(cx + (radius + length) * np.cos(angle))
            y2 = int(cy - (radius + length) * np.sin(angle))
            shade = int(60 + 195 * strength)
            cv2.line(canvas, (x1, y1), (x2, y2), (shade // 3, shade // 2, shade),
                     2, cv2.LINE_AA)

        heading = state.heading_state.heading_rad
        hx = int(cx + (radius - 6) * np.cos(heading))
        hy = int(cy - (radius - 6) * np.sin(heading))
        cv2.arrowedLine(canvas, (cx, cy), (hx, hy), ACCENT, 2, cv2.LINE_AA,
                        tipLength=0.3)

        tx = cx + radius + 60
        cv2.putText(
            canvas, f"heading {np.degrees(heading):+7.1f} deg",
            (tx, cy - 14), FONT, 0.5, INK, 1, cv2.LINE_AA,
        )
        confidence = state.heading_state.confidence
        colour = GOOD if confidence > 0.5 else WARN
        cv2.putText(
            canvas, f"confidence {confidence:.2f}",
            (tx, cy + 10), FONT, 0.5, colour, 1, cv2.LINE_AA,
        )

        # Path integration: the vector home, which is what makes GPS-denied
        # return possible and is worth showing explicitly.
        if state.path_state is not None:
            gps = "GPS" if state.path_state.gps_available else "GPS DENIED"
            gps_colour = MUTED if state.path_state.gps_available else WARN
            cv2.putText(
                canvas,
                f"{gps}   home {state.path_state.distance_to_home:5.1f}m",
                (tx + 260, cy - 14), FONT, 0.5, gps_colour, 1, cv2.LINE_AA,
            )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _panel_frame(self, canvas, x0, y0, title):
        cv2.rectangle(
            canvas, (x0, y0), (self._width, y0 + self._panel_h), PANEL_BG, -1
        )
        cv2.line(canvas, (x0, y0), (self._width, y0), RULE, 1)
        cv2.line(canvas, (x0, y0), (x0, y0 + self._panel_h), RULE, 1)
        self._label(canvas, title, x0 + 12, y0 + 24)

    def _label(self, canvas, text, x, y):
        cv2.putText(canvas, text, (x, y), FONT, 0.46, MUTED, 1, cv2.LINE_AA)

    def _banner(self, canvas, reason, height):
        """Override banner. Red, because it means the mission has stopped."""
        text = reason.replace("_", " ").upper()
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
        x, y = 14, 54
        cv2.rectangle(
            canvas, (x - 6, y - th - 8), (x + tw + 10, y + 8), DANGER, -1
        )
        cv2.putText(canvas, text, (x, y), FONT, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)

    @staticmethod
    def _fit(image, max_w, max_h):
        """Scale to fit while preserving aspect ratio."""
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return image
        scale = min(max_w / w, max_h / h)
        return cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @classmethod
    def from_config(cls, cfg) -> "FivePanelView":
        return cls(upscale=cfg.demo.upscale_factor)
