# ui.py 

from __future__ import annotations # Allows type hints that reference classes before they are defined.

import time
from typing import Dict, List, Tuple # Enables clear type hints for lists, dictionaries, and tuples.

import tkinter as tk
from tkinter import ttk, messagebox

from probability_model import Point, Click
import results as results_module


CANVAS_WIDTH = 640
CANVAS_HEIGHT = 360

CLICK_RADII = [32, 24, 16, 12, 10, 8, 7, 6]


def current_millis() -> float:
    return time.time() * 1000.0


class TremorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Little Tremors, Big Stories – CS109 Challenge Project | Aditya Garg")
        self.geometry("1920x1080")

        # Data store: same structure expected by probability_model / results
        self.data_store: Dict[str, List] = {
            "line": [],
            "circle": [],
            "spiral": [],
            "eight": [],
            "clicks": [],
        }

        self.click_radii = CLICK_RADII
        self.click_targets: List[Tuple[float, float, float]] = []  # (x, y, r)

        # Current drawing state
        self.mode = "line"
        self.current_stroke: List[Point] | None = None

        self._build_ui()
        self._setup_bindings()
        self._compute_click_targets()
        self._redraw_all()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Top-level layout: left (canvas + controls) and right (summary/results)
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        # --- Left: buttons, status, canvas ---
        button_frame = ttk.Frame(left)
        button_frame.pack(fill="x")

        self.mode_var = tk.StringVar(value="line")

        def make_mode_button(text: str, mode: str) -> ttk.Radiobutton:
            return ttk.Radiobutton(
                button_frame,
                text=text,
                value=mode,
                variable=self.mode_var,
                command=self._on_mode_changed,
            )

        make_mode_button("1. Straight line", "line").grid(row=0, column=0, padx=4, pady=2)
        make_mode_button("2. Circle", "circle").grid(row=0, column=1, padx=4, pady=2)
        make_mode_button("3. Spiral", "spiral").grid(row=0, column=2, padx=4, pady=2)
        make_mode_button("4. Figure-8", "eight").grid(row=0, column=3, padx=4, pady=2)
        make_mode_button("5. Click targets", "click").grid(row=0, column=4, padx=4, pady=2)

        self.analyze_button = ttk.Button(
            button_frame,
            text="Analyze my pattern",
            command=self._on_analyze_clicked,
        )
        self.analyze_button.grid(row=0, column=5, padx=10, pady=2)

        self.reset_button = ttk.Button(
            button_frame,
            text="Reset data",
            command=self._on_reset_clicked,
        )
        self.reset_button.grid(row=0, column=6, padx=4, pady=2)

        self.status_label = ttk.Label(
            left,
            text="Task: trace the grey horizontal line from left to right.",
        )
        self.status_label.pack(fill="x", pady=4)

        # Canvas
        self.canvas = tk.Canvas(
            left,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#050811",
            highlightthickness=1,
            highlightbackground="#444444",
        )
        self.canvas.pack(fill="both", expand=False)

        # Short note
        self.note_label = ttk.Label(
            left,
            text=(
                "You can repeat each task several times. "
                "\n"
                "You need to complete at least 5 strokes for Straight line, Circle, Spiral, and Figure-8. "
                "\n"
                "You need at least 65 clicks in total across all button sizes and atleast 8 clicks on each size. "
                "\n"
                "The analysis uses all strokes and clicks together, "
                "so more data usually leads to a more stable estimate."
            ),
            wraplength=CANVAS_WIDTH,
        )
        self.note_label.pack(fill="x", pady=4)

        # Data summary and time
        summary_frame = ttk.Frame(left)
        summary_frame.pack(fill="both", expand=True, pady=(4, 0))

        ttk.Label(summary_frame, text="Data collected so far:").pack(anchor="w")
        self.data_summary_text = tk.Text(
            summary_frame, height=14, wrap="word", font=("Courier", 12)
        )
        self.data_summary_text.pack(fill="x", pady=2)

        ttk.Label(summary_frame, text="Time spent (approximate):").pack(anchor="w")
        self.time_summary_text = tk.Text(
            summary_frame, height=6, wrap="word", font=("Courier", 12)
        )
        self.time_summary_text.pack(fill="x", pady=2)

        # --- Right: explanation / results ---
        about_frame = ttk.Frame(right)
        about_frame.pack(fill="both", expand=True)

        about_title = ttk.Label(
            about_frame,
            text="About this Project",
            font=("TkDefaultFont", 12, "bold"),
        )
        about_title.pack(anchor="w")

        about_text = (
            "I grew up around people (family members, neighbours, and even strangers at the local "
            "convenience store) who lived with small hand tremors or difficulty keeping their hands "
            "steady. Some were older, some were younger, and many were never formally diagnosed. "
            "Millions of people around the world experience motor challenges like these without ever "
            "receiving a name for what they feel, yet these small movements shape how they interact "
            "with the digital world every day.\n\n"

            "Little Tremors, Big Stories is an educational CS109 probability experiment designed to "
            "make those micro-movements visible. As you draw lines, circles, spirals, and figure-8s, or "
            "click on different button sizes, the app records your cursor positions and timing. These "
            "become random variables describing your movement patterns: how steady your line is, how "
            "smoothly your direction changes, how your radius expands and contracts, and how close your "
            "clicks land to their targets.\n\n"

            "When you analyze your data, the full results are saved in a dedicated output folder. Inside "
            "you’ll find:\n"
            "  • A written summary explaining your movement features.\n"
            "  • A Gaussian Naive Bayes posterior showing which synthetic motor profile your data "
            "resembles most closely.\n"
            "  • Button-by-button click statistics comparing your actual hit rates with the Rayleigh "
            "model’s predicted probabilities.\n"
            "  • Several graphs that visualize your variability, the model’s assumptions, and how "
            "target size interacts with click noise.\n\n"

            "This project is meant to be thoughtful, approachable, and curiosity-driven. It is not a "
            "diagnostic test, not a medical evaluation, and not a substitute for professional advice. "
            "Its purpose is simply to show how probability can help us understand everyday motor noise "
            "and encourage more empathetic, accessible interface design."
        )
        about_label = ttk.Label(about_frame, text=about_text, wraplength=800)
        about_label.pack(fill="x", pady=(2, 6))

        ttk.Label(about_frame, text="Analysis results:").pack(anchor="w")

        self.results_text = tk.Text(
            about_frame, height=22, wrap="word", font=("Courier", 12)
        )
        self.results_text.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Event bindings
    # ------------------------------------------------------------------

    def _setup_bindings(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

    # ------------------------------------------------------------------
    # Mode / buttons
    # ------------------------------------------------------------------

    def _on_mode_changed(self) -> None:
        self.mode = self.mode_var.get()
        if self.mode == "line":
            self.status_label.config(
                text="Task: trace the grey horizontal line from left to right."
            )
        elif self.mode == "circle":
            self.status_label.config(
                text="Task: draw a round circle following the guide."
            )
        elif self.mode == "spiral":
            self.status_label.config(
                text="Task: draw a spiral along the guide."
            )
        elif self.mode == "eight":
            self.status_label.config(
                text="Task: draw a sideways 8 along the guide."
            )
        elif self.mode == "click":
            self.status_label.config(
                text="Task: click inside the circles (big to small) several times each (at least 8 times each and 65 times total)."
            )

        self._redraw_all()

    def _on_reset_clicked(self) -> None:
        if messagebox.askyesno("Reset data", "Clear all strokes and clicks?"):
            self.data_store = {
                "line": [],
                "circle": [],
                "spiral": [],
                "eight": [],
                "clicks": [],
            }
            self.current_stroke = None
            self.results_text.delete("1.0", "end")
            self._redraw_all()
            self._update_summaries()

    def _on_analyze_clicked(self) -> None:
        try:
            analysis = results_module.analyze_and_save(
                self.data_store,
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
                self.click_radii,
                base_output_dir="outputs",
            )
        except results_module.NotEnoughDataError as e:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"Not enough data yet:\n{e}\n")
            return
        except Exception as e:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"Error during analysis:\n{e}\n")
            return

        # Show summary and where files were saved
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", analysis.summary_text)
        self.results_text.insert(
            "end",
            f"\n\nGraphs and data have been saved in:\n  {analysis.output_dir}\n",
        )

    # ------------------------------------------------------------------
    # Canvas drawing & guides
    # ------------------------------------------------------------------

    def _compute_click_targets(self) -> None:
        """
        Compute target centers horizontally across the canvas.
        """
        center_y = CANVAS_HEIGHT / 2.0
        n = len(self.click_radii)
        spacing = CANVAS_WIDTH / (n + 1)
        self.click_targets = []
        for i, r in enumerate(self.click_radii):
            x = spacing * (i + 1)
            y = center_y
            self.click_targets.append((x, y, r))

    def _draw_guides(self) -> None:
        c = self.canvas
        c.delete("guide")

        if self.mode == "line":
            y = CANVAS_HEIGHT / 2.0
            c.create_line(
                40,
                y,
                CANVAS_WIDTH - 40,
                y,
                fill="#666b88",
                dash=(6, 4),
                width=2,
                tags="guide",
            )

        elif self.mode == "circle":
            cx = CANVAS_WIDTH / 2.0
            cy = CANVAS_HEIGHT / 2.0
            r = 80
            c.create_oval(
                cx - r,
                cy - r,
                cx + r,
                cy + r,
                outline="#666b88",
                dash=(6, 4),
                width=2,
                tags="guide",
            )

        elif self.mode == "spiral":
            # Accurate Archimedean-like spiral guide
            cx = CANVAS_WIDTH / 2.0
            cy = CANVAS_HEIGHT / 2.0
            coords = []
            r = 20.0
            a_step = 0.08  # angle step (rad)
            for k in range(int(4 * 3.14159 / a_step) + 1):
                a = k * a_step
                x = cx + r * math.cos(a)
                y = cy + r * math.sin(a)
                coords.extend([x, y])
                # increase radius slightly for each step
                r += 0.8

            if len(coords) >= 4:
                c.create_line(
                    *coords,
                    fill="#666b88",
                    dash=(6, 4),
                    width=2,
                    tags="guide",
                    smooth=True,
                )

        elif self.mode == "eight":
            # Sideways figure-8 (Gerono lemniscate style)
            cx = CANVAS_WIDTH / 2.0
            cy = CANVAS_HEIGHT / 2.0
            a = 60  # size parameter
            coords = []
            step = 0.03
            t_val = 0.0
            while t_val <= 2 * math.pi + 1e-6:
                x = cx + a * math.sin(t_val)
                y = cy + a * math.sin(t_val) * math.cos(t_val)
                coords.extend([x, y])
                t_val += step
            if len(coords) >= 4:
                c.create_line(
                    *coords,
                    fill="#666b88",
                    dash=(6, 4),
                    width=2,
                    tags="guide",
                    smooth=True,
                )

        elif self.mode == "click":
            # Draw click targets
            self._compute_click_targets()
            for (x, y, r) in self.click_targets:
                c.create_oval(
                    x - r,
                    y - r,
                    x + r,
                    y + r,
                    outline="#c0c4ff",
                    width=2,
                    tags="guide",
                )
                diam = 2 * r
                c.create_text(
                    x,
                    y + r + 12,
                    text=f"{diam:.0f}px",
                    fill="#ddddff",
                    font=("TkDefaultFont", 8),
                    tags="guide",
                )

    def _draw_strokes_for_current_mode(self) -> None:
        if self.mode == "click":
            return  # we only show targets for click mode

        strokes: List[List[Point]] = self.data_store.get(self.mode, [])
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            coords = []
            for p in stroke:
                coords.extend([p.x, p.y])
            self.canvas.create_line(
                *coords,
                fill="#66bb6a",
                width=2,
                tags="stroke",
                smooth=True,
            )

        # If currently drawing, show it in a lighter color
        if self.current_stroke and len(self.current_stroke) >= 2:
            coords = []
            for p in self.current_stroke:
                coords.extend([p.x, p.y])
            self.canvas.create_line(
                *coords,
                fill="#88ff88",
                width=2,
                tags="stroke",
                smooth=True,
            )

    def _redraw_all(self) -> None:
        self.canvas.delete("all")
        self._draw_guides()
        self._draw_strokes_for_current_mode()
        self._update_summaries()

    # ------------------------------------------------------------------
    # Canvas event handlers
    # ------------------------------------------------------------------

    def _on_canvas_press(self, event: tk.Event) -> None:
        x, y = float(event.x), float(event.y)
        t_ms = current_millis()

        if self.mode == "click":
            self._handle_click(x, y, t_ms)
            return

        # Start a new stroke
        self.current_stroke = [Point(x=x, y=y, t_ms=t_ms)]
        self._redraw_all()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self.mode == "click":
            return
        if self.current_stroke is None:
            return

        x, y = float(event.x), float(event.y)
        t_ms = current_millis()
        self.current_stroke.append(Point(x=x, y=y, t_ms=t_ms))
        self._redraw_all()

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self.mode == "click":
            return
        if not self.current_stroke:
            return

        # Only record strokes with a few points
        if len(self.current_stroke) > 4:
            self.data_store[self.mode].append(self.current_stroke)

        self.current_stroke = None
        self._redraw_all()

    def _handle_click(self, x: float, y: float, t_ms: float) -> None:
        if not self.click_targets:
            self._compute_click_targets()

        # Find nearest target
        best_idx = 0
        best_dist = float("inf")
        for i, (cx, cy, r) in enumerate(self.click_targets):
            dx = x - cx
            dy = y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        target_x, target_y, r = self.click_targets[best_idx]
        hit = best_dist <= r
        click = Click(
            x=x,
            y=y,
            t_ms=t_ms,
            target_index=best_idx,
            hit=hit,
            error_px=best_dist,
        )
        self.data_store["clicks"].append(click)

        # Redraw targets and highlight nearest circle
        self._redraw_all()
        color = "#66bb6a" if hit else "#ff5555"
        self.canvas.create_oval(
            target_x - r,
            target_y - r,
            target_x + r,
            target_y + r,
            outline=color,
            width=3,
            tags="stroke",
        )

    # ------------------------------------------------------------------
    # Summaries (data + time)
    # ------------------------------------------------------------------

    def _update_summaries(self) -> None:
        self._update_data_summary()
        self._update_time_summary()

    def _update_data_summary(self) -> None:
        n_line = len(self.data_store["line"])
        n_circle = len(self.data_store["circle"])
        n_spiral = len(self.data_store["spiral"])
        n_eight = len(self.data_store["eight"])
        clicks: List[Click] = self.data_store["clicks"]
        total_clicks = len(clicks)

        counts = [
            {"hits": 0, "total": 0} for _ in self.click_radii
        ]
        for c in clicks:
            if 0 <= c.target_index < len(self.click_radii):
                counts[c.target_index]["total"] += 1
                if c.hit:
                    counts[c.target_index]["hits"] += 1

        lines = []
        lines.append(f"Lines:     {n_line} strokes")
        lines.append(f"Circles:   {n_circle} strokes")
        lines.append(f"Spirals:   {n_spiral} strokes")
        lines.append(f"Figure-8s: {n_eight} strokes")
        lines.append(f"Clicks:    {total_clicks} total")
        for i, r in enumerate(self.click_radii):
            diam = 2 * r
            lines.append(
                f"  • {diam:>3.0f}px buttons: {counts[i]['total']} clicks"
            )

        self.data_summary_text.delete("1.0", "end")
        self.data_summary_text.insert("1.0", "\n".join(lines))

    def _update_time_summary(self) -> None:
        def stroke_time_info(strokes: List[List[Point]]) -> Tuple[int, float]:
            if not strokes:
                return 0, 0.0
            total_ms = 0.0
            for s in strokes:
                if len(s) < 2:
                    continue
                total_ms += s[-1].t_ms - s[0].t_ms
            return len(strokes), total_ms

        n_line, t_line = stroke_time_info(self.data_store["line"])
        n_circ, t_circ = stroke_time_info(self.data_store["circle"])
        n_sp, t_sp = stroke_time_info(self.data_store["spiral"])
        n_ei, t_ei = stroke_time_info(self.data_store["eight"])

        clicks: List[Click] = self.data_store["clicks"]
        if len(clicks) > 1:
            t_click_ms = clicks[-1].t_ms - clicks[0].t_ms
        else:
            t_click_ms = 0.0

        def fmt(ms: float) -> str:
            return f"{ms / 1000.0:.1f}"

        lines = []
        lines.append(
            f"Line drawing:   {n_line} strokes, ~{fmt(t_line)} s total"
        )
        lines.append(
            f"Circles:        {n_circ} strokes, ~{fmt(t_circ)} s total"
        )
        lines.append(
            f"Spirals:        {n_sp} strokes, ~{fmt(t_sp)} s total"
        )
        lines.append(
            f"Figure-8s:      {n_ei} strokes, ~{fmt(t_ei)} s total"
        )
        lines.append(
            f"Click task:     {len(clicks)} clicks over ~{fmt(t_click_ms)} s"
        )

        self.time_summary_text.delete("1.0", "end")
        self.time_summary_text.insert("1.0", "\n".join(lines))


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import math  # used in spiral / figure-8

    app = TremorApp()
    app.mainloop()
