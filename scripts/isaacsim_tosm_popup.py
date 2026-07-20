"""Isaac Sim / Omniverse Kit: show TOSM info when an object is clicked.

Sketch. Paste into Isaac Sim's Script Editor (Window > Script Editor) after
opening tosm_scene.usda, or wrap as a Kit extension. On every selection change
it reads the selected prim's `customData` (written by usd_export.py) and shows
it in a docked panel. No popup-window plumbing needed beyond omni.ui.

Each object prim carries customData like:
    string type = "chair"; bool isKeyObject = false; double3 dimensions_lwh = ...
    string rel_isNextTo = "machine_004, ..."; string rel_isInsideOf = "place_1"
plus matching custom `tosm:*` attributes and `rel tosm:*` relationships.
"""
import omni.usd
import omni.ui as ui

_KEYS = ["name", "type", "id", "color", "confidence", "isKeyObject",
         "isMovable", "pose_xyz", "dimensions_lwh", "rel_isInsideOf",
         "rel_isOn", "rel_isAboveOf", "rel_isNextTo", "symbolicReason"]


class TosmInspector:
    def __init__(self):
        self._win = ui.Window("TOSM Object Info", width=380, height=460)
        self._labels = {}
        with self._win.frame:
            with ui.VStack(spacing=4):
                self._title = ui.Label("(click an object)",
                                       style={"font_size": 18})
                ui.Separator()
                for k in _KEYS:
                    with ui.HStack(height=0):
                        ui.Label(k, width=130,
                                 style={"color": 0xFF9AA0A6})
                        self._labels[k] = ui.Label("", word_wrap=True)
        # subscribe to selection changes
        self._sub = (omni.usd.get_context().get_stage_event_stream()
                     .create_subscription_to_pop(self._on_event,
                                                  name="tosm_inspector"))

    def _on_event(self, e):
        if e.type != int(omni.usd.StageEventType.SELECTION_CHANGED):
            return
        ctx = omni.usd.get_context()
        sel = ctx.get_selection().get_selected_prim_paths()
        stage = ctx.get_stage()
        if not sel or stage is None:
            return
        prim = stage.GetPrimAtPath(sel[0])
        cd = prim.GetCustomData() if prim and prim.IsValid() else {}
        self._title.text = cd.get("name", prim.GetName() if prim else "?")
        for k in _KEYS:
            v = cd.get(k, "")
            self._labels[k].text = str(v) if v != "" else "-"

    def destroy(self):
        self._sub = None
        self._win = None


# instantiate (keep a global ref so the subscription stays alive)
inspector = TosmInspector()
