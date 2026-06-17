"""Export a unified TOSM graph to a USD (.usda) scene for Isaac Sim.

Sketch / first cut. Writes a text-USD (.usda) stage — no pxr dependency, so it
runs anywhere; Isaac Sim / usdview opens the .usda directly. Mapping:

  place   -> Xform + thin floor slab over its bbox  (/World/Places/<id>)
  object  -> Xform + Cube scaled to its OBB, posed at (poseX,poseY,poseZ),
             yaw = poseTheta, displayColor from the object color
             (/World/Objects/<name>)
  TOSM attributes -> custom prim attributes (tosm:type, tosm:isKeyObject,
             tosm:isMovable, tosm:confidence, ...)
  relations -> USD relationships on the subject prim, so the scene graph is
             queryable in Isaac Sim:
               object  rel tosm:isInsideOf  = </World/Places/place_1>
               object  rel tosm:isOn        = [ </World/Objects/...> ]
               object  rel tosm:isNextTo    = [ ... ]
               place   rel tosm:isConnectedTo = [ </World/Places/...> ]

Units: metersPerUnit = 1, upAxis = Z (Isaac Sim convention). poseZ is the OBB
centre height (floor-referenced), so the Cube sits at the right elevation.
"""
import re

_COLORS = {
    "black": (0.05, 0.05, 0.05), "white": (0.92, 0.92, 0.92),
    "gray": (0.5, 0.5, 0.5), "grey": (0.5, 0.5, 0.5),
    "brown": (0.45, 0.30, 0.18), "red": (0.7, 0.1, 0.1),
    "green": (0.1, 0.6, 0.2), "blue": (0.15, 0.35, 0.7),
    "yellow": (0.85, 0.75, 0.1), "silver": (0.75, 0.78, 0.8),
}
_DEFAULT_COLOR = (0.6, 0.6, 0.62)


def _safe(name):
    """Valid USD prim name: alnum/underscore, not leading with a digit."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return "_" + s if s and s[0].isdigit() else (s or "unnamed")


def _color(name):
    return _COLORS.get(str(name).lower(), _DEFAULT_COLOR)


def _targets(paths):
    return "[ " + ", ".join(paths) + " ]"


def to_usda(graph, stage_name="TOSM"):
    objects = graph["objects"]
    places = graph.get("places", [])
    obj_path = {o["name"]: f"/World/Objects/{_safe(o['name'])}" for o in objects}
    place_path = {p["id"]: f"/World/Places/{_safe(p['id'])}" for p in places}

    # group relations by subject + predicate
    by_subj = {}
    for e in graph.get("relations", []):
        by_subj.setdefault((e["subject"], e["predicate"]), []).append(e["object"])

    def rels_for(subj):
        out = []
        for (s, pred), objs in by_subj.items():
            if s != subj:
                continue
            tgt = [obj_path.get(o) or place_path.get(o) for o in objs]
            tgt = [t for t in tgt if t]
            if tgt:
                out.append(f'            rel tosm:{pred} = {_targets(tgt)}')
        return "\n".join(out)

    L = []
    L.append('#usda 1.0')
    L.append('(')
    L.append(f'    defaultPrim = "World"')
    L.append('    metersPerUnit = 1')
    L.append('    upAxis = "Z"')
    L.append(f'    customLayerData = {{ string generator = "blk360seg.usd_export" }}')
    L.append(')')
    L.append('')
    L.append('def Xform "World"')
    L.append('{')

    # ---- places ----
    L.append('    def Scope "Places"')
    L.append('    {')
    for p in places:
        bb = p.get("bbox")
        cx, cy = p["centroid"]
        conn = rels_for(p["id"])
        L.append(f'        def Xform "{_safe(p["id"])}" (kind = "group")')
        L.append('        {')
        L.append(f'            custom string tosm:id = "{p["id"]}"')
        L.append(f'            custom string tosm:name = "{p.get("name","")}"')
        L.append(f'            custom string tosm:type = "{p.get("type","room")}"')
        L.append(f'            custom double tosm:area_m2 = {p.get("area_m2",0.0)}')
        if conn:
            L.append(conn)
        if bb:
            x0, y0, x1, y1 = bb
            sx, sy = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
            L.append(f'            double3 xformOp:translate = ({cx}, {cy}, 0.0)')
            L.append('            uniform token[] xformOpOrder = ["xformOp:translate"]')
            L.append('            def Mesh "floor"')
            L.append('            {')
            L.append('                int[] faceVertexCounts = [4]')
            L.append('                int[] faceVertexIndices = [0, 1, 2, 3]')
            hx, hy = sx / 2.0, sy / 2.0
            L.append(f'                point3f[] points = [({-hx},{-hy},0), '
                     f'({hx},{-hy},0), ({hx},{hy},0), ({-hx},{hy},0)]')
            L.append('                color3f[] primvars:displayColor = [(0.85, 0.82, 0.7)]')
            L.append('                float[] primvars:displayOpacity = [0.25]')
            L.append('            }')
        L.append('        }')
    L.append('    }')

    # ---- objects ----
    L.append('    def Scope "Objects"')
    L.append('    {')
    for o in objects:
        pr = o.get("properties", {})
        d = o["dimensions"]
        z = pr.get("poseZ", 0.0)
        th = o.get("poseTheta", 0.0) * 180.0 / 3.141592653589793
        r, g, b = _color(o.get("color"))
        rels = rels_for(o["name"])
        L.append(f'        def Xform "{_safe(o["name"])}" (kind = "component")')
        L.append('        {')
        L.append(f'            custom string tosm:type = "{o.get("type","")}"')
        L.append(f'            custom string tosm:id = "{o.get("id","")}"')
        L.append(f'            custom double tosm:confidence = {o.get("confidence",0.0)}')
        L.append(f'            custom bool tosm:isKeyObject = '
                 f'{"true" if pr.get("isKeyObject", False) else "false"}')
        L.append(f'            custom bool tosm:isMovable = '
                 f'{"true" if pr.get("isMovable", False) else "false"}')
        L.append(f'            custom string tosm:color = "{o.get("color","")}"')
        if rels:
            L.append(rels)
        L.append(f'            double3 xformOp:translate = '
                 f'({o["poseX"]}, {o["poseY"]}, {z})')
        L.append(f'            double xformOp:rotateZ = {round(th, 3)}')
        L.append(f'            double3 xformOp:scale = '
                 f'({d["length"]}, {d["width"]}, {d["height"]})')
        L.append('            uniform token[] xformOpOrder = '
                 '["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]')
        L.append('            def Cube "geom"')
        L.append('            {')
        L.append('                double size = 1')
        L.append(f'                color3f[] primvars:displayColor = '
                 f'[({r}, {g}, {b})]')
        L.append('            }')
        L.append('        }')
    L.append('    }')
    L.append('}')
    return "\n".join(L) + "\n"
