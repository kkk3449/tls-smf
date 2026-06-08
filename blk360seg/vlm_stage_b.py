"""Stage B — VLM refinement of the semanticObject message (Claude Sonnet 4.6).

Two forced-tool-use calls per object on its 4-view render:
  Prompt 1 (symbolic):  verify/correct `type` (fixes Uni3D errors, e.g.
                        chair mislabeled 'stair'); geometry is read-only.
  Prompt 2 (implicit):  infer TOSM implicit model — isKeyObject, isMovable,
                        and (doors only) isOpen / canBeOpen.

Forced tool-use guarantees schema-valid JSON (no parsing/regex). The explicit
model (pose/size) is shown to the VLM as context but never modified here.
"""
import base64
import json
import os
import re

SYMBOLIC_TOOL = {
    "name": "report_symbolic",
    "description": "Choose the object's final type: a preferred candidate if one fits better, else keep the original Uni3D label.",
    "input_schema": {
        "type": "object",
        "properties": {
            "corrected_type": {"type": "string",
                               "description": "Final type: a candidate if one matches the object better, "
                                              "otherwise the original uni3d_type unchanged."},
            "changed": {"type": "boolean",
                        "description": "True if corrected_type differs from the original uni3d_type."},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                           "description": "Confidence in corrected_type, 0..1."},
            "reason": {"type": "string",
                       "description": "<=1 sentence: why this type (and why switched/kept)."},
        },
        "required": ["corrected_type", "changed", "confidence", "reason"],
    },
}

IMPLICIT_TOOL = {
    "name": "report_implicit",
    "description": "Report the TOSM implicit/affordance properties of the object.",
    "input_schema": {
        "type": "object",
        "properties": {
            "isKeyObject": {"type": "boolean",
                            "description": "Fixed, distinctive landmark useful for localization/place recognition."},
            "isMovable": {"type": "boolean",
                          "description": "Could be physically displaced in normal operation."},
            "isOpen": {"type": ["boolean", "null"],
                       "description": "Door only: currently open? null if not a door."},
            "canBeOpen": {"type": ["boolean", "null"],
                          "description": "Door only: can it be opened if closed? null if not a door."},
            "reason": {"type": "string", "description": "<=1 sentence."},
        },
        "required": ["isKeyObject", "isMovable", "isOpen", "canBeOpen", "reason"],
    },
}

_SYS_SYMBOLIC = (
    "You are a semantic-mapping verifier for an indoor mobile robot (AGV). You are "
    "shown rendered image(s) of ONE segmented 3D object (a point cloud with a red 3D "
    "bounding box), the original label from an automatic 3D classifier "
    "(`uni3d_type`), and an OPTIONAL list of user-preferred candidate classes.\n\n"
    "DECISION RULE (follow exactly):\n"
    "1. If one of the candidate classes matches the object in the image BETTER than "
    "`uni3d_type`, set corrected_type to that candidate (changed=true).\n"
    "2. Otherwise KEEP `uni3d_type` unchanged (changed=false).\n"
    "When candidates are provided, corrected_type MUST be either one of the "
    "candidates or exactly `uni3d_type` — never 'unknown' and never any other "
    "label. (If NO candidates are provided, instead correct `uni3d_type` to the most "
    "accurate common-noun label you can, open-vocabulary.)\n"
    "- The classifier is known to confuse chairs with 'stair'/'machine'; look for "
    "seat+backrest+legs before deciding.\n"
    "- SIZE SANITY CHECK: the measured `dimensions` (metres, length x width x "
    "height) are a HARD constraint. Do NOT switch to a candidate whose typical "
    "real-world size is incompatible with them (e.g. a single chair is roughly "
    "0.4-0.9 m wide and under ~1.3 m tall; a 2 m+ wide object is not one chair, it "
    "is likely several merged objects or a different class). If every candidate's "
    "real size contradicts the dimensions, keep `uni3d_type`.\n"
    "- Respond ONLY via the report_symbolic tool."
)

_SYS_IMPLICIT = (
    "You infer implicit robot-task (affordance) properties of ONE indoor object from "
    "its multi-view images and confirmed type, for AGV planning.\n"
    "DEFINITIONS (TOSM):\n"
    "- isKeyObject: a crucial, fixed, visually distinctive landmark useful for "
    "localization / place recognition (door, pillar, large fixed machine). Small or "
    "movable clutter is NOT a key object.\n"
    "- isMovable: could be physically displaced during normal operation (chair, box, "
    "pallet = true; wall cabinet, pipe, beam, mounted panel = false).\n"
    "- isOpen (doors only): is the door currently open?\n"
    "- canBeOpen (doors only): if closed, can it be opened?\n"
    "Set isOpen and canBeOpen to null unless the type is a door.\n"
    "Respond ONLY via the report_implicit tool."
)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_") or "object"


def _img_block(path):
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data}}


class SemanticVLM:
    def __init__(self, model="claude-sonnet-4-6", api_key=None, max_tokens=512):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def _call(self, system, content, tool):
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": content}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise RuntimeError("no tool_use in response")

    def build_symbolic_request(self, obj, image_paths, allowed_types=None):
        ctx = {"uni3d_type": obj.get("type"), "dimensions": obj.get("dimensions")}
        if allowed_types:
            vocab = (f"Preferred candidate classes: {allowed_types}\n"
                     f"If one of these fits the object better than uni3d_type "
                     f"('{obj.get('type')}'), switch to it; otherwise keep "
                     f"'{obj.get('type')}'.")
        else:
            vocab = ("No candidates given — correct uni3d_type to the most accurate "
                     "label if it is wrong, else keep it.")
        n = len(image_paths)
        imgdesc = ("The image shows the object." if n == 1 else
                   f"The {n} images are the SAME object from azimuths around it.")
        text = (f"Object context (geometry is measured, do not change):\n"
                f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
                f"{vocab}\n"
                f"{imgdesc} Decide corrected_type per the rule.")
        content = [_img_block(p) for p in image_paths] + [{"type": "text", "text": text}]
        return content

    def verify_symbolic(self, obj, image_paths, allowed_types=None):
        content = self.build_symbolic_request(obj, image_paths, allowed_types)
        return self._call(_SYS_SYMBOLIC, content, SYMBOLIC_TOOL)

    def infer_implicit(self, obj_type, image_paths):
        n = len(image_paths)
        imgdesc = ("The image shows the object." if n == 1 else
                   f"The {n} images are the SAME object from several azimuths.")
        text = (f"Confirmed type: {obj_type}.\n"
                f"{imgdesc} Infer the implicit properties.")
        content = [_img_block(p) for p in image_paths] + [{"type": "text", "text": text}]
        return self._call(_SYS_IMPLICIT, content, IMPLICIT_TOOL)


def apply_symbolic(obj, result):
    """Update type/name/confidence/symbolicVerified from a report_symbolic result."""
    obj["type"] = result["corrected_type"]
    obj["name"] = f"{slug(result['corrected_type'])}_{int(obj['id']):03d}"
    obj["confidence"] = round(float(result["confidence"]), 3)
    obj.setdefault("properties", {})["symbolicVerified"] = True
    obj["properties"]["symbolicReason"] = result.get("reason", "")
    return obj


def apply_implicit(obj, result):
    p = obj.setdefault("properties", {})
    p["isKeyObject"] = bool(result["isKeyObject"])
    p["isMovable"] = bool(result["isMovable"])
    if result.get("isOpen") is not None:
        p["isOpen"] = bool(result["isOpen"])
    if result.get("canBeOpen") is not None:
        p["canBeOpen"] = bool(result["canBeOpen"])
    p["implicitReason"] = result.get("reason", "")
    return obj
