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
    "description": "Report whether the candidate type matches the object, and correct it if wrong.",
    "input_schema": {
        "type": "object",
        "properties": {
            "matches": {"type": "boolean",
                        "description": "True if the candidate type already matches the object."},
            "corrected_type": {"type": "string",
                               "description": "The correct object class (= candidate if it was right)."},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                           "description": "Confidence in corrected_type, 0..1."},
            "reason": {"type": "string",
                       "description": "<=1 sentence: the visual feature that decided it."},
        },
        "required": ["matches", "corrected_type", "confidence", "reason"],
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
    "shown rendered multi-view images of ONE segmented 3D object (a point cloud with "
    "a red 3D bounding box) plus a candidate JSON record from an automatic 3D "
    "classifier.\n\n"
    "Your ONLY job: judge whether the candidate `type` matches what is visibly the "
    "object, and correct it if wrong.\n"
    "RULES:\n"
    "- The geometric fields (poseX/Y, poseTheta, dimensions) are physically measured. "
    "Never change or question them; use them only as size context.\n"
    "- The classifier is known to confuse chairs with 'stair'/'machine'. Look for "
    "seat+backrest+legs before trusting the candidate.\n"
    "- If the object is ambiguous or only partially scanned, keep the candidate type "
    "but lower the confidence.\n"
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
        candidate = {k: obj[k] for k in ("type", "name") if k in obj}
        candidate["dimensions"] = obj.get("dimensions")
        vocab = (f"`corrected_type` MUST be exactly one of: {allowed_types} "
                 "(or 'unknown' if none clearly fit)."
                 if allowed_types else
                 "Use a concise common-noun class for `corrected_type`.")
        text = (f"Candidate record (geometry is measured, do not change):\n"
                f"{json.dumps(candidate, ensure_ascii=False)}\n\n"
                f"{vocab}\n"
                f"The {len(image_paths)} images are the SAME object from "
                f"azimuths around it. Does `type` match? Correct if needed.")
        content = [_img_block(p) for p in image_paths] + [{"type": "text", "text": text}]
        return content

    def verify_symbolic(self, obj, image_paths, allowed_types=None):
        content = self.build_symbolic_request(obj, image_paths, allowed_types)
        return self._call(_SYS_SYMBOLIC, content, SYMBOLIC_TOOL)

    def infer_implicit(self, obj_type, image_paths):
        text = (f"Confirmed type: {obj_type}.\n"
                f"The {len(image_paths)} images are the SAME object from several "
                f"azimuths. Infer the implicit properties.")
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
