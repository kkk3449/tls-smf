#!/usr/bin/env python3
"""One-click E57 -> 3D semantic model orchestrator (driven by the console).

  .venv/bin/python scripts/semantic_pipeline.py --job job.json

job.json:
  {"name": "site_2026-09-17", "e57": "/path/scan.e57", "mode": "new"|"epoch",
   "map_id": "testroom", "vlm": true, "budget_usd": 5.0, "overhead": true,
   "kg_graph": "outputs/testroom_epochs_kg.json"   (epoch mode),
   "kg_map_yaml": "/home/caselab/ammr_twin/map_vis_n2_1.yaml" (epoch mode),
   "bounds": "outputs/vis_n2_room_bounds.json" (epoch mode),
   "ref_clean": "outputs/vis_n2_det_filt/clean.ply" (epoch mode)}

Writes <outdir>/pipeline_status.json continuously (steps, log tail, cost) so
the console can poll it. Every stage is an existing script run as a
subprocess with the pipeline's own venv; the only new logic is sequencing,
budget guarding, and file plumbing.
"""
import argparse, datetime, json, os, shlex, subprocess, sys, time, glob

SEG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = "/home/caselab/blk360_ros2_ws"
PY = os.path.join(SEG, ".venv", "bin", "python")
TWIN = os.path.expanduser("~/ammr_twin")
OVERHEAD_VOCAB = ("light fixture,fluorescent lamp,air conditioner,ventilation duct,cable tray,"
                  "beam,pipe,sprinkler head,projector,speaker,sign,surveillance camera,"
                  "smoke detector,ceiling panel,curtain rail")
CALL_USD = 0.0097          # measured mean per Stage-B call (claude-sonnet-4-6)


class Status:
    def __init__(self, path, job):
        self.path, self.job = path, job
        self.d = {"state": "running", "job": job, "started": now(), "steps": [],
                  "log": [], "cost_usd": 0.0, "outputs": {}, "notes": []}
        self.flush()

    def flush(self):
        tmp = self.path + ".tmp"
        json.dump(self.d, open(tmp, "w"), indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    def log(self, line):
        line = line.rstrip()
        if not line:
            return
        print(line, flush=True)
        self.d["log"].append(f"{datetime.datetime.now():%H:%M:%S} {line}")
        self.d["log"] = self.d["log"][-120:]
        self.flush()

    def step(self, sid, label):
        s = {"id": sid, "label": label, "state": "running", "started": now(), "ended": None, "note": ""}
        self.d["steps"].append(s); self.flush()
        return s

    def end(self, s, state="done", note=""):
        s["state"], s["ended"], s["note"] = state, now(), note
        self.flush()


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def run(st, cmd, cwd=SEG, env=None):
    st.log("$ " + " ".join(shlex.quote(c) for c in cmd))
    e = dict(os.environ); e["PYTHONUNBUFFERED"] = "1"
    if env:
        e.update(env)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)
    for line in p.stdout:
        st.log(line)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {cmd[1] if len(cmd) > 1 else cmd[0]}")


def count_objs(path):
    d = json.load(open(path)); return len(d["semanticObjects"] if isinstance(d, dict) else d)


def usage_cost(path):
    try:
        return float(json.load(open(path))["cost_usd"])
    except (OSError, ValueError, KeyError):
        return 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--job", required=True)
    a = ap.parse_args()
    job = json.load(open(a.job))
    name = job["name"]; e57 = job["e57"]; mode = job.get("mode", "new")
    map_id = job.get("map_id", name); vlm = bool(job.get("vlm", True))
    budget = float(job.get("budget_usd", 5.0)); overhead = bool(job.get("overhead", True))
    D = os.path.join(SEG, "outputs", f"{name}_objects"); os.makedirs(D, exist_ok=True)
    VIEWS = os.path.join(SEG, "outputs", f"{name}_views")
    st = Status(os.path.join(D, "pipeline_status.json"), job)
    st.d["outputs"]["objects_dir"] = D
    epoch = mode == "epoch"
    graph = job.get("kg_graph") if epoch else os.path.join(SEG, "outputs", f"{map_id}_kg.json")
    graph = os.path.join(SEG, graph) if graph and not os.path.isabs(graph) else graph
    ts = now()
    try:
        # 1 Stage A ------------------------------------------------------------
        s = st.step("extract", "Stage A: structure removal + object decomposition")
        run(st, [PY, "scripts/extract_objects.py", "--input", e57, "--out", D, "--save-clean"])
        st.end(s, note=f"{len(glob.glob(D + '/obj_*.ply'))} objects")
        s = st.step("classify", "Stage A: Uni3D provisional labels (industrial vocabulary)")
        run(st, [PY, "scripts/classify_objects.py", "--objects-dir", D, "--classes", "industrial"])
        st.end(s)
        s = st.step("build", "Stage A: TOSM records (VDA 5050 message)")
        run(st, [PY, "scripts/build_semantic_objects.py", "--objects-dir", D,
                 "--classification", os.path.join(D, "classification.csv"), "--map-id", map_id])
        stageA = os.path.join(D, "semanticObjects.json")
        st.end(s, note=f"{count_objs(stageA)} records")
        # 2 map + bounds (new site) / registration (epoch) --------------------
        if epoch:
            s = st.step("register", "Register scan into the existing KG frame (FPFH+RANSAC, ICP)")
            Tp = os.path.join(D, "to_kg_T.npy")
            run(st, [PY, "scripts/register_scan.py", "--src", os.path.join(D, "clean.ply"),
                     "--dst", os.path.join(SEG, job["ref_clean"]), "--out", Tp])
            reg = json.load(open(Tp.replace(".npy", ".json")))
            st.end(s, note=f"ICP fitness {reg['icp_fitness']:.2f}, rmse {reg['icp_rmse_m']*100:.1f} cm, yaw {reg['yaw_deg']:.1f} deg")
            map_yaml = job["kg_map_yaml"]; bounds = os.path.join(SEG, job["bounds"])
        else:
            s = st.step("occ", "Occupancy grid for navigation / place layer (build_occ_from_e57)")
            run(st, [PY, os.path.join(WS, "scripts", "build_occ_from_e57.py"), e57, "--outdir", TWIN], cwd=WS)
            base = os.path.splitext(os.path.basename(e57))[0]
            map_yaml = os.path.join(TWIN, f"map_{base}.yaml")
            if not os.path.exists(map_yaml):     # build_occ names by stem; fall back to newest
                cands = sorted(glob.glob(os.path.join(TWIN, "map_*.yaml")), key=os.path.getmtime)
                map_yaml = cands[-1]
            st.end(s, note=os.path.basename(map_yaml))
            s = st.step("bounds", "Room bounds from the map's largest free region")
            bounds = os.path.join(SEG, "outputs", f"{map_id}_room_bounds.json")
            run(st, [PY, "scripts/make_room_bounds.py", "--map", map_yaml, "--out", bounds])
            st.end(s)
            Tp = None
        st.d["outputs"]["map_yaml"] = map_yaml; st.d["outputs"]["bounds"] = bounds
        # room scoping in the KG frame
        s = st.step("scope", "Room scoping (KG frame)")
        cmd = [PY, "scripts/roomscope_any.py", "--src", stageA, "--bounds", bounds, "--tol", "0.0"]
        if Tp: cmd += ["--T", Tp]
        run(st, cmd)
        room_native_names = open(os.path.join(D, "pipeline_status.json")).name  # placeholder
        scoped = stageA[:-5] + ".kgframe.room.json"
        n_room = count_objs(scoped)
        st.end(s, note=f"{n_room} in-room objects")
        # 3 Stage B -------------------------------------------------------------
        s = st.step("render", "Render 4 base + 8 escalation views per object")
        run(st, [PY, "scripts/render_object_views.py", "--semantic", stageA, "--objects-dir", D, "--out", VIEWS, "--views", "8"])
        run(st, [PY, "scripts/render_object_views.py", "--semantic", stageA, "--objects-dir", D, "--out", VIEWS,
                 "--views", "4", "--zoom", "1.2", "--suffix", "_z"])
        st.end(s)
        sel = ",".join(o["name"] for o in json.load(open(scoped))["semanticObjects"])
        est = n_room * 13 * CALL_USD
        lf = stageA[:-5] + ".lf_esc.json"
        if vlm and est <= budget:
            s = st.step("verify", f"Stage B: multi-view VLM verification + escalation (est. ${est:.2f})")
            run(st, [PY, "scripts/vlm_late_fusion.py", "--input", stageA, "--views-dir", VIEWS, "--escalate",
                     "--model", "claude-sonnet-4-6", "--workers", "3", "--select", sel, "--out", lf])
            c = usage_cost(lf[:-5] + ".usage.json"); st.d["cost_usd"] += c
            st.end(s, note=f"${c:.2f}")
            verified = lf
        else:
            s = st.step("verify", "Stage B skipped" + ("" if vlm else " (VLM disabled)") +
                        (f" (estimate ${est:.2f} > budget ${budget:.2f})" if vlm else ""))
            st.end(s, state="skipped"); verified = stageA
            st.d["notes"].append("records ingested with provisional labels only (unverified)")
        # transform verified records to the KG frame + room
        cmd = [PY, "scripts/roomscope_any.py", "--src", verified, "--bounds", bounds, "--tol", "0.0"]
        if Tp: cmd += ["--T", Tp]
        run(st, cmd)
        kg_in = verified[:-5] + ".kgframe.room.json"
        # 4 knowledge graph ------------------------------------------------------
        s = st.step("upsert", "Knowledge graph upsert (mint-once identities, relations)")
        if epoch and os.path.exists(graph):
            bak = graph[:-5] + f".bak_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
            import shutil; shutil.copy(graph, bak); st.log(f"backup -> {bak}")
        run(st, [PY, "scripts/kg_upsert.py", "--input", kg_in, "--graph", graph, "--map-id", map_id,
                 "--timestamp", ts, "--cypher", graph[:-5] + ".cypher"])
        st.d["outputs"]["kg"] = graph; st.end(s)
        # 5 place layer ------------------------------------------------------------
        s = st.step("places", "Place layer: SLIC regions + LLM ring-code names")
        places = os.path.join(SEG, "outputs", f"place_layer_{name}_slic.json")
        run(st, [PY, "scripts/place_slic_segment.py", "--map", map_yaml, "--bounds", bounds,
                 "--objects", kg_in, "--epoch", name, "--out", places])
        naming = os.path.join(SEG, "outputs", "place_ring_naming.json")
        run(st, [PY, "scripts/place_ring_naming.py", "--places", places, "--epoch", f"{name}_slic", "--out", naming])
        st.d["outputs"]["places"] = places; st.d["outputs"]["naming"] = naming; st.d["outputs"]["naming_key"] = f"{name}_slic"
        st.end(s)
        # 6 overhead pass ------------------------------------------------------------
        if overhead:
            s = st.step("overhead", "Overhead layer: ceiling-mounted structure")
            OD = os.path.join(SEG, "outputs", f"{name}_overhead"); OV = OD + "_views"
            run(st, [PY, "scripts/overhead_extract.py", "--ref", os.path.join(D, "ceilingref.ply"), "--raw", e57,
                     "--clean", os.path.join(D, "clean.ply"), "--conn-eps", "0.15", "--out", OD])
            oj = os.path.join(OD, "semanticObjects.json")
            if count_objs(oj) > 0:
                run(st, [PY, "scripts/classify_objects.py", "--objects-dir", OD, "--classes", OVERHEAD_VOCAB])
                # merge provisional labels
                import pandas as pd
                cls = {r["file"]: (r["top1"], float(r["score1"])) for _, r in pd.read_csv(os.path.join(OD, "classification.csv")).iterrows()}
                doc = json.load(open(oj))
                for o in doc["semanticObjects"]:
                    t, c = cls[o["properties"]["imageFile"]]; o["type"], o["confidence"] = t, round(c, 3)
                    o["name"] = f"{t.replace(' ', '_')}_{int(o['id']):03d}"; o["properties"]["layer"] = "overhead"
                json.dump(doc, open(oj, "w"), indent=1, ensure_ascii=False)
                cmd = [PY, "scripts/roomscope_any.py", "--src", oj, "--bounds", bounds, "--tol", "0.5"]
                if Tp: cmd += ["--T", Tp]
                run(st, cmd)
                oroom = oj[:-5] + ".kgframe.room.json"; n_o = count_objs(oroom)
                if n_o:
                    run(st, [PY, "scripts/render_object_views.py", "--semantic", oj, "--objects-dir", OD, "--out", OV, "--views", "8", "--elev", "-30"])
                    run(st, [PY, "scripts/render_object_views.py", "--semantic", oj, "--objects-dir", OD, "--out", OV, "--views", "4", "--elev", "-30", "--zoom", "1.2", "--suffix", "_z"])
                    osel = ",".join(o["name"] for o in json.load(open(oroom))["semanticObjects"])
                    est_o = n_o * 13 * CALL_USD
                    if vlm and st.d["cost_usd"] + est_o <= budget:
                        floor_z = json.load(open(oj)).get("overhead", {}).get("floorZ", 0.0)
                        olf = oj[:-5] + ".lf_esc.json"
                        run(st, [PY, "scripts/vlm_late_fusion.py", "--input", oj, "--views-dir", OV, "--escalate",
                                 "--model", "claude-sonnet-4-6", "--floor-z", str(floor_z), "--height-level",
                                 "--candidates", OVERHEAD_VOCAB, "--workers", "3", "--select", osel, "--out", olf])
                        st.d["cost_usd"] += usage_cost(olf[:-5] + ".usage.json")
                        cmd = [PY, "scripts/roomscope_any.py", "--src", olf, "--bounds", bounds, "--tol", "0.5"]
                        if Tp: cmd += ["--T", Tp]
                        run(st, cmd); oin = olf[:-5] + ".kgframe.room.json"
                    else:
                        oin = oroom; st.d["notes"].append("overhead records ingested unverified (VLM skipped)")
                    run(st, [PY, "scripts/kg_upsert.py", "--input", oin, "--graph", graph, "--map-id", map_id, "--timestamp", ts])
                st.end(s, note=f"{n_o} in-room overhead objects")
            else:
                st.end(s, note="no overhead candidates (flat ceiling)")
        st.d["state"] = "done"; st.d["ended"] = now(); st.flush()
        st.log(f"DONE in {(datetime.datetime.fromisoformat(st.d['ended']) - datetime.datetime.fromisoformat(st.d['started'])).seconds} s, cost ${st.d['cost_usd']:.2f}")
    except Exception as e:  # noqa: BLE001
        for s in st.d["steps"]:
            if s["state"] == "running":
                st.end(s, "failed", str(e))
        st.d["state"] = "failed"; st.d["error"] = str(e); st.d["ended"] = now(); st.flush()
        st.log(f"FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
