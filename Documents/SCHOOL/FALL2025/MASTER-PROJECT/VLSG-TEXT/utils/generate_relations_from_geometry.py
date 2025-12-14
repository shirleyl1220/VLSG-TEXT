import json
import numpy as np
import os

# --------------------------
# Relation thresholds
# --------------------------
DIR_THRESHOLD = 0.20   # meters required to say left/right/front/back
HEIGHT_THRESHOLD = 0.15
NEAR_THRESHOLD = 0.80   # objects within this distance = "near"


def load_semseg(path):
    with open(path, "r") as f:
        data = json.load(f)
    id_to_label = {str(obj["objectId"]): obj["label"].lower() for obj in data["segGroups"]}
    return id_to_label


def load_scene_graph(path):
    with open(path, "r") as f:
        return json.load(f)


def get_direction_relation(ci, cj):
    """Return list of directional relations between i→j."""
    dx, dy, dz = cj - ci
    rels = []

    # Horizontal (left/right)
    if abs(dx) > DIR_THRESHOLD:
        if dx > 0:
            rels.append("right_of")
        else:
            rels.append("left_of")

    # Depth (front/back)
    if abs(dz) > DIR_THRESHOLD:
        if dz > 0:
            rels.append("in_front_of")
        else:
            rels.append("behind")

    # Vertical (above/below)
    if abs(dy) > HEIGHT_THRESHOLD:
        if dy > 0:
            rels.append("above")
        else:
            rels.append("below")

    return rels


def get_distance_relation(ci, cj, ri, rj):
    dist = np.linalg.norm(cj - ci)
    touch_dist = ri + rj

    if dist <= touch_dist + 0.05:
        return ["touching"]

    if dist < NEAR_THRESHOLD:
        return ["near"]

    return []


def make_symmetric(subject, object, relations):
    """Expand relations into symmetric versions when meaningful."""
    sym = []

    for r in relations:
        if r == "left_of":
            sym.append((object, subject, "right_of"))
        elif r == "right_of":
            sym.append((object, subject, "left_of"))
        elif r == "in_front_of":
            sym.append((object, subject, "behind"))
        elif r == "behind":
            sym.append((object, subject, "in_front_of"))
        elif r == "above":
            sym.append((object, subject, "below"))
        elif r == "below":
            sym.append((object, subject, "above"))
        elif r in ["touching", "near"]:
            sym.append((object, subject, r))  # symmetric
        # others can be added

    return sym


def generate_relations(scene_json_path, semseg_path, output_path):
    scene = load_scene_graph(scene_json_path)
    id_to_label = load_semseg(semseg_path)

    nodes = scene["nodes"]

    # Build object info
    objects = {}
    for obj_id, nd in nodes.items():
        centroid = np.array(nd["centroid"], dtype=float)
        radius = float(nd.get("radius", 0.1))
        label = id_to_label.get(obj_id, f"obj_{obj_id}")
        objects[obj_id] = {
            "centroid": centroid,
            "radius": radius,
            "label": label
        }

    # Build relations
    edges = []

    obj_ids = list(objects.keys())

    for i in range(len(obj_ids)):
        for j in range(len(obj_ids)):
            if i == j:
                continue

            oi = obj_ids[i]
            oj = obj_ids[j]

            ci = objects[oi]["centroid"]
            cj = objects[oj]["centroid"]
            ri = objects[oi]["radius"]
            rj = objects[oj]["radius"]

            base_relations = []

            # Directional
            base_relations.extend(get_direction_relation(ci, cj))

            # Distance / touching
            base_relations.extend(get_distance_relation(ci, cj, ri, rj))

            # Add all relations as edges
            for r in base_relations:
                edges.append({
                    "subject": oi,
                    "object": oj,
                    "relation": r
                })

            # Add symmetric expansions
            for (sub, obj, r2) in make_symmetric(objects[oi]["label"], objects[oj]["label"], base_relations):
                # sub and obj are labels; convert back to ids
                edges.append({
                    "subject": oi if sub == objects[oi]["label"] else oj,
                    "object": oi if obj == objects[oi]["label"] else oj,
                    "relation": r2
                })

    # Save result
    scene_out = dict(scene)
    scene_out["edges_text"] = edges

    with open(output_path, "w") as f:
        json.dump(scene_out, f, indent=2)

    print(f"[OK] Relations written to {output_path}")
    print(f"Generated {len(edges)} relations.")


# --------------------------
# Main Usage
# --------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Path to scene_graph.json")
    ap.add_argument("--semseg", required=True, help="Path to semseg.v2.json")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    generate_relations(args.scene, args.semseg, args.out)