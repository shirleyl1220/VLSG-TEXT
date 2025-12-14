import os
import json
import numpy as np
import trimesh

import torch
from torch.utils.data import Dataset


class TextNodeSceneDataset(Dataset):
    """
    Dataset that aligns:
    - text node labels (for CLIP/SigLIP encoder)
    - scene graph geometry (for MultiModalSingleScanEncoder)
    
    Uses:
        all_descriptions.json   (your metadata)
        <scene_id>/labels.instances.annotated.v2.ply  (real geometry)
    """

    def __init__(
        self,
        json_path,
        scene_root,
        points_per_object=300,
        split="train"
    ):
        """
        Args:
            json_path: path to all_descriptions.json
            scene_root: root folder of 3RScan scenes
            points_per_object: number of points sampled per object
        """
        super().__init__()

        self.scene_root = scene_root
        self.points_per_object = points_per_object

        # Load frames metadata
        with open(json_path, "r") as f:
            self.frames = json.load(f)

        # simple 80/20 split
        if split == "train":
            self.frames = self.frames[: int(0.8 * len(self.frames))]
        else:
            self.frames = self.frames[int(0.8 * len(self.frames)) :]

        # SceneGraphLoc embedding dims
        self.attr_dim = 128
        self.rel_dim = 128

        # cache of loaded scene point clouds
        self.scene_cache = {}   # scene_id → {instance_id: Nx3 array}

    def __len__(self):
        return len(self.frames)

    # ----------------------------------------------------------------------
    # Load real geometry from PLY (cached per scene)
    # ----------------------------------------------------------------------
    def load_scene_points(self, scene_id):
        """
        Loads PLY and groups points by instance id.
        Cache per scene for performance.
        """
        if scene_id in self.scene_cache:
            return self.scene_cache[scene_id]

        ply_path = os.path.join(
            self.scene_root,
            scene_id,
            "labels.instances.annotated.v2.ply"
        )

        mesh = trimesh.load(ply_path, process=False)

        # Extract per-vertex arrays
        vertex = mesh.metadata["ply_raw"]["vertex"]["data"]

        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)  # Nx3
        inst = vertex["objectId"].astype(int)                            # Nx1

        # Group points by instance ID
        instance_points = {}
        for oid in np.unique(inst):
            pts = xyz[inst == oid]
            instance_points[oid] = pts

        self.scene_cache[scene_id] = instance_points
        return instance_points

    # ----------------------------------------------------------------------
    # Main __getitem__
    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        frame = self.frames[idx]

        scene_id = frame["scene_index"]
        visible_objects = frame["visible_objects"]     # dict
        relations = frame["spatial_relations"]

        # ---------------------------------------------------------------
        # 1. TEXT NODE LABELS
        # ---------------------------------------------------------------
        text_node_labels = []
        object_ids = []   # instance IDs as integers

        for obj_id_str, obj in visible_objects.items():
            label = obj["label"]
            if label.strip() == "":
                continue
            oid = int(obj_id_str)
            object_ids.append(oid)
            text_node_labels.append(label)

        num_objs = len(object_ids)
        if num_objs == 0:
            text_node_labels = ["unknown"]
            object_ids = [0]
            num_objs = 1

        # ---------------------------------------------------------------
        # 2. SCENE GEOMETRY FROM PLY
        # ---------------------------------------------------------------
        inst_points = self.load_scene_points(scene_id)

        object_pointclouds = []
        for oid in object_ids:
            if oid in inst_points:
                pts = inst_points[oid]
            else:
                # fallback: dummy centroid
                pts = np.array([[0, 0, 0]])

            # downsample / upsample to fixed number of points
            if pts.shape[0] >= self.points_per_object:
                idxs = np.random.choice(pts.shape[0], self.points_per_object, replace=False)
            else:
                idxs = np.random.choice(pts.shape[0], self.points_per_object, replace=True)
            pts_fixed = pts[idxs]  # [P,3]

            object_pointclouds.append(pts_fixed)

        # stack into shape [3, total_points]
        tot_obj_pts = np.stack(object_pointclouds, axis=0)       # [N_obj, P, 3]
        tot_obj_pts = tot_obj_pts.transpose(0, 2, 1)             # [N_obj, 3, P]

        # ---------------------------------------------------------------
        # 3. BOW ATTRIBUTE + RELATION FEATURES (simple version)
        # ---------------------------------------------------------------
        bow_attr = np.zeros((num_objs, self.attr_dim), dtype=np.float32)
        bow_rel = np.zeros((num_objs, self.rel_dim), dtype=np.float32)

        # Build mapping from relation string to index
        rel_vocab = {}
        next_rel = 0

        # Build edges
        edges_list = []
        object_labels = text_node_labels  # used to index by label string

        for rel in relations:
            subj = rel["subject"]
            obj = rel["object"]
            r = rel["relation"]

            # find object indices by label
            try:
                si = object_labels.index(subj)
                oi = object_labels.index(obj)
            except:
                continue

            edges_list.append([si, oi])

            # relation one-hot
            if r not in rel_vocab:
                rel_vocab[r] = next_rel % self.rel_dim
                next_rel += 1

            bow_rel[si, rel_vocab[r]] = 1.0

        if len(edges_list) == 0:
            edges_list = [[0, 0]]

        edges = np.array(edges_list, dtype=np.int64)

        # ---------------------------------------------------------------
        # 4. RELATIVE POSE FEATURES (use centroids from JSON)
        # ---------------------------------------------------------------
        rel_pose_list = []
        for oid_str in visible_objects:
            xyz = visible_objects[oid_str]["centroid_world"]
            rel_pose_list.append([xyz[0], xyz[1], xyz[2]])

        tot_rel_pose = np.array(rel_pose_list, dtype=np.float32)  # [N,3]

        # ---------------------------------------------------------------
        # 5. Pack into data_dict expected by MultiModalSingleScanEncoder
        # ---------------------------------------------------------------
        data_dict = {
            # TEXT
            "text_node_labels": [text_node_labels],

            # GEOMETRY
            "tot_obj_pts": torch.tensor(tot_obj_pts).float(),     # [N_obj, 3, P]
            "tot_bow_vec_object_attr_feats": torch.tensor(bow_attr).float(),
            "tot_bow_vec_object_edge_feats": torch.tensor(bow_rel).float(),
            "tot_rel_pose": torch.tensor(tot_rel_pose).float(),

            # GRAPH STRUCTURE
            "graph_per_obj_count": torch.tensor([[num_objs]]).int(),
            "graph_per_edge_count": torch.tensor([[edges.shape[0]]]).int(),
            "edges": torch.tensor(edges).long(),

            "batch_size": 1
        }

        return data_dict