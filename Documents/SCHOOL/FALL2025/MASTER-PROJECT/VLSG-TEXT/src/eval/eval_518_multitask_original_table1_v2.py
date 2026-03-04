"""
Fast evaluation using precomputed embeddings.
use this! you can run either table 1 or 2.
Usage:
  # Top-k out of 10 (Table 1 protocol)
  python eval_cached.py --checkpoint /path/to/ckpt.pth --mode top10

  # Top-k out of all scenes (Table 2 protocol)
  python eval_cached.py --checkpoint /path/to/ckpt.pth --mode full
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
from tqdm import tqdm
import sys
from pathlib import Path

# ── Args ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',      type=str,   required=True)
parser.add_argument('--cache_dir',       type=str,   default='/content/drive/MyDrive/VLSG_Files')
parser.add_argument('--mode',            type=str,   default='top10',
                    choices=['top10', 'full'],
                    help='top10: rank against 10 candidates (Table 1), '
                         'full: rank against all 55 test scenes (Table 2)')
parser.add_argument('--w_emb',           type=float, default=0.33)
parser.add_argument('--w_scene',         type=float, default=0.33)
parser.add_argument('--w_jac',           type=float, default=0.34)
parser.add_argument('--eval_iters',      type=int,   default=10)
parser.add_argument('--eval_iter_count', type=int,   default=100)
parser.add_argument('--out_of',          type=int,   default=10,
                    help='Only used in top10 mode')
parser.add_argument('--seed',            type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from src.eval.scene_graph import SceneGraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Eval mode:    {args.mode}")

# ── Helper ─────────────────────────────────────────────────
def get_base_label(label):
    parts   = label.split('_')
    spatial = {'north','south','east','west','center','upper','middle','lower'}
    base    = []
    for part in parts:
        if part in spatial:
            break
        base.append(part)
    return '_'.join(base) if base else label

def score_pair(q_cache, db_cache, w_emb, w_scene, w_jac):
    emb_sim   = (q_cache['emb'] * db_cache['emb']).sum().item()
    scene_sim = F.cosine_similarity(
        q_cache['scene_clip'], db_cache['scene_clip']
    ).item()
    overlap   = len(q_cache['labels'] & db_cache['labels'])
    if len(q_cache['labels']) > 0 and len(db_cache['labels']) > 0:
        precision = overlap / len(db_cache['labels'])
        recall    = overlap / len(q_cache['labels'])
        f1        = (2 * precision * recall) / (precision + recall + 1e-8)
    else:
        f1 = 0
    return w_emb * emb_sim + w_scene * scene_sim + w_jac * f1

# ── Load cached embeddings ─────────────────────────────────
print("\nLoading cached embeddings...")
db_emb_cache    = torch.load(f'{args.cache_dir}/db_emb_cache.pt',    weights_only=False)
query_emb_cache = torch.load(f'{args.cache_dir}/query_emb_cache.pt', weights_only=False)
print(f"✓ DB embeddings:    {len(db_emb_cache)} scenes")
print(f"✓ Query embeddings: {len(query_emb_cache)} queries")

# ── Load pool graphs for scene_id lookup ──────────────────
print("\nLoading pool graphs for scene ID lookup...")
scanscribe_218 = torch.load(f'{args.cache_dir}/scanscribe_cleaned_original_518D.pt',
                             weights_only=False, map_location='cpu')
pool_graphs = {}
for sid in tqdm(scanscribe_218, desc="Pool"):
    for tid in scanscribe_218[sid].keys():
        key = f"{sid}_{str(tid).zfill(5)}"
        pool_graphs[key] = SceneGraph(sid, txt_id=tid, graph_type='scanscribe',
                                      graph=scanscribe_218[sid][tid],
                                      embedding_type='word2vec', use_attributes=True)
pool_graphs = {k: v for k, v in pool_graphs.items() if len(v.edge_idx[0]) >= 1}
print(f"✓ Loaded {len(pool_graphs)} pool graphs")

# ── Build buckets ──────────────────────────────────────────
# Query buckets: scene_id → list of query keys (from 55-scene test set)
query_buckets = {}
for key, cache in query_emb_cache.items():
    sid = cache['scene_id']
    if sid not in query_buckets:
        query_buckets[sid] = []
    query_buckets[sid].append(key)

# Pool buckets: scene_id → list of pool keys (218 scenes)
pool_buckets = {}
for key, g in pool_graphs.items():
    if g.scene_id not in pool_buckets:
        pool_buckets[g.scene_id] = []
    pool_buckets[g.scene_id].append(key)

# 55 test scene IDs (intersection of query scenes and DB)
test_scene_ids = [sid for sid in query_buckets.keys() if sid in db_emb_cache]

print(f"\nQuery scenes:     {len(query_buckets)}")
print(f"Pool scenes:      {len(pool_buckets)}")
print(f"Test scenes in DB:{len(test_scene_ids)}")

# ── Eval: top10 mode ───────────────────────────────────────
def eval_top10(query_emb_cache, db_emb_cache, query_buckets, pool_buckets,
               eval_iters, eval_iter_count, out_of, valid_top_k,
               w_emb, w_scene, w_jac):
    """Table 1 protocol: rank query against 10 candidates (1 correct + 9 random)."""

    all_valid   = {k: [] for k in valid_top_k}
    debug_count = 0

    for eval_round in tqdm(range(eval_iters), desc="Eval rounds [top10]"):
        valid = {k: [] for k in valid_top_k}

        for batch_idx in range(eval_iter_count):
            query_scene_id = random.choice(list(query_buckets.keys()))
            query_key      = random.choice(query_buckets[query_scene_id])
            q_cache        = query_emb_cache[query_key]

            other_pool_scenes = [s for s in pool_buckets.keys() if s != query_scene_id]
            sampled_scenes    = random.sample(other_pool_scenes, out_of - 1)
            candidate_scenes  = [query_scene_id] + sampled_scenes

            match_scores = []
            scene_ids    = []

            for scene_id in candidate_scenes:
                if scene_id not in db_emb_cache:
                    continue
                final_score = score_pair(q_cache, db_emb_cache[scene_id],
                                         w_emb, w_scene, w_jac)
                match_scores.append(final_score)
                scene_ids.append(scene_id)

            if len(match_scores) == 0:
                continue

            match_scores   = np.array(match_scores)
            sorted_indices = np.argsort(match_scores)[::-1]

            if debug_count < 3:
                print(f"\n{'='*60}")
                print(f"DEBUG {debug_count+1} | Query: {query_scene_id}")
                for rank_idx, idx in enumerate(sorted_indices[:5]):
                    sid  = scene_ids[idx]
                    tag  = "✓ CORRECT" if sid == query_scene_id else "✗ wrong"
                    print(f"  Rank {rank_idx+1}: {sid[:40]} "
                          f"score={match_scores[idx]:.4f} {tag}")
                gt_rank = next((r+1 for r, idx in enumerate(sorted_indices)
                                if scene_ids[idx] == query_scene_id), None)
                print(f"GT rank: {gt_rank}/{len(sorted_indices)}")
                debug_count += 1

            for k in valid_top_k:
                top_k_scenes = [scene_ids[idx] for idx in sorted_indices[:k]]
                valid[k].append(1 if query_scene_id in top_k_scenes else 0)

        for k in valid_top_k:
            all_valid[k].append(np.mean(valid[k]))

    return {k: (np.mean(all_valid[k]), np.std(all_valid[k])) for k in valid_top_k}


# ── Eval: full mode ────────────────────────────────────────
def eval_full(query_emb_cache, db_emb_cache, query_buckets, test_scene_ids,
              eval_iters, eval_iter_count, valid_top_k,
              w_emb, w_scene, w_jac):
    """Table 2 protocol: rank query against ALL 55 test scenes."""

    all_valid   = {k: [] for k in valid_top_k}
    debug_count = 0

    for eval_round in tqdm(range(eval_iters), desc="Eval rounds [full]"):
        valid = {k: [] for k in valid_top_k}

        for batch_idx in range(eval_iter_count):
            query_scene_id = random.choice(list(query_buckets.keys()))
            query_key      = random.choice(query_buckets[query_scene_id])
            q_cache        = query_emb_cache[query_key]

            # Rank against ALL 55 test scenes
            match_scores = []
            scene_ids    = []

            for scene_id in test_scene_ids:
                if scene_id not in db_emb_cache:
                    continue
                final_score = score_pair(q_cache, db_emb_cache[scene_id],
                                         w_emb, w_scene, w_jac)
                match_scores.append(final_score)
                scene_ids.append(scene_id)

            if len(match_scores) == 0:
                continue

            match_scores   = np.array(match_scores)
            sorted_indices = np.argsort(match_scores)[::-1]

            if debug_count < 3:
                print(f"\n{'='*60}")
                print(f"DEBUG {debug_count+1} | Query: {query_scene_id}")
                print(f"Ranking against {len(scene_ids)} scenes")
                for rank_idx, idx in enumerate(sorted_indices[:5]):
                    sid  = scene_ids[idx]
                    tag  = "✓ CORRECT" if sid == query_scene_id else "✗ wrong"
                    print(f"  Rank {rank_idx+1}: {sid[:40]} "
                          f"score={match_scores[idx]:.4f} {tag}")
                gt_rank = next((r+1 for r, idx in enumerate(sorted_indices)
                                if scene_ids[idx] == query_scene_id), None)
                print(f"GT rank: {gt_rank}/{len(sorted_indices)}")
                debug_count += 1

            for k in valid_top_k:
                top_k_scenes = [scene_ids[idx] for idx in sorted_indices[:k]]
                valid[k].append(1 if query_scene_id in top_k_scenes else 0)

        for k in valid_top_k:
            all_valid[k].append(np.mean(valid[k]))

    return {k: (np.mean(all_valid[k]), np.std(all_valid[k])) for k in valid_top_k}


# ── Run ────────────────────────────────────────────────────
valid_top_k_of_10 = [1, 2, 3, 5]
valid_top_k_full   = [5,10,20,30]

print(f"\nWeights: emb={args.w_emb}, scene={args.w_scene}, jac={args.w_jac}")

if args.mode == 'top10':
    print(f"Protocol: top-k out of {args.out_of} candidates (Table 1)\n")
    results = eval_top10(
        query_emb_cache  = query_emb_cache,
        db_emb_cache     = db_emb_cache,
        query_buckets    = query_buckets,
        pool_buckets     = pool_buckets,
        eval_iters       = args.eval_iters,
        eval_iter_count  = args.eval_iter_count,
        out_of           = args.out_of,
        valid_top_k      = valid_top_k_of_10,
        w_emb            = args.w_emb,
        w_scene          = args.w_scene,
        w_jac            = args.w_jac
    )

elif args.mode == 'full':
    print(f"Protocol: top-k out of all {len(test_scene_ids)} test scenes (Table 2)\n")
    results = eval_full(
        query_emb_cache  = query_emb_cache,
        db_emb_cache     = db_emb_cache,
        query_buckets    = query_buckets,
        test_scene_ids   = test_scene_ids,
        eval_iters       = args.eval_iters,
        eval_iter_count  = args.eval_iter_count,
        valid_top_k      = valid_top_k_full,
        w_emb            = args.w_emb,
        w_scene          = args.w_scene,
        w_jac            = args.w_jac
    )

print(f"\n{'='*60}")
print(f"FINAL RESULTS [{args.mode.upper()}]")
print(f"{'='*60}")
valid_top_k_to_print = valid_top_k_full if args.mode == 'full' else valid_top_k_of_10
for k in valid_top_k_to_print:
    mean, std = results[k]
    print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
print(f"{'='*60}")