"""
Evaluation script for SimpleGraphMatcher (518-dim + Scene CLIP fusion).

For IMG-generated ScanScribe test set (Table 4).
Now with CACHE support for fast evaluation!
"""

import time
import argparse
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import clip
from pathlib import Path

# Fix imports
REPO_ROOT = Path('/content/VLSG-TEXT/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT')
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src' / 'eval'))

from scene_graph import SceneGraph

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Helper Functions
# ============================================================

def get_base_label(label):
    """Remove spatial modifiers from label."""
    parts = label.split('_')
    spatial = {'north','south','east','west','center','upper','middle','lower'}
    base = []
    for part in parts:
        if part in spatial:
            break
        base.append(part)
    return '_'.join(base) if base else label


def score_pair(q_cache, db_cache, w_emb, w_scene, w_jac, return_components=False):
    """Compute score between query and database using cached embeddings."""
    # 1. Embedding similarity
    emb_sim = (q_cache['emb'] * db_cache['emb']).sum().item()

    # 2. Scene CLIP similarity
    scene_sim = F.cosine_similarity(
        q_cache['scene_clip'],
        db_cache['scene_clip']
    ).item()

    # 3. Label overlap (F1 score)
    matched_labels = q_cache['labels'] & db_cache['labels']
    overlap = len(matched_labels)
    if len(q_cache['labels']) > 0 and len(db_cache['labels']) > 0:
        precision = overlap / len(db_cache['labels'])
        recall = overlap / len(q_cache['labels'])
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    else:
        f1 = 0

    total = w_emb * emb_sim + w_scene * scene_sim + w_jac * f1
    if return_components:
        return total, emb_sim, scene_sim, f1, matched_labels
    return total


# ============================================================
# Evaluation Function
# ============================================================

def eval_with_cache(query_emb_cache, db_emb_cache,
                    eval_iters=10, eval_iter_count=100, out_of=10,
                    valid_top_k=[1, 2, 3, 5],
                    w_emb=0.33, w_scene=0.33, w_jac=0.34,
                    debug_n=0):
    """
    Fast evaluation using precomputed embeddings.
    debug_n: print score breakdown for the first N top-1 failures.
    """
    
    print(f"\n{'='*70}")
    print(f"Evaluating with CACHED embeddings")
    print(f"Fusion weights: emb={w_emb:.2f}, scene={w_scene:.2f}, label={w_jac:.2f} (F1)")
    print(f"{'='*70}")
    
    # Build buckets
    query_buckets = {}
    for key, cache in query_emb_cache.items():
        sid = cache['scene_id']
        if sid not in query_buckets:
            query_buckets[sid] = []
        query_buckets[sid].append(key)
    
    # Negatives sampled from the same image-desc scenes as the query (matches baseline protocol)
    pool_buckets = query_buckets

    print(f"Query scenes: {len(query_buckets)}")
    print(f"Pool scenes: {len(pool_buckets)}")
    print(f"DB scenes: {len(db_emb_cache)}")
    
    # Evaluation
    all_valid = {k: [] for k in valid_top_k}
    debug_count = 0
    
    for eval_round in tqdm(range(eval_iters), desc="Eval rounds"):
        valid = {k: [] for k in valid_top_k}
        
        for batch_idx in range(eval_iter_count):
            # Sample query
            query_scene_id = random.choice(list(query_buckets.keys()))
            query_key = random.choice(query_buckets[query_scene_id])
            q_cache = query_emb_cache[query_key]
            
            # Sample candidates (1 correct + 9 random)
            other_pool_scenes = [s for s in pool_buckets.keys() if s != query_scene_id]
            sampled_scenes = random.sample(other_pool_scenes, out_of - 1)
            candidate_scenes = [query_scene_id] + sampled_scenes
            
            match_scores = []
            scene_ids = []
            score_components = []  # (emb_sim, scene_sim, f1, matched_labels)

            for scene_id in candidate_scenes:
                if scene_id not in db_emb_cache:
                    continue

                total, emb_sim, scene_sim, f1, matched = score_pair(
                    q_cache, db_emb_cache[scene_id],
                    w_emb, w_scene, w_jac, return_components=True
                )
                match_scores.append(total)
                scene_ids.append(scene_id)
                score_components.append((emb_sim, scene_sim, f1, matched))

            if len(match_scores) == 0:
                continue

            # Sort by score (high to low)
            match_scores = np.array(match_scores)
            sorted_indices = np.argsort(match_scores)[::-1]

            # Ground truth rank
            gt_rank = next(
                (r + 1 for r, idx in enumerate(sorted_indices) if scene_ids[idx] == query_scene_id),
                None
            )

            # Print debug breakdown for top-1 failures
            is_failure = gt_rank is None or gt_rank > 1
            if debug_count < debug_n and is_failure:
                debug_count += 1
                q_labels = sorted(q_cache['labels'])
                print(f"\n{'='*70}")
                print(f"DEBUG FAILURE #{debug_count}  (round {eval_round}, batch {batch_idx})")
                print(f"  Query scene : {query_scene_id}")
                print(f"  Query labels: {q_labels}")
                print(f"  GT ranked   : {gt_rank}/{len(sorted_indices)}")
                print(f"  {'Rank':<5} {'Scene ID':<40} {'Total':>7} {'Emb':>7} {'SceneCLIP':>10} {'F1':>6}  Matched labels")
                print(f"  {'-'*5} {'-'*40} {'-'*7} {'-'*7} {'-'*10} {'-'*6}  {'-'*30}")
                for rank_idx, idx in enumerate(sorted_indices):
                    sid = scene_ids[idx]
                    total_s = match_scores[idx]
                    emb_s, scene_s, f1_s, matched = score_components[idx]
                    tag = " <-- GT" if sid == query_scene_id else ""
                    db_labels = sorted(db_emb_cache[sid]['labels'])
                    print(f"  {rank_idx+1:<5} {sid:<40} {total_s:>7.4f} {emb_s:>7.4f} {scene_s:>10.4f} {f1_s:>6.4f}  "
                          f"matched={sorted(matched)}  db_labels={db_labels}{tag}")
                print(f"{'='*70}\n")
            
            # Check top-k
            for k in valid_top_k:
                if k <= len(sorted_indices):
                    top_k_scenes = [scene_ids[idx] for idx in sorted_indices[:k]]
                    valid[k].append(1 if query_scene_id in top_k_scenes else 0)
        
        for k in valid_top_k:
            if len(valid[k]) > 0:
                all_valid[k].append(np.mean(valid[k]))
    
    # Results
    accuracy = {k: (np.mean(all_valid[k]), np.std(all_valid[k])) 
                for k in valid_top_k if len(all_valid[k]) > 0}
    
    print(f"\nResults:")
    for k in accuracy:
        mean, std = accuracy[k]
        print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    
    return accuracy


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, 
                       default='/content/drive/MyDrive/VLSG_Files',
                       help='Directory containing cached embeddings')
    parser.add_argument('--cache_suffix', type=str, default='_img',
                       help='Cache file suffix (e.g., _img for IMG dataset)')
    parser.add_argument('--w_emb', type=float, default=0.33,
                       help='Weight for embedding similarity')
    parser.add_argument('--w_scene', type=float, default=0.33,
                       help='Weight for scene CLIP similarity')
    parser.add_argument('--w_jac', type=float, default=0.34,
                       help='Weight for F1 label overlap')
    parser.add_argument('--eval_iters', type=int, default=10,
                       help='Number of evaluation rounds')
    parser.add_argument('--eval_iter_count', type=int, default=100,
                       help='Number of samples per round')
    parser.add_argument('--out_of', type=int, default=10,
                       help='Number of candidates to rank')
    parser.add_argument('--debug_n', type=int, default=0,
                       help='Print score breakdown for the first N top-1 failures (0 = off)')
    args = parser.parse_args()
    
    print(f"\nCache directory: {args.cache_dir}")
    print(f"Cache suffix: {args.cache_suffix}")
    print(f"Fusion weights: emb={args.w_emb:.2f}, scene={args.w_scene:.2f}, label={args.w_jac:.2f}\n")
    
    # ============================================================
    # Load cached embeddings
    # ============================================================
    
    print("Loading cached embeddings...")
    db_emb_cache = torch.load(
        f'{args.cache_dir}/db_emb_cache{args.cache_suffix}.pt',
        weights_only=False
    )
    query_emb_cache = torch.load(
        f'{args.cache_dir}/query_emb_cache{args.cache_suffix}.pt',
        weights_only=False
    )
    print(f"✓ DB embeddings:    {len(db_emb_cache)} scenes")
    print(f"✓ Query embeddings: {len(query_emb_cache)} queries")

    query_scene_ids = set(cache['scene_id'] for cache in query_emb_cache.values())
    missing = query_scene_ids - set(db_emb_cache.keys())
    print(f"✓ Scene IDs in query but missing from DB: {len(missing)}")
    if missing:
        print(f"  WARNING: {len(missing)} query scenes have no DB entry — they will be skipped as candidates")
        print(f"  Missing: {list(missing)[:5]}")
    
    # ============================================================
    # Load pool graphs (for scene_id lookup)
    # ============================================================
    
    print("\nLoading pool graphs for scene ID lookup...")
    
    # Determine which pool file to use based on suffix
    if args.cache_suffix == '_img':
        pool_file = 'scanscribe_graphs_test_518D_gpt_labels.pt'
        print("Using IMG dataset pool")
    else:
        pool_file = 'scanscribe_cleaned_original_518D.pt'
        print("Using original text dataset pool")
    
    scanscribe_pool = torch.load(
        f'{args.cache_dir}/{pool_file}',
        weights_only=False, 
        map_location='cpu'
    )
    
    pool_graphs = {}
    for sid in tqdm(scanscribe_pool, desc="Loading pool"):
        for tid in scanscribe_pool[sid].keys():
            key = f"{sid}_{str(tid).zfill(5)}"
            pool_graphs[key] = SceneGraph(
                sid, txt_id=tid, graph_type='scanscribe',
                graph=scanscribe_pool[sid][tid],
                embedding_type='word2vec', use_attributes=True
            )
    
    pool_graphs = {k: v for k, v in pool_graphs.items() if len(v.edge_idx[0]) >= 1}
    print(f"✓ Loaded {len(pool_graphs)} pool graphs")
    
    # ============================================================
    # Run evaluation
    # ============================================================
    
    # Best weights found via grid search: (0.33, 0.33, 0.34)
    best_weights = (args.w_emb, args.w_scene, args.w_jac)

    # ============================================================
    # Final eval
    # ============================================================
    print(f"\nRunning eval with weights: emb={best_weights[0]:.2f}, scene={best_weights[1]:.2f}, jac={best_weights[2]:.2f}...")
    results = eval_with_cache(
        query_emb_cache=query_emb_cache,
        db_emb_cache=db_emb_cache,
        eval_iters=args.eval_iters,
        eval_iter_count=args.eval_iter_count,
        out_of=args.out_of,
        valid_top_k=[1, 2, 3, 5],
        w_emb=best_weights[0],
        w_scene=best_weights[1],
        w_jac=best_weights[2],
        debug_n=args.debug_n,
    )

    print(f"\n{'='*70}")
    print("FINAL RESULTS - ScanScribe IMG (Table 4)")
    print(f"{'='*70}")
    print(f"Weights: emb={best_weights[0]:.2f}, scene={best_weights[1]:.2f}, jac={best_weights[2]:.2f} (F1)")
    for k in [1, 2, 3, 5]:
        if k in results:
            mean, std = results[k]
            print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    print(f"{'='*70}\n")