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


def score_pair(q_cache, db_cache, w_emb, w_scene, w_jac):
    """Compute score between query and database using cached embeddings."""
    # 1. Embedding similarity
    emb_sim = (q_cache['emb'] * db_cache['emb']).sum().item()
    
    # 2. Scene CLIP similarity
    scene_sim = F.cosine_similarity(
        q_cache['scene_clip'], 
        db_cache['scene_clip']
    ).item()
    
    # 3. Label overlap (F1 score)
    overlap = len(q_cache['labels'] & db_cache['labels'])
    if len(q_cache['labels']) > 0 and len(db_cache['labels']) > 0:
        precision = overlap / len(db_cache['labels'])
        recall = overlap / len(q_cache['labels'])
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    else:
        f1 = 0
    
    # Combined score
    return w_emb * emb_sim + w_scene * scene_sim + w_jac * f1


# ============================================================
# Evaluation Function
# ============================================================

def eval_with_cache(query_emb_cache, db_emb_cache, pool_graphs, 
                    eval_iters=10, eval_iter_count=100, out_of=10,
                    valid_top_k=[1, 2, 3, 5],
                    w_emb=0.33, w_scene=0.33, w_jac=0.34):
    """
    Fast evaluation using precomputed embeddings.
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
    
    pool_buckets = {}
    for key, g in pool_graphs.items():
        if g.scene_id not in pool_buckets:
            pool_buckets[g.scene_id] = []
        pool_buckets[g.scene_id].append(key)
    
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
            
            for scene_id in candidate_scenes:
                if scene_id not in db_emb_cache:
                    continue
                
                final_score = score_pair(q_cache, db_emb_cache[scene_id],
                                        w_emb, w_scene, w_jac)
                match_scores.append(final_score)
                scene_ids.append(scene_id)
            
            if len(match_scores) == 0:
                continue
            
            # Sort by score (high to low)
            match_scores = np.array(match_scores)
            sorted_indices = np.argsort(match_scores)[::-1]
            
            # DEBUG: Show first 5 batches
            if debug_count < 5:
                print(f"\n{'='*70}")
                print(f"DEBUG Batch {debug_count + 1} (Round {eval_round})")
                print(f"{'='*70}")
                print(f"Query scene: {query_scene_id}")
                
                print(f"\n🎯 TOP 10 PREDICTIONS:")
                for rank_idx, idx in enumerate(sorted_indices[:min(10, len(sorted_indices))]):
                    sid = scene_ids[idx]
                    score = match_scores[idx]
                    is_correct = "✓ CORRECT" if sid == query_scene_id else "✗ wrong"
                    print(f"  Rank {rank_idx+1}: {sid:40s} score={score:.4f} {is_correct}")
                
                # Ground truth rank
                gt_rank = None
                for rank_idx, idx in enumerate(sorted_indices):
                    if scene_ids[idx] == query_scene_id:
                        gt_rank = rank_idx + 1
                        break
                
                print(f"\n  📊 Ground truth ranked at: {gt_rank}/{len(sorted_indices)}")
                
                if gt_rank and gt_rank <= 3:
                    print(f"  ✅ GOOD!")
                elif gt_rank and gt_rank <= 5:
                    print(f"  ⚠️  OKAY")
                else:
                    print(f"  ❌ POOR")
                
                print(f"{'='*70}\n")
                debug_count += 1
            
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
    
    # ============================================================
    # Load pool graphs (for scene_id lookup)
    # ============================================================
    
    print("\nLoading pool graphs for scene ID lookup...")
    
    # Determine which pool file to use based on suffix
    if args.cache_suffix == '_img':
        pool_file = 'scanscribe_graphs_test_518D.pt'
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
    
    print(f"\n✅ Using cached embeddings with weights:")
    print(f"   emb={args.w_emb:.2f}, scene={args.w_scene:.2f}, label={args.w_jac:.2f} (F1)\n")
    
    results = eval_with_cache(
        query_emb_cache=query_emb_cache,
        db_emb_cache=db_emb_cache,
        pool_graphs=pool_graphs,
        eval_iters=args.eval_iters,
        eval_iter_count=args.eval_iter_count,
        out_of=args.out_of,
        valid_top_k=[1, 2, 3, 5],
        w_emb=args.w_emb,
        w_scene=args.w_scene,
        w_jac=args.w_jac
    )
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS - ScanScribe IMG (Table 4)")
    print(f"{'='*70}")
    print(f"Weights: emb={args.w_emb:.2f}, scene={args.w_scene:.2f}, label={args.w_jac:.2f} (F1)")
    for k in [1, 2, 3, 5]:
        if k in results:
            mean, std = results[k]
            print(f"  Top-{k}: {mean*100:.2f}% ± {std*100:.2f}%")
    print(f"{'='*70}\n")