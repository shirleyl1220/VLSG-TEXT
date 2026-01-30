"""
Diagnostic script to check if training file loads data properly
"""
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append('/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT-1/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT')

from src.datasets.dual_scene_graph_dataset_518_v2 import DualSceneGraphDataset
from src.trainval.train_with_scene_clip_518_v2 import collate_graph_batch_with_scene_clip

print("="*80)
print("TESTING TRAINING DATA LOADING")
print("="*80)

# Load dataset
dataset = DualSceneGraphDataset(
    dataset_dir='/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT-1/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/scene_graph_clip',
    metadata_path='/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/meta_files/3RScan.json',
    augment_ratio=0.0
)

print(f"\n✓ Dataset loaded: {len(dataset)} scenes\n")

# Create dataloader with training collate function
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    collate_fn=collate_graph_batch_with_scene_clip
)

print(f"✓ DataLoader created\n")

# Test loading one batch
print("="*80)
print("TESTING BATCH LOADING")
print("="*80)

try:
    batch = next(iter(dataloader))
    print("\n✓ Successfully loaded batch!\n")
    
    print("BATCH CONTENTS:")
    print("-" * 80)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:25s}: shape={str(value.shape):20s} dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"  {key:25s}: list with {len(value)} items")
        elif isinstance(value, int):
            print(f"  {key:25s}: {value}")
        else:
            print(f"  {key:25s}: {type(value)}")
    
    # Check critical fields
    print("\n" + "="*80)
    print("FIELD VALIDATION")
    print("="*80)
    
    required_fields = [
        'node_feats_src', 'node_feats_ref',
        'geom_edges_src', 'geom_edges_ref',
        'geom_attr_src', 'geom_attr_ref',
        'text_edges_src', 'text_edges_ref',
        'text_attr_src', 'text_attr_ref',
        'src_batch', 'ref_batch',
        'batch_size',
        'scene_clip_src', 'scene_clip_ref',
        'room_ids'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in batch:
            missing_fields.append(field)
            print(f"  ❌ Missing field: {field}")
        else:
            print(f"  ✓ Has field: {field}")
    
    if missing_fields:
        print(f"\n❌ PROBLEM: Missing {len(missing_fields)} required fields!")
    else:
        print(f"\n✓ All required fields present!")
    
    # Check node features dimension
    print("\n" + "="*80)
    print("NODE FEATURES VALIDATION")
    print("="*80)
    
    node_feats_src = batch['node_feats_src']
    node_feats_ref = batch['node_feats_ref']
    
    print(f"\nSource nodes: {node_feats_src.shape}")
    print(f"Reference nodes: {node_feats_ref.shape}")
    
    if node_feats_src.shape[1] == 518:
        print(f"✓ Source node features are 518D")
    else:
        print(f"❌ Source node features are {node_feats_src.shape[1]}D, expected 518D!")
    
    if node_feats_ref.shape[1] == 518:
        print(f"✓ Reference node features are 518D")
    else:
        print(f"❌ Reference node features are {node_feats_ref.shape[1]}D, expected 518D!")
    
    # Check scene CLIP
    print("\n" + "="*80)
    print("SCENE CLIP VALIDATION")
    print("="*80)
    
    scene_clip_src = batch['scene_clip_src']
    scene_clip_ref = batch['scene_clip_ref']
    batch_size = batch['batch_size']
    
    print(f"\nBatch size: {batch_size}")
    print(f"scene_clip_src shape: {scene_clip_src.shape}")
    print(f"scene_clip_ref shape: {scene_clip_ref.shape}")
    
    if scene_clip_src.shape == (batch_size, 512):
        print(f"✓ scene_clip_src has correct shape [batch_size, 512]")
    else:
        print(f"❌ scene_clip_src shape is wrong! Expected [{batch_size}, 512]")
    
    if scene_clip_ref.shape == (batch_size, 512):
        print(f"✓ scene_clip_ref has correct shape [batch_size, 512]")
    else:
        print(f"❌ scene_clip_ref shape is wrong! Expected [{batch_size}, 512]")
    
    # Check norms
    src_norm = scene_clip_src.norm(dim=1).mean().item()
    ref_norm = scene_clip_ref.norm(dim=1).mean().item()
    
    print(f"\nscene_clip_src avg norm: {src_norm:.4f}")
    print(f"scene_clip_ref avg norm: {ref_norm:.4f}")
    
    if src_norm > 0.9 and src_norm < 1.1:
        print(f"✓ scene_clip_src norms look good (normalized)")
    else:
        print(f"❌ scene_clip_src norms look wrong!")
    
    if ref_norm > 0.9 and ref_norm < 1.1:
        print(f"✓ scene_clip_ref norms look good (normalized)")
    else:
        print(f"❌ scene_clip_ref norms look wrong!")
    
    # Check room IDs
    print("\n" + "="*80)
    print("ROOM IDS VALIDATION")
    print("="*80)
    
    room_ids = batch['room_ids']
    print(f"\nRoom IDs type: {type(room_ids)}")
    print(f"Room IDs length: {len(room_ids)}")
    print(f"Room IDs: {room_ids}")
    
    if len(room_ids) == batch_size:
        print(f"✓ Correct number of room IDs")
    else:
        print(f"❌ Wrong number of room IDs! Expected {batch_size}, got {len(room_ids)}")
    
    # Check batch indices
    print("\n" + "="*80)
    print("BATCH INDICES VALIDATION")
    print("="*80)
    
    src_batch = batch['src_batch']
    ref_batch = batch['ref_batch']
    
    print(f"\nsrc_batch shape: {src_batch.shape}")
    print(f"ref_batch shape: {ref_batch.shape}")
    print(f"src_batch unique values: {src_batch.unique().tolist()}")
    print(f"ref_batch unique values: {ref_batch.unique().tolist()}")
    
    expected_batch_ids = list(range(batch_size))
    src_batch_ids = src_batch.unique().tolist()
    ref_batch_ids = ref_batch.unique().tolist()
    
    if sorted(src_batch_ids) == expected_batch_ids:
        print(f"✓ src_batch has correct batch indices")
    else:
        print(f"❌ src_batch has wrong indices! Expected {expected_batch_ids}, got {src_batch_ids}")
    
    if sorted(ref_batch_ids) == expected_batch_ids:
        print(f"✓ ref_batch has correct batch indices")
    else:
        print(f"❌ ref_batch has wrong indices! Expected {expected_batch_ids}, got {ref_batch_ids}")
    
    # Count nodes per graph
    print(f"\nNodes per graph:")
    for i in range(batch_size):
        src_count = (src_batch == i).sum().item()
        ref_count = (ref_batch == i).sum().item()
        print(f"  Graph {i}: src={src_count:3d} nodes, ref={ref_count:3d} nodes")
    
    print("\n" + "="*80)
    print("TESTING FORWARD PASS COMPATIBILITY")
    print("="*80)
    
    # Simulate what the training loop does
    print("\nSimulating training loop operations...")
    
    # Test room label creation (from training script)
    unique_rooms = list(set(room_ids))
    room_id_to_label = {room_id: i for i, room_id in enumerate(unique_rooms)}
    room_labels_list = [room_id_to_label[room_id] for room_id in room_ids]
    room_labels = torch.tensor(room_labels_list * 2)
    
    print(f"✓ Created room labels: {room_labels.tolist()}")
    print(f"  Unique rooms in batch: {len(unique_rooms)}")
    print(f"  Room labels shape: {room_labels.shape}")
    
    if room_labels.shape[0] == batch_size * 2:
        print(f"✓ Room labels have correct length (batch_size * 2)")
    else:
        print(f"❌ Room labels have wrong length!")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    issues = []
    
    if missing_fields:
        issues.append(f"Missing fields: {missing_fields}")
    
    if node_feats_src.shape[1] != 518:
        issues.append(f"Source nodes wrong dimension: {node_feats_src.shape[1]}")
    
    if node_feats_ref.shape[1] != 518:
        issues.append(f"Reference nodes wrong dimension: {node_feats_ref.shape[1]}")
    
    if scene_clip_src.shape != (batch_size, 512):
        issues.append(f"scene_clip_src wrong shape: {scene_clip_src.shape}")
    
    if scene_clip_ref.shape != (batch_size, 512):
        issues.append(f"scene_clip_ref wrong shape: {scene_clip_ref.shape}")
    
    if len(room_ids) != batch_size:
        issues.append(f"Wrong number of room IDs: {len(room_ids)}")
    
    if issues:
        print("\n❌ FOUND ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ ALL CHECKS PASSED!")
        print("Training file should load data correctly!")

except Exception as e:
    print(f"\n❌ ERROR loading batch:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
