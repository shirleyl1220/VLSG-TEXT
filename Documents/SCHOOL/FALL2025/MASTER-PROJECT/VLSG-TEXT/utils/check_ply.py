


"""Check PLY file structure"""
from plyfile import PlyData
import numpy as np

ply_path = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan/c92fb5ab-f771-2064-842c-c342564aabcc/labels.instances.annotated.v2.ply"

ply = PlyData.read(ply_path)
v = ply["vertex"].data

print("Vertex attributes:", v.dtype.names)
print()

if "objectId" in v.dtype.names:
    obj_ids = v["objectId"]
    unique_ids = np.unique(obj_ids)
    print(f"objectId values: {unique_ids[:20]}")
    print(f"Total unique: {len(unique_ids)}")
    print()
    
    # Check distribution
    for uid in unique_ids[:10]:
        count = (obj_ids == uid).sum()
        print(f"  objectId {uid}: {count} points")
