import torch
import sys

pt1 = sys.argv[1]
pt2 = sys.argv[2]

data1 = torch.load(pt1, map_location='cpu', weights_only=False)
data2 = torch.load(pt2, map_location='cpu', weights_only=False)

ids1 = set(data1.keys())
ids2 = set(data2.keys())

overlap = ids1 & ids2
only_in_1 = ids1 - ids2
only_in_2 = ids2 - ids1

print(f"File 1: {pt1}")
print(f"  Scene IDs: {len(ids1)}")
print(f"File 2: {pt2}")
print(f"  Scene IDs: {len(ids2)}")
print(f"\nOverlap: {len(overlap)}")
print(f"Only in file 1: {len(only_in_1)}")
print(f"Only in file 2: {len(only_in_2)}")

# Uncomment to print the actual IDs:
print(f"\nOverlapping IDs: {sorted(overlap)}")
print(f"\nOnly in file 1: {sorted(only_in_1)}")
print(f"\nOnly in file 2: {sorted(only_in_2)}")