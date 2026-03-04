import clip
import torch

clip_model, _ = clip.load("ViT-B/32", device='cpu')

labels_scanscribe = ['table', 'chair', 'bench', 'stool']
labels_3dssg = ['table', 'chair', 'bench', 'stool', 
                'kitchen cabinet', 'standing on', 'wall']

with torch.no_grad():
    for l1 in labels_scanscribe:
        t1 = clip.tokenize([l1])
        e1 = clip_model.encode_text(t1)
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        
        sims = []
        for l2 in labels_3dssg:
            t2 = clip.tokenize([l2])
            e2 = clip_model.encode_text(t2)
            e2 = e2 / e2.norm(dim=-1, keepdim=True)
            sim = (e1 * e2).sum().item()
            sims.append((l2, sim))
        
        sims.sort(key=lambda x: x[1], reverse=True)
        print(f"\n'{l1}' most similar to:")
        for label, sim in sims[:4]:
            print(f"  {label}: {sim:.3f}")

relations_scanscribe = ['left_of', 'right_of', 'above', 'below', 'in_front_of', 'touching']
relations_3dssg = ['standing on', 'attached to', 'hanging on', 'supported by', 
                   'left', 'right', 'front', 'behind', 'close by']

with torch.no_grad():
    for r1 in relations_scanscribe:
        t1 = clip.tokenize([r1])
        e1 = clip_model.encode_text(t1)
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        
        sims = []
        for r2 in relations_3dssg:
            t2 = clip.tokenize([r2])
            e2 = clip_model.encode_text(t2)
            e2 = e2 / e2.norm(dim=-1, keepdim=True)
            sim = (e1 * e2).sum().item()
            sims.append((r2, sim))
        
        sims.sort(key=lambda x: x[1], reverse=True)
        print(f"\n'{r1}' most similar 3DSSG relation:")
        for label, sim in sims[:3]:
            print(f"  {label}: {sim:.3f}")