from plyfile import PlyData

ply_path = "/Users/shirley/Documents/SCHOOL/FALL2025/MASTER-PROJECT/3RScan/fcf66dbc-622d-291c-8481-6e8761c93e21/labels.instances.annotated.v2.ply"
ply = PlyData.read(ply_path)

print("ELEMENTS:")
for element in ply.elements:
    print(" -", element.name)
    print("   Properties:")
    for prop in element.properties:
        print("     •", prop.name, "(", prop.val_dtype, ")")