import os
import trimesh

class Exporter:
    def __init__(self, output_dir="d:/3d_model/assets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_3d(self, mesh, filename="avatar", formats=["obj", "glb"]):
        paths = []
        for fmt in formats:
            path = os.path.join(self.output_dir, f"{filename}.{fmt}")
            if fmt == "obj":
                mesh.export(path)
            elif fmt == "glb":
                mesh.export(path, file_type="glb")
            paths.append(path)
            print(f"Exported 3D model to {path}")
        return paths

    def export_2d(self, image, filename="avatar_2d"):
        path = os.path.join(self.output_dir, f"{filename}.png")
        image.save(path)
        print(f"Exported 2D avatar to {path}")
        return path

if __name__ == "__main__":
    exporter = Exporter()
    print("Exporter module ready.")
