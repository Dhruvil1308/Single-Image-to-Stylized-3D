import os
import sys
from PIL import Image
import trimesh
import pyrender
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess.extractor import FaceExtractor
from src.recon.mesh_fit import MeshFitter

def main():
    print("Testing Pyrender lighting & texturing...")
    os.makedirs("assets", exist_ok=True)
    
    extractor = FaceExtractor()
    input_path = "temp_input.jpg"
    aligned = extractor.align_and_crop(input_path)
    
    fitter = MeshFitter()
    mesh = fitter.fit([aligned], None)
    
    # Render with various settings
    try:
        # 1. Manual Material Injection
        tex_img = getattr(mesh.visual.material, 'baseColorTexture', None)
        material = None
        if tex_img is not None:
            img_np = np.array(tex_img.convert('RGB'))
            tex = pyrender.Texture(source=img_np, source_channels='RGB')
            material = pyrender.MetallicRoughnessMaterial(
                baseColorTexture=tex,
                metallicFactor=0.0,
                roughnessFactor=0.7,
                alphaMode='OPAQUE'
            )
        
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene = pyrender.Scene(ambient_light=[0.6, 0.6, 0.6], bg_color=[1.0, 1.0, 1.0, 1.0])
        scene.add(py_mesh)
        
        t_scene = trimesh.Scene()
        t_scene.add_geometry(mesh)
        t_scene.set_camera()
        camera_pose = t_scene.camera_transform.copy()
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        scene.add(camera, pose=camera_pose)
        
        renderer = pyrender.OffscreenRenderer(512, 512)
        color, _ = renderer.render(scene)
        Image.fromarray(color).save("assets/test_light_1_ambient.png")
        print("Saved test_light_1_ambient.png")
        
        # 2. Add directional light, lower ambient
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0, 1.0])
        scene.add(py_mesh)
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        scene.add(light, pose=camera_pose)
        color, _ = renderer.render(scene)
        Image.fromarray(color).save("assets/test_light_2_directional.png")
        print("Saved test_light_2_directional.png")
        
        # 3. Check what mesh.visual is
        print("Trimesh visual type:", type(mesh.visual))
        if hasattr(mesh.visual, 'material'):
            print("Material type:", type(mesh.visual.material))
            
    except Exception as e:
        print("Error during test:", e)

if __name__ == "__main__":
    main()
