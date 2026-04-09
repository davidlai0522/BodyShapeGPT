import os
import torch
import smplx
import trimesh
import argparse

# Example: python load_smpl.py --betas 1.2 -0.4 0.8 0 0 0 0 0 0 0

def parse_args():
    parser = argparse.ArgumentParser(description="Load and visualize an SMPL mesh.")
    parser.add_argument(
        "--gender",
        choices=["male", "female", "neutral"],
        default="neutral",
        help="SMPL model gender to load.",
    )
    parser.add_argument(
        "--betas",
        nargs=10,
        type=float,
        metavar=("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"),
        default=[3.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        help="SMPL shape parameters (10 floats).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Dynamically get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Loading model from: {current_dir}")
    print(f"Using gender: {args.gender}")

    # 2. Load the SMPL model
    try:
        model = smplx.create(model_path=current_dir, model_type='smpl', gender=args.gender)
    except Exception as e:
        print(f"Error loading model. Make sure SMPL files are in {current_dir}.")
        raise e

    # 3. Define a set of 'betas' (shape parameters)
    # SMPL uses 10 parameters to control the body shape (e.g., height, weight, proportions).
    # Shape must be (batch_size, num_betas) -> (1, 10).
    # Positive values in the first parameter usually make the mesh taller/larger.
    custom_betas = torch.tensor([args.betas], dtype=torch.float32)
    output = model(betas=custom_betas, return_verts=True)
    
    vertices = output.vertices.detach().cpu().numpy().squeeze() 
    faces = model.faces

    print("Opening 3D viewer...")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Optional: Give the mesh a distinct color (R, G, B, Alpha)
    mesh.visual.vertex_colors = [249, 228, 212, 255] 
    
    # This will pop open an interactive 3D window
    mesh.show()

if __name__ == "__main__":
    
    main()
