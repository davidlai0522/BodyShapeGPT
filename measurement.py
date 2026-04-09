import torch
import numpy as np
np.bool = np.bool_
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.unicode = str

import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
import smplx
import os

class BodyMeasurements:
    def __init__(self, model_folder="smpl", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Using the neutral model by default
        model_path = os.path.join(model_folder, "SMPL_NEUTRAL.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SMPL model not found at {model_path}. Run fetch_smpl.sh first.")
            
        # load SMPL model
        self.smpl = smplx.create(model_path='.', model_type='smpl', gender='neutral', 
                                 ext='pkl', use_pca=False, num_betas=10).to(self.device)
        self.smpl.eval()

    def get_measurements(self, betas):
        """
        Calculates basic measurements from SMPL betas.
        betas: shape (B, 10)
        Returns: tensor of shape (B, 3) contaning [height, upper_body_width, bmi_surrogate]
        """
        betas = betas.to(self.device)
        B = betas.shape[0]
        with torch.no_grad():
            output = self.smpl(
                betas=betas,
                global_orient=torch.zeros(B, 3, device=self.device),
                body_pose=torch.zeros(B, 69, device=self.device)
            )
            vertices = output.vertices # (B, 6890, 3)
            
            # Simple heuristic measurements:
            # 1. Height: max Y - min Y
            height = vertices[:, :, 1].max(dim=1)[0] - vertices[:, :, 1].min(dim=1)[0]
            
            # 2. Width (Shoulder/Upper body): max X - min X
            width = vertices[:, :, 0].max(dim=1)[0] - vertices[:, :, 0].min(dim=1)[0]
            
            # 3. Depth: max Z - min Z
            depth = vertices[:, :, 2].max(dim=1)[0] - vertices[:, :, 2].min(dim=1)[0]
            
            # 4. Volume heuristic (Bounding Box volume) ~ proportional to mass
            vol = height * width * depth
            
            # 5. BMI heuristic: mass / height^2 ~ vol / height^2 = (width * depth) / height
            bmi = vol / (height ** 2)
            
        return torch.stack([height, width, bmi], dim=1)

    def categorize_measurements(self, measurements):
        """
        Categorizes continuous measurements into 5 bins (Very Low, Low, Average, High, Very High)
        We use empirical thresholds based on the neutral model default size.
        """
        # Empirical mean/std can be better, but lets define simple static bins
        # Default neutral height is ~1.7, width ~0.5, bmi ~0.08
        
        # Bins: 0=Very Low, 1=Low, 2=Average, 3=High, 4=Very High
        def assign_bins(val, mean, std):
            bins = torch.zeros_like(val, dtype=torch.long)
            bins[val < mean - 1.5 * std] = 0
            bins[(val >= mean - 1.5 * std) & (val < mean - 0.5 * std)] = 1
            bins[(val >= mean - 0.5 * std) & (val < mean + 0.5 * std)] = 2
            bins[(val >= mean + 0.5 * std) & (val < mean + 1.5 * std)] = 3
            bins[val >= mean + 1.5 * std] = 4
            return bins
            
        height_bins = assign_bins(measurements[:, 0], mean=1.70, std=0.08)
        width_bins = assign_bins(measurements[:, 1], mean=0.45, std=0.05)
        bmi_bins = assign_bins(measurements[:, 2], mean=0.075, std=0.015)
        
        return torch.stack([height_bins, width_bins, bmi_bins], dim=1)

    def compute_reward(self, beta_hat, beta_gt):
        """
        Computes negative Cross Entropy (reward) between predictions and ground truth.
        """
        # (B, 3) continuous
        meas_hat = self.get_measurements(beta_hat)
        meas_gt = self.get_measurements(beta_gt)
        
        # Categorize
        cat_hat = self.categorize_measurements(meas_hat)
        cat_gt = self.categorize_measurements(meas_gt)
        
        # Instead of strict CE on probabilities (since we have discrete categories), 
        # we can use continuous distance, or -abs(cat_hat - cat_gt)
        # If the paper strictly used CE, they were treating predictions as a distribution over classes,
        # but since we parse text -> float -> discrete class deterministic,
        # the best reward is higher when categories match exactly or are close.
        # R = - MSE(cat_hat, cat_gt)
        
        diff = torch.abs(cat_hat - cat_gt).float()
        # Max diff is 4, so max reward is 0, min reward is -4
        reward = -diff.mean(dim=1) 
        return reward
