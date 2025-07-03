import numpy as np
import torch

from deltashap.explainer.explainers import BaseExplainer, ExplainerConfig


class TIMINGExplainer(BaseExplainer):
    """
    The explainer for TIMING (Time Integrated Gradients) method.
    Generates random contiguous time segments with one dimension each.
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)
        self.explainer = None

    def _initialize_explainer(self) -> None:
        """No separate explainer initialization needed for TIMING"""
        pass

    def attribute(self, input) -> np.ndarray:
        self.setup_attribution()
        
        x, mask, timesteps = input
        x = x.to(self.device)
        baseline = self.create_baseline(x)
        
        # Default parameters from config or use defaults
        n_samples = self.config.n_samples
        num_segments = 50  # Default number of segments
        max_seg_len = None  # Use full sequence length by default
        min_seg_len = 10
        
        # Get additional parameters from config if available
        if self.config.additional_args:
            num_segments = self.config.additional_args.get("num_segments", num_segments)
            max_seg_len = self.config.additional_args.get("max_seg_len", max_seg_len)
            min_seg_len = self.config.additional_args.get("min_seg_len", min_seg_len)
        
        score = self._attribute_timing(
            inputs=x,
            baselines=baseline,
            targets=None,  # No specific target for now
            additional_forward_args=(mask, timesteps, False),  # Assuming return_all=False
            n_samples=n_samples,
            num_segments=num_segments,
            max_seg_len=max_seg_len,
            min_seg_len=min_seg_len
        )
        
        self.cleanup_attribution()
        return score.detach().cpu().numpy()

    def _attribute_timing(
        self,
        inputs: torch.Tensor,  # [B, T, D]
        baselines: torch.Tensor,  # [B, T, D]
        targets: torch.Tensor = None,  # [B]
        additional_forward_args = None,
        n_samples: int = 50,
        num_segments: int = 3,  # how many time segments (one dimension each) to fix per sample
        max_seg_len: int = None,  # optional maximum length for each time segment
        min_seg_len: int = 1,
    ):
        """
        Generates random contiguous time segments (each segment picks ONE random dimension).
        BUT crucially, each sample i uses the SAME random segments for the *entire batch*.

        Steps:
        1) Interpolate from baselines -> inputs using n_samples alpha steps
        2) For each sample i (i.e. alpha step), create `num_segments` random slices
            - each slice picks a single dimension, plus time range [t_start : t_end)
            - fix that dimension/time range for ALL batch items
        3) Forward pass & gather target logit => sum => compute gradients
        4) Multiply by (inputs - baselines), optionally scale by how often (t,d) was free
        """
        if inputs.shape != baselines.shape:
            raise ValueError("Inputs and baselines must have the same shape.")

        B, T, D = inputs.shape
        device = inputs.device

        mask, timesteps, return_all = additional_forward_args

        # -------------------------------------------------------
        # 1) Build interpolation from baseline -> inputs
        # -------------------------------------------------------
        alphas = torch.linspace(0, 1 - 1 / n_samples, n_samples, device=device).view(-1, 1, 1, 1)
        
        expanded_inputs = inputs.unsqueeze(0)
        expanded_baselines = baselines.unsqueeze(0)
        # Interpolate with batch-specific alphas
        noisy_inputs = expanded_baselines + alphas * (expanded_inputs - expanded_baselines)
        noise = torch.randn_like(noisy_inputs) * 1e-4
        noisy_inputs = noisy_inputs + noise
        
        if max_seg_len is None:
            max_seg_len = T

        # Generate batch-specific masks
        dims = torch.randint(0, D, (n_samples, B, num_segments), device=device)
        seg_lens = torch.randint(min_seg_len, max_seg_len+1, (n_samples, B, num_segments), device=device)
        
        t_starts = (torch.rand(n_samples, B, num_segments, device=device) * (T - seg_lens)).long()

        # Initialize mask
        time_mask = torch.ones_like(noisy_inputs)

        # Create indices tensor
        batch_indices = torch.arange(B, device=device)
        sample_indices = torch.arange(n_samples, device=device)

        # Create mask via scatter
        for s in range(num_segments):
            max_len = seg_lens[:,:,s].max()
            base_range = torch.arange(max_len, device=device)
            base_range = base_range.unsqueeze(0).unsqueeze(0)
            
            indices = t_starts[:,:,s].unsqueeze(-1) + base_range

            end_points = t_starts[:,:,s] + seg_lens[:,:,s]  # shape [n_samples, B]
            end_points = end_points.unsqueeze(-1)           # shape [n_samples, B, 1]

            valid_indices = (indices < end_points) & (indices < T)
            time_mask[sample_indices.view(-1,1,1), batch_indices.view(1,-1,1), indices * valid_indices, dims[:,:,s].unsqueeze(-1)] = 0

        # Combine masked inputs
        fixed_inputs = expanded_inputs.detach()
        masked_inputs = time_mask * noisy_inputs + (1 - time_mask) * fixed_inputs
        masked_inputs.requires_grad = True

        # -------------------------------------------------------
        # 3) Forward pass & gather target logits
        # -------------------------------------------------------
        predictions = self.base_model.predict(
            masked_inputs.view(-1, T, D),
            mask=mask.repeat(n_samples, 1, 1) if mask is not None else None,
            timesteps=timesteps.repeat(n_samples, 1) if timesteps is not None else None,
        )
        
        # Ensure shape => [n_samples, B, num_classes]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        predictions = predictions.view(n_samples, B, -1)

        # If targets not provided, use argmax of the original prediction
        if targets is None:
            with torch.no_grad():
                original_preds = self.base_model.predict(
                    inputs, 
                    mask=mask,
                    timesteps=timesteps
                )
                targets = original_preds.argmax(dim=-1) if original_preds.dim() > 1 else torch.zeros(B, device=device).long()

        # Gather only the target logit for each example
        gathered = predictions.gather(
            dim=2, index=targets.unsqueeze(0).unsqueeze(-1).expand(n_samples, B, 1)
        ).squeeze(-1)

        total_for_target = gathered.sum()
        
        grad = torch.autograd.grad(outputs=total_for_target, inputs=masked_inputs, retain_graph=True)[0]
        grad[time_mask == 0] = 0

        grads = grad.sum(dim=0)  # Proper Riemann sum
        final_attr = grads * (inputs - baselines) / time_mask.sum(dim=0)
            
        return final_attr

    def _get_base_name(self) -> str:
        return "TIMING"
