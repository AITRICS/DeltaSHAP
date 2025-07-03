import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import GradientShap
# from captum.attr import GradientShap, ShapleyValueSampling
from captum.attr._utils.attribution import PerturbationAttribution
from captum._utils.common import _format_additional_forward_args

from deltashap.explainer.explainers import BaseExplainer


class ScoreDeltaWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, last_data, last_mask, rest_data, rest_mask):
        data = torch.cat([rest_data, last_data], dim=1)
        mask = torch.cat([rest_mask, last_mask], dim=1)
        return self.base_model(data, mask)

    def predict(self, **kwargs):
        return F.sigmoid(self.forward(**kwargs))


class ShapleyValueSampling(PerturbationAttribution):
    def __init__(self, forward_func):
        PerturbationAttribution.__init__(self, forward_func)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        feature_mask=None,
        n_samples=25,
        normalize=True,
    ):
        if baselines is None:
            baselines = torch.zeros_like(inputs)
        additional_forward_args = _format_additional_forward_args(additional_forward_args)

        if feature_mask is None:
            feature_indices = torch.arange(inputs.shape[-1], device=inputs.device)
            feature_mask = feature_indices.expand(inputs.shape)

        with torch.no_grad():
            device = inputs.device
            
            total_features = feature_mask.max().item() + 1
            all_permutations = torch.arange(total_features, device=device).expand(n_samples, total_features).clone()
            
            for i in range(total_features - 1):
                j = torch.randint(i, total_features, (n_samples,), device=device)
                rows = torch.arange(n_samples, device=device)
                
                temp = all_permutations[rows, i].clone()
                all_permutations[rows, i] = all_permutations[rows, j].clone()
                all_permutations[rows, j] = temp

            feature_masks_expanded = torch.stack([(feature_mask == f) for f in range(total_features)]).to(device)
            feature_diff = inputs - baselines
            
            input_shape = baselines.shape
            cumulative_masks = torch.zeros((n_samples, total_features + 1, *input_shape), device=device)
            
            expanded_baseline = baselines.unsqueeze(0).expand(n_samples, *input_shape)
            cumulative_masks[:, 0] = expanded_baseline
            
            current_tensor = expanded_baseline.clone()
            
            last_mask = additional_forward_args[0]
            
            expanded_last_mask = torch.zeros_like(last_mask).unsqueeze(0).expand(n_samples, *last_mask.shape)
            cumulative_masks_for_last_mask = torch.zeros((n_samples, total_features + 1, *last_mask.shape), device=device)
            
            current_mask = expanded_last_mask.clone()
            
            # import pdb; pdb.set_trace()

            for pos in range(total_features):
                feat_indices = all_permutations[:, pos]
                
                current_tensor = current_tensor + feature_masks_expanded[feat_indices] * feature_diff.expand_as(current_tensor)
                cumulative_masks[:, pos + 1] = current_tensor
                
                current_mask = current_mask + feature_masks_expanded[feat_indices] * last_mask.expand_as(current_mask)
                cumulative_masks_for_last_mask[:, pos + 1] = current_mask
            
            seq_len = baselines.shape[1]
            feature_dim = baselines.shape[2]
            
            expanded_last_data = cumulative_masks.reshape(-1, seq_len, feature_dim)
            expanded_last_mask = cumulative_masks_for_last_mask.reshape(-1, *last_mask.shape[1:])

            rest_data = additional_forward_args[1]
            rest_mask = additional_forward_args[2]
            repeat_factor = expanded_last_data.shape[0] // rest_data.shape[0]
            expanded_rest_data = rest_data.repeat_interleave(repeat_factor, dim=0)
            expanded_rest_mask = rest_mask.repeat_interleave(repeat_factor, dim=0)
            forward_args = [expanded_last_data, expanded_last_mask, expanded_rest_data, expanded_rest_mask]
            all_outputs = self._run_forward(self.forward_func, forward_args, target)

            all_outputs = all_outputs.view(n_samples, total_features + 1, -1)
            marginal_contributions = all_outputs[:, 1:] - all_outputs[:, :-1]
            feature_masks = feature_masks_expanded[all_permutations]

            marginal_contributions = marginal_contributions.unsqueeze(-1).unsqueeze(-1)
            attribution_tensor = (marginal_contributions * feature_masks).sum(dim=(0, 1)) / n_samples

            if normalize:
                self.original_output = all_outputs[0, -1, :]
                self.baseline_output = all_outputs[0, 0, :]
                
                pred_diff = self.original_output - self.baseline_output
                attribution_sum = attribution_tensor.sum(dim=(1, 2))

                non_zero_mask = (attribution_sum != 0)
                scale_factor = torch.ones_like(attribution_sum)
                scale_factor[non_zero_mask] = pred_diff[non_zero_mask] / attribution_sum[non_zero_mask]
                scale_factor = scale_factor.view(-1, 1, 1)

                attribution_tensor = attribution_tensor * scale_factor
            return attribution_tensor

    def _run_forward(self, forward_func, args, target=None):
        result = forward_func(*args)
        if target is not None:
            return result[target]
        return result


class DeltaShapExplainer(BaseExplainer):
    def __init__(self, config, base_model, **kwargs):
        # import pdb; pdb.set_trace()
        super().__init__(config)
        self.base_model = ScoreDeltaWrapper(base_model)

    def _initialize_explainer(self) -> None:
        # self.explainer = GradientShap(self.base_model.predict)
        self.explainer = ShapleyValueSampling(self.base_model.predict)

    def attribute(self, input) -> np.ndarray:
        self.base_model.eval()
        self.base_model.zero_grad()

        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        data, mask, _ = input
        last_data, rest_data = data[:, -1, :].unsqueeze(1), data[:, :-1, :]
        last_mask, rest_mask = mask[:, -1, :].unsqueeze(1), mask[:, :-1, :]

        baseline = self.create_baseline(data)
        last_baseline = baseline[:, -1, :].unsqueeze(1)

        score = torch.zeros(*data.shape)
        attribution = self.explainer.attribute(
            last_data,
            baselines=last_baseline,
            additional_forward_args=(last_mask, rest_data, rest_mask),
            n_samples=self.config.n_samples,
            normalize=self.config.additional_args["normalize"],
        )
        score[:, -1, :] = attribution.squeeze(1)

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score.detach().cpu().numpy()

    def _get_base_name(self) -> str:
        return "DeltaSHAP"
