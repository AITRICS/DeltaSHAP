from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


class Masker:
    """
    A class used to compute performance drop using different masking methods.
    """

    def __init__(
        self,
        mask_method: str,
        top: int | float,
        balanced: bool,
        seed: int,
        absolutize: bool,
        aggregate_method: str = "mean",
        last_obs_only: bool = True,
    ):
        """
        Constructor

        Args:
            mask_method:
                The masking method. Must be "end", "std" or "end_fit". Note that "end" will
                mask the all subsequent observations if the observation is deemed important. "std"
                will mask all subsequent observations until the value changed by 1 STD of the
                feature distribution. "end_fit" is like "end", but with the code of the original
                FIT repository and started masking when t >= 10.
            top:
               If int, it will mask the top `top` observations of each time series. If float, it
               will mask the top `top*100` percent of the observations for all time series.
            balanced:
               If True, the number of masked will be balanced std. It is used for the STD-BAL
               masking method in the paper.
            seed:
               The random seed.
            absolutize:
               Indicate whether we should absolutize the feature importance for determining the
               top features.
            aggregate_method:
               For features that contain a window size, i.e. WinIT, this describes the aggregation
               method. It can be "absmax", "max", "mean".
        """
        # if mask_method not in ["end", "std", "end_fit", "mam"]:
        #     raise NotImplementedError(f"Mask method {mask_method} unrecognized")
        self.mask_method = mask_method
        self.top = top

        self.balanced = balanced
        self.seed = seed
        self.local = isinstance(top, int)
        min_time_dict = {"std": 1, "end": 1, "end_fit": 10, "mam": 1, "point": 1}
        self.min_time = min_time_dict[self.mask_method]
        self.importance_threshold = -1000
        self.absolutize = absolutize
        self.aggregate_method = aggregate_method
        self.last_obs_only = last_obs_only
        self.select_top = True  # Default to selecting top features
        self.mask_value = 0  # Add this line - default mask value is 0
        # assert not balanced or self.local and mask_method in ["std", "end"]

        self.start_masked_count = None
        self.all_masked_count = None
        self.feature_masked = None

    def get_name(self):
        return self.mask_method

    def mask(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        importances: Dict[int, np.ndarray],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Mask the input based on importance scores by carrying forward previous values.
        Ensures the number of masked elements doesn't exceed the number of observed points.
        Fully vectorized implementation without batch iteration.
        """
        if self.last_obs_only:
            return self.mask_last_timestep(x, mask, importances)

        new_xs = {}
        new_masks = {}
        importance_masks = {}
        self.start_masked_count = {}
        self.all_masked_count = {}

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        batch_size, seq_len, feat_dim = x.shape

        for cv, importance_scores in importances.items():
            # Reshape importance scores to (batch, features, time) to prioritize earlier times
            importance_scores = importance_scores.reshape(batch_size, feat_dim, -1)
            if self.absolutize:
                importance_scores = np.abs(importance_scores)

            new_x = x.copy()
            new_mask = mask.copy()

            # Calculate the number of observed points for each sample in the batch
            if not self.last_obs_only:
                observed_points_count = np.sum(mask, axis=(1, 2))
            else:
                timewise_count = np.sum(mask, axis=2)
                timewise_idx = timewise_count != 0
                has_nonzero = timewise_idx.any(axis=1)
                flip_idx = np.argmax(timewise_idx[:, ::-1], axis=1)
                last_idx = (timewise_idx.shape[1] - 1) - flip_idx
                last_idx[~has_nonzero] = 0
                observed_points_count = timewise_count[
                    np.arange(mask.shape[0]), last_idx
                ]

            # Initialize importance mask
            importance_mask = np.zeros((batch_size, seq_len, feat_dim), dtype=bool)

            # Reshape importance scores for easier processing
            if not self.last_obs_only:
                batch_scores = importance_scores.reshape(batch_size, -1)
            else:
                batch_scores = importance_scores

            # Calculate max elements to mask for each batch item
            max_to_mask = np.minimum(
                (batch_scores.shape[-1] * self.top // 100),
                observed_points_count.astype(int),
            )

            # Create a mask of valid indices for each batch item
            valid_mask = np.zeros_like(batch_scores, dtype=bool)

            # Get sorted indices for each batch element (either ascending or descending)
            if self.select_top:
                # Descending order (large values first)
                sorted_indices = np.argsort(-batch_scores, axis=-1)
            else:
                # Ascending order (small values first)
                sorted_indices = np.argsort(batch_scores, axis=-1)

            # For each batch item, mark the top/bottom indices as valid based on max_to_mask
            for b in range(batch_size):
                if not self.last_obs_only:
                    valid_mask[b, sorted_indices[b, : max_to_mask[b]]] = True
                else:
                    valid_mask[
                        b, last_idx[b], sorted_indices[b, last_idx[b], : max_to_mask[b]]
                    ] = True

            # Find the coordinates of all valid indices
            if not self.last_obs_only:
                valid_batch_indices, valid_flat_indices = np.where(valid_mask)

                # Convert flat indices to time and feature indices
                valid_time_indices = valid_flat_indices // feat_dim
                valid_feature_indices = valid_flat_indices % feat_dim
            else:
                valid_batch_indices, valid_time_indices, valid_feature_indices = (
                    np.where(valid_mask)
                )

            # Set all valid points in the importance mask
            importance_mask[
                valid_batch_indices, valid_time_indices, valid_feature_indices
            ] = True

            # Apply masking - we still need to handle carry-forward logic
            # For t=0, set to NORMAL_VALUE
            t0_mask = valid_time_indices == 0
            if np.any(t0_mask):
                b_indices = valid_batch_indices[t0_mask]
                f_indices = valid_feature_indices[t0_mask]

                for i in range(len(b_indices)):
                    b, f = b_indices[i], f_indices[i]
                    new_x[b, 0, f] = 0

            # For t>0, carry forward previous values
            for t in range(1, seq_len):
                t_mask = valid_time_indices == t
                if np.any(t_mask):
                    b_indices = valid_batch_indices[t_mask]
                    f_indices = valid_feature_indices[t_mask]

                    # Vectorized carry-forward
                    new_x[b_indices, t, f_indices] = new_x[b_indices, t - 1, f_indices]

            # Set all masked values in the mask to 0
            new_mask[valid_batch_indices, valid_time_indices, valid_feature_indices] = 0

            new_xs[cv] = new_x
            new_masks[cv] = new_mask
            importance_masks[cv] = importance_mask

            self.start_masked_count[cv] = importance_mask
            self.all_masked_count[cv] = importance_mask
        return new_xs, new_masks, importance_masks

    def mask_last_timestep(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        importances: Dict[int, np.ndarray],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        new_xs = {}
        new_masks = {}
        importance_masks = {}
        self.start_masked_count = {}
        self.all_masked_count = {}

        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        batch_size, seq_len, feat_dim = x.shape
        last_timestep = seq_len - 1

        for cv, importance_scores in importances.items():
            last_ts_scores = importance_scores[:, last_timestep, :]
            if self.absolutize:
                last_ts_scores = np.abs(last_ts_scores)

            new_x = x.copy()
            new_mask = mask.copy()
            importance_mask = np.zeros((batch_size, feat_dim))

            observed_mask = mask[:, last_timestep, :] > 0
            observed_points_count = np.sum(observed_mask, axis=1)

            for b in range(batch_size):
                instance_observed = observed_mask[b]
                num_observed = observed_points_count[b]
                if num_observed == 0:
                    continue  # Skip if no observed points

                instance_scores = last_ts_scores[b].copy()
                if self.select_top:
                    max_to_mask = np.ceil(0.01 * self.top * num_observed).astype(int)
                    sorted_indices = np.argsort(-instance_scores)  # Descending order (highest first)
                    to_mask = sorted_indices[:max_to_mask]
                    importance_mask[b, to_mask] = 1
                else:
                    max_to_mask = np.ceil(0.01 * self.top * num_observed + (feat_dim - num_observed)).astype(int)
                    sorted_indices = np.argsort(instance_scores)  # Ascending order (lowest first)
                    to_mask = sorted_indices[:max_to_mask]
                    importance_mask[b, to_mask] = 1

            valid_batch_indices, valid_feature_indices = np.where(importance_mask)
            for i in range(len(valid_batch_indices)):
                b, f = valid_batch_indices[i], valid_feature_indices[i]
                if last_timestep == 0:
                    new_x[b, 0, f] = 0
                else:
                    new_x[b, last_timestep, f] = new_x[b, last_timestep - 1, f]
            new_mask[valid_batch_indices, last_timestep, valid_feature_indices] = 0
            full_importance_mask = np.zeros((batch_size, seq_len, feat_dim))
            full_importance_mask[:, last_timestep, :] = importance_mask

            new_xs[cv] = new_x
            new_masks[cv] = new_mask
            importance_masks[cv] = full_importance_mask

            self.start_masked_count[cv] = importance_mask
            self.all_masked_count[cv] = importance_mask
        return new_xs, new_masks, importance_masks

    def _generate_arg_sort(self, scores, randomize_ties=True):
        """
        Returns a list of coordinates that is the argument of the sorting. In descending order.
        If local is True, the list of coordinates will be sorted within each sample. i.e.,
        the first (num_features * num_times) coordinates would always correspond to the first
        sample, the next (num_features * num_times) coordinates would correspond to the second
        sample, etc.

        Args:
            scores:
               Importance scores of shape (num_samples, num_features, num_times)
            min_time:
               The minimum timesteps to sort.
            local:
               Indicates whether the sorting is local or global. i.e. along all axes, or just the
               feature and time axes.
            randomize_ties:
               If there is a tie in the scores (which happens very often in Dynamask), we randomly
               permute the coordinates across the tie.

        Returns:
            A array of shape (all_coordinate_length, 3), for each row is the coordinate.
        """
        truncated_scores = scores[:, :, self.min_time :]
        if self.local:
            flattened_scores = truncated_scores.reshape(scores.shape[0], -1)
            argsorted_ravel_local = np.argsort(flattened_scores)[:, ::-1]
            if randomize_ties:
                self._shuffle_ties(argsorted_ravel_local, flattened_scores)
            feature_index = argsorted_ravel_local // truncated_scores.shape[2]
            time_index = (
                argsorted_ravel_local % truncated_scores.shape[2]
            ) + self.min_time
            arange = (
                np.arange(scores.shape[0])
                .reshape(-1, 1)
                .repeat(feature_index.shape[1], 1)
            )
            coordinate_list = np.stack([arange, feature_index, time_index]).reshape(
                3, -1
            )  # (3, all_coordinate_length)
            return coordinate_list.transpose()
        else:
            flattened_scores = truncated_scores.ravel()
            argsorted_ravel_global = np.argsort(flattened_scores)[::-1]
            if randomize_ties:
                self._shuffle_ties_global(argsorted_ravel_global, flattened_scores)
            coordinate_list = np.stack(
                np.unravel_index(argsorted_ravel_global, truncated_scores.shape)
            )  # (3, all_coordinate_length)
            coordinate_list[2, :] += self.min_time
            return coordinate_list.transpose()

    def _shuffle_ties_global(self, argsorted_ravel_global, flattened_scores):
        sorted_scores = flattened_scores[argsorted_ravel_global]
        repeated = np.r_[False, sorted_scores[1:] == sorted_scores[:-1]]
        indices = np.where(np.diff(repeated))[0]
        if len(indices) % 2 == 1:
            indices = np.r_[indices, len(sorted_scores) - 1]
        indices = indices.reshape(-1, 2)
        indices[:, 1] += 1
        rng = np.random.default_rng(self.seed)
        for repeated_index in indices:
            from_index, to_index = repeated_index
            rng.shuffle(argsorted_ravel_global[from_index:to_index])

    def _shuffle_ties(self, argsorted_ravel_local, flattened_scores):
        sorted_scores = np.take_along_axis(
            flattened_scores, argsorted_ravel_local, axis=1
        )
        repeated = np.concatenate(
            [
                np.zeros((len(sorted_scores), 1), dtype=bool),
                sorted_scores[:, 1:] == sorted_scores[:, :-1],
            ],
            axis=1,
        )
        indices = np.where(np.diff(repeated, axis=1))
        previous_sample_id = -1
        left = -1
        rng = np.random.default_rng(self.seed)
        for sample_id, x in zip(*indices):
            if sample_id != previous_sample_id:
                # new sample seen
                if left != -1:
                    # still have right bound left.
                    right = sorted_scores.shape[1]
                    rng.shuffle(argsorted_ravel_local[previous_sample_id, left:right])
                left = x
            else:
                # same sample
                if left != -1:
                    rng.shuffle(argsorted_ravel_local[previous_sample_id, left : x + 1])
                    left = -1
                else:
                    left = x
            previous_sample_id = sample_id
        if left != -1:
            right = sorted_scores.shape[1]
            rng.shuffle(argsorted_ravel_local[previous_sample_id, left:right])

    def _carry_forward(self, timestep, time_series, mask):
        assert len(time_series.shape) == 1
        ts_length = time_series.shape[0]

        assert timestep != 0, "Carry forward is not defined at index 0"

        if mask == "std":
            threshold = np.std(time_series)
            old = time_series[timestep]
            segment = np.abs(time_series[timestep:] - old) > threshold
            over = np.where(segment)[0]
            new_timestep = ts_length if len(over) == 0 else over[0] + timestep
        elif mask == "end":
            new_timestep = ts_length
        else:
            raise NotImplementedError(
                f"Mask method {mask} not recognized for carry forward"
            )

        time_series[timestep:new_timestep] = time_series[timestep - 1]
        return new_timestep, time_series

    def get_name(self) -> str:
        if self.local:
            if self.balanced:
                return f"bal{int(self.top)}_{self.mask_method}_{self.aggregate_method}"
            return f"top{int(self.top)}_{self.mask_method}_{self.aggregate_method}"
        return (
            f"globaltop{int(self.top * 100)}_{self.mask_method}_{self.aggregate_method}"
        )
