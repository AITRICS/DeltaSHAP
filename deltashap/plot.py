from __future__ import annotations

import os
import gc
import pathlib
from typing import Dict, List
import traceback
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontProperties

from deltashap.utils import aggregate_scores


class BoxPlotter:
    """
    A class for plotting various box plots as plotExampleBox in FIT repo.
    """

    def __init__(self, dataset, plot_path, num_to_plot=20, explainer_name=""):
        self.dataset = dataset
        self.plot_path = plot_path
        self.num_to_plot = num_to_plot
        self.explainer_name = explainer_name
        testset = list(dataset.test_loader.dataset)
        self.x_test = torch.stack([x[0] for x in testset]).cpu().numpy()
        self.y_test = torch.stack([x[1] for x in testset]).cpu().numpy()
        self.mask_test = torch.stack([x[2] for x in testset]).cpu().numpy()
        self.plot_path.mkdir(parents=True, exist_ok=True)

    def plot_combined_visualization(
        self,
        importances: Dict[int, np.ndarray],
        aggregate_method: str,
        x_test: np.ndarray,
        mask_test: np.ndarray,
        new_xs: Dict[int, np.ndarray],
        new_masks: Dict[int, np.ndarray],
        importance_masks: Dict[int, np.ndarray],
        orig_preds: Dict[int, np.ndarray],
        new_preds: Dict[int, np.ndarray],
        y_test: np.ndarray,
        mask_name: str,
    ) -> None:
        """Plot combined visualization for both original and masked data"""
        # Set memory-efficient backend
        plt.switch_backend("Agg")

        # Helper function to safely transpose arrays
        def safe_transpose(arr, axes=(0, 2, 1)):
            if arr is None:
                return None
            if arr.ndim == 3:  # Only transpose 3D arrays
                return np.transpose(arr, axes)
            return arr

        # Safely transpose all arrays
        importances = {cv: safe_transpose(importances[cv]) for cv in importances}
        x_test = safe_transpose(x_test)
        mask_test = safe_transpose(mask_test)
        new_xs = {cv: safe_transpose(new_xs[cv]) for cv in new_xs}
        new_masks = {cv: safe_transpose(new_masks[cv]) for cv in new_masks}
        importance_masks = {cv: safe_transpose(importance_masks[cv]) for cv in importance_masks}

        # optional: scale importances and predictions by 100 for better visualization
        importances = {cv: importances[cv] * 100 for cv in importances}
        orig_preds = {cv: orig_preds[cv] * 100 for cv in orig_preds}
        new_preds = {cv: new_preds[cv] * 100 for cv in new_preds}

        # Set up fonts
        font_path = "scripts/Times New Roman.ttf"
        TITLE_SIZE = 28
        SUBTITLE_SIZE = 24
        LABEL_SIZE = 18

        times_new_roman_title = FontProperties(fname=font_path, size=TITLE_SIZE)
        times_new_roman_subtitle = FontProperties(fname=font_path, size=SUBTITLE_SIZE)
        times_new_roman_label = FontProperties(fname=font_path, size=LABEL_SIZE)

        plt.rcParams["figure.max_open_warning"] = False

        # Use self.plot_path directly as it's set from the command line argument
        base_path = pathlib.Path(str(self.plot_path))  # Convert to Path object
        for cv, importance_unaggregated in importances.items():
            importance_scores = aggregate_scores(
                importance_unaggregated, aggregate_method
            )

            for i in range(self.num_to_plot):
                try:
                    # Get label value
                    label_val = float(y_test[i])
                    if isinstance(orig_preds[cv], (int, float, np.number)):
                        orig_pred_val = float(orig_preds[cv])
                    else:
                        orig_pred_val = float(orig_preds[cv][i])
                    if isinstance(new_preds[cv], (int, float, np.number)):
                        new_pred_val = float(new_preds[cv])
                    else:
                        new_pred_val = float(new_preds[cv][i])

                    # Create figure with 3x2 subplots
                    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

                    # Main title with Times New Roman font
                    title_text = f"Sample {i} (Label: {'Positive' if label_val == 1 else 'Negative'}, Original Prediction: {orig_pred_val:.3f}, Masked Prediction: {new_pred_val:.3f})"
                    fig.suptitle(title_text, fontproperties=times_new_roman_title, y=0.95)

                    # Function to add calibrated indices with Times New Roman font
                    def add_calibrated_indices(ax, data):
                        num_rows, num_cols = data.shape
                        ax.set_yticks(range(num_rows))
                        ax.set_yticklabels(
                            range(num_rows),
                            fontproperties=times_new_roman_label,
                            fontsize=10,
                        )
                        ax.set_xticks(range(num_cols))
                        ax.set_xticklabels(
                            range(num_cols),
                            fontproperties=times_new_roman_label,
                            fontsize=10,
                            # rotation=90,
                        )
                        ax.tick_params(axis="both", which="major", length=0)

                    # Plot configurations
                    plot_configs = [
                        {
                            "data": (
                                x_test[i].cpu().numpy()
                                if torch.is_tensor(x_test[i])
                                else x_test[i]
                            ),
                            "title": "Original Input",
                            "cmap": "Reds",
                            "pos": (0, 0),
                            "symmetric": True,
                            "vmin": 0,
                        },
                        {
                            "data": (
                                new_xs[cv][i].cpu().numpy()
                                if torch.is_tensor(new_xs[cv][i])
                                else new_xs[cv][i]
                            ),
                            "title": "Masked Input",
                            "cmap": "Reds",
                            "pos": (0, 1),
                            "symmetric": True,
                            "vmin": 0,
                        },
                        {
                            "data": (
                                mask_test[i].cpu().numpy()
                                if torch.is_tensor(mask_test[i])
                                else mask_test[i]
                            ),
                            "title": "Original Mask",
                            "cmap": "Reds",
                            "pos": (1, 0),
                            "symmetric": False,
                            "vmin": 0,
                            "vmax": 1,
                        },
                        {
                            "data": (
                                new_masks[cv][i].cpu().numpy()
                                if torch.is_tensor(new_masks[cv][i])
                                else new_masks[cv][i]
                            ),
                            "title": "New Mask",
                            "cmap": "Reds",
                            "pos": (1, 1),
                            "symmetric": False,
                            "vmin": 0,
                            "vmax": 1,
                        },
                        {
                            "data": (
                                importance_unaggregated[i].cpu().numpy()
                                if torch.is_tensor(importance_unaggregated[i])
                                else importance_unaggregated[i]
                            ),
                            "title": "Feature Attributions",
                            "cmap": "Blues",
                            "pos": (2, 0),
                            "symmetric": False,
                        },
                        {
                            "data": (
                                importance_masks[cv][i].cpu().numpy()
                                if torch.is_tensor(importance_masks[cv][i])
                                else importance_masks[cv][i]
                            ),
                            "title": "Removed Points",
                            "cmap": "gray_r",
                            "pos": (2, 1),
                            "symmetric": False,
                            "vmin": 0,
                            "vmax": 1,
                        },
                    ]

                    # Create all subplots
                    for config in plot_configs:
                        ax = axes[config["pos"]]
                        data = np.asarray(config["data"], dtype=np.float32)

                        # Set vmin/vmax based on config
                        if config["symmetric"]:
                            abs_max = float(np.max(np.abs(data)))
                            vmin = config.get("vmin", -abs_max)
                            vmax = abs_max
                        else:
                            vmin = config.get("vmin", None)
                            vmax = config.get("vmax", None)

                        im = ax.imshow(
                            data,
                            cmap=config["cmap"],
                            interpolation="nearest",
                            aspect="auto",
                            vmin=vmin,
                            vmax=vmax,
                        )

                        # Add calibrated indices
                        add_calibrated_indices(ax, data)

                        # Add grid for cells
                        num_rows, num_cols = data.shape
                        ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
                        ax.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
                        ax.grid(
                            which="minor",
                            color="gray",
                            linestyle="-",
                            linewidth=0.5,
                            alpha=0.3,
                        )

                        ax.set_title(
                            config["title"],
                            fontproperties=times_new_roman_subtitle,
                            pad=10,
                        )

                        cbar = plt.colorbar(im, ax=ax)
                        cbar.ax.tick_params(labelsize=LABEL_SIZE)
                        for label in cbar.ax.get_yticklabels():
                            label.set_fontproperties(times_new_roman_label)

                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.5)

                    plt.tight_layout(rect=[0, 0, 1, 0.95])

                    # Create save directory with proper path handling
                    save_dir = (
                        base_path
                        / self.explainer_name
                        / aggregate_method
                        / mask_name
                        / f"cv_{cv}"
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"sample_{i}.png"

                    logging.info(f"Saving plot to: {save_path}")  # Debug print

                    plt.savefig(
                        str(save_path),
                        bbox_inches="tight",
                        dpi=200,
                    )

                except Exception as e:
                    print(f"Error plotting sample {i}: {str(e)}")
                    traceback.print_exc()

                finally:
                    plt.close("all")

                if i % 5 == 0:
                    gc.collect()

    def plot_importances(
        self, importances: Dict[int, np.ndarray], aggregate_method: str
    ) -> None:
        """
        Plot the importance for all cv and save it to files.

        Args:
            importances:
                A dictionary from CV to feature importances.
            aggregate_method:
                The aggregation method for WinIT.
        """
        for cv, importance_unaggregated in importances.items():
            importance_scores = aggregate_scores(
                importance_unaggregated, aggregate_method
            )
            for i in range(self.num_to_plot):
                prefix = (
                    f"{self.explainer_name}_{aggregate_method}_cv_{cv}_attributions"
                )
                self._plot_single_visualization(
                    sample_idx=i,
                    explainer=f"{self.explainer_name}_{aggregate_method}_cv_{cv}_attributions",
                    pred=importance_scores[i],
                    label=self.y_test[i],
                    attr_map=importance_unaggregated[i],
                    save_dir=str(self.plot_path),
                    timewise=False,
                )

    def plot_ground_truth_importances(self, ground_truth_importance: np.ndarray):
        """
        Plot the ground truth importances for all cv and save it to files.

        Args:
            ground_truth_importance:
                The ground truth importance.
        """
        prefix = "ground_truth_attributions"
        for i in range(self.num_to_plot):
            self._plot_single_visualization(
                sample_idx=i,
                explainer=f"{self.explainer_name}_{prefix}_cv_{i}_attributions",
                pred=ground_truth_importance[i],
                label=self.y_test[i],
                attr_map=ground_truth_importance[i],
                save_dir=str(self.plot_path),
                timewise=False,
            )

    def plot_labels(self):
        """
        Plot the labels and save it to files. If the label is one-dimensional, skip the plotting.
        """
        if self.y_test.ndim != 2:
            return
        for i in range(self.num_to_plot):
            self._plot_single_visualization(
                sample_idx=i,
                explainer="labels",
                pred=self.y_test[i],
                label=self.y_test[i],
                attr_map=self.y_test[i],
                save_dir=str(self.plot_path),
                timewise=False,
            )

    def plot_x_pred(
        self,
        x: np.ndarray | Dict[int, np.ndarray] | None,
        preds: Dict[int, np.ndarray],
        prefix: str = None,
    ):
        """
        Plot the data and the corresponding predictions and save it to files.

        Args:
            x:
                The data. Can be a numpy array of a dictionary of CV to numpy arrays.
            preds:
                The predictions. A dictionary of CV to numpy arrays. (In case of only 1 data,
                the predictions are the predictions of the model of the corresponding CV
                on the same data.
            prefix:
                The prefix of the name of the files to be saved.
        """
        if x is None:
            x = self.x_test

        if isinstance(x, np.ndarray):
            for i in range(self.num_to_plot):
                filename_prefix = prefix if prefix is not None else "data"
                self._plot_single_visualization(
                    sample_idx=i,
                    explainer=f"{self.explainer_name}_{filename_prefix}_cv_{i}",
                    pred=x[i],
                    label=self.y_test[i],
                    attr_map=x[i],
                    save_dir=str(self.plot_path),
                    timewise=False,
                )
        elif isinstance(x, dict):
            for cv, xin in x.items():
                for i in range(self.num_to_plot):
                    filename_prefix = (
                        f"data_cv_{cv}" if prefix is None else f"{prefix}_cv_{cv}"
                    )
                    self._plot_single_visualization(
                        sample_idx=i,
                        explainer=f"{self.explainer_name}_{filename_prefix}_cv_{i}",
                        pred=xin[i],
                        label=self.y_test[i],
                        attr_map=xin[i],
                        save_dir=str(self.plot_path),
                        timewise=False,
                    )

        for cv, pred in preds.items():
            for i in range(self.num_to_plot):
                filename_prefix = (
                    f"preds_cv_{cv}" if prefix is None else f"{prefix}_cv_{cv}"
                )
                self._plot_single_visualization(
                    sample_idx=i,
                    explainer=f"{self.explainer_name}_{filename_prefix}_cv_{i}",
                    pred=pred[i],
                    label=self.y_test[i],
                    attr_map=pred[i],
                    save_dir=str(self.plot_path),
                    timewise=False,
                )


def visualize_per_sample_attribution(
    attributions,
    predictions,
    carry_forward_predictions,
    mask_test,
    inputs=None,
    carry_forward_inputs=None,
    feature_names=None,
    save_dir="plots/attributions",
    visualize_last_timestep=True,
):
    """
    Visualize attribution scores for prediction differences.
    
    Args:
        attributions: Attribution scores
        predictions: Model predictions
        carry_forward_predictions: Baseline predictions
        mask_test: Masks indicating observed features
        inputs: Current input values (optional)
        carry_forward_inputs: Baseline input values (optional)
        feature_names: Names of features
        save_dir: Directory to save plots
        visualize_last_timestep: Whether to visualize only the last timestep
    """
    # Set up fonts
    font_path = "scripts/Times New Roman.ttf"
    TITLE_SIZE = 20
    LABEL_SIZE = 16

    times_new_roman_title = FontProperties(fname=font_path, size=TITLE_SIZE)
    times_new_roman_label = FontProperties(fname=font_path, size=LABEL_SIZE)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(attributions.shape[2])]

    batch_size = attributions.shape[0]
    num_timesteps = attributions.shape[1]
    os.makedirs(save_dir, exist_ok=True)

    attributions = 100 * attributions
    predictions = 100 * predictions
    carry_forward_predictions = 100 * carry_forward_predictions

    # For each sample
    for sample_idx in range(min(batch_size, len(predictions))):
        # Determine which timesteps to visualize
        if visualize_last_timestep:
            timesteps_to_viz = [num_timesteps - 1]  # Just the last timestep
        else:
            timesteps_to_viz = range(1, num_timesteps)  # All timesteps except the first

        # For selected timesteps
        for t in timesteps_to_viz:
            # Get the prediction values safely
            orig_pred = predictions[sample_idx]
            base_pred = carry_forward_predictions[sample_idx]
            curr_attrs = attributions[sample_idx, t, :]

            order = np.argsort(np.abs(curr_attrs))
            active_mask = np.abs(curr_attrs[order]) > 0
            order = order[active_mask]
            num_features = len(order)

            # Skip plotting if there are no features to show
            if num_features == 0:
                continue

            # Set a maximum figure height to prevent excessively tall figures
            max_fig_height = 20  # Maximum height in inches
            row_height = 0.5
            base_height = max(3, num_features * row_height)  # Calculate base height

            # Apply scaling but cap at maximum height
            fig_height = min(base_height * 1.5, max_fig_height)
            plt.figure(figsize=(12, fig_height))

            # Plot variables
            rng = range(num_features)
            prev_loc = base_pred
            loc = base_pred

            # Get figure parameters for text positioning
            fig = plt.gcf()
            ax = plt.gca()
            renderer = fig.canvas.get_renderer()
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width = bbox.width
            xlen = orig_pred - base_pred
            bbox_to_xscale = xlen / width if abs(xlen) > 1e-6 else 1.0

            for i, idx in enumerate(order):
                sval = float(curr_attrs[idx])  # Convert to float to ensure it's a scalar
                color = "#E9608E" if sval >= 0 else "#4AADDE"

                # Draw vertical line connecting previous bar end to current bar start
                if i > 0:
                    plt.plot(
                        [prev_loc, prev_loc],
                        [rng[i] - 1 - 0.4, rng[i] + 0.4],
                        color="gray",
                        linestyle="--",
                        linewidth=0.5,
                        zorder=-1,
                    )

                # Create rectangle instead of arrow
                height = 0.8 if num_features > 1 else 0.4
                rect = plt.Rectangle(
                    (loc, rng[i] - height / 2),  # (x, y)
                    sval,  # width
                    height,  # height
                    facecolor=color,
                    edgecolor="none",
                )

                ax.add_patch(rect)

                # Add value label (without input values)
                txt_obj = plt.text(
                    loc + sval / 2,
                    rng[i],
                    f"{sval:+.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                    fontproperties=times_new_roman_label,
                )

                # Handle text overflow
                text_bbox = txt_obj.get_window_extent(renderer=renderer)
                rect_bbox = rect.get_window_extent(renderer=renderer)
                if text_bbox.width > rect_bbox.width:
                    txt_obj.remove()
                    if sval >= 0:
                        plt.text(
                            loc + sval + bbox_to_xscale * 0.1,
                            rng[i],
                            f"{sval:+.2f}",
                            ha="left",
                            va="center",
                            color="black",
                            fontsize=12,
                            fontproperties=times_new_roman_label,
                        )
                    else:
                        plt.text(
                            loc + sval - bbox_to_xscale * 0.1,
                            rng[i],
                            f"{sval:+.2f}",
                            ha="right",
                            va="center",
                            color="black",
                            fontsize=12,
                            fontproperties=times_new_roman_label,
                        )

                # Update locations for next iteration
                prev_loc = loc + sval
                loc += sval

            # Add special connecting line for base value
            plt.plot(
                [orig_pred, orig_pred],
                [rng[-1] - 0.4, rng[-1] + 0.4],
                color="gray",
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )

            # Customize plot
            plt.gca().set_yticks(range(num_features))
            
            # Create enhanced y-tick labels with unnormalized feature values
            y_tick_labels = []

            for idx in order:  # Only create labels for the features we're actually showing
                feature_name = feature_names[idx]
                if inputs is not None and carry_forward_inputs is not None:
                    # Get normalized values
                    curr_input_norm = inputs[sample_idx, -1, idx].item()
                    base_input_norm = carry_forward_inputs[sample_idx, -1, idx].item()
                    
                    # Get mean and std for unnormalization
                    # feature_idx = FEATURE_MAP.get(feature_name, idx)
                    mean = FEATURE_MEAN[feature_name] if feature_name in FEATURE_MEAN.keys() else 0
                    std = FEATURE_STD[feature_name] if feature_name in FEATURE_STD.keys() else 1
                    
                    # Unnormalize values
                    curr_input_unnorm = curr_input_norm * std + mean
                    base_input_unnorm = base_input_norm * std + mean
                    
                    # Format based on the specific feature requirements
                    if "Glascow" in feature_name or feature_name in [
                        "Hours", "Diastolic Blood Pressure", "Systolic Blood Pressure", 
                        "Mean Arterial Pressure", "Heart Rate", "Glucose", 
                        "Oxygen Saturation", "Respiratory Rate", "Capillary Refill Rate"
                    ]:
                        # Integer format
                        feature_name = f"{feature_name} ({int(base_input_unnorm)} → {int(curr_input_unnorm)})"
                    elif feature_name in ["Height", "Weight", "pH", "Temperature"]:
                        # 2 decimal places
                        feature_name = f"{feature_name} ({base_input_unnorm:.2f} → {curr_input_unnorm:.2f})"
                    else:
                        # Default to 1 decimal place for any other features
                        feature_name = f"{feature_name} ({base_input_unnorm:.1f} → {curr_input_unnorm:.1f})"
                
                y_tick_labels.append(feature_name)
            
            plt.gca().set_yticklabels(y_tick_labels, fontproperties=times_new_roman_label)

            # Set x-axis ticks font
            ax.tick_params(axis="x", labelsize=12)
            for tick in ax.get_xticklabels():
                tick.set_fontproperties(times_new_roman_label)

            # Add reference lines and styling with labels
            # Bottom label for "Previous"
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks([base_pred])
            ax2.set_xticklabels(
                [f"Base Score: {base_pred:.2f}"], fontproperties=times_new_roman_label
            )
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")
            ax2.tick_params(axis="x", pad=20)
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)

            # Top label for "Current"
            ax3 = ax.twiny()
            ax3.set_xlim(ax.get_xlim())
            ax3.set_xticks([orig_pred])
            ax3.set_xticklabels(
                [f"Current Score: {orig_pred:.2f}"],
                fontproperties=times_new_roman_label,
            )
            ax3.xaxis.set_ticks_position("top")
            ax3.xaxis.set_label_position("top")
            ax3.tick_params(axis="x", pad=10)
            ax3.spines["right"].set_visible(False)
            ax3.spines["top"].set_visible(False)
            ax3.spines["left"].set_visible(False)
            ax3.spines["bottom"].set_visible(False)

            y_max = (rng[0] + 0.4 - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
            plt.axvline(base_pred, 0, y_max, color="gray", linestyle="--", linewidth=0.5, zorder=-1)
            plt.axvline(
                orig_pred,
                0,
                1,
                color="gray",
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )

            for i in range(num_features):
                plt.axhline(i, color="gray", lw=0.5, dashes=(1, 5), zorder=-1)

            # Clean up axes
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Find the last time where any feature was observed (mask sum > 0)
            last_observed_time = -1
            mask_numpy = mask_test[sample_idx]  # Already transposed above

            for prev_t in range(t - 1, -1, -1):
                # Sum across features for the timestep
                if prev_t < mask_numpy.shape[0] and np.sum(mask_numpy[prev_t, :]) > 0:
                    last_observed_time = prev_t
                    break

            # Format title with the last observed time
            title_time = last_observed_time if last_observed_time != -1 else t - 1
            plt.title(
                f"Score Evolution: {base_pred:.2f} → {orig_pred:.2f} (Difference = {orig_pred - base_pred:+.2f})",
                fontproperties=times_new_roman_title,
            )

            # Special filename for last timestep
            if t == num_timesteps - 1:
                save_path = os.path.join(save_dir, f"sample_{sample_idx:03d}_last_timestep.png")
            else:
                save_path = os.path.join(save_dir, f"sample_{sample_idx:03d}_{t:03d}.png")
                
            logging.info(f"Saving plot to: {save_path}")
            plt.savefig(
                save_path,
                bbox_inches="tight",
                dpi=300,
            )
            plt.close("all")


def visualize_temporal_evolution(
    feature_values,
    masks,
    predictions,
    feature_names=None,
    save_dir="plots/temporal",
):
    # Convert tensors to numpy and transpose from (B,F,T) to (B,T,F)
    if torch.is_tensor(feature_values):
        feature_values = feature_values.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    
    # Transpose from (B,F,T) to (B,T,F)
    feature_values = np.transpose(feature_values, (0, 2, 1))
    masks = np.transpose(masks, (0, 2, 1))
    predictions = np.round(predictions * 100, 2)

    # Set up fonts with increased sizes
    font_path = "scripts/Times New Roman.ttf"
    TITLE_SIZE = 48
    LEGEND_SIZE = 18
    LABEL_SIZE = 24
    FEATURE_LEGEND_SIZE = 24

    times_new_roman_title = FontProperties(fname=font_path, size=TITLE_SIZE)
    times_new_roman_legend = FontProperties(fname=font_path, size=LEGEND_SIZE)
    times_new_roman_label = FontProperties(fname=font_path, size=LABEL_SIZE)
    times_new_roman_feature_legend = FontProperties(
        fname=font_path, size=FEATURE_LEGEND_SIZE
    )

    # Color palette
    # base_colors = [
    #     "#000000",  # Black (PULSE)
    #     "#1A1A1A",  # Dark Gray (RESP)
    #     "#D62728",  # Bright Red (SBP)
    #     "#FF9896",  # Light Red (DBP)
    #     "#2CA02C",  # Dark Green (TEMP)
    #     "#98DF8A",  # Light Green (SpO2)
    #     "#1F77B4",  # Dark Blue (GCS)
    #     "#AEC7E8",  # Light Blue (BILIRUBIN)
    #     "#FF7F0E",  # Dark Orange (LACTATE)
    #     "#FFBB78",  # Light Orange (CREATININE)
    #     "#9467BD",  # Dark Purple (PLATELET)
    #     "#C5B0D5",  # Light Purple (APH)
    #     "#8C564B",  # Dark Brown (SODIUM)
    #     "#C49C94",  # Light Brown (POTASSIUM)
    #     "#E377C2",  # Dark Pink (HEMATOCRIT)
    #     "#F7B6D2",  # Light Pink (WBC)
    #     "#7F7F7F",  # Dark Gray (HCO3)
    #     "#C7C7C7",  # Light Gray (CRP)
    # ]

    base_colors = [
        # Base colors
        "#000000",  # Black (Height)
        "#1F77B4",  # Blue (Hours)
        "#FF7F0E",  # Orange (DBP)
        "#2CA02C",  # Green (FiO2)
        "#D62728",  # Red (Glucose)
        "#9467BD",  # Purple (HR)
        "#8C564B",  # Brown (MBP)
        "#E377C2",  # Pink (SpO2)
        "#7F7F7F",  # Gray (RR)
        "#BCBD22",  # Olive (SBP)
        "#17BECF",  # Cyan (Temp)
        "#AEC7E8",  # Light Blue (Weight)
        "#FFBB78",  # Light Orange (pH)
        
        # Extended palette for categorical features
        "#98DF8A",  # Light Green (Cap. Refill)
        
        # GCS Eye - Blue family
        "#1F77B4",  # Blue
        "#6BAED6",  # Medium Blue
        "#9ECAE1",  # Light Blue
        "#C6DBEF",  # Very Light Blue
        
        # GCS Motor - Red family
        "#D62728",  # Red
        "#E6550D",  # Dark Orange
        "#FD8D3C",  # Medium Orange
        "#FDAE6B",  # Light Orange
        "#FDD0A2",  # Very Light Orange
        "#FEE6CE",  # Pale Orange
        
        # GCS Total - Green family
        "#2CA02C",  # Green
        "#31A354",  # Darker Green
        "#74C476",  # Medium Green
        "#A1D99B",  # Light Green
        "#C7E9C0",  # Very Light Green
        "#41AB5D",  # Forrest Green
        "#78C679",  # Grass Green
        "#ADDD8E",  # Mint Green
        "#D9F0A3",  # Pale Green
        "#F7FCB9",  # Lightest Green
        "#FFFFCC",  # Very Pale Yellow
        "#EDF8B1",  # Pale Yellow
        "#C7E9B4",  # Very Pale Green
        
        # GCS Verbal - Purple family
        "#9467BD",  # Purple
        "#8C6BB1",  # Medium Purple
        "#9E9AC8",  # Light Purple
        "#DADAEB",  # Very Light Purple
        "#F2F0F7"   # Pale Purple
    ]

    MAES_COLOR = "#1F4D78"  # Navy blue for MAES plot

    os.makedirs(save_dir, exist_ok=True)

    for sample_idx in range(feature_values.shape[0]):
        sample_dir = os.path.join(save_dir, f"sample_{sample_idx:03d}")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        for end_time in range(1, feature_values.shape[2]):
            print(f"{sample_idx=} {end_time=}")
            print(f"{predictions[end_time, sample_idx]=} {predictions[end_time - 1, sample_idx]=}")
            if (
                predictions[end_time, sample_idx]
                == predictions[end_time - 1, sample_idx]
            ):
                continue

            # Create figure with wider aspect ratio for side-by-side plots
            fig = plt.figure(figsize=(36, 10))  # Keep overall figure size

            # Create GridSpec with side-by-side layout and less spacing
            gs = plt.GridSpec(
                1, 2, width_ratios=[1, 1], wspace=0.1
            )  # Reduced wspace from 0.3 to 0.2

            # Create subplots with GridSpec
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            time_points = np.arange(end_time + 1)  # Only up to current timestep
            valid_timesteps = np.zeros(len(time_points), dtype=bool)

            # Collect valid timesteps up to current time
            for feat_idx, feature in enumerate(feature_names):
                feat_idx_data = FEATURE_MAP[feature]
                # Access using transposed indices (t,f) instead of (f,t)
                mask_data = masks[sample_idx, :end_time + 1, feat_idx_data]
                valid_timesteps = valid_timesteps | (mask_data > 0)

            # Plot MAES predictions up to current time
            valid_time_points = time_points[valid_timesteps]
            valid_pred_data = predictions[: end_time + 1, sample_idx][valid_timesteps]

            if len(valid_time_points) > 0:
                ax1.plot(
                    valid_time_points,
                    valid_pred_data,
                    color=MAES_COLOR,
                    linewidth=2,
                    marker="o",
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    zorder=2,
                    label="MAES",
                )

            # MAES plot styling
            ax1.set_title("MAES", fontproperties=times_new_roman_title, y=1.2)
            ax1.set_xlabel("Timestep", fontproperties=times_new_roman_label)
            ax1.grid(True, linestyle="--", alpha=0.3, color="gray")
            ax1.set_axisbelow(True)
            ax1.set_ylim(-10, 110)
            ax1.set_xlim(-1, 49)
            ax1.set_xticks(valid_time_points)
            ax1.set_xticklabels(valid_time_points, fontproperties=times_new_roman_label)

            # Plot features up to current time
            lines = []
            labels = []
            for feat_idx, feature in enumerate(feature_names):
                feat_idx_data = FEATURE_MAP[feature]
                # Access using transposed indices (t,f) instead of (f,t)
                mask_data = masks[sample_idx, :end_time + 1, feat_idx_data]
                feature_data = feature_values[sample_idx, :end_time + 1, feat_idx_data]

                valid_points = mask_data > 0
                if np.any(valid_points):
                    line = ax2.plot(
                        time_points[valid_points],
                        feature_data[valid_points],
                        label=feature,
                        marker="o",
                        markersize=6,
                        linewidth=2,
                        color=base_colors[feat_idx],
                        markeredgecolor="white",
                        markeredgewidth=1,
                    )[0]
                    lines.append(line)
                    labels.append(feature)

            # Features plot styling
            ax2.set_title(
                "Features", fontproperties=times_new_roman_title, y=1.3
            )  # Increased y from 1.1 to 1.3
            ax2.set_xlabel("Timestep", fontproperties=times_new_roman_label)
            ax2.grid(True, linestyle="--", alpha=0.3, color="gray")
            ax2.set_axisbelow(True)
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_xlim(-1, 49)
            ax2.set_xticks(valid_time_points)
            ax2.set_xticklabels(valid_time_points, fontproperties=times_new_roman_label)

            # Add legend
            if lines:
                ax2.legend(
                    lines,
                    labels,
                    prop=times_new_roman_feature_legend,
                    bbox_to_anchor=(0.5, 1.3),  # Keep legend position below title
                    loc="upper center",
                    ncol=6,
                    frameon=False,
                )

            # Set consistent styling for both axes
            for ax in [ax1, ax2]:
                ax.tick_params(axis="both", which="major", labelsize=LEGEND_SIZE)
                for tick in ax.get_xticklabels():
                    tick.set_fontproperties(times_new_roman_label)
                for tick in ax.get_yticklabels():
                    tick.set_fontproperties(times_new_roman_legend)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # Adjust positions of subplots for side-by-side layout
            pos1 = ax1.get_position()
            pos2 = ax2.get_position()

            # Make each subplot wider by adjusting position and width
            ax1.set_position(
                [pos1.x0 - 0.08, pos1.y0, pos1.width + 0.1, pos1.height]
            )  # Increased width and adjusted position
            ax2.set_position(
                [pos2.x0 + 0.02, pos2.y0, pos2.width + 0.1, pos2.height]
            )  # Increased width and adjusted position

            save_path = os.path.join(
                sample_dir,
                f"sample_{sample_idx:03d}_temporal_evolution_upto_{end_time:03d}.png",
            )
            print(f"Saving temporal plot to: {save_path}")
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()


# import os
# import pickle
# import numpy as np
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime


# def load_attribution_data(pid):
#     """Load attribution data for a given patient ID from shap directory"""
#     shap_dir = (
#         "/nfs/potter/shared/for_andy/score_delta_model_result/score_delta_shap_pos"
#     )
#     shap_file = f"{pid}.pkl"
#     shap_path = os.path.join(shap_dir, shap_file)

#     try:
#         with open(shap_path, "rb") as f:
#             attribution_data = pickle.load(f)
#         return attribution_data
#     except Exception as e:
#         print(f"Error loading attribution data for {pid}: {e}")
#         return None


# def process_ts_data(all_data):
#     """
#     Process time series data from last_v/ti/ty and rest_v/ti/ty.
#     Returns integrated data for the last 49 timesteps using forward/backward fill.
#     """
#     # Extract components
#     last_v = np.array(all_data["last_v"])
#     last_ti = np.array(all_data["last_ti"])
#     last_ty = np.array(all_data["last_ty"])

#     rest_v = np.array(all_data["rest_v"])
#     rest_ti = np.array(all_data["rest_ti"])
#     rest_ty = np.array(all_data["rest_ty"])

#     print(
#         f"Shapes - last_v: {last_v.shape}, last_ti: {last_ti.shape}, last_ty: {last_ty.shape}"
#     )
#     print(
#         f"Shapes - rest_v: {rest_v.shape}, rest_ti: {rest_ti.shape}, rest_ty: {rest_ty.shape}"
#     )

#     batch_size = len(last_v)
#     x_test = np.zeros(
#         (batch_size, 18, 49)
#     )  # Note: transposed from original for consistency
#     mask_test = np.zeros((batch_size, 18, 49))

#     # Reshape rest data
#     rest_v = rest_v.reshape(batch_size, -1)
#     rest_ti = rest_ti.reshape(batch_size, -1)
#     rest_ty = rest_ty.reshape(batch_size, -1)

#     for idx in range(batch_size):
#         # Process last data
#         lv, lti, lty = last_v[idx], last_ti[idx], last_ty[idx]
#         mask = lty != -1
#         lv, lti, lty = lv[mask], lti[mask], lty[mask]

#         # Process rest data
#         rv = rest_v[idx]
#         rti = rest_ti[idx]
#         rty = rest_ty[idx]
#         mask = rty != -1
#         rv, rti, rty = rv[mask], rti[mask], rty[mask]

#         # Create frame for data
#         np_arr_frame = np.full((73, 18), np.nan)

#         # Fill rest data
#         valid_indices = rti < 73
#         rti = rti[valid_indices].astype(int)
#         rty = rty[valid_indices].astype(int)
#         rv = rv[valid_indices]
#         np_arr_frame[rti, rty] = rv

#         # Fill last data
#         if len(lti) > 0:
#             valid_indices = lti < 73
#             lti = lti[valid_indices].astype(int)
#             lty = lty[valid_indices].astype(int)
#             lv = lv[valid_indices]
#             np_arr_frame[lti, lty] = lv

#         # Forward and backward fill
#         filled_arr = (
#             pd.DataFrame(np_arr_frame)
#             .fillna(method="ffill")
#             .fillna(method="bfill")
#             .values[-49:]
#         )

#         # Transpose to match expected shape (18, 49)
#         x_test[idx] = filled_arr.T
#         mask_test[idx] = ~np.isnan(filled_arr.T)

#     return x_test, mask_test


# def get_batch_predictions_and_attributions(pids):
#     """Collect predictions and attributions for all samples in batch"""
#     all_preds = []
#     all_prev_preds = []
#     all_delayed = []
#     all_attributions = []
#     all_pred_counts = []  # Store number of predictions per sample

#     for pid in pids:
#         attribution_data = load_attribution_data(pid)
#         if attribution_data is not None:
#             if (
#                 "current_output" in attribution_data
#                 and "delayed_effect" in attribution_data
#             ):
#                 preds = attribution_data["current_output"]
#                 prev_preds = attribution_data["previous_output"]
#                 delayed = attribution_data["delayed_effect"]
#                 if len(preds) > 0:
#                     all_preds.append(preds)
#                     all_prev_preds.append(prev_preds)
#                     all_delayed.append(delayed)
#                     all_pred_counts.append(len(preds))

#                     if "dynamic_feature_importance" in attribution_data:
#                         all_attributions.append(
#                             attribution_data["dynamic_feature_importance"]
#                         )

#     return all_preds, all_prev_preds, all_delayed, all_attributions, all_pred_counts


# def process_pickle_data(
#     directory_or_file="/nfs/potter/shared/for_andy/score_delta_model_result/pos_set_sample",
#     single_file=None,
#     feature_names=None,
#     FEATURE_MAP=None,
#     save_dir="",
# ):
#     """Import and process pickle files for visualization."""
#     # Load sample data
#     if single_file:
#         file_path = single_file
#     elif os.path.isfile(directory_or_file):
#         file_path = directory_or_file
#     else:
#         directory_path = directory_or_file
#         pickle_files = [f for f in os.listdir(directory_path) if f.endswith(".pkl")]
#         if not pickle_files:
#             print(f"No pickle files found in {directory_path}.")
#             return None
#         file_path = os.path.join(directory_path, pickle_files[0])

#     try:
#         with open(file_path, "rb") as f:
#             all_data = pickle.load(f)
#     except Exception as e:
#         print(f"Error loading sample data: {e}")
#         return None

#     # Extract patient IDs
#     pids = all_data.get("pid", [])
#     if not pids:
#         print("No patient IDs found in sample data")
#         return None

#     # Process time series data
#     x_test, mask_test = process_ts_data(all_data)
#     if x_test is None or mask_test is None:
#         return None

#     # Get feature names
#     if feature_names is None and FEATURE_MAP is not None:
#         feature_names = list(FEATURE_MAP.keys())
#     if feature_names is None:
#         feature_names = [f"Feature {i}" for i in range(x_test.shape[1])]

#     # Process all predictions and attributions in batch
#     all_preds, all_prev_preds, all_delayed, all_attributions, pred_counts = (
#         get_batch_predictions_and_attributions(pids)
#     )

#     all_preds, all_prev_preds, all_delayed, all_attributions = (
#         np.array(all_preds),
#         np.array(all_prev_preds),
#         np.array(all_delayed),
#         np.array(all_attributions),
#     )
    
#     import pdb; pdb.set_trace()

#     # # Create output directory for each sample
#     # for i, (preds, prev_preds, delayed, attributions, n_preds) in enumerate(
#     #     zip(all_preds, all_prev_preds, all_delayed, all_attributions, pred_counts)
#     # ):
#     #     sample_dir = os.path.join(save_dir, f"sample_{pids[i]}")
#     #     os.makedirs(sample_dir, exist_ok=True)

#     #     # import pdb; pdb.set_trace()

#     #     # Clip prediction values
#     #     vis_orig_preds = preds
#     #     all_prev_preds = prev_preds
#     #     carry_preds = preds - delayed

#     #     # Reshape predictions for visualization (timesteps, samples)
#     #     n_timesteps = x_test.shape[2]  # Should be 49
#     #     vis_orig_preds_reshaped = vis_orig_preds
#     #     vis_carry_preds_reshaped = vis_carry_preds

#     #     # import pdb; pdb.set_trace()

#     #     print(f"Reshaped predictions shape: {vis_orig_preds_reshaped.shape}")

#     all_carry_preds = np.array(all_preds) - np.array(all_delayed)

#     visualize_temporal_evolution(
#         feature_values=x_test,
#         masks=mask_test,
#         predictions=all_preds,
#         feature_names=feature_names,
#         save_dir=save_dir,
#     )

#     visualize_per_sample_attribution(
#         attributions=all_attributions,
#         predictions=all_preds,
#         carry_forward_predictions=all_carry_preds,
#         mask_test=mask_test,
#         feature_names=feature_names,
#         save_dir=save_dir,
#     )

#     # Process static features and labels
#     static_test = (
#         np.array(all_data.get("age", [])).reshape(-1, 1) if "age" in all_data else None
#     )
#     y_test = np.array(all_data.get("label", [])) if "label" in all_data else None

#     # Convert to tensors
#     x_test_tensor = torch.tensor(x_test)
#     mask_test_tensor = torch.tensor(mask_test)
#     delta_test_tensor = torch.ones_like(x_test_tensor)
#     static_test_tensor = torch.tensor(static_test) if static_test is not None else None

#     # Flatten predictions for return value
#     orig_preds = np.concatenate(all_preds)
#     new_preds = orig_preds.copy()

#     return (
#         x_test_tensor,
#         mask_test_tensor,
#         delta_test_tensor,
#         static_test_tensor,
#         x_test,
#         mask_test,
#         delta_test_tensor.numpy(),
#         static_test,
#         orig_preds,
#         new_preds,
#         y_test,
#         all_data,
#     )


# def process_all_samples():
#     # Set up directories
#     base_dir = "/nfs/potter/shared/for_andy/score_delta_model_result"
#     sample_dir = f"{base_dir}/pos_set_sample"
#     output_base_dir = "plots/marcus"

#     # Create output directory if it doesn't exist
#     os.makedirs(output_base_dir, exist_ok=True)

#     # Get list of all pickle files
#     pickle_files = [f for f in os.listdir(sample_dir) if f.endswith(".pkl")]

#     for pkl_file in pickle_files:
#         # Get patient ID from filename
#         pid = pkl_file.replace(".pkl", "")
#         print(f"\nProcessing patient {pid}")

#         # Create patient-specific output directory
#         patient_output_dir = os.path.join(output_base_dir, pid)
#         os.makedirs(patient_output_dir, exist_ok=True)

#         # Process the patient's data
#         process_pickle_data(
#             single_file=os.path.join(sample_dir, pkl_file),
#             FEATURE_MAP=FEATURE_MAP,
#             save_dir=patient_output_dir,
#         )

#     print("\nProcessing completed. Results saved in:", output_base_dir)


# if __name__ == "__main__":
#     process_all_samples()
