from __future__ import annotations

import logging
import pathlib
import pickle as pkl
from typing import Any, Dict, List
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn import metrics
from torch.utils.data import DataLoader

from deltashap.dataloader import DeltaSHAPDataset, SimulatedData
from deltashap.explainer.masker import Masker
from deltashap.explainer.explainers import (
    ExplainerConfig,
    BaselineType,
    BaseExplainer,
    MockExplainer,
    RandomExplainer,
    LIMEExplainer,
    IGExplainer,
    DeepLiftExplainer,
    ShapExplainer,
    KernelShapExplainer,
    DeepLiftShapExplainer,
    GradientShapExplainer,
    FOExplainer,
    AFOExplainer,
)
from deltashap.explainer.deltashapexplainer import DeltaShapExplainer
from deltashap.explainer.fitexplainers import FITExplainer
from deltashap.explainer.winitexplainers import WinITExplainer
from deltashap.explainer.timingexplainer import TIMINGExplainer
from deltashap.explainer.generator.generator import GeneratorTrainingResults
from deltashap.modeltrainer import ModelTrainerWithCv
from deltashap.plot import (
    BoxPlotter,
    visualize_per_sample_attribution,
    visualize_temporal_evolution,
)
from deltashap.utils import aggregate_scores
from deltashap.config import FEATURE_MAP


class ExplanationRunner:
    """
    Our main class for training the model, training the generator, running explanations and
    evaluating explanations for various datasets.
    """

    def __init__(
        self,
        args,
        dataset: DeltaSHAPDataset,
        device,
        out_path: pathlib.Path,
        ckpt_path: pathlib.Path,
        plot_path: pathlib.Path,
    ):
        """
        Constructor

        Args:
            dataset:
                The dataset wished to be run.
            device:
                The torch device.
            out_path:
                The path of the files containing the results of the explainer.
            ckpt_path:
                The path of the files containing the model and the generator checkpoints.
            plot_path:
                The path of the files containing plots or numpy arrays.
        """
        self.args = args
        self.dataset = dataset
        self.device = device
        self.out_path = out_path
        self.ckpt_path = ckpt_path
        self.plot_path = plot_path

        # if not self.out_path.exists() and not self.out_path.is_file():
        #     self.out_path.mkdir(parents=True, exist_ok=True)
        # if not self.ckpt_path.exists() and not self.ckpt_path.is_file():
        #     self.ckpt_path.mkdir(parents=True, exist_ok=True)
        # if not self.plot_path.exists() and not self.plot_path.is_file():
        #     self.plot_path.mkdir(parents=True, exist_ok=True)

        self.model_trainers: ModelTrainerWithCv | None = None
        self.explainers: Dict[int, BaseExplainer] | None = None
        self.importances: Dict[int, np.ndarray] | None = None
        self.elapsed_times: Dict[int, float] | None = None

        self.log = logging.getLogger(ExplanationRunner.__name__)

    def init_model(
        self,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        model_type: str = "GRU",
        verbose_eval: int = 10,
        early_stopping: bool = True,
        num_ensemble: int = 10,
        multi_gpu: bool = True,
    ) -> None:
        """
        Initialize the base models.

        Args:
            hidden_size:
                The hidden size of the models
            dropout:
                The dropout rate of the models.
            num_layers:
                The number of layers of the models for GRU or LSTM.
            model_type:
                The model type of the models. GRU, LSTM or CONV.
            verbose_eval:
               Training metrics is logged at every given verbose_eval epoch.
            early_stopping:
               Whether apply early stopping or not.
            num_ensemble:
               Number of ensemble models to train.
            multi_gpu:
               Whether to use multiple GPUs for the model.
        """
        self.model_trainers = ModelTrainerWithCv(
            self.dataset,
            self.ckpt_path,
            hidden_size,
            dropout,
            num_layers,
            model_type,
            self.device,
            verbose_eval,
            early_stopping,
            num_ensemble,
            multi_gpu=multi_gpu,
        )

    def train_model(
        self,
        num_epochs: int,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        use_all_times: bool = True,
    ) -> None:
        """
        Train the base models and log the test results.

        Args:
            num_epochs:
                The number of epochs to train the models.
            lr:
                The learning rate.
            weight_decay:
                The weight decay.
            use_all_times:
                Whether we use all timesteps or only the last timesteps to train the models.
        """
        if self.model_trainers is None:
            raise RuntimeError("Initialize the model first.")
        self.model_trainers.train_models(
            num_epochs, lr, weight_decay, use_all_times=use_all_times
        )
        self.model_trainers.load_model()
        self._get_test_results(use_all_times)

    def load_model(self, use_all_times: bool = True) -> None:
        """
        Load the base models and log the test results.

        Args:
            use_all_times:
                Whether we use all timesteps or only the last timesteps to train the models.
        """
        if self.model_trainers is None:
            raise RuntimeError("Initialize the model first.")
        self.model_trainers.load_model()
        self._get_test_results(use_all_times)

    def _get_test_results(self, use_all_times: bool) -> None:
        test_results = self.model_trainers.get_test_results(use_all_times)
        test_accs = [round(v.accuracy, 6) for v in test_results.values()]
        test_aucs = [round(v.auc, 6) for v in test_results.values()]
        test_auprcs = [round(v.auprc, 6) for v in test_results.values()]
        test_f1s = [round(v.f1, 6) for v in test_results.values()]
        self.log.info(
            f"Average Accuracy = {np.mean(test_accs):.4f}\u00b1{np.std(test_accs):.4f}"
        )
        self.log.info(f"Model Accuracy on Tests = {test_accs}.")
        self.log.info(
            f"Average AUROC = {np.mean(test_aucs):.4f}\u00b1{np.std(test_aucs):.4f}"
        )
        self.log.info(f"Model AUROC on Tests = {test_aucs}.")
        self.log.info(
            f"Average AUPRC = {np.mean(test_auprcs):.4f}\u00b1{np.std(test_auprcs):.4f}"
        )
        self.log.info(f"Model AUPRC on Tests = {test_auprcs}.")
        self.log.info(
            f"Average F1 = {np.mean(test_f1s):.4f}\u00b1{np.std(test_f1s):.4f}"
        )
        self.log.info(f"Model F1 on Tests = {test_f1s}.")

    def run_inference(
        self,
        dataloader: DataLoader,
    ) -> Dict[int, np.ndarray]:
        """
        Run inference using a dataloader.

        Args:
            dataloader: DataLoader containing the data

        Returns:
            Dictionary of CV to numpy arrays of predictions
        """
        return self.model_trainers.run_inference(dataloader)

    def clean_up(self, clean_importance=True, clean_explainer=True, clean_model=False):
        """
        Clean up.

        Args:
            clean_importance:
                indicate whether we clean the importance stored.
            clean_explainer:
                indicate whether we clean the explainer stored.
            clean_model:
                indicate whether we clean the model stored.
        """
        if clean_model and self.model_trainers is not None:
            del self.model_trainers
            self.model_trainers = None
        if clean_explainer and self.explainers is not None:
            del self.explainers
            self.explainers = None
        if clean_importance and self.importances is not None:
            del self.importances
            self.importances = None
        torch.cuda.empty_cache()

    def get_explainers(
        self,
        args,
        explainer_name: str,
        explainer_dict: Dict[str, Any],
    ) -> None:
        """Initialize explainers for each CV fold

        Args:
            args: General arguments
            explainer_name: Name of the explainer to use
            explainer_dict: Additional arguments for the explainer
        """
        # Add required parameters for FIT explainer
        if explainer_name == "fit":
            explainer_dict.update(
                {
                    "feature_size": self.dataset.feature_size,
                    "data_name": self.dataset.get_name(),
                    "path": args.path if hasattr(args, "path") else None,
                }
            )
        config = ExplainerConfig(
            device=self.device,
            baseline_type=BaselineType.ZERO,
            # baseline_type=BaselineType.TIME_DELTA,
            additional_args=explainer_dict,
        )
        config.last_timestep_only = args["last_timestep_only"]

        # import pdb; pdb.set_trace()

        self.explainers = {}
        for cv in self.dataset.cv_to_use():
            if explainer_name == "afo":
                train_loader = self.dataset.train_loaders[cv]
                explainer = AFOExplainer(config, train_loader)
            elif explainer_name == "fit":
                explainer = FITExplainer(
                    config,
                    device=config.device,
                    feature_size=config.additional_args["feature_size"],
                    data_name=config.additional_args["data_name"],
                    path=self._get_generator_path(cv),
                    counterfactual_strategy="generator",
                    # counterfactual_strategy="carry_forwrad",
                    # counterfactual_strategy=config.baseline_type,
                )
            elif explainer_name == "winit":
                explainer = WinITExplainer(
                    config,
                    num_features=self.dataset.feature_size,
                    data_name=self.dataset.get_name(),
                    path=self._get_generator_path(cv),
                    train_loader=None,
                    window_size=(
                        args.window_size if hasattr(args, "window_size") else 10
                    ),
                    num_samples=args.num_samples if hasattr(args, "num_samples") else 3,
                    conditional=(
                        args.conditional if hasattr(args, "conditional") else False
                    ),
                    joint=args.joint if hasattr(args, "joint") else False,
                    metric=args.metric if hasattr(args, "metric") else "pd",
                    random_state=(
                        args.random_state if hasattr(args, "random_state") else None
                    ),
                    # counterfactual_strategy=config.baseline_type,
                    counterfactual_strategy="generator",
                    args=args,
                )
            elif explainer_name == "deltashap":
                if args["deltashap_baseline"] == "carryforward":
                    config.baseline_type = BaselineType.CARRY_FORWARD
                elif args["deltashap_baseline"] == "time_delta":
                    config.baseline_type = BaselineType.TIME_DELTA
                elif args["deltashap_baseline"] == "zero":
                    config.baseline_type = BaselineType.ZERO
                config.n_samples = args["deltashap_n_samples"]
                config.additional_args["normalize"] = args["deltashap_normalize"]
                explainer = DeltaShapExplainer(
                    config,
                    self.model_trainers.model_trainers[cv].model,
                )
            else:
                explainer_map = {
                    "mock": MockExplainer,
                    "random": RandomExplainer,
                    "lime": LIMEExplainer,
                    "shap": ShapExplainer,
                    "kernelshap": KernelShapExplainer,
                    "deepliftshap": DeepLiftShapExplainer,
                    "gradientshap": GradientShapExplainer,
                    "deeplift": DeepLiftExplainer,
                    "ig": IGExplainer,
                    "fo": FOExplainer,
                    "afo": AFOExplainer,
                    "timing": TIMINGExplainer,
                }
                explainer_class = explainer_map.get(explainer_name)
                explainer = explainer_class(config)
            self.explainers[cv] = explainer

    def train_generators(
        self, num_epochs: int
    ) -> Dict[int, GeneratorTrainingResults] | None:
        """
        Train the generator if applicable. Test the generator and save the generator
        training results.

        Args:
            num_epochs:
                Train the generator for number of epochs.

        Returns:
            The generator training results. None if the explainer has no generator to train.
        """
        if self.explainers is None:
            raise RuntimeError(
                "explainer is not initialized. Call get_explainer to initialize."
            )
        results = {}
        generator_array_path = self._get_generator_array_path()
        generator_array_path.mkdir(parents=True, exist_ok=True)
        for cv in self.dataset.cv_to_use():
            self.log.info(f"Training generator for cv={cv}")
            gen_result = self.explainers[cv].train_generators(
                self.dataset.train_loaders[cv],
                self.dataset.valid_loaders[cv],
                num_epochs,
            )
            self.explainers[cv].test_generators(self.dataset.test_loader)
            if gen_result is not None:
                results[cv] = gen_result
                np.save(
                    generator_array_path / f"{gen_result.name}_train_loss_cv_{cv}.npy",
                    gen_result.train_loss_trends,
                )
                np.save(
                    generator_array_path / f"{gen_result.name}_valid_loss_cv_{cv}.npy",
                    gen_result.valid_loss_trends,
                )
                np.save(
                    generator_array_path / f"{gen_result.name}_best_epoch_cv_{cv}.npy",
                    gen_result.best_epochs,
                )
        if len(results) > 0:
            return results
        return None

    def load_generators(self):
        """
        Load the generator and print the test results.
        """
        if self.explainers is None:
            raise RuntimeError(
                "explainer is not initialized. Call get_explainer to initialize."
            )

        for cv in self.dataset.cv_to_use():
            self.explainers[cv].load_generators()
            # self.explainers[cv].test_generators(self.dataset.test_loader)

    def _get_generator_path(self, cv: int) -> pathlib.Path:
        return pathlib.Path(self.ckpt_path).parent / self.dataset.get_name() / str(cv)

    def _get_importance_path(self) -> pathlib.Path:
        return self.out_path / self.dataset.get_name()

    def _get_importance_file_name(self, cv: int):
        return f"{self.explainers[cv].get_name()}_test_importance_scores_{cv}.pkl"

    def set_model_for_explainer(self, set_eval: bool = True):
        """
        Set the base model for the explainer.

        Args:
            set_eval:
                Set the model to eval mode. If False, leave the model as is.
        """
        if self.explainers is None:
            raise RuntimeError(
                "explainer is not initialized. Call get_explainer to initialize."
            )

        for cv in self.dataset.cv_to_use():
            # Get the model, unwrapping from DataParallel and PredictWrapper if needed
            model = self.model_trainers.model_trainers[cv].model
            self.explainers[cv].set_model(model, set_eval=set_eval)

    def run_attributes(self) -> None:
        """
        Run attribution method for the explainer on the test set.
        """
        if self.explainers is None:
            raise RuntimeError(
                "explainer is not initialized. Call get_explainer to initialize."
            )

        self.importances, self.elapsed_times = self._run_attributes_recursive(
            self.dataset.test_loader
        )

    def _run_attributes_recursive(
        self, dataloader: DataLoader
    ) -> Dict[int, np.ndarray]:
        """
        A convenient method to run attributes when we adjust the batch size if cuda is out of
        memory

        Args:
            dataloader:
                The data loader for the input for attributes

        Returns:
            A dictionary of CV to the attribution.
        """
        try:
            return self._run_attributes(dataloader)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # reduce batch size
                new_batch_size = dataloader.batch_size // 2
                if 0 < new_batch_size < dataloader.batch_size:
                    self.log.warning(
                        f"CUDA out of memory! Reducing batch size from "
                        f"{dataloader.batch_size} to {new_batch_size}"
                    )
                    new_loader = DataLoader(dataloader.dataset, new_batch_size)
                    # self.test_loader.batch_size = new_batch_size
                    return self._run_attributes_recursive(new_loader)
            raise e

    def _run_attributes(self, dataloader: DataLoader) -> Dict[int, np.ndarray]:
        """
        Run feature attribution.

        Args:
            dataloader:
                The data loader for the input for attributes

        Returns:
            A dictionary of CV to the attribution and elapsed times.
        """
        all_importance_scores = {}
        elapsed_times = {}  # New dictionary to store elapsed times

        for cv in self.dataset.cv_to_use():
            importance_scores = []
            total_time = 0  # Track total time for this CV
            num_batches = 0  # Count number of batches

            for i, batch in tqdm(enumerate(dataloader)):
                # Unpack the batch components
                batch = [x.to(self.device) for x in batch]
                values, masks = batch[0], batch[1]

                start_time = time.time()

                # Run attribution
                score = self.explainers[cv].attribute(batch)

                # import pdb; pdb.set_trace()

                end_time = time.time()
                total_time += end_time - start_time
                num_batches += 1

                # # Handle masking if needed (for specific model types)
                # if isinstance(self.explainers[cv].base_model, (mTAND, SeFT)) and masks is not None:
                #     score[masks.cpu() == 0] = float("-inf")

                importance_scores.append(score)

                # Calculate temporal predictions if visualization is enabled
                # if self.args.get("vis", False) and i == 0:
                #     # import pdb; pdb.set_trace()
                #     with torch.no_grad():
                #         values, masks = values[:self.args["num_vis"]], masks[:self.args["num_vis"]]
                #         carry_forward_values = values.clone()
                #         carry_forward_values[:, -1, :] = carry_forward_values[:, -2, :]

                #         predictions = self.explainers[cv].explainer.original_output[:self.args["num_vis"]]
                #         carry_forward_predictions = self.explainers[cv].explainer.baseline_output[:self.args["num_vis"]]

                #     visualize_per_sample_attribution(
                #         attributions=score[:self.args["num_vis"]],
                #         predictions=predictions.cpu().numpy(),
                #         carry_forward_predictions=carry_forward_predictions.cpu().numpy(),
                #         mask_test=masks.cpu().numpy(),
                #         inputs=values.cpu().numpy()[:self.args["num_vis"]],
                #         carry_forward_inputs=carry_forward_values.cpu().numpy()[:self.args["num_vis"]],
                #         feature_names=list(FEATURE_MAP.keys()),
                #         save_dir=self.args["vis_dir"]
                #     )

                #     # visualize_temporal_evolution(
                #     #     feature_values=values.cpu().numpy(),
                #     #     masks=masks.cpu().numpy(),
                #     #     predictions=predictions.cpu().numpy(),
                #     #     feature_names=list(FEATURE_MAP.keys()),
                #     # )

            importance_scores = np.concatenate(importance_scores, 0)
            all_importance_scores[cv] = importance_scores
            elapsed_times[cv] = total_time / num_batches

            self.log.info(
                f"CV {cv}: Average attribution time per batch: {elapsed_times[cv]:.4f} seconds"
            )

        return all_importance_scores, elapsed_times

    def save_importance(self):
        """
        Save the feature importance.
        """
        pass

        # if self.importances is None:
        #     return

        # importance_path = self._get_importance_path()
        # importance_path.mkdir(parents=True, exist_ok=True)

        # for cv, importance_scores in self.importances.items():
        #     importance_file_name = importance_path / self._get_importance_file_name(cv)
        #     self.log.info(f"Saving file to {importance_file_name}")
        #     with importance_file_name.open("wb") as f:
        #         pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)

    def load_importance(self):
        """
        Load the importance from the file.
        """
        importances = {}
        for cv in self.dataset.cv_to_use():
            importance_file_name = (
                self._get_importance_path() / self._get_importance_file_name(cv)
            )
            with importance_file_name.open("rb") as f:
                importance_scores = pkl.load(f)
            importances[cv] = importance_scores
        self.importances = importances

    def evaluate_simulated_importance(self, aggregate_methods) -> pd.DataFrame:
        """
        Run evaluation for importance for Simulated Data. The metrics are AUROC, AVPR, AUPRC,
        mean rank, mean rank (min) and positive ratios. Save the example boxes as well.

        Args:
            aggregate_methods:
                The aggregation method for WinIT.

        Returns:
            A DataFrame of shape (num_cv * aggregate_method, num_metric=6).
        """
        if not isinstance(self.dataset, SimulatedData):
            raise ValueError(
                "non simulated dataset does not have simulated importances."
            )

        if self.importances is None:
            raise ValueError(
                "No importances is loaded. Call load_importance or run_attribute first."
            )

        ground_truth_importance = self.dataset.load_ground_truth_importance()

        absolutize = not isinstance(
            next(iter(self.explainers.values())),
            (
                WinITExplainer,
                FITExplainer,
            ),
        )
        df = self._evaluate_importance_with_gt(
            ground_truth_importance, absolutize, aggregate_methods
        )
        self._plot_boxes(num_to_plot=20, aggregate_methods=aggregate_methods)
        return df

    def evaluate_performance_drop(
        self,
        maskers: List[Masker],
    ) -> pd.DataFrame:
        """
        Evaluate the importances on non simulated dataset by performance drop.
        """
        # Load test data
        orig_preds = self.run_inference(self.dataset.test_loader)

        testset = list(self.dataset.test_loader.dataset)
        x_test, mask_test, y_test = (
            torch.stack([x[0] for x in testset]).cpu().numpy(),
            torch.stack([x[1] for x in testset]).cpu().numpy(),
            torch.stack([x[-1] for x in testset]).cpu().numpy(),
        )
        dfs = {}
        for masker in maskers:
            new_xs, new_masks, importance_masks = masker.mask(
                x_test, mask_test, self.importances
            )
            new_xs = {k: torch.from_numpy(v) for k, v in new_xs.items()}
            new_masks = {k: torch.from_numpy(v) for k, v in new_masks.items()}
            importance_masks = {
                k: torch.from_numpy(v) for k, v in importance_masks.items()
            }

            # create new testset
            masked_testset = [
                (new_xs[0][i], new_masks[0][i]) + item[2:]
                for i, item in enumerate(testset)
            ]
            batch_size = self.dataset.test_loader.batch_size
            masked_test_loader = DataLoader(
                masked_testset, batch_size=batch_size, shuffle=False
            )

            new_preds = self.run_inference(masked_test_loader)

            # Create DataFrame for this masker
            df = pd.DataFrame()
            for cv in self.dataset.cv_to_use():
                orig_pred = orig_preds[cv]
                orig_pred = orig_pred.reshape(-1)
                y_test_reshaped = y_test.reshape(-1)
                new_pred = new_preds[cv].reshape(-1)

                # Calculate metrics
                orig_auc = metrics.roc_auc_score(
                    y_test_reshaped, orig_pred, average="macro"
                )
                new_auc = metrics.roc_auc_score(
                    y_test_reshaped, new_pred, average="macro"
                )

                orig_auprc = metrics.average_precision_score(
                    y_test_reshaped, orig_pred, average="macro"
                )
                new_auprc = metrics.average_precision_score(
                    y_test_reshaped, new_pred, average="macro"
                )

                # For original predictions
                best_f1 = 0
                best_threshold = 0.5
                thresholds = np.linspace(0.01, 0.99, 99)
                for threshold in thresholds:
                    temp_pred_binary = (orig_pred > threshold).astype(int)
                    temp_f1 = metrics.f1_score(
                        y_test_reshaped, temp_pred_binary, average="macro"
                    )
                    if temp_f1 > best_f1:
                        best_f1 = temp_f1
                        best_threshold = threshold
                print(f"Best F1: {best_f1} at threshold: {best_threshold}")

                original_pred_binary = (orig_pred > best_threshold).astype(int)
                original_f1 = metrics.f1_score(
                    y_test_reshaped, original_pred_binary, average="macro"
                )

                # Store results
                df[cv] = pd.Series(
                    {
                        "auc_drop": orig_auc - new_auc,
                        "auprc_drop": orig_auprc - new_auprc,
                        "f1_drop": original_f1 - original_f1,
                        "avg_pred_diff": np.abs(orig_pred - new_pred).mean(),
                        "avg_masked_count": (
                            masker.all_masked_count[cv].sum() / len(x_test)
                        ).item(),
                    }
                )

            df = df.transpose()
            df.index.name = "cv"
            dfs[masker.get_name()] = df

            # Visualization code remains unchanged
            if self.args["vis"]:
                plotter = BoxPlotter(
                    self.dataset,
                    self.plot_path,
                    num_to_plot=min(self.args["num_vis"], len(x_test)),
                    explainer_name=self.get_explainer_name(),
                )

                for cv in self.dataset.cv_to_use():
                    cv_x_test = x_test
                    cv_mask_test = mask_test
                    cv_new_xs = (
                        new_xs[cv].cpu().numpy()
                        if torch.is_tensor(new_xs[cv])
                        else new_xs[cv]
                    )
                    cv_new_masks = (
                        new_masks[cv].cpu().numpy()
                        if torch.is_tensor(new_masks[cv])
                        else new_masks[cv]
                    )
                    cv_importance_masks = (
                        importance_masks[cv].cpu().numpy()
                        if torch.is_tensor(importance_masks[cv])
                        else importance_masks[cv]
                    )
                    cv_orig_preds = orig_preds[cv]
                    cv_new_preds = new_preds[cv]
                    cv_y_test = y_test

                    plotter.plot_combined_visualization(
                        self.importances,
                        masker.aggregate_method,
                        cv_x_test,
                        cv_mask_test,
                        cv_new_xs,
                        cv_new_masks,
                        cv_importance_masks,
                        cv_orig_preds,
                        cv_new_preds,
                        cv_y_test,
                        masker.get_name(),
                    )
        return dfs

    def evaluate_performance_drop_cum(
        self,
        maskers: List[Masker],
    ) -> pd.DataFrame:
        """
        Evaluate the importances on non simulated dataset by cumulative performance drop
        and prediction preservation.
        """
        # import pdb; pdb.set_trace()
        orig_preds = {}
        for cv in self.dataset.cv_to_use():
            orig_preds[cv] = self.run_inference(self.dataset.test_loader)

        testset = list(self.dataset.test_loader.dataset)
        x_test, mask_test, y_test = (
            torch.stack([x[0] for x in testset]).cpu().numpy(),
            torch.stack([x[1] for x in testset]).cpu().numpy(),
            torch.stack([x[-1] for x in testset]).cpu().numpy(),
        )

        dfs = {}
        for masker in maskers:
            self.log.info(
                f"Beginning performance drop for mask={masker.get_name()} in cumulative setting"
            )
            total = masker.top

            # For performance drop (highest attributions)
            all_preds_drop = []
            # For prediction preservation (lowest attributions)
            all_preds_preserve = []

            last_new_xs_drop = None
            last_new_masks_drop = None
            last_importance_masks_drop = None

            last_new_xs_preserve = None
            last_new_masks_preserve = None
            last_importance_masks_preserve = None

            # Collect all predictions first
            for i in range(total):
                # For performance drop (highest attributions)
                masker.top = i + 1
                masker.select_top = True  # Select highest attributions
                new_xs_drop, new_masks_drop, importance_masks_drop = masker.mask(
                    x_test, mask_test, self.importances
                )
                new_xs_drop = {k: torch.from_numpy(v) for k, v in new_xs_drop.items()}
                new_masks_drop = {
                    k: torch.from_numpy(v) for k, v in new_masks_drop.items()
                }

                # Create masked dataset and loader for performance drop
                masked_drop_testset = [
                    (new_xs_drop[0][i], new_masks_drop[0][i]) + item[2:]
                    for i, item in enumerate(testset)
                ]
                batch_size = self.dataset.test_loader.batch_size
                masked_drop_loader = DataLoader(
                    masked_drop_testset, batch_size=batch_size, shuffle=False
                )

                new_preds_drop = self.run_inference(masked_drop_loader)
                all_preds_drop = {k: new_preds_drop for k in self.dataset.cv_to_use()}

                # For prediction preservation (lowest attributions)
                masker.select_top = False  # Select lowest attributions
                # For instance-wise masking, we need to calculate p% of observed points per instance
                masker.instance_wise = (
                    True  # Enable instance-wise masking for preservation
                )
                masker.p_percent = (i + 1) / total  # Gradually increase percentage

                new_xs_preserve, new_masks_preserve, importance_masks_preserve = (
                    masker.mask(x_test, mask_test, self.importances)
                )
                new_xs_preserve = {
                    k: torch.from_numpy(v) for k, v in new_xs_preserve.items()
                }
                new_masks_preserve = {
                    k: torch.from_numpy(v) for k, v in new_masks_preserve.items()
                }

                # Create masked dataset and loader for prediction preservation
                masked_preserve_testset = [
                    (new_xs_preserve[0][i], new_masks_preserve[0][i]) + item[2:]
                    for i, item in enumerate(testset)
                ]
                masked_preserve_loader = DataLoader(
                    masked_preserve_testset, batch_size=batch_size, shuffle=False
                )

                new_preds_preserve = self.run_inference(masked_preserve_loader)
                all_preds_preserve = {k: new_preds_preserve for k in self.dataset.cv_to_use()}

                if i == total - 1:  # Store last iteration values
                    last_new_xs_drop = new_xs_drop
                    last_new_masks_drop = new_masks_drop
                    last_importance_masks_drop = importance_masks_drop

                    last_new_xs_preserve = new_xs_preserve
                    last_new_masks_preserve = new_masks_preserve
                    last_importance_masks_preserve = importance_masks_preserve

            df = pd.DataFrame()
            for cv in self.dataset.cv_to_use():
                # import pdb; pdb.set_trace()
                orig_pred = orig_preds[cv].flatten()
                y_test_reshaped = np.array(y_test).flatten()

                # Calculate original scores
                original_auc = metrics.roc_auc_score(
                    y_test_reshaped, orig_pred, average="macro"
                )
                original_auprc = metrics.average_precision_score(
                    y_test_reshaped, orig_pred, average="macro"
                )
                # For original predictions
                best_f1 = 0
                best_threshold = 0.5
                thresholds = np.linspace(0.01, 0.99, 99)
                for threshold in thresholds:
                    temp_pred_binary = (orig_pred > threshold).astype(int)
                    temp_f1 = metrics.f1_score(
                        y_test_reshaped, temp_pred_binary, average="macro"
                    )
                    if temp_f1 > best_f1:
                        best_f1 = temp_f1
                        best_threshold = threshold
                self.log.info(
                    f"Best F1: {best_f1:.4f} at threshold: {best_threshold:.4f}"
                )

                original_pred_binary = (orig_pred > best_threshold).astype(int)
                original_f1 = metrics.f1_score(
                    y_test_reshaped, original_pred_binary, average="macro"
                )

                # For performance drop
                before_pred = orig_pred
                before_auc = original_auc
                before_auprc = original_auprc
                before_f1 = original_f1

                pred_change_list = []
                auc_change_list = []
                auprc_change_list = []
                f1_change_list = []

                # For prediction preservation
                pred_preserve_list = []
                auc_preserve_list = []
                auprc_preserve_list = []
                f1_preserve_list = []

                before_preserve_pred = orig_pred
                before_preserve_auc = original_auc
                before_preserve_auprc = original_auprc
                before_preserve_f1 = original_f1

                for i in range(total):
                    # import pdb; pdb.set_trace()

                    current_pred = all_preds_drop[cv].flatten()

                    # Calculate current scores
                    current_auc = metrics.roc_auc_score(
                        y_test_reshaped, current_pred, average="macro"
                    )
                    current_auprc = metrics.average_precision_score(
                        y_test_reshaped, current_pred, average="macro"
                    )
                    current_pred_binary = (current_pred > best_threshold).astype(int)
                    current_f1 = metrics.f1_score(
                        y_test_reshaped, current_pred_binary, average="macro"
                    )

                    # For predictions: calculate cumulative differences
                    avg_pred_diff = np.abs(before_pred - current_pred).mean().item()
                    # For metrics: calculate direct differences from original
                    auc_drop = before_auc - current_auc
                    auprc_drop = before_auprc - current_auprc
                    f1_drop = before_f1 - current_f1

                    # Store changes
                    pred_change_list.append(
                        avg_pred_diff
                    )  # This will be cumulative since before_pred updates
                    auc_change_list.append(auc_drop)  # Direct difference from original
                    auprc_change_list.append(
                        auprc_drop
                    )  # Direct difference from original
                    f1_change_list.append(f1_drop)  # Direct difference from original

                    # Update before_pred for cumulative prediction differences
                    before_pred = current_pred

                    # Process prediction preservation metrics similarly
                    preserve_pred = all_preds_preserve[cv].flatten()

                    # Calculate preservation scores
                    preserve_auc = metrics.roc_auc_score(
                        y_test_reshaped, preserve_pred, average="macro"
                    )
                    preserve_auprc = metrics.average_precision_score(
                        y_test_reshaped, preserve_pred, average="macro"
                    )
                    preserve_pred_binary = (preserve_pred > best_threshold).astype(int)
                    preserve_f1 = metrics.f1_score(
                        y_test_reshaped, preserve_pred_binary, average="macro"
                    )

                    # For predictions: calculate cumulative differences
                    pred_preserve = (
                        np.abs(before_preserve_pred - preserve_pred).mean().item()
                    )
                    # For metrics: calculate direct differences from original
                    auc_preserve = before_preserve_auc - preserve_auc
                    auprc_preserve = before_preserve_auprc - preserve_auprc
                    f1_preserve = before_preserve_f1 - preserve_f1

                    # Store preservation metrics
                    pred_preserve_list.append(pred_preserve)  # This will be cumulative
                    auc_preserve_list.append(
                        auc_preserve
                    )  # Direct difference from original
                    auprc_preserve_list.append(
                        auprc_preserve
                    )  # Direct difference from original
                    f1_preserve_list.append(
                        f1_preserve
                    )  # Direct difference from original

                    # Update before_preserve_pred for cumulative prediction differences
                    before_preserve_pred = preserve_pred

                # Convert lists to arrays
                pred_change_list = np.cumsum(np.array(pred_change_list))
                auc_change_list = np.cumsum(np.array(auc_change_list))
                auprc_change_list = np.cumsum(np.array(auprc_change_list))
                f1_change_list = np.cumsum(np.array(f1_change_list))

                pred_preserve_list = np.cumsum(np.array(pred_preserve_list))
                auc_preserve_list = np.cumsum(np.array(auc_preserve_list))
                auprc_preserve_list = np.cumsum(np.array(auprc_preserve_list))
                f1_preserve_list = np.cumsum(np.array(f1_preserve_list))

                # import pdb; pdb.set_trace()

                # Save arrays
                cum_array_path = self._get_cum_array_path()
                cum_array_path.mkdir(parents=True, exist_ok=True)

                # Calculate final metrics
                avg_mask_count = (
                    (masker.all_masked_count[cv].sum() / len(x_test)).item()
                    if hasattr(masker, "all_masked_count")
                    else 0
                )

                # Calculate area under curve metrics for performance drop
                area_pred_diff = np.trapz(pred_change_list) / total
                area_auc_drop = np.trapz(np.abs(auc_change_list)) / total
                area_auprc_drop = np.trapz(np.abs(auprc_change_list)) / total
                area_f1_drop = np.trapz(np.abs(f1_change_list)) / total

                # Calculate area under curve metrics for prediction preservation
                area_pred_preserve = np.trapz(pred_preserve_list) / total
                area_auc_preserve = np.trapz(np.abs(auc_preserve_list)) / total
                area_auprc_preserve = np.trapz(np.abs(auprc_preserve_list)) / total
                area_f1_preserve = np.trapz(np.abs(f1_preserve_list)) / total

                df[cv] = pd.Series(
                    {
                        "avg_masked_count": avg_mask_count,
                        # Area under curve metrics for performance drop
                        "AUPD": area_pred_diff,
                        "AUAUCD": area_auc_drop,
                        "AUAPRD": area_auprc_drop,
                        "AUF1D": area_f1_drop,
                        # Area under curve metrics for prediction preservation
                        "APPP": area_pred_preserve,
                        "AUAUCP": area_auc_preserve,
                        "AUAPRP": area_auprc_preserve,
                        "AUF1P": area_f1_preserve,
                    }
                )

            df = df.transpose()
            df.index.name = "cv"
            dfs[masker.get_name()] = df

            if self.args.get("vis", False):
                plotter = BoxPlotter(
                    self.dataset,
                    pathlib.Path(self.plot_path),
                    num_to_plot=min(self.args["num_vis"], len(x_test)),
                    explainer_name=self.get_explainer_name(),
                )

                for cv in self.dataset.cv_to_use():
                    # Fix: Handle both torch tensors and numpy arrays correctly
                    def safe_convert_to_numpy(obj):
                        if isinstance(obj, dict):
                            return {k: safe_convert_to_numpy(v) for k, v in obj.items()}
                        elif isinstance(obj, torch.Tensor):
                            return obj.cpu().numpy()
                        else:
                            return obj  # Already numpy or other type

                    # Convert all objects safely
                    xs_drop_np = safe_convert_to_numpy(last_new_xs_drop)
                    masks_drop_np = safe_convert_to_numpy(last_new_masks_drop)
                    importance_masks_drop_np = safe_convert_to_numpy(
                        last_importance_masks_drop
                    )

                    xs_preserve_np = safe_convert_to_numpy(last_new_xs_preserve)
                    masks_preserve_np = safe_convert_to_numpy(last_new_masks_preserve)
                    importance_masks_preserve_np = safe_convert_to_numpy(
                        last_importance_masks_preserve
                    )

                    # import pdb; pdb.set_trace()

                    plotter.plot_combined_visualization(
                        self.importances,
                        masker.aggregate_method,
                        x_test,
                        mask_test,
                        xs_drop_np,
                        masks_drop_np,
                        importance_masks_drop_np,
                        orig_preds,
                        all_preds_drop,
                        y_test,
                        f"{masker.get_name()}_drop",
                    )

                    plotter.plot_combined_visualization(
                        self.importances,
                        masker.aggregate_method,
                        x_test,
                        mask_test,
                        xs_preserve_np,
                        masks_preserve_np,
                        importance_masks_preserve_np,
                        orig_preds,
                        all_preds_preserve,
                        y_test,
                        f"{masker.get_name()}_preserve",
                    )

        if dfs:
            dfs = pd.concat(dfs, axis=0)
            dfs.index.name = "mask method"
        return dfs

    def _plot_boxes(
        self,
        num_to_plot,
        aggregate_methods: List[str],
        x_other: List[Dict[int, torch.Tensor]] | None = None,
        mask_other: List[Dict[int, torch.Tensor]] | None = None,
        importance_mask_other: List[Dict[int, torch.Tensor]] | None = None,
        mask_name: str = "",
    ) -> None:
        """
        Plot comprehensive visualization including original/masked inputs, masks, and importances.
        """
        explainer_name = self.get_explainer_name()
        plotter = BoxPlotter(self.dataset, self.plot_path, num_to_plot, explainer_name)

        if self.importances is not None and x_other is not None:
            for aggregate_method in aggregate_methods:
                testset = list(self.dataset.test_loader.dataset)
                x_test = torch.stack([x[0] for x in testset]).cpu().numpy()
                mask_test = torch.stack([x[1] for x in testset]).cpu().numpy()
                y_test = torch.stack([x[-1] for x in testset]).cpu().numpy()

                # Get original predictions
                orig_preds = self.run_inference(self.dataset.test_loader)

                # Create masked testset for inference
                masked_testset = [
                    (x_other[0][i], mask_other[0][i]) + item[2:]
                    for i, item in enumerate(testset)
                ]
                batch_size = self.dataset.test_loader.batch_size
                masked_test_loader = DataLoader(
                    masked_testset, batch_size=batch_size, shuffle=False
                )

                new_preds = self.run_inference(masked_test_loader)

                plotter.plot_combined_visualization(
                    self.importances,
                    aggregate_method,
                    x_test,
                    mask_test,
                    {k: v.cpu().numpy() for k, v in x_other[0].items()},
                    {k: v.cpu().numpy() for k, v in mask_other[0].items()},
                    {k: v.cpu().numpy() for k, v in importance_mask_other[0].items()},
                    orig_preds,
                    new_preds,
                    y_test,
                    mask_name,
                )

    def _evaluate_importance_with_gt(
        self,
        ground_truth_importance: np.ndarray,
        absolutize: bool,
        aggregate_methods: List[str],
    ):
        ground_truth_importance = ground_truth_importance[:, :, 1:].reshape(
            len(ground_truth_importance), -1
        )
        dfs = {}
        for aggregate_method in aggregate_methods:
            df = pd.DataFrame()
            for cv, importance_unaggregated in self.importances.items():
                importance_scores = aggregate_scores(
                    importance_unaggregated, aggregate_method
                )
                importance_scores = importance_scores[:, :, 1:].reshape(
                    len(importance_scores), -1
                )

                # compute mean ranks
                ranks = rankdata(-importance_scores, axis=1)
                ranks_min = rankdata(-importance_scores, axis=1, method="min")
                gt_positions = np.where(ground_truth_importance)
                gt_ranks = ranks[gt_positions]
                gt_ranks_min = ranks_min[gt_positions]
                mean_rank = np.mean(gt_ranks)
                mean_rank_min = np.mean(gt_ranks_min)

                gt_score = ground_truth_importance.flatten()
                explainer_score = importance_scores.flatten()

                if absolutize:
                    explainer_score = np.abs(explainer_score)

                if np.any(np.isnan(explainer_score)):
                    self.log.warning("NaNs appear in explainer scores!")

                explainer_score = np.nan_to_num(explainer_score)
                auc_score = metrics.roc_auc_score(gt_score, explainer_score)
                aupr_score = metrics.average_precision_score(gt_score, explainer_score)
                prec_score, rec_score, thresholds = metrics.precision_recall_curve(
                    gt_score, explainer_score
                )
                auprc_score = (
                    metrics.auc(rec_score, prec_score) if rec_score.shape[0] > 1 else -1
                )

                pos_ratio = ground_truth_importance.sum() / len(ground_truth_importance)
                result = {
                    "Auroc": auc_score,
                    "Avpr": aupr_score,
                    "Auprc": auprc_score,
                    "Mean rank": mean_rank,
                    "Mean rank (min)": mean_rank_min,
                    "Pos ratio": pos_ratio,
                }
                self.log.info(f"cv={cv}")
                for k, v in result.items():
                    self.log.info(f"{k:20}: {v:.4f}")
                df[cv] = pd.Series(result)
            df = df.transpose()
            df.index.name = "cv"
            dfs[aggregate_method] = df
        df_all = pd.concat(dfs, axis=0)
        df_all.index.name = "aggregate method"
        return df_all

    def _get_mask_array_path(self) -> pathlib.Path:
        return self.plot_path / self.dataset.get_name() / "array"

    def _get_cum_array_path(self) -> pathlib.Path:
        return self.plot_path / self.dataset.get_name() / "cum_array"

    def _get_generator_array_path(self) -> pathlib.Path:
        return self.plot_path / self.dataset.get_name() / "generator_array"

    def get_explainer_name(self) -> str:
        if self.explainers is None:
            return ""
        return next(iter(self.explainers.values())).get_name()
