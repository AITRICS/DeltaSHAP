from __future__ import annotations

import logging
import pathlib
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from deltashap.explainer.explainers import BaseExplainer
from deltashap.explainer.generator.generator import (
    FeatureGenerator,
    BaseFeatureGenerator,
    GeneratorTrainingResults,
)
from deltashap.explainer.generator.jointgenerator import JointFeatureGenerator
from deltashap.config import NORMAL_VALUE, FEATURE_MAP


class WinITExplainer(BaseExplainer):
    """
    The explainer for our method WinIT
    """

    def __init__(
        self,
        device,
        num_features: int,
        data_name: str,
        path: pathlib.Path,
        train_loader: DataLoader | None = None,
        window_size: int = 10,
        num_samples: int = 3,
        conditional: bool = False,
        joint: bool = False,
        metric: str = "pd",
        random_state: int | None = None,
        counterfactual_strategy: str = "zero",
        feature_strategies: dict = None,
        aggregate_method: str = "mean",
        args=None,
        **kwargs,
    ):
        """
        Construtor

        Args:
            device:
                The torch device.
            num_features:
                The number of features.
            data_name:
                The name of the data.
            path:
                The path indicating where the generator to be saved.
            train_loader:
                The train loader if we are using the data distribution instead of a generator
                for generating counterfactual. Default=None.
            window_size:
                The window size for the WinIT
            num_samples:
                The number of Monte-Carlo samples for generating counterfactuals.
            conditional:
                Indicate whether the individual feature generator we used are conditioned on
                the current features. Default=False
            joint:
                Indicate whether we are using the joint generator.
            metric:
                The metric for the measures of comparison of the two distributions for i(S)_a^b
            random_state:
                The random state.
            counterfactual_strategy: Default strategy if feature_strategies not provided.
                Must be one of ["generator", "carry_forward", "zero"]. Default="generator"
            feature_strategies: Dict mapping feature indices to strategies.
                Each strategy must be one of ["generator", "carry_forward", "zero"].
                If None, uses counterfactual_strategy for all features.
            aggregate_method:
                The aggregation method for WinIT scores. One of ["mean", "max", "absmax"].
                Default="mean"
            **kwargs:
                There should be no additional kwargs.
        """
        super().__init__(device)

        # Initialize logger first
        self.log = logging.getLogger(WinITExplainer.__name__)

        # Basic attributes
        self.window_size = window_size
        self.num_samples = num_samples
        self.num_features = num_features
        self.data_name = data_name
        self.joint = joint
        self.conditional = conditional
        self.metric = metric
        self.args = args
        self.rng = np.random.default_rng(random_state)
        self.aggregate_method = aggregate_method

        # Set counterfactual strategy
        if path is None and counterfactual_strategy == "generator":
            self.log.warning(
                "No path provided for generator. Falling back to carry_forward strategy."
            )
            counterfactual_strategy = "carry_forward"
        self.counterfactual_strategy = counterfactual_strategy

        # Set counterfactual strategies per feature
        if feature_strategies is None:
            # Use same strategy for all features
            self.feature_strategies = {
                i: self.counterfactual_strategy for i in range(num_features)
            }
        else:
            # Validate and use provided strategies, filling in missing features with default strategy
            valid_strategies = {"generator", "carry_forward", "zero"}
            self.feature_strategies = {}
            for i in range(num_features):
                if i in feature_strategies:
                    strategy = feature_strategies[i]
                    if strategy not in valid_strategies:
                        raise ValueError(
                            f"Invalid strategy '{strategy}' for feature {i}"
                        )
                    self.feature_strategies[i] = strategy
                else:
                    self.feature_strategies[i] = self.counterfactual_strategy

        # Set first timestep strategies
        self.first_timestep_strategies = {}
        for f in range(num_features):
            strategy = self.feature_strategies[f]
            if strategy in ["generator", "zero"]:
                self.first_timestep_strategies[f] = "zero"
            elif strategy == "carry_forward":
                self.first_timestep_strategies[f] = "normal"

        self.generators: BaseFeatureGenerator | None = None
        self.path = path
        if train_loader is not None:
            self.data_distribution = (
                torch.stack([x[0] for x in train_loader.dataset]).detach().cpu().numpy()
            )
        else:
            self.data_distribution = None

        if len(kwargs):
            self.log.warning(f"kwargs is not empty. Unused kwargs={kwargs}")

    def _model_predict(self, x, mask=None, timesteps=None):
        """
        Run predict on base model. If the output is binary, i.e. num_class = 1, we will make it
        into a probability distribution by append (p, 1-p) to it.
        """

        p = self.base_model.predict(x, mask, timesteps, return_all=False)
        if self.base_model.num_states == 1:
            prob_distribution = torch.cat((1 - p, p), dim=1)
            return prob_distribution
        return p

    def attribute(self, input) -> np.ndarray:
        """
        Compute WinIT attribution.

        Args:
            input: Tuple of (x, mask, label) where:
                x: Input tensor of shape (batch_size, time_steps, features)
                mask: Optional mask tensor of same shape as x
                label: Optional label tensor
        """
        x, mask, _ = input
        self.base_model.eval()
        self.base_model.zero_grad()

        batch_size, num_timesteps, num_features = x.shape
        score = np.zeros((batch_size, num_features, num_timesteps))
        with torch.no_grad():
            timesteps = (
                torch.linspace(0, 1, num_timesteps, device=x.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )

            scores = []
            x_first = x[:, 0:1, :]
            p_y = self._model_predict(x_first, mask[:, 0:1, :], timesteps[:, 0:1])

            first_score = np.zeros((batch_size, num_features, self.window_size))
            for f in range(num_features):
                if self.config.last_timestep_only and f != num_features - 1:
                    continue

                x_baseline = x_first.clone()
                x_baseline[:, 0, f] = 0.0

                mask_baseline = None
                if mask is not None:
                    mask_baseline = mask[:, 0:1, :].clone()
                    mask_baseline[:, 0, f] = 1

                p_y_baseline = self._model_predict(
                    x_baseline, mask_baseline, timesteps[:, 0:1]
                )
                if self.metric == "kl":
                    score = torch.sum(
                        torch.nn.KLDivLoss(reduction="none")(
                            torch.log(p_y_baseline), p_y
                        ),
                        -1,
                    )
                elif self.metric == "js":
                    average = (p_y_baseline + p_y) / 2
                    lhs = torch.nn.KLDivLoss(reduction="none")(
                        torch.log(average), p_y_baseline
                    )
                    rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y)
                    score = torch.sum((lhs + rhs) / 2, -1)
                else:  # pd
                    score = torch.sum(torch.abs(p_y_baseline - p_y), -1)
                first_score[:, f, 0] = score.detach().cpu().numpy()

            scores.append(first_score)
            for t in range(1, num_timesteps):
                if self.config.last_timestep_only and t != num_timesteps - 1:
                    scores.append(
                        np.zeros((batch_size, num_features, self.window_size))
                    )
                    continue

                window_size = min(t, self.window_size)
                if t == 0:
                    scores.append(
                        np.zeros((batch_size, num_features, self.window_size))
                    )
                    continue

                p_y = self._model_predict(
                    x[:, : t + 1, :], mask[:, : t + 1, :], timesteps[:, : t + 1]
                )
                iS_array = np.zeros(
                    (num_features, window_size, batch_size), dtype=float
                )
                for n in range(window_size):
                    time_past = t - n
                    time_forward = n + 1
                    counterfactuals = self._generate_counterfactuals(
                        time_forward, x[:, :time_past, :], x[:, time_past : t + 1, :]
                    )

                    for f in range(num_features):
                        x_hat_in = (
                            x[:, : t + 1, :]
                            .unsqueeze(0)
                            .repeat(self.num_samples, 1, 1, 1)
                        )
                        x_hat_in[:, :, time_past : t + 1, f] = counterfactuals[
                            f, :, :, :
                        ]

                        mask_hat_in = (
                            mask[:, : t + 1, :]
                            .unsqueeze(0)
                            .repeat(self.num_samples, 1, 1, 1)
                        )
                        mask_hat_in[:, :, time_past : t + 1, f] = 0  # Values exist

                        time_hat_in = (
                            timesteps[:, : t + 1]
                            .unsqueeze(0)
                            .repeat(self.num_samples, 1, 1)
                        )

                        p_y_hat = self._model_predict(
                            x_hat_in.reshape(
                                self.num_samples * batch_size, t + 1, num_features
                            ),
                            mask_hat_in.reshape(
                                self.num_samples * batch_size, t + 1, num_features
                            ),
                            time_hat_in.reshape(self.num_samples * batch_size, t + 1),
                        )
                        p_y_exp = (
                            p_y.unsqueeze(0)
                            .repeat(self.num_samples, 1, 1)
                            .reshape(self.num_samples * batch_size, p_y.shape[-1])
                        )

                        iSab_sample = self._compute_metric(p_y_exp, p_y_hat).reshape(
                            self.num_samples, batch_size
                        )
                        iSab = torch.mean(iSab_sample, dim=0).detach().cpu().numpy()
                        iSab = np.clip(iSab, -1e6, 1e6)
                        iS_array[f, n, :] = iSab

                # Compute the I(S) array
                b = iS_array[:, 1:, :] - iS_array[:, :-1, :]
                iS_array[:, 1:, :] = b

                score = iS_array[:, ::-1, :].transpose(2, 0, 1)
                if score.shape[2] < self.window_size:
                    score = np.pad(
                        score, ((0, 0), (0, 0), (self.window_size - score.shape[2], 0))
                    )
                scores.append(score)

            scores = np.stack(scores).transpose((1, 0, 2, 3))
            return self.aggregate_scores(scores, self.aggregate_method)

    def _compute_metric(
        self, p_y_exp: torch.Tensor, p_y_hat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the metric for comparisons of two distributions.

        Args:
            p_y_exp:
                The current expected distribution. Shape = (batch_size, num_states)
            p_y_hat:
                The modified (counterfactual) distribution. Shape = (batch_size, num_states)

        Returns:
            The result Tensor of shape (batch_size).

        """
        if self.metric == "kl":
            return torch.sum(
                torch.nn.KLDivLoss(reduction="none")(torch.log(p_y_hat), p_y_exp), -1
            )
        if self.metric == "js":
            average = (p_y_hat + p_y_exp) / 2
            lhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_hat)
            rhs = torch.nn.KLDivLoss(reduction="none")(torch.log(average), p_y_exp)
            return torch.sum((lhs + rhs) / 2, -1)
        if self.metric == "pd":
            diff = torch.abs(p_y_hat - p_y_exp)
            return torch.sum(diff, -1)
        raise Exception(f"unknown metric. {self.metric}")

    def _init_generators(self):
        if self.joint:
            gen_path = self.path / "joint_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = JointFeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=self.num_features * 3,
                prediction_size=self.window_size,
                data=self.data_name,
            )
        else:
            gen_path = self.path / "feature_generator"
            gen_path.mkdir(parents=True, exist_ok=True)
            self.generators = FeatureGenerator(
                self.num_features,
                self.device,
                gen_path,
                hidden_size=50,
                prediction_size=self.window_size,
                conditional=self.conditional,
                data=self.data_name,
            )

    def train_generators(
        self, train_loader, valid_loader, num_epochs=300
    ) -> GeneratorTrainingResults:
        self._init_generators()
        return self.generators.train_generator(train_loader, valid_loader, num_epochs)

    def test_generators(self, test_loader) -> float:
        test_loss = self.generators.test_generator(test_loader)
        self.log.info(f"Generator Test MSE Loss: {test_loss}")
        return test_loss

    def load_generators(self) -> None:
        self._init_generators()
        self.generators.load_generator()

    def _generate_counterfactuals(
        self, time_forward: int, x_in: torch.Tensor, x_current: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x_in: Input tensor of shape (batch_size, time_steps, features)
            time_forward: Number of timesteps to generate counterfactuals for
            x_current: Optional current state tensor of shape (batch_size, time_steps, features)
        Returns:
            Counterfactuals tensor of shape (features, n_samples, batch_size, time_forward)
        """
        batch_size, num_time, _ = x_in.shape
        counterfactuals = torch.zeros(
            (self.num_features, self.num_samples, batch_size, time_forward),
            device=self.device,
        )

        # For t=0 or single timestep, use zero strategy
        if num_time <= 1:
            self.log.info("Using zero strategy for t=0 or single timestep")
            return counterfactuals

        # Generate generator-based counterfactuals
        generator_features = [
            f for f, s in self.feature_strategies.items() if s == "generator"
        ]
        if generator_features and self.generators is not None:
            # Input is already in correct shape (B x T x F)
            x_in_gen = x_in
            x_current_gen = x_current if x_current is not None else None

            # self.log.info(f"x_in_gen shape: {x_in_gen.shape}")

            if isinstance(self.generators, FeatureGenerator):
                # Handle FeatureGenerator
                mu, std = self.generators.forward(
                    x_current_gen, x_in_gen, deterministic=True
                )
                mu = mu[:, :time_forward, :]  # (bs, time_forward, f)
                std = std[:, :time_forward, :]  # (bs, time_forward, f)

                # Generate samples
                samples = mu.unsqueeze(0) + torch.randn(
                    self.num_samples,
                    batch_size,
                    time_forward,
                    self.num_features,
                    device=self.device,
                ) * std.unsqueeze(0)

                # Reshape to match expected output format (f, ns, bs, time_forward)
                counterfactuals = samples.permute(3, 0, 1, 2)

            elif isinstance(self.generators, JointFeatureGenerator):
                # Handle JointFeatureGenerator
                for f in generator_features:
                    mu_z, std_z = self.generators.get_z_mu_std(x_in_gen)
                    gen_out, _ = (
                        self.generators.forward_conditional_multisample_from_z_mu_std(
                            x_in_gen,
                            x_current_gen,
                            list(set(range(self.num_features)) - {f}),
                            mu_z,
                            std_z,
                            self.num_samples,
                        )
                    )

                    if gen_out is None:
                        self.log.warning(
                            f"Generator output is None for feature {f}. Using carry_forward."
                        )
                        self.feature_strategies[f] = "carry_forward"
                        continue

                    # gen_out shape (ns, bs, time_forward, f)
                    counterfactuals[f, :, :, :] = gen_out[:, :, :, f]
            else:
                raise ValueError(f"Unknown generator type: {type(self.generators)}")

        # Handle carry_forward features
        last_observation = x_in[:, -1, :]
        is_timestep_zero = num_time == 1

        for f, strategy in self.feature_strategies.items():
            if strategy == "carry_forward" and not is_timestep_zero:
                counterfactuals[f, :, :, :] = (
                    last_observation[:, f]
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(self.num_samples, 1, time_forward)
                )

        return counterfactuals

    def get_name(self):
        # builder = ["winit", "window", str(self.window_size)]
        # if self.num_samples != 3:
        #     builder.extend(["samples", str(self.num_samples)])
        # if self.conditional:
        #     builder.append("cond")
        # if self.joint:
        #     builder.append("joint")
        # builder.append(self.metric)
        # if self.data_distribution is not None:
        #     builder.append("usedatadist")
        # return "_".join(builder)
        return "WinIT"

    def _get_base_name(self) -> str:
        """Return the base name for this explainer"""
        return "WinIT"

    def _initialize_explainer(self):
        """Initialize the explainer-specific components"""
        # Basic initialization
        self.window_indices = None
        self.importance_scores = None
        self.feature_importance = None
        # Any additional initialization can be added here
        pass

    def aggregate_scores(
        self, scores: np.ndarray, aggregate_method: str | None = None
    ) -> np.ndarray:
        """
        Aggregate WinIT importance scores across the window dimension.

        Args:
            scores: The input importance scores. Shape = (num_samples, num_features, num_times, window_size)
                   or (num_samples, num_features, num_times).
            aggregate_method: The aggregation method - one of "absmax", "max", or "mean".
                            If None, uses self.aggregate_method.

        Returns:
            Aggregated scores as numpy array with shape (num_samples, num_features, num_times)
        """
        # Use default aggregate_method if none provided
        if aggregate_method is None:
            aggregate_method = self.aggregate_method

        if scores.ndim == 3:
            return scores

        num_samples, num_features, num_times, window_size = scores.shape
        aggregated_scores = np.zeros((num_samples, num_features, num_times))

        for t in range(num_times):
            # Get windows where current observation is included
            relevant_windows = np.arange(t, min(t + window_size, num_times))
            # Get position of current observation within each window
            relevant_obs = -relevant_windows + t - 1
            # Extract relevant scores
            relevant_scores = scores[:, :, relevant_windows, relevant_obs]
            relevant_scores = np.nan_to_num(relevant_scores)

            if aggregate_method == "absmax":
                # Take value with largest absolute magnitude
                score_max = relevant_scores.max(axis=-1)
                score_min = relevant_scores.min(axis=-1)
                aggregated_scores[:, :, t] = np.where(
                    -score_min > score_max, score_min, score_max
                )
            elif aggregate_method == "max":
                # Take maximum value
                aggregated_scores[:, :, t] = relevant_scores.max(axis=-1)
            elif aggregate_method == "mean":
                # Take mean value
                aggregated_scores[:, :, t] = relevant_scores.mean(axis=-1)
            else:
                raise NotImplementedError(
                    f"Aggregation method {aggregate_method} unrecognized"
                )

        return aggregated_scores
