from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import logging
import pdb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import (
    Attribution,
    Lime,
    ShapleyValueSampling,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    DeepLift,
)

from deltashap.models import TorchModel
from deltashap.explainer.generator.generator import GeneratorTrainingResults
from deltashap.utils import resolve_device
# from winit.config import NORMAL_VALUE, FEATURE_MAP

class BaselineType(Enum):
    ZERO = "zero"
    CARRY_FORWARD = "carry_forward"
    TIME_DELTA = "time_delta"


@dataclass
class ExplainerConfig:
    """Configuration for explainers"""

    device: Optional[str] = None
    n_samples: int = 50  # Default matches current usage in runner
    baseline_type: BaselineType = BaselineType.CARRY_FORWARD
    additional_args: Dict[str, Any] = None


class BaseExplainer(abc.ABC):
    """
    A base class for explainer.
    """

    def __init__(self, config: ExplainerConfig):
        self.base_model: Optional[TorchModel] = None
        self.device = resolve_device(config.device)
        self.config = config
        self.explainer: Optional[Attribution] = None

    def create_baseline(self, x, time_delta=1) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        if self.config.baseline_type == BaselineType.ZERO:
            baseline = torch.zeros_like(x)
        elif self.config.baseline_type == BaselineType.TIME_DELTA:
            # import pdb; pdb.set_trace()
            baseline = x.clone()
            baseline[:, :-time_delta, :] = x[:, :-time_delta, :]
            baseline[:, -time_delta:, :] = x[:, -time_delta-1:-1, :]
        else:
            baseline = x.clone()
            baseline[:, 1:, :] = x[:, :-1, :]
        return baseline

    def setup_attribution(self) -> None:
        """Common setup for attribution"""
        self.base_model.zero_grad()
        self.base_model.eval()
        self.orig_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"

    def cleanup_attribution(self) -> None:
        """Common cleanup after attribution"""
        torch.backends.cudnn.enabled = self.orig_cudnn

    def set_model(self, model: TorchModel, set_eval: bool = True) -> None:
        """Set the model to explain"""
        self.base_model = model
        if set_eval:
            self.base_model.eval()
        self.base_model.to(self.device)
        self._initialize_explainer()

    @abc.abstractmethod
    def _initialize_explainer(self) -> None:
        """Initialize the specific explainer implementation"""

    @abc.abstractmethod
    def attribute(self, x) -> np.ndarray:
        """Compute attributions"""

    @abc.abstractmethod
    def _get_base_name(self) -> str:
        """Return base name of the explainer"""

    def get_name(self) -> str:
        """Return the full name of the explainer, including baseline type"""
        base_name = self._get_base_name()
        baseline_suffix = f"_{self.config.baseline_type.value}" if self.config.baseline_type else ""
        return f"{base_name}{baseline_suffix}"

    def train_generators(
        self, train_loader, valid_loader, num_epochs=300
    ) -> GeneratorTrainingResults | None:
        """
        If the explainer or attribution method needs a generator, this will train the generator.

        Args:
            train_loader:
                The dataloader for training
            valid_loader:
                The dataloader for validation.
            num_epochs:
                The number of epochs.

        Returns:
            The training results for the generator, if applicable. This includes the
            training curves.

        """
        return None

    def test_generators(self, test_loader) -> float | None:
        """
        If the explainer or attribution method needs a generator, this will return the performance
        of the generator on the test set.

        Args:
            test_loader:
                The dataloader for testing.

        Returns:
            The test result (MSE) for the generator, if applicable.

        """
        return None

    def load_generators(self) -> None:
        """
        If the explainer or attribution method needs a generator, this will load the generator from
        the disk.
        """


class MockExplainer(BaseExplainer):
    """
    Class for mock explainer. The mock explainer returns all the attributes to 0.
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)

    def _initialize_explainer(self) -> None:
        pass

    def attribute(self, x) -> np.ndarray:
        return np.zeros(x.shape)

    def _get_base_name(self) -> str:
        return "Mock"


class RandomExplainer(BaseExplainer):
    """
    Class for random explainer. The random explainer returns random attributes.
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)

    def _initialize_explainer(self) -> None:
        pass

    def attribute(self, x) -> np.ndarray:
        return np.random.randn(*x.shape)

    def _get_base_name(self) -> str:
        return "Random"


class LIMEExplainer(BaseExplainer):
    """
    The explainer for LIME (Local Interpretable Model-agnostic Explanations).
    Uses Captum's implementation with a linear interpretable model.
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)
        self.explainer = None

    def _initialize_explainer(self) -> None:
        from captum._utils.models.linear_model import SkLearnLinearModel

        # Wrap the model to ensure output is in the expected format for LIME
        def wrapper_fn(*args, **kwargs):
            output = self.base_model.predict(*args, **kwargs)
            # Add an extra dimension to match Captum's expectations
            return output.unsqueeze(-1)

        self.explainer = Lime(
            forward_func=wrapper_fn,
            interpretable_model=SkLearnLinearModel("linear_model.Ridge", alpha=1.0),
        )

    def preprocess_tensor(self, x):
        """
        Reshape tensor and handle NaN values with forward and backward fill

        Args:
            x (torch.Tensor): Input tensor of shape [batch, time, features]
            device (torch.device): Target device

        Returns:
            torch.Tensor: Cleaned and reshaped tensor
        """
        # Reshape to 2D: [batch * time, features]
        x_reshaped = x.reshape(-1, x.shape[-1])
        x_np = x_reshaped.detach().cpu().numpy()

        df = pd.DataFrame(x_np)
        df_forward = df.ffill()
        df_backward = df_forward.bfill()

        x_cleaned = torch.tensor(df_backward.values, dtype=x.dtype)
        x_restored = x_cleaned.reshape(x.shape)
        return x_restored

    def attribute(self, input) -> np.ndarray:
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        x, mask, _ = input
        x = self.preprocess_tensor(x).to(self.device)

        baseline = self.create_baseline(x)
        score = self.explainer.attribute(
            inputs=x,
            baselines=baseline,
            additional_forward_args=(mask,),
            # n_samples=self.config.n_samples,
            # perturbations_per_eval=1,
            show_progress=False,
        )
        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score.detach().cpu().numpy()

    def _get_base_name(self) -> str:
        return "LIME"


class ShapExplainer(BaseExplainer):
    """
    The explainer for Shapley Values using carry-forward as the baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)
        self.explainer = None

    def _initialize_explainer(self) -> None:
        self.explainer = ShapleyValueSampling(self.base_model.predict)

    def attribute(self, input) -> np.ndarray:
        self.base_model.zero_grad()
        self.base_model.eval()

        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        # import pdb; pdb.set_trace()

        x, mask, _ = input
        x = x.to(self.device)
        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"

        baseline = self.create_baseline(x)
        score = self.explainer.attribute(
            x,
            baselines=baseline,
            additional_forward_args=(mask,),
            show_progress=False,
        )

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score.detach().cpu().numpy()

    def _get_base_name(self) -> str:
        return "Shapley"


from typing import Callable, cast, Generator, Optional, Tuple, Union

import torch
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.lime import construct_feature_mask, Lime
from captum.attr._utils.common import _format_input_baseline
from captum.log import log_usage
from torch import Tensor
from torch.distributions.categorical import Categorical


class KernelShap(Lime):
    r"""
    Kernel SHAP is a method that uses the LIME framework to compute
    Shapley Values. Setting the loss function, weighting kernel and
    regularization terms appropriately in the LIME framework allows
    theoretically obtaining Shapley Values more efficiently than
    directly computing Shapley Values.

    More information regarding this method and proof of equivalence
    can be found in the original paper here:
    https://arxiv.org/abs/1705.07874
    """

    def __init__(self, forward_func: Callable[..., Tensor]) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        Lime.__init__(
            self,
            forward_func,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=self.kernel_shap_similarity_kernel,
            perturb_func=self.kernel_shap_perturb_generator,
        )
        self.inf_weight = 1000000.0

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above,
        training an interpretable model based on KernelSHAP and returning a
        representation of the interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME / KernelShap
        is generally used for sample-based interpretability, training a separate
        interpretable model to explain a model's prediction on each individual example.

        A batch of inputs can also be provided as inputs, similar to
        other perturbation-based attribution methods. In this case, if forward_fn
        returns a scalar per example, attributions will be computed for each
        example independently, with a separate interpretable model trained for each
        example. Note that provided similarity and perturbation functions will be
        provided each example separately (first dimension = 1) in this case.
        If forward_fn returns a scalar per batch (e.g. loss), attributions will
        still be computed using a single interpretable model for the full batch.
        In this case, similarity and perturbation functions will be provided the
        same original input containing the full batch.

        The number of interpretable features is determined from the provided
        feature mask, or if none is provided, from the default feature mask,
        which considers each scalar input as a separate feature. It is
        generally recommended to provide a feature mask which groups features
        into a small number of interpretable features / components (e.g.
        superpixels in images).

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which KernelShap
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define the reference value which replaces each
                        feature when the corresponding interpretable feature
                        is set to 0.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.
                        Default: None
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which surrogate model is trained
                        (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            feature_mask (Tensor or tuple[Tensor, ...], optional):
                        feature_mask defines a mask for the input, grouping
                        features which correspond to the same
                        interpretable feature. feature_mask
                        should contain the same number of tensors as inputs.
                        Each tensor should
                        be the same size as the corresponding input or
                        broadcastable to match the input tensor. Values across
                        all tensors should be integers in the range 0 to
                        num_interp_features - 1, and indices corresponding to the
                        same feature should have the same value.
                        Note that features are grouped across tensors
                        (unlike feature ablation and occlusion), so
                        if the same index is used in different tensors, those
                        features are still grouped and added simultaneously.
                        If None, then a feature mask is constructed which assigns
                        each scalar within a tensor as a separate feature.
                        Default: None
            n_samples (int, optional): The number of samples of the original
                        model used to train the surrogate interpretable model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
                        to be processed simultaneously in one call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function returns a single scalar per batch,
                        perturbations_per_eval must be set to 1.
                        Default: 1
            return_input_shape (bool, optional): Determines whether the returned
                        tensor(s) only contain the coefficients for each interp-
                        retable feature from the trained surrogate model, or
                        whether the returned attributions match the input shape.
                        When return_input_shape is True, the return type of attribute
                        matches the input shape, with each element containing the
                        coefficient of the corresponding interpretable feature.
                        All elements with the same value in the feature mask
                        will contain the same coefficient in the returned
                        attributions. If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpretable model, with length
                        num_interp_features.
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The attributions with respect to each input feature.
                        If return_input_shape = True, attributions will be
                        the same size as the provided inputs, with each value
                        providing the coefficient of the corresponding
                        interpretale feature.
                        If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpreatable models, with length
                        num_interp_features.
        Examples::
            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()

            >>> # Generating random input with size 1 x 4 x 4
            >>> input = torch.randn(1, 4, 4)

            >>> # Defining KernelShap interpreter
            >>> ks = KernelShap(net)
            >>> # Computes attribution, with each of the 4 x 4 = 16
            >>> # features as a separate interpretable feature
            >>> attr = ks.attribute(input, target=1, n_samples=200)

            >>> # Alternatively, we can group each 2x2 square of the inputs
            >>> # as one 'interpretable' feature and perturb them together.
            >>> # This can be done by creating a feature mask as follows, which
            >>> # defines the feature groups, e.g.:
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # With this mask, all inputs with the same value are set to their
            >>> # baseline value, when the corresponding binary interpretable
            >>> # feature is set to 0.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])

            >>> # Computes KernelSHAP attributions with feature mask.
            >>> attr = ks.attribute(input, target=1, feature_mask=feature_mask)
        """
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )
        num_features_list = torch.arange(num_interp_features, dtype=torch.float)
        denom = num_features_list * (num_interp_features - num_features_list)
        probs = torch.tensor((num_interp_features - 1)) / denom
        probs[0] = 0.0
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            num_select_distribution=Categorical(probs),
            show_progress=show_progress,
        )

    # pyre-fixme[24] Generic type `Callable` expects 2 type parameters.
    def attribute_future(self) -> Callable:
        r"""
        This method is not implemented for KernelShap.
        """
        raise NotImplementedError("attribute_future is not implemented for KernelShap")

    def kernel_shap_similarity_kernel(
        self,
        _,
        __,
        interpretable_sample: Tensor,
        **kwargs: object,
    ) -> Tensor:
        assert (
            "num_interp_features" in kwargs
        ), "Must provide num_interp_features to use default similarity kernel"
        num_selected_features = int(interpretable_sample.sum(dim=1).item())
        num_features = kwargs["num_interp_features"]
        if num_selected_features == 0 or num_selected_features == num_features:
            # weight should be theoretically infinite when
            # num_selected_features = 0 or num_features
            # enforcing that trained linear model must satisfy
            # end-point criteria. In practice, it is sufficient to
            # make this weight substantially larger so setting this
            # weight to 1000000 (all other weights are 1).
            similarities = self.inf_weight
        else:
            similarities = 1.0
        return torch.tensor([similarities])

    def kernel_shap_perturb_generator(
        self,
        original_inp: Union[Tensor, Tuple[Tensor, ...]],
        **kwargs: object,
    ) -> Generator[Tensor, None, None]:
        r"""
        Perturbations are sampled by the following process:
         - Choose k (number of selected features), based on the distribution
                p(k) = (M - 1) / (k * (M - k))

            where M is the total number of features in the interpretable space

         - Randomly select a binary vector with k ones, each sample is equally
            likely. This is done by generating a random vector of normal
            values and thresholding based on the top k elements.

         Since there are M choose k vectors with k ones, this weighted sampling
         is equivalent to applying the Shapley kernel for the sample weight,
         defined as:
         k(M, k) = (M - 1) / (k * (M - k) * (M choose k))
        """
        assert (
            "num_select_distribution" in kwargs and "num_interp_features" in kwargs
        ), (
            "num_select_distribution and num_interp_features are necessary"
            " to use kernel_shap_perturb_func"
        )
        if isinstance(original_inp, Tensor):
            device = original_inp.device
        else:
            device = original_inp[0].device
        num_features = cast(int, kwargs["num_interp_features"])
        
        # First yield the edge cases (all ones and all zeros)
        yield torch.ones(1, num_features, device=device, dtype=torch.long)
        yield torch.zeros(1, num_features, device=device, dtype=torch.long)
        
        # Batch size for parallel processing
        batch_size = 32  # Adjust based on memory constraints
        
        while True:
            # Generate multiple samples at once
            batch_masks = []
            for _ in range(batch_size):
                num_selected_features = cast(
                    Categorical, kwargs["num_select_distribution"]
                ).sample()
                rand_vals = torch.randn(1, num_features)
                threshold = torch.kthvalue(
                    rand_vals, num_features - num_selected_features
                ).values.item()
                batch_masks.append((rand_vals > threshold).to(device=device).long())
            
            # Stack masks into a single batch
            stacked_masks = torch.cat(batch_masks, dim=0)
            
            # Yield each mask in the batch
            for i in range(stacked_masks.shape[0]):
                yield stacked_masks[i:i+1]


class KernelShapExplainer(BaseExplainer):
    """
    The explainer for KernelSHAP using zeros as the baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)
        self.explainer = None

    def _initialize_explainer(self) -> None:
        self.explainer = KernelShap(self.base_model.predict)

    def attribute(self, input) -> np.ndarray:
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        x, mask, _ = input
        x = x.to(self.device)
        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"
        
        perturbations_per_eval = len(x)
        
        baseline = self.create_baseline(x)
        score = self.explainer.attribute(
            x,
            n_samples=25,
            baselines=baseline,
            additional_forward_args=(mask,),
            perturbations_per_eval=perturbations_per_eval,
            show_progress=False,  # Enable progress bar to monitor execution
        )
        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score.detach().cpu().numpy()

    def _get_base_name(self) -> str:
        return "KernelSHAP"


class GradientShapExplainer(BaseExplainer):
    """
    The explainer for GradientSHAP using zeros as the baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)
        self.explainer = None

    def _initialize_explainer(self) -> None:
        self.explainer = GradientShap(self.base_model.predict, multiply_by_inputs=False)

    def attribute(self, input) -> np.ndarray:
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        x, mask, _ = input
        x = x.to(self.device)
        baseline = self.create_baseline(x)
        score = self.explainer.attribute(
            x,
            baselines=torch.cat([0 * x, 1 * x]),
            # baselines=baseline,
            additional_forward_args=(mask,),
        )

        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score.detach().cpu().numpy()

    def _get_base_name(self) -> str:
        return "GradientSHAP"


class DeepLiftShapExplainer(BaseExplainer):
    """
    The explainer for DeepLiftSHAP using zeros as the baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)
        self.explainer = None

    def _initialize_explainer(self) -> None:
        # Wrap the predict method in a module if it's a function
        class ModelWrapper(nn.Module):
            def __init__(self, predict_func):
                super().__init__()
                self.predict = predict_func

            def forward(self, x, mask, *args, **kwargs):
                return self.predict(x, mask, *args, **kwargs)

        wrapped_model = ModelWrapper(self.base_model.predict)
        self.explainer = DeepLiftShap(wrapped_model)

    def attribute(self, input) -> np.ndarray:
        self.base_model.zero_grad()
        self.base_model.eval()

        # Save and restore cudnn enabled
        orig_cudnn_setting = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        x, mask, _ = input
        x = x.to(self.device)
        assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"
        baseline = self.create_baseline(x)
        score = self.explainer.attribute(
            x,
            # n_samples=10,
            baselines=baseline,
            additional_forward_args=(mask,),
        )
        torch.backends.cudnn.enabled = orig_cudnn_setting
        return score.detach().cpu().numpy()

    def _get_base_name(self) -> str:
        return "DeepLiftSHAP"


# class GradientShapExplainer(BaseExplainer):
#     """
#     The explainer for GradientSHAP using zeros as the baseline and the captum
#     implementation. Multiclass case is not implemented.
#     """

#     def __init__(self, config: ExplainerConfig):
#         super().__init__(config)
#         self.explainer = None

#     def _initialize_explainer(self) -> None:
#         self.explainer = GradientShap(self.base_model.predict)

#     def attribute(self, input) -> np.ndarray:
#         self.base_model.zero_grad()
#         self.base_model.eval()

#         # Save and restore cudnn enabled
#         orig_cudnn_setting = torch.backends.cudnn.enabled
#         torch.backends.cudnn.enabled = False

#         x, mask, _ = input
#         x = x.to(self.device)
#         assert self.base_model.num_states == 1, "TODO: Implement for > 1 class"
#         baseline = self.create_baseline(x)
#         score = self.explainer.attribute(
#             x,
#             # n_samples=10,
#             baselines=baseline,
#             additional_forward_args=(mask,),
#         )
#         torch.backends.cudnn.enabled = orig_cudnn_setting
#         return score.detach().cpu().numpy()

#     def _get_base_name(self) -> str:
#         return "GradientSHAP"


class IGExplainer(BaseExplainer):
    """
    The explainer for integrated gradients using configurable baseline and the captum
    implementation. Multiclass case is not implemented.
    """

    def _initialize_explainer(self) -> None:
        self.explainer = IntegratedGradients(self.base_model.predict)

    def attribute(self, input) -> np.ndarray:
        self.setup_attribution()
        x, mask, _ = input
        baseline = self.create_baseline(x)

        # import pdb; pdb.set_trace()

        # Add memory optimization parameters
        score = self.explainer.attribute(
            x,
            baselines=baseline,
            additional_forward_args=(mask,),
            # internal_batch_size=x.shape[0],
        )

        self.cleanup_attribution()
        return score.detach().cpu().numpy()

    def _get_base_name(self) -> str:
        return "IG"
        
    def get_name(self) -> str:
        """Return the full name of the explainer, including baseline type"""
        base_name = self._get_base_name()
        baseline_suffix = f"_{self.config.baseline_type.value}" if self.config.baseline_type else ""
        return f"{base_name}{baseline_suffix}"


class DeepLiftExplainer(BaseExplainer):
    """
    The explainer for the DeepLift method using zeros as the baseline and captum for the
    implementation.
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)
        self.explainer = None

    def _initialize_explainer(self) -> None:
        # # Wrap the predict method in a module if it's a function
        class ModelWrapper(nn.Module):
            def __init__(self, predict_func):
                super().__init__()
                self.predict = predict_func

            def forward(self, x, mask, *args, **kwargs):
                return self.predict(x, mask, *args, **kwargs)

        wrapped_model = ModelWrapper(self.base_model.predict)
        self.explainer = DeepLift(wrapped_model)

    def attribute(self, input) -> np.ndarray:
        # import pdb; pdb.set_trace()

        # self.base_model.zero_grad()
        # self.base_model.eval()

        # orig_cudnn_setting = torch.backends.cudnn.enabled
        # torch.backends.cudnn.enabled = False

        # assert (
        #     self.base_model.num_states == 1
        # ), "TODO: Implement retrospective for > 1 class"

        self.setup_attribution()
        x, mask, _ = input
        baseline = self.create_baseline(x)
        score = self.explainer.attribute(
            x,
            baselines=baseline,
            additional_forward_args=(mask,),
        )

        self.cleanup_attribution()
        return score.detach().cpu().numpy()

    def _get_base_name(self) -> str:
        return "DeepLIFT"


class FOExplainer(BaseExplainer):
    """
    The explainer for feature occlusion. The implementation is simplified from the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(self, config: ExplainerConfig):
        super().__init__(config)
        self.explainer = None

    def _initialize_explainer(self) -> None:
        pass

    def attribute(self, input) -> np.ndarray:
        self.base_model.eval()
        self.base_model.zero_grad()

        x, mask, _ = input
        batch_size, t_len, n_features = x.shape
        score = np.zeros(x.shape)

        timesteps = (
            torch.linspace(0, 1, t_len, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        # import pdb; pdb.set_trace()
        
        for t in range(t_len):
            # Fixed order of arguments: x, mask, delta, static, timesteps
            p_y_t = self.base_model.predict(
                x[:, : t + 1, :],
                mask[:, : t + 1, :],
                timesteps[:, : t + 1],
            )
            for i in range(n_features):
                x_hat = x[:, : t + 1, :].clone()
                mask_hat = mask[:, : t + 1, :].clone()
                kl_score = 0
                if self.config.baseline_type == BaselineType.CARRY_FORWARD:
                    if t == 0:
                        x_hat[:, t, i] = 0
                        mask_hat[:, t, i] = 0  # Value exists
                    else:
                        x_hat[:, t, i] = x_hat[:, t, i - 1]
                        mask_hat[:, t, i] = mask_hat[:, t, i - 1]
                else:
                    x_hat[:, t, i] = 0
                    mask_hat[:, t, i] = 0  # Value doesn't exist

                y_hat_t = self.base_model.predict(
                    x_hat,
                    mask_hat,
                    timesteps[:, : t + 1],
                )
                kl = torch.abs(y_hat_t - p_y_t)
                kl_score = np.mean(kl.detach().cpu().numpy(), -1)
                score[:, t, i] = kl_score
        return score

    def _get_base_name(self) -> str:
        return "FO"


class AFOExplainer(BaseExplainer):
    """
    The explainer for augmented feature occlusion. The implementation is simplified from
    the FIT repository.
    https://github.com/sanatonek/time_series_explainability/blob/master/TSX/explainers.py
    """

    def __init__(self, config: ExplainerConfig, train_loader=None):
        super().__init__(config)
        if train_loader is not None:
            trainset = list(train_loader.dataset)
            self.data_distribution = torch.stack([x[0] for x in trainset])
        else:
            self.log.warning("No train_loader provided for AFO. Using zero baseline.")
            self.data_distribution = None

    def _initialize_explainer(self) -> None:
        """Initialize the AFO explainer."""
        pass

    def attribute(self, input) -> np.ndarray:
        self.base_model.eval()
        self.base_model.zero_grad()

        x, mask, _ = input
        batch_size, t_len, n_features = x.shape
        score = np.zeros(x.shape)

        timesteps = (
            torch.linspace(0, 1, t_len, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        for t in range(t_len):
            if self.config.last_timestep_only and t != t_len - 1:
                continue

            p_y_t = self.base_model.predict(
                x[:, : t + 1, :],
                mask[:, : t + 1, :],
                timesteps[:, : t + 1],
            )
            for i in range(n_features):
                x_hat = x[:, : t + 1, :].clone()
                mask_hat = mask[:, : t + 1, :].clone()
                kl_score = 0
                if self.config.baseline_type == BaselineType.CARRY_FORWARD:
                    if t == 0:
                        # Set first timestep to normal/median value for this feature
                        # feature_name = [k for k, v in FEATURE_MAP.items() if v == i][0]
                        # x_hat[:, t, i] = NORMAL_VALUE[feature_name]
                        x_hat[:, t, i] = 0
                        mask_hat[:, t, i] = 0  # Value exists
                    else:
                        # Carry forward the previous value
                        x_hat[:, t, i] = x_hat[:, t, i - 1]
                        mask_hat[:, t, i] = mask_hat[:, t, i - 1]
                else:
                    if self.data_distribution is not None:
                        feature_dist = np.array(
                            self.data_distribution[:, :, i]
                        ).reshape(-1)
                        x_hat[:, t, i] = torch.Tensor(
                            np.random.choice(feature_dist, size=(len(x),))
                        ).to(self.device)
                    else:
                        # Fallback to zero if no data distribution available
                        x_hat[:, t, i] = torch.zeros_like(x_hat[:, t, i])
                    mask_hat[:, t, i] = 1  # Value exists

                y_hat_t = self.base_model.predict(
                    x_hat,
                    mask_hat[:, : t + 1, :],
                    timesteps[:, : t + 1],
                    return_all=False,
                )
                kl = torch.abs(y_hat_t - p_y_t)
                kl_score = np.mean(kl.detach().cpu().numpy(), -1)
                score[:, t, i] = kl_score
        return score

    def _get_base_name(self) -> str:
        return "AFO"