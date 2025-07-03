from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from deltashap.dataloader import DeltaSHAPDataset
from deltashap.models import (
    TorchModel,
    StateClassifier,
    ConvClassifier,
    mTAND,
    SeFT,
    TransformerGRUD,
    LSTM,
    PredictWrapper,

)
from deltashap.utils import resolve_device


@dataclasses.dataclass(frozen=True)
class EpochResult:
    epoch_loss: float
    accuracy: float
    auc: float
    auprc: float
    f1: float
    precision: float
    recall: float

    def __str__(self):
        return f"Loss: {self.epoch_loss}, Acc: {100 * self.accuracy:.2f}%, Auc: {self.auc:.4f}"


class ModelTrainer:
    """
    A class for training one model.
    """

    def __init__(
        self,
        feature_size: int,
        num_timesteps: int,
        num_classes: int,
        batch_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        model_file_name: pathlib.Path,
        model_type: str = "GRU",
        device: str | torch.device | None = None,
        verbose_eval: int = 10,
        early_stopping: bool = True,
        num_ensemble: int = 10,
        multi_gpu: bool = True,
    ):
        """
        Constructor

        Args:
            feature_size:
               The number of features.
            num_classes:
               The number of classes (output nodes, i.e. num_states)
            batch_size:
               The batch size for training the model
            hidden_size:
               The hidden size for the model
            dropout:
               The dropout rate of the model in case of GRU or LSTM
            num_layers:
               The number of layers of the models in case of GRU or LSTM.
            model_file_name:
               The model file name (pathlib.Path)
            model_type:
               The model type. Can be "GRU", "LSTM" or "CONV"
            device:
               The torch device.
            verbose_eval:
               Training metrics is logged at every given verbose_eval epoch.
            early_stopping:
               Whether apply early stopping or not.
            multi_gpu:
               Whether to use multiple GPUs for the model.
        """
        self.feature_size = feature_size
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.model_type = model_type
        self.model: TorchModel | None = None
        self.num_ensemble = num_ensemble
        self.multi_gpu = multi_gpu
        self.log = logging.getLogger(ModelTrainer.__name__)

        # if model_type in ["GRU", "LSTM"] or model_type is None:
        if model_type == "GRU" or model_type is None:
            self.model = StateClassifier(
                feature_size=self.feature_size,
                num_states=self.num_classes,
                hidden_size=hidden_size,
                device=self.device,
                rnn=model_type,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_type == "CONV":
            self.model = ConvClassifier(
                feature_size=self.feature_size,
                num_states=self.num_classes,
                hidden_size=hidden_size,
                kernel_size=10,
                device=self.device,
            )
        elif model_type == "MTAND":
            self.model = mTAND(
                feature_size=self.feature_size,
                num_timesteps=self.num_timesteps,
                num_states=self.num_classes,
                device=self.device,
            )
        elif model_type == "SEFT":
            self.model = SeFT(
                feature_size=self.feature_size,
                num_timesteps=self.num_timesteps,
                num_states=self.num_classes,
                device=self.device,
            )
        elif model_type == "TRANSFORMERGRUD":
            self.model = TransformerGRUD()
        elif model_type == "LSTM":
            self.model = LSTM(
                num_inputs=self.feature_size,
                num_classes=self.num_classes,
            )
        else:
            raise ValueError(f"Invalid model type ({model_type}).")

        self.verbose_eval = verbose_eval
        self.early_stopping = early_stopping
        self.model_file_name = model_file_name
        # self.model_file_name.parent.mkdir(parents=True, exist_ok=True)

        # If using multi-GPU, determine which GPU to use for this trainer
        if self.multi_gpu and torch.cuda.device_count() > 1:
            # Extract device index if device is in format 'cuda:X'
            if isinstance(device, str) and ':' in device:
                device_idx = int(device.split(':')[1])
            else:
                device_idx = 0
            
            # Set the device for this trainer
            torch.cuda.set_device(device_idx)
            self.log.info(f"Setting CUDA device to {device_idx}")

    def _init_model(self):
        """Initialize the model and prepare it for multi-GPU if needed"""
        # Use the existing PredictWrapper from models.py
        self.model = self.model.to(self.device)
        
        # Wrap with DataParallel if multi_gpu is enabled
        if self.multi_gpu and torch.cuda.device_count() > 1:
            gpu_count = torch.cuda.device_count()
            self.log.info(f"Using {gpu_count} GPUs for model training")
            self.model = nn.DataParallel(self.model)
            self.model = PredictWrapper(self.model)

    def train_model(
        self,
        train_loader,
        valid_loader,
        test_loader,
        num_epochs,
        lr=0.001,
        weight_decay=0.001,
        use_all_times: bool = True,
    ) -> None:
        """
        Train the model on train loader, perform validation on valid loader. Evaluate
        the model on test loader after the model has been trained.

        Args:
            train_loader:
               DataLoader for training
            valid_loader:
               DataLoader for validation
            test_loader:
               DataLoader for testing
            num_epochs:
               Number of epochs of training
            lr:
               Learning rate
            weight_decay:
               Weight decay
            use_all_times:
               Use all the timestep for training. Otherwise, only use the last timestep.

        """
        # Initialize model for multi-GPU if needed
        self._init_model()
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        train_results_trend = []
        valid_results_trend = []

        best_auc = 0
        for epoch in range(num_epochs):
            train_results = self._run_one_epoch(
                train_loader,
                run_train=True,
                optimizer=optimizer,
                use_all_times=use_all_times,
            )
            valid_results = self._run_one_epoch(
                valid_loader,
                run_train=False,
                optimizer=None,
                use_all_times=use_all_times,
            )
            train_results_trend.append(train_results)
            valid_results_trend.append(valid_results)

            if epoch % self.verbose_eval == 0:
                self.log.info(f"Epoch {epoch + 1}")
                self.log.info(f"Training   ===> {train_results}")
                self.log.info(f"Validation ===> {valid_results}")

            if self.early_stopping:
                if valid_results.auc > best_auc:
                    best_auc = valid_results.auc
                    torch.save(self.model.state_dict(), str(self.model_file_name))

        if self.early_stopping:
            # retrieve best model
            self.load_model()
        else:
            # save the model at the end only
            torch.save(self.model.state_dict(), str(self.model_file_name))

        test_results = self._run_one_epoch(
            test_loader, run_train=False, optimizer=None, use_all_times=use_all_times
        )

        self.log.info(f"Test ===> {test_results}")

    def load_model(self) -> None:
        """
        Load the model for the file.
        """
        checkpoint = torch.load(str(self.model_file_name), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model = self.model.to(self.device)

        # import pdb; pdb.set_trace()

        if self.multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.model = PredictWrapper(self.model)

    def get_test_results(self, test_loader, use_all_times: bool) -> EpochResult:
        """
        Return the test results on the test set.

        Args:
            test_loader:
                The test loader
            use_all_times:
                Whether to output the test results only on the last timesteps.

        Returns:
            An EpochResult object containing all the test results.
        """
        test_results = self._run_one_epoch(
            test_loader, run_train=False, optimizer=None, use_all_times=use_all_times
        )
        return test_results

    def run_inference(
        self, data: DataLoader | torch.Tensor, with_activation=True, return_all=True
    ) -> np.ndarray:
        """
        Run inference.

        Args:
            data:
                The data to be run. Shape of the batch = (num_samples, num_features, num_times)
            with_activation:
                Whether activation should be used.
            return_all:
                Whether return all the timesteps or the last one.

        Returns:
            For binary classification (num_classes <= 2):
                A numpy array of shape (num_samples,) containing probabilities of positive class if with_activation=True,
                or logits if with_activation=False
            For multiclass (num_classes > 2):
                A numpy array of shape (num_samples, num_classes) containing class probabilities if with_activation=True,
                or logits if with_activation=False
        """
        self.model.eval()
        self.model.to(self.device)
        outputs = []

        for batch in data:
            x, mask, _ = batch
            x, mask = x.to(self.device), mask.to(self.device)
            
            # Use forward method which will be parallelized by DataParallel
            logits = self.model(x, mask)

            # For binary classification, output is shape [B, 1]
            if self.num_classes <= 2:
                logits = logits.squeeze(-1)  # Remove last dimension if present
                if with_activation:
                    probs = torch.sigmoid(logits)
            else:
                if with_activation:
                    probs = torch.softmax(logits, dim=1)

            probs = probs.detach().cpu().numpy()
            outputs.append(probs)
        return np.concatenate(outputs, axis=0)

    def _run_one_epoch(
        self,
        dataloader: DataLoader,
        run_train: bool = True,
        optimizer: Optimizer = None,
        use_all_times: bool = True,
    ) -> EpochResult:
        """
        Run one epoch of forward pass. Run backward and update if run_train is true.
        """
        # Treat num_classes=2 as binary classification
        multiclass = self.num_classes > 2
        self.model = self.model.to(self.device)
        if run_train:
            self.model.train()
            if optimizer is None:
                raise ValueError("optimizer is none in train mode.")
        else:
            self.model.eval()
        epoch_loss = 0

        # Choose loss function based on classification type
        if multiclass:
            loss_criterion = torch.nn.CrossEntropyLoss()
        else:
            loss_criterion = torch.nn.BCEWithLogitsLoss()

        all_labels, all_probs = [], []
        for batch in dataloader:
            # Unpack the batch
            x, mask, labels = batch
            x, mask, labels = (
                x.to(self.device),
                mask.to(self.device),
                labels.to(self.device),
            )

            if run_train:
                optimizer.zero_grad()

            # Pass separated signals to model
            logits = self.model(x, mask)

            if not multiclass:
                # Binary classification case (including num_classes=2)
                logits = logits.squeeze(-1)  # Remove last dimension if present
                loss = loss_criterion(logits, labels)
                # Store sigmoid of logits for metrics calculation
                probs_numpy = torch.sigmoid(logits).detach().cpu().numpy()
            else:
                # Multiclass case (num_classes > 2)
                # Ensure prob has shape [batch_size, num_classes]
                if logits.dim() == 3:  # If shape is [batch_size, 1, num_classes]
                    logits = logits.squeeze(1)
                elif logits.dim() == 1:  # If shape is [batch_size]
                    logits = logits.unsqueeze(1)  # Make it [batch_size, 1]

                # Ensure labels are proper indices
                labels = labels.long().clamp(0, self.num_classes - 1)

                loss = loss_criterion(logits, labels)
                # Apply softmax with numerical stability
                probs_numpy = (
                    torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
                )

                # Verify probabilities sum to 1
                prob_sums = np.sum(probs_numpy, axis=1)
                if not np.allclose(prob_sums, 1.0):
                    print(
                        f"Warning: Probabilities don't sum to 1! Range: {prob_sums.min():.4f} to {prob_sums.max():.4f}"
                    )
                    # Normalize probabilities
                    probs_numpy = probs_numpy / prob_sums[:, np.newaxis]

            # Check for NaN values
            if np.any(np.isnan(probs_numpy)):
                print(f"Warning: NaN detected in probabilities!")
                print(f"Probability shape: {probs_numpy.shape}")
                print(f"Non-finite values: {np.sum(~np.isfinite(probs_numpy))}")
                # Replace NaN values with 0
                probs_numpy = np.nan_to_num(probs_numpy, nan=0.0)

            epoch_loss += loss.item()

            if run_train:
                loss.backward()
                optimizer.step()

            all_probs.append(probs_numpy)
            all_labels.append(labels.detach().cpu().numpy())

        # compile results
        all_labels = np.concatenate(all_labels).astype(int)
        all_probs = np.concatenate(all_probs)

        if multiclass:
            # Multiclass metrics (num_classes > 2)
            auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
            auprc = average_precision_score(all_labels, all_probs, average="macro")
            all_preds = np.argmax(all_probs, axis=1)
            f1 = f1_score(all_labels, all_preds, average="macro")
            recall = recall_score(all_labels, all_preds, average="macro")
            precision = precision_score(all_labels, all_preds, average="macro")
            accuracy = float(np.mean(all_labels == all_preds))
        else:
            # Binary classification metrics (including num_classes=2)
            auroc = roc_auc_score(all_labels.reshape(-1), all_probs.reshape(-1))
            auprc = average_precision_score(
                all_labels.reshape(-1), all_probs.reshape(-1)
            )

            # Find optimal cutoff for F1 score
            thresholds = np.linspace(0.01, 0.99, 99)
            f1_scores = []
            for threshold in thresholds:
                preds = (all_probs.reshape(-1) >= threshold).astype(int)
                f1_scores.append(
                    f1_score(all_labels.reshape(-1), preds, zero_division=0)
                )
            optimal_cutoff = thresholds[np.argmax(f1_scores)]

            # Use optimal cutoff for all metrics
            all_preds = (all_probs.reshape(-1) >= optimal_cutoff).astype(int)
            f1 = f1_score(all_labels.reshape(-1), all_preds, zero_division=0)
            recall = recall_score(all_labels.reshape(-1), all_preds, zero_division=0)
            precision = precision_score(
                all_labels.reshape(-1), all_preds, zero_division=0
            )
            accuracy = float(np.mean(all_labels.reshape(-1) == all_preds))

        return EpochResult(
            epoch_loss=epoch_loss,
            accuracy=accuracy,
            auc=auroc,
            auprc=auprc,
            f1=f1,
            precision=precision,
            recall=recall,
        )

    def calculate_metrics(self, y_true, y_pred_proba, cutoff):
        """
        Calculate classification metrics for a given cutoff.

        Args:
            y_true: Array of true binary labels (0 or 1)
            y_pred_proba: Array of predicted probabilities
            cutoff: Classification cutoff between 0 and 1

        Returns:
            Dictionary with metrics (accuracy, precision, recall, f1)
        """
        y_pred = (y_pred_proba >= cutoff).astype(int)

        # Calculate confusion matrix
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "cutoff": cutoff,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def calculate_optimal_cutoff(self, y_true, y_pred_proba, metric="f1"):
        """
        Find optimal cutoff based on specified metric.

        Args:
            y_true: Array of true binary labels (0 or 1)
            y_pred_proba: Array of predicted probabilities
            metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')

        Returns:
            optimal_cutoff, metrics_at_optimal_cutoff
        """
        cutoffs = np.linspace(0, 1, 100)
        metrics_list = [
            self.calculate_metrics(y_true, y_pred_proba, t) for t in cutoffs
        ]

        # Find cutoff that maximizes the specified metric
        optimal_idx = np.argmax([m[metric] for m in metrics_list])
        # return cutoffs[optimal_idx], metrics_list[optimal_idx]
        return cutoffs[optimal_idx]


class ModelTrainerWithCv:
    """
    A class for training a separate model for each CV for a dataset.
    """

    def __init__(
        self,
        dataset: DeltaSHAPDataset,
        ckpt_path: pathlib.Path,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        model_type: str = "GRU",
        device: str | torch.device | None = None,
        verbose_eval: int = 10,
        early_stopping: bool = True,
        num_ensemble: int = 10,
        multi_gpu: bool = True,
    ):
        """
        Constructor

        Args:
            dataset:
                The specified dataset.
            hidden_size:
                The size of the hidden units of the models.
            dropout:
                The dropout rate of the models.
            num_layers:
                The number of layers of the models if model type is RNN or LSTM.
            model_type:
                The model type of the models. It can be "RNN", "LSTM" or "CONV".
            device:
                The torch device.
            verbose_eval:
               Training metrics is logged at every given verbose_eval epoch.
            early_stopping:
               Whether apply early stopping or not.
            ckpt_path:
               The ckeckpoint path.
        """
        self.dataset = dataset
        # self.model_path = ckpt_path / dataset.get_name()
        # self.model_path.mkdir(parents=True, exist_ok=True)
        self.model_path = ckpt_path
        print(f"{self.model_path=}")
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_args = {
            "batch_size": dataset.batch_size,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "num_layers": num_layers,
            "model_type": model_type,
        }
        self.model_trainers: Dict[int, ModelTrainer] = {}
        self.log = logging.getLogger(ModelTrainerWithCv.__name__)

        for cv in self.dataset.cv_to_use():
            self.model_trainers[cv] = ModelTrainer(
                dataset.feature_size,
                dataset.num_timesteps,
                dataset.num_classes,
                dataset.batch_size,
                hidden_size,
                dropout,
                num_layers,
                # self._model_file_name(cv),
                # self.model_path / f"model.{cv}.pth.tar",
                self.model_path,
                model_type,
                device,
                verbose_eval,
                early_stopping,
                num_ensemble,
                multi_gpu,
            )

    def train_models(
        self, num_epochs, lr=0.001, weight_decay=0.001, use_all_times: bool = True
    ) -> None:
        """
        Train the models on the dataset for each CV. Evaluate the model on the test set afterwards.

        Args:
            num_epochs:
               Number of epochs of training
            lr:
               Learning rate
            weight_decay:
               Weight decay
            use_all_times:
               Use all the timestep for training. Otherwise, only use the last timestep.
        """
        for cv, model_trainer in self.model_trainers.items():
            self.log.info(f"Training model for cv={cv}")
            model_trainer.train_model(
                self.dataset.train_loaders[cv],
                self.dataset.valid_loaders[cv],
                self.dataset.test_loader,
                num_epochs,
                lr,
                weight_decay,
                use_all_times,
            )

    def load_model(self) -> None:
        """
        Load all the models from the disk.
        """
        for cv, model_trainer in self.model_trainers.items():
            model_trainer.load_model()

    def get_test_results(self, use_all_times: bool) -> Dict[int, EpochResult]:
        """
        Return the results of each model on the test sets.

        Args:
            use_all_times:
                Indicates whether we use all timesteps for test results.
        Returns:
            A dictionary from CV to EpochResult indicating the test results for each CV.
        """
        accuracies = {}
        for cv, model_trainer in self.model_trainers.items():
            accuracies[cv] = model_trainer.get_test_results(
                self.dataset.test_loader, use_all_times=use_all_times
            )
        return accuracies

    def run_inference(
        self,
        dataloader: DataLoader,
    ) -> Dict[int, np.ndarray]:
        """
        Run inference using a dataloader.

        Args:
            dataloader: DataLoader containing the data
            with_activation: Whether activation should be used
            return_all: Whether return all timesteps or last one

        Returns:
            Dictionary of CV to numpy arrays of predictions
        """
        return self.model_trainers[0].run_inference(dataloader)