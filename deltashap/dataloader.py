from __future__ import annotations

import abc
import os
import pathlib
import pickle
from typing import List

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset


class DeltaSHAPDataset(abc.ABC):
    """
    Dataset abstract class that needed to run using our code.
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        batch_size: int,
        testbs: int | None,
        deterministic: bool,
        cv_to_use: List[int] | int | None,
        seed: int | None,
    ):
        """
        Constructor

        Args:
            data_path:
                The path of the data.
            batch_size:
                The batch size of the train loader and valid loader.
            testbs:
                The batch size for the test loader.
            deterministic:
                Indicate whether deterministic algorithm is to be used. GLOBAL behaviour!
            cv_to_use:
                Indicate which cv to use. CV are from 0 to 4.
            seed:
                The random seed.

        """
        self.data_path = data_path
        self.train_loaders: List[DataLoader] | None = None
        self.valid_loaders: List[DataLoader] | None = None
        self.test_loader: DataLoader | None = None
        self.feature_size: int | None = None

        self.seed = seed
        self.batch_size = batch_size
        self.testbs = testbs
        self._cv_to_use = cv_to_use

        torch.set_printoptions(precision=8)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True

    def _get_loaders(
        self,
        train_data: np.ndarray,
        train_label: np.ndarray,
        test_data: np.ndarray,
        test_label: np.ndarray,
        train_mask: np.ndarry = None,
        test_mask: np.ndarry = None,
    ):
        """
        Get the train loader, valid loader and the test loaders. The "train_data" and "train_label"
        will be split to be the training set and the validation set.

        Args:
            train_data:
                The train data
            train_label:
                The train label
            test_data:
                The test data
            test_label:
                The test label
        """
        feature_size = train_data.shape[1]
        num_timesteps = train_data.shape[2]
        train_tensor_dataset = TensorDataset(
            torch.Tensor(train_data),
            torch.Tensor(train_label),
            torch.Tensor(train_mask),
        )
        test_tensor_dataset = TensorDataset(
            torch.Tensor(test_data),
            torch.Tensor(test_label),
            torch.Tensor(test_mask),
        )

        kf = KFold(n_splits=5)
        train_loaders = []
        valid_loaders = []
        for train_indices, valid_indices in kf.split(train_data):
            train_subset = Subset(train_tensor_dataset, train_indices)
            valid_subset = Subset(train_tensor_dataset, valid_indices)
            train_loaders.append(DataLoader(train_subset, batch_size=self.batch_size))
            valid_loaders.append(DataLoader(valid_subset, batch_size=self.batch_size))
        testbs = self.testbs if self.testbs is not None else len(test_data)
        test_loader = DataLoader(test_tensor_dataset, batch_size=testbs)
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.test_loader = test_loader
        self.feature_size = feature_size
        self.num_timesteps = num_timesteps

    @abc.abstractmethod
    def load_data(self) -> None:
        """
        Load the data from the file.
        """

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the dataset.
        """

    @property
    @abc.abstractmethod
    def data_type(self) -> str:
        """
        Return the type of the dataset. (Not currently used)
        """

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        """
        Return the number of classes
        """

    def num_cv(self) -> int:
        """
        Return the total number of CV
        """
        if self.train_loaders is None:
            return 0
        return len(self.train_loaders)

    def cv_to_use(self) -> List[int]:
        """
        Return a list of CV to use.
        """
        if self.train_loaders is None:
            return [0]
        num_cv = self.num_cv()
        if self._cv_to_use is None:
            return list(range(num_cv))
        if isinstance(self._cv_to_use, int) and 0 <= self._cv_to_use < num_cv:
            return [self._cv_to_use]
        if isinstance(self._cv_to_use, list) and all(
            0 <= c < num_cv for c in self._cv_to_use
        ):
            return self._cv_to_use
        raise ValueError("CV to use range is invalid.")

    def get_train_loader(self, cv: int) -> DataLoader | None:
        """
        Get the train loader to the corresponding CV.
        """
        if self.train_loaders is None:
            return None
        return self.train_loaders[cv]

    def get_valid_loader(self, cv: int) -> DataLoader | None:
        """
        Get the valid loader to the corresponding CV.
        """
        if self.valid_loaders is None:
            return None
        return self.valid_loaders[cv]

    def get_test_loader(self) -> DataLoader | None:
        """
        Return the test loader.
        """
        return self.test_loader


class Mimic(DeltaSHAPDataset):
    """
    The pre-processed Mimic mortality dataset.
    Num Features = 31, Num Times = 48, Num Classes = 1
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name: str = "patient_vital_preprocessed_mask_reversed.pkl",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name = file_name

    def load_data(self, train_ratio=0.8):
        with (self.data_path / self.file_name).open("rb") as f:
            data = pickle.load(f)
        feature_size = len(data[0][0])

        n_train = int(train_ratio * len(data))

        X = np.array([datum[0] for datum in data])
        train_data = X[0:n_train]
        test_data = X[n_train:]
        train_label = np.array([datum[1] for datum in data[0:n_train]])
        test_label = np.array([datum[1] for datum in data[n_train:]])
        train_mask = np.array([datum[3] for datum in data[0:n_train]])
        test_mask = np.array([datum[3] for datum in data[n_train:]])

        train_data, test_data = self.normalize(train_data, test_data, feature_size)
        self._get_loaders(
            train_data, train_label, test_data, test_label, train_mask, test_mask
        )

    @staticmethod
    def normalize(train_data, test_data, feature_size):
        d = np.stack(train_data, axis=0)
        num_timesteps = train_data.shape[1]

        feature_means = np.mean(d.reshape(-1, feature_size), axis=0)
        feature_std = np.std(d.reshape(-1, feature_size), axis=0)

        feature_means = feature_means[None, None, :]
        feature_std = feature_std[None, None, :]

        np.seterr(divide="ignore", invalid="ignore")
        train_data = np.where(
            feature_std == 0,
            train_data - feature_means,
            (train_data - feature_means) / feature_std,
        )
        test_data = np.where(
            feature_std == 0,
            test_data - feature_means,
            (test_data - feature_means) / feature_std,
        )
        return train_data, test_data

    def get_name(self) -> str:
        return "mimic"

    @property
    def data_type(self) -> str:
        return "mimic"

    @property
    def num_classes(self) -> int:
        return 1


class SimulatedData(DeltaSHAPDataset, abc.ABC):
    """
    An abstract class for simulated data.
    """

    def __init__(
        self,
        data_path: pathlib.Path,
        batch_size: int,
        testbs: int | None,
        deterministic: bool,
        file_name_prefix: str,
        ground_truth_prefix: str,
        cv_to_use: List[int] | int | None,
        seed: int | None,
    ):
        """
        Constructor

        Args:
            data_path:
                The path of the data.
            batch_size:
                The batch size of the train loader and valid loader.
            testbs:
                The batch size for the test loader.
            deterministic:
                Indicate whether deterministic algorithm is to be used. GLOBAL behaviour!
            file_name_prefix:
                The file name prefix for the train and the test data. The names of the files will
                be [PREFIX]x_train.pkl, [PREFIX]y_train.pkl, [PREFIX]x_test.pkl and
                [PREFIX]y_test.pkl.
            ground_truth_prefix:
                The ground truth importance file prefix. The file name will be [PREFIX]_test.pkl
            cv_to_use:
                Indicate which cv to use. CV are from 0 to 4.
            seed:
                The random seed.
        """
        super().__init__(data_path, batch_size, testbs, deterministic, cv_to_use, seed)
        self.file_name_prefix = file_name_prefix
        self.ground_truth_prefix = ground_truth_prefix

    def load_data(self) -> None:
        with (self.data_path / f"{self.file_name_prefix}x_train.pkl").open("rb") as f:
            train_data = pickle.load(f)
        with (self.data_path / f"{self.file_name_prefix}y_train.pkl").open("rb") as f:
            train_label = pickle.load(f)
        with (self.data_path / f"{self.file_name_prefix}x_test.pkl").open("rb") as f:
            test_data = pickle.load(f)
        with (self.data_path / f"{self.file_name_prefix}y_test.pkl").open("rb") as f:
            test_label = pickle.load(f)

        rng = np.random.default_rng(seed=self.seed)
        perm = rng.permutation(train_data.shape[0])
        train_data = train_data[perm]
        train_label = train_label[perm]

        self._get_loaders(train_data, train_label, test_data, test_label)

    @property
    def num_classes(self) -> int:
        return 1

    def load_ground_truth_importance(self) -> np.ndarray:
        with open(
            os.path.join(self.data_path, self.ground_truth_prefix + "_test.pkl"), "rb"
        ) as f:
            gt = pickle.load(f)
        return gt


class SimulatedState(SimulatedData):
    """
    Simulated State data
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/simulated_state_data"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "state_dataset_",
        ground_truth_prefix: str = "state_dataset_importance",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )

    def get_name(self) -> str:
        return "simulated_state"

    @property
    def data_type(self) -> str:
        return "state"


class SimulatedSwitch(SimulatedData):
    """
    Simulated Switch data
    """

    def __init__(
        self,
        data_path: pathlib.Path = pathlib.Path("./data/simulated_switch_data"),
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "state_dataset_",
        ground_truth_prefix: str = "state_dataset_importance",
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )

    def get_name(self) -> str:
        return "simulated_switch"

    @property
    def data_type(self) -> str:
        return "switch"


class SimulatedSpike(SimulatedData):
    """
    Simulated Spike data, with possible delay involved.
    """

    def __init__(
        self,
        data_path: pathlib.Path = None,
        batch_size: int = 100,
        testbs: int | None = None,
        deterministic: bool = False,
        file_name_prefix: str = "",
        ground_truth_prefix: str = "gt",
        delay: int = 0,
        cv_to_use: List[int] | int | None = None,
        seed: int | None = 1234,
    ):
        if delay > 0:
            data_path = pathlib.Path(f"./data/simulated_spike_data_delay_{delay}")
        elif delay == 0:
            data_path = pathlib.Path("./data/simulated_spike_data")
        else:
            raise ValueError("delay must be non-negative.")
        self.delay = delay

        super().__init__(
            data_path,
            batch_size,
            testbs,
            deterministic,
            file_name_prefix,
            ground_truth_prefix,
            cv_to_use,
            seed,
        )

    def get_name(self) -> str:
        if self.delay == 0:
            return "simulated_spike"
        return f"simulated_spike_delay_{self.delay}"

    @property
    def data_type(self) -> str:
        return "spike"


class Mimic3o(DeltaSHAPDataset):
    def __init__(
        self,
        args,
        data_path,
        batch_size=100,
        testbs=None,
        deterministic=False,
        cv_to_use=None,
        seed=None
    ):
        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            testbs=testbs,
            deterministic=deterministic,
            cv_to_use=cv_to_use,
            seed=seed,
        )
        self.args = args
        self.feature_size = 42
        self.num_timesteps = 48

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        pid = self._pid_list[index]
        label = self._label_list[index]
        data = self._data_list[index]
        return pid, data, label

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def data_type(self) -> str:
        return "mimic3o"

    def get_name(self) -> str:
        return "mimic3o"

    def load_data(self):
        if self.args.get("vis", False):
            # TODO: debugging (remove)
            mini_data_path = os.path.join(self.data_path, "test_debug.pkl")
            if os.path.exists(mini_data_path):
                with open(mini_data_path, 'rb') as f:
                    test_data_pkl = pickle.load(f)
            else:
                with open(os.path.join(self.data_path, "test.pkl"), 'rb') as _f:
                    test_data_pkl = pickle.load(_f)

                pids = np.array(test_data_pkl['pid'])
                data = np.array(test_data_pkl['data'])
                mask = np.array(test_data_pkl['mask'])
                labels = np.array(test_data_pkl['label'])

                positive_indices = np.where(labels == 1)[0][:100]
                negative_indices = np.where(labels == 0)[0][:100]
                selected_indices = np.concatenate((positive_indices, negative_indices))

                mini_data = {
                    'pid': pids[selected_indices],
                    'data': data[selected_indices],
                    'mask': mask[selected_indices],
                    'label': labels[selected_indices],
                }            
                with open(mini_data_path, 'wb') as f:
                    pickle.dump(mini_data, f)
                test_data_pkl = mini_data

            test_data_list = np.array(test_data_pkl['data'])
            test_mask_list = np.array(test_data_pkl['mask'])
            test_label_list = np.array(test_data_pkl['label'])
            self.test_loader = DataLoader(
                TensorDataset(torch.Tensor(test_data_list), torch.Tensor(test_mask_list), torch.Tensor(test_label_list)),
                batch_size=self.testbs if self.testbs is not None else len(test_data_list),
                shuffle=False
            )    
            self.train_loaders = self.valid_loaders = [self.test_loader]

        # ================================================
        else:
            test_data_pkl = pickle.load(open(os.path.join(self.data_path, "test.pkl"), 'rb'))
            test_data_list = np.array(test_data_pkl['data'])
            test_mask_list = np.array(test_data_pkl['mask'])
            test_label_list = np.array(test_data_pkl['label'])
            
            # import pdb; pdb.set_trace()
            
            self.test_loader = DataLoader(
                TensorDataset(torch.Tensor(test_data_list), torch.Tensor(test_mask_list), torch.Tensor(test_label_list)),
                batch_size=self.testbs if self.testbs is not None else len(test_data_list),
                shuffle=False
            )

            # Load train and validation data if needed
            if self.args.get("train", False) or self.args.get("traingen", False) or "afo" in self.args.get("explainer", []):
                # Load training data
                train_path = os.path.join(self.data_path, "train.pkl")
                with open(train_path, 'rb') as f:
                    train_data_pkl = pickle.load(f)

                train_data = np.array(train_data_pkl['data'])
                train_mask = np.array(train_data_pkl['mask'])
                train_labels = np.array(train_data_pkl['label'])
                
                self.train_loaders = [
                    DataLoader(
                        TensorDataset(torch.Tensor(train_data), torch.Tensor(train_mask), torch.Tensor(train_labels)),
                        batch_size=self.batch_size,
                        shuffle=True
                    )
                ]

                # Load validation data
                valid_path = os.path.join(self.data_path, "val.pkl")
                with open(valid_path, 'rb') as f:
                    valid_data_pkl = pickle.load(f)

                valid_data = np.array(valid_data_pkl['data'])
                valid_mask = np.array(valid_data_pkl['mask'])
                valid_labels = np.array(valid_data_pkl['label'])
                
                self.valid_loaders = [
                    DataLoader(
                        TensorDataset(torch.Tensor(valid_data), torch.Tensor(valid_mask), torch.Tensor(valid_labels)),
                        batch_size=self.batch_size,
                        shuffle=False
                    )
                ]
            else:
                self.train_loaders = None
                self.valid_loaders = None


class Physionet19(Mimic3o):
    """
    Physionet19 dataset class that inherits from Mimic3o since they share the same structure.
    Only overrides necessary attributes and methods.
    """
    def __init__(
        self,
        args,
        data_path,
        batch_size=100,
        testbs=None,
        deterministic=False,
        cv_to_use=None,
        seed=None
    ):
        super().__init__(
            args,
            data_path=data_path,
            batch_size=batch_size,
            testbs=testbs,
            deterministic=deterministic,
            cv_to_use=cv_to_use,
            seed=seed,
        )
        # Override feature size and timesteps if different from Mimic3o
        self.feature_size = 40  # Update if different
        self.num_timesteps = 48  # Update if different

    @property
    def data_type(self) -> str:
        return "physionet19"

    def get_name(self) -> str:
        return "physionet19"