from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

import pytorch_lightning as pl

from .dataset import LabelledGraphDataset
from .loader import GraphDataLoader


class GraphDataModule(pl.LightningDataModule, ABC):
    def __init__(self, test_names: list[str]):
        super().__init__()
        self.test_names = test_names

    @abstractmethod
    def has_stage(self, stage: Literal["train", "val", "test"]) -> bool:
        """
        Check if the data module is able to provide data for a given stage.

        Expect to be called at any time after :meth:`prepare_data`.

        Parameters
        ----------
        stage
            The stage to check for.
        """

    def prepare_data(self) -> None:
        return super().prepare_data()

    @abstractmethod
    def train_dataset(self) -> LabelledGraphDataset:
        """
        Get the training dataset. This useful for e.g. pre-processing steps.
        Assume that ``self.has_stage("train")`` and that :meth:`prepare_data`
        has been called.
        """

    ## add type hints ##
    def train_dataloader(self) -> GraphDataLoader:
        return super().train_dataloader()

    def val_dataloader(self) -> GraphDataLoader:
        return super().val_dataloader()

    def test_dataloader(self) -> Sequence[GraphDataLoader]:
        return super().test_dataloader()


class FixedDatasets(GraphDataModule):
    def __init__(
        self,
        train: LabelledGraphDataset | None = None,
        val: LabelledGraphDataset | None = None,
        tests: dict[str, LabelledGraphDataset] | None = None,
        **dataloader_kwargs,
    ):
        tests = tests or {}
        super().__init__(test_names=list(tests.keys()))

        self.train = train
        self.val = val
        self.tests = tests

        default_kwargs = {
            "batch_size": 32,
            "num_workers": 4,
            "persistent_workers": True,
        }
        self.dataloader_kwargs = {**default_kwargs, **dataloader_kwargs}

    def has_stage(self, stage: Literal["train", "val", "test"]) -> bool:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.tests,
        }.get(stage) is not None

    def train_dataset(self) -> LabelledGraphDataset:
        assert self.train is not None
        return self.train

    def train_dataloader(self) -> GraphDataLoader:
        assert self.train is not None
        kwargs = {**self.dataloader_kwargs, "shuffle": True}
        return GraphDataLoader(self.train, **kwargs)

    def val_dataloader(self) -> GraphDataLoader:
        assert self.val is not None
        kwargs = {**self.dataloader_kwargs, "shuffle": False}
        return GraphDataLoader(self.val, **kwargs)

    def test_dataloader(self) -> Sequence[GraphDataLoader]:
        assert self.tests is not None
        kwargs = {**self.dataloader_kwargs, "shuffle": False}
        return [GraphDataLoader(test, **kwargs) for test in self.tests.values()]
