import argparse
import os
import logging

from typing import List
from datasets import Dataset, load_dataset

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class TextClassificationDataset:
    """ Base class for sentence-level dataset """
    task = None
    name = None
    subset = None
    seed = None
    column_map = None
    dataset = None
    id2label = None

    def get_train_dataset(self, sample_rate=None) -> Dataset:
        """Get a object of :class:`Dataset` for the train set."""
        raise NotImplementedError()

    def get_dev_dataset(self, sample_rate=None) -> Dataset:
        """Get a object of :class:`Dataset` for the dev set."""
        raise NotImplementedError()

    def get_test_dataset(self, sample_rate=None) -> Dataset:
        """Get a object of :class:`Dataset` for the test set."""
        raise NotImplementedError()

    def get_label_list(self) -> List[str]:
        """Get a list of string for the labels."""
        raise NotImplementedError()

    @classmethod
    def sample(cls, dataset, sample_rate=None, seed=42) -> Dataset:
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            num_sample = int(len(dataset) * sample_rate)
            dataset = dataset.shuffle(seed=seed)
            dataset = Dataset.from_dict(dataset[:num_sample])
        return dataset

    @classmethod
    def rename(cls, dataset, column_map=None) -> Dataset:
        if column_map and len(column_map) > 0 and isinstance(column_map, dict):
            for col in dataset.features:
                if col not in column_map:
                    continue
                if col == column_map[col]:
                    continue
                dataset = dataset.rename_column(col, column_map[col])
            add_columns = list(column_map.values())
            remove_columns = [col for col in dataset.features if col not in add_columns]
            dataset = dataset.remove_columns(remove_columns)
        return dataset

    @classmethod
    def convert(cls, dataset, column_name=None, value_map=None) -> Dataset:
        def convert_value(examples):
            for col in examples.keys():
                if col != column_name:
                    continue
                examples[col] = [value_map[val] for val in examples[col]]
            return examples
        dataset = dataset.map(convert_value, batched=True)
        return dataset

    @classmethod
    def spilt(cls, dataset) -> Dataset:
        def split_row(examples, indices):
            results = {col: [] for col in examples.keys()}
            results["idx_0"], results["idx_1"] = [], []
            for i, col in enumerate(examples.keys()):
                for d, idx in zip(examples[col], indices):
                    results[col].extend(d)
                    if i > 0:
                        continue
                    results["idx_0"].extend([idx] * len(d))
                    results["idx_1"].extend(range(len(d)))
            return results
        dataset = dataset.map(split_row, batched=True, with_indices=True)
        return dataset


class ClincOosIntentDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "intent"
        self.name = "clinc_oos"
        self.subset = "plus"
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "intent": "label",
            "text": "text",
        }
        self.dataset = load_dataset(self.name, self.subset)
        class_names = self.dataset["train"].features["intent"].names
        self.id2label = {i: l for i, l in enumerate(class_names)}

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_label_list(self) -> List[str]:
        label_list = list(self.id2label.values())
        return label_list


class Sst2SentimentDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "sentiment"
        self.name = "glue"
        self.subset = "sst2"
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "idx": "id",
            "label": "label",
            "sentence": "text",
        }
        self.id2label = {
            0: "negative",
            1: "positive",
        }
        self.dataset = load_dataset(self.name, self.subset)

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_label_list(self) -> List[str]:
        label_list = list(self.id2label.values())
        return label_list


class MultimodalEmotionLinesSentimentDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "sentiment"
        self.name = "silicone"
        self.subset = "meld_s"
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "Idx": "id",
            "Dialogue_ID": "d_id",
            "Utterance_ID": "u_id",
            "Label": "label",
            "Utterance": "text",
        }
        self.id2label = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }
        self.dataset = load_dataset(self.name, self.subset)

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_label_list(self) -> List[str]:
        label_list = list(self.id2label.values())
        return label_list


class SemaineSentimentDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "sentiment"
        self.name = "silicone"
        self.subset = "sem"
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "Idx": "id",
            "Dialogue_ID": "d_id",
            "SpeechTurn": "u_id",
            "Label": "label",
            "Utterance": "text",
        }
        self.id2label = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }
        self.dataset = load_dataset(self.name, self.subset)

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_label_list(self) -> List[str]:
        label_list = list(self.id2label.values())
        return label_list


class AmazonReviewsSentimentDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "sentiment"
        self.name = "amazon_reviews_multi"
        self.language = kwargs.get("language", 'en')
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "review_id": "id",
            "stars": "label",
            "review_body": "text",
        }
        self.id2label = {
            1: "extreme_negative",
            2: "negative",
            3: "neutral",
            4: "positive",
            5: "extreme_positive",
        }
        self.dataset = load_dataset(self.name, self.language)

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_label_list(self) -> List[str]:
        label_list = list(self.id2label.values())
        return label_list


class AmazonReviewsTopicDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "topic"
        self.name = "amazon_reviews_multi"
        self.language = kwargs.get("language", 'en')
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "review_id": "id",
            "product_category": "label",
            "review_body": "text",
        }
        self.dataset = load_dataset(self.name, self.language)
        self.label_list = self.dataset["train"].unique("product_category")

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        return dataset

    def get_label_list(self) -> List[str]:
        return self.label_list


class SwitchboardTopicDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "topic"
        self.name = "silicone"
        self.subset = "swda"
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "Idx": "id",
            "Conv_ID": "d_id",
            "Topic": "label",
            "Utterance": "text",
        }
        self.dataset = load_dataset(self.name, self.subset)
        self.label_list = self.dataset["train"].unique("Topic")

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        return dataset

    def get_label_list(self) -> List[str]:
        return self.label_list


class DailyDialogQuestionDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "question"
        self.name = "daily_dialog"
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "idx_0": "d_id",
            "idx_1": "u_id",
            "act": "label",
            "dialog": "text",
        }
        self.dataset = load_dataset(self.name)
        class_names = self.dataset["train"].features["act"].feature.names
        self.id2label = {i: l for i, l in enumerate(class_names)}

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.spilt(dataset)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.spilt(dataset)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.spilt(dataset)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_label_list(self) -> List[str]:
        label_list = list(self.id2label.values())
        return label_list


class IcsiMrdaQuestionDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "question"
        self.name = "silicone"
        self.subset = "mrda"
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "Utterance_ID": "id",
            "Label": "label",
            "Utterance": "text",
        }
        self.dataset = load_dataset(self.name, self.subset)
        class_names = self.dataset["train"].features["Label"].names
        self.id2label = {i: l for i, l in enumerate(class_names)}

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_label_list(self) -> List[str]:
        label_list = list(self.id2label.values())
        return label_list


class SwitchboardQuestionDataset(TextClassificationDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.task = "question"
        self.name = "silicone"
        self.subset = "swda"
        self.seed = kwargs.get("seed", 42)
        self.column_map = {
            "Idx": "id",
            "Conv_ID": "d_id",
            "Label": "label",
            "Utterance": "text",
        }
        self.dataset = load_dataset(self.name, self.subset)
        class_names = self.dataset["train"].features["Label"].names
        self.id2label = {i: l for i, l in enumerate(class_names)}

    def get_train_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["train"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_dev_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["validation"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_test_datasets(self, sample_rate=None) -> Dataset:
        dataset = self.dataset["test"]
        if sample_rate and sample_rate > 0.0 and sample_rate < 1.0:
            dataset = self.sample(dataset, sample_rate, self.seed)
        dataset = self.rename(dataset, self.column_map)
        dataset = self.convert(dataset, "label", self.id2label)
        return dataset

    def get_label_list(self) -> List[str]:
        label_list = list(self.id2label.values())
        return label_list


supported_datasets = {
    "intent": {
        "clinc_oos": ClincOosIntentDataset,
    },
    "sentiment": {
        "sst2": Sst2SentimentDataset,
        "meld_s": MultimodalEmotionLinesSentimentDataset,
        "sem": SemaineSentimentDataset,
        "amazon_reviews_multi": AmazonReviewsSentimentDataset,
    },
    "topic": {
        "amazon_reviews_multi": AmazonReviewsTopicDataset,
        "swda": SwitchboardTopicDataset,
    },
    "question": {
        "daily_dialog": DailyDialogQuestionDataset,
        "mrda": IcsiMrdaQuestionDataset,
        "swda": SwitchboardQuestionDataset,
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, help='Task name')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--sample_rate', type=float, help='Sample rate', required=False)
    parser.add_argument('--seed', type=int, help='Randon seed', default=42)
    args = parser.parse_args()

    if args.task_name not in supported_datasets:
        raise ValueError(f"unsupported task name: {args.task_name}")
    if args.dataset_name not in supported_datasets[args.task_name]:
        raise ValueError(f"unsupported dataset name: {args.dataset_name}")

    dataset = supported_datasets[args.task_name][args.dataset_name](seed=args.seed)

    train_file = os.path.join(args.output_dir, "train.json")
    train_dataset = dataset.get_train_datasets(args.sample_rate)
    train_dataset.to_json(train_file)
    dev_file = os.path.join(args.output_dir, "dev.json")
    dev_dataset = dataset.get_dev_datasets(args.sample_rate)
    dev_dataset.to_json(dev_file)
    test_file = os.path.join(args.output_dir, "test.json")
    test_dataset = dataset.get_test_datasets(args.sample_rate)
    test_dataset.to_json(test_file)
    label_file = os.path.join(args.output_dir, "labels.txt")
    label_list = dataset.get_label_list()
    with open(label_file, "w", encoding="utf-8") as f:
        for l in label_list:
            f.write(f"{l}\n")


if __name__ == "__main__":
    main()
