import pytest
from unittest.mock import patch, PropertyMock
from datasets import Dataset, DatasetDict
from datasets import Sequence, Features, Value, ClassLabel
from training.src.create_dataset import (
    ClincOosIntentDataset,
    Sst2SentimentDataset,
    MultimodalEmotionLinesSentimentDataset,
    SemaineSentimentDataset,
    AmazonReviewsSentimentDataset,
    AmazonReviewsTopicDataset,
    SwitchboardTopicDataset,
    DailyDialogQuestionDataset,
    IcsiMrdaQuestionDataset,
    SwitchboardQuestionDataset,
)


# Unit Testing for `ClincOosIntentDataset`
@pytest.fixture
def clinc_oos_intent_dataset():
    with patch("training.src.create_dataset.ClincOosIntentDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "text": Value(dtype="string", id=None),
            "intent": ClassLabel(num_classes=2, names=["intent_0", "intent_1"], id=None),
        })
        train_set = {
            "text": ["train_0", "train_1", "train_2", "train_3", "train_4"],
            "intent": [0, 1, 0, 0, 1]
        }
        val_set = {
            "text": ["val_0", "val_1", "val_2"],
            "intent": [0, 1, 0]
        }
        test_set = {
            "text": ["test_0", "test_1"],
            "intent": [0, 1]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = ClincOosIntentDataset(seed=0)
        yield dataset


def test_clinc_oos_intent_dataset_get_train_datasets(clinc_oos_intent_dataset):
    dataset = clinc_oos_intent_dataset
    train_dataset = dataset.get_train_datasets(0.6)
    assert len(train_dataset) == 3
    assert train_dataset[0]["text"] == "train_2" and train_dataset[0]["label"] == "intent_0"
    assert train_dataset[1]["text"] == "train_4" and train_dataset[1]["label"] == "intent_1"
    assert train_dataset[2]["text"] == "train_3" and train_dataset[2]["label"] == "intent_0"


def test_clinc_oos_intent_dataset_get_dev_datasets(clinc_oos_intent_dataset):
    dataset = clinc_oos_intent_dataset
    dev_dataset = dataset.get_dev_datasets(0.8)
    assert len(dev_dataset) == 2
    assert dev_dataset[0]["text"] == "val_2" and dev_dataset[0]["label"] == "intent_0"
    assert dev_dataset[1]["text"] == "val_0" and dev_dataset[1]["label"] == "intent_0"


def test_clinc_oos_intent_dataset_get_test_datasets(clinc_oos_intent_dataset):
    dataset = clinc_oos_intent_dataset
    test_dataset = dataset.get_test_datasets(0.5)
    assert len(test_dataset) == 1
    assert test_dataset[0]["text"] == "test_0" and test_dataset[0]["label"] == "intent_0"


def test_clinc_oos_intent_dataset_get_label_list(clinc_oos_intent_dataset):
    dataset = clinc_oos_intent_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 2
    assert label_list[0] == "intent_0"
    assert label_list[1] == "intent_1"


# Unit Testing for `Sst2SentimentDataset`
@pytest.fixture
def glue_sst2_sentiment_dataset():
    with patch("training.src.create_dataset.Sst2SentimentDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "idx": Value(dtype="int32", id=None),
            "sentence": Value(dtype="string", id=None),
            "label": ClassLabel(num_classes=2, names=["negative", "positive"], id=None),
        })
        train_set = {
            "idx": [0, 1, 2, 3, 4, 5],
            "sentence": ["train_0", "train_1", "train_2", "train_3", "train_4", "train_5"],
            "label": [0, 0, 1, 1, 1, 0]
        }
        val_set = {
            "idx": [0, 1, 2, 3],
            "sentence": ["val_0", "val_1", "val_2", "val_3"],
            "label": [1, 0, 0, 1]
        }
        test_set = {
            "idx": [0, 1],
            "sentence": ["test_0", "test_1"],
            "label": [1, 0]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = Sst2SentimentDataset(seed=0)
        yield dataset


def test_glue_sst2_sentiment_dataset_get_train_datasets(glue_sst2_sentiment_dataset):
    dataset = glue_sst2_sentiment_dataset
    train_dataset = dataset.get_train_datasets(0.8)
    assert len(train_dataset) == 4
    assert train_dataset[0]["id"] == 3 and \
        train_dataset[0]["text"] == "train_3" and train_dataset[0]["label"] == "positive"
    assert train_dataset[1]["id"] == 2 and \
        train_dataset[1]["text"] == "train_2" and train_dataset[1]["label"] == "positive"
    assert train_dataset[2]["id"] == 5 and \
        train_dataset[2]["text"] == "train_5" and train_dataset[2]["label"] == "negative"
    assert train_dataset[3]["id"] == 4 and \
        train_dataset[3]["text"] == "train_4" and train_dataset[3]["label"] == "positive"


def test_glue_sst2_sentiment_dataset_get_dev_datasets(glue_sst2_sentiment_dataset):
    dataset = glue_sst2_sentiment_dataset
    dev_dataset = dataset.get_dev_datasets(0.5)
    assert len(dev_dataset) == 2
    assert dev_dataset[0]["id"] == 2 and \
        dev_dataset[0]["text"] == "val_2" and dev_dataset[0]["label"] == "negative"
    assert dev_dataset[1]["id"] == 0 and \
        dev_dataset[1]["text"] == "val_0" and dev_dataset[1]["label"] == "positive"


def test_glue_sst2_sentiment_dataset_get_test_datasets(glue_sst2_sentiment_dataset):
    dataset = glue_sst2_sentiment_dataset
    test_dataset = dataset.get_test_datasets(0.8)
    assert len(test_dataset) == 1
    assert test_dataset[0]["id"] == 0 and \
        test_dataset[0]["text"] == "test_0" and test_dataset[0]["label"] == "positive"


def test_glue_sst2_sentiment_dataset_get_label_list(glue_sst2_sentiment_dataset):
    dataset = glue_sst2_sentiment_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 2
    assert label_list[0] == "negative"
    assert label_list[1] == "positive"


# Unit Testing for `MultimodalEmotionLinesSentimentDataset`
@pytest.fixture
def silicone_meld_s_sentiment_dataset():
    with patch("training.src.create_dataset.MultimodalEmotionLinesSentimentDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "Idx": Value(dtype="int32", id=None),
            "Utterance": Value(dtype="string", id=None),
            "Label": ClassLabel(num_classes=3, names=["negative", "neutral", "positive"], id=None),
        })
        train_set = {
            "Idx": [0, 1, 2, 3, 4, 5, 6],
            "Utterance": ["train_0", "train_1", "train_2", "train_3", "train_4", "train_5", "train_6"],
            "Label": [1, 0, 2, 0, 1, 0, 1]
        }
        val_set = {
            "Idx": [0, 1, 2, 3],
            "Utterance": ["val_0", "val_1", "val_2", "val_3"],
            "Label": [0, 2, 1, 1]
        }
        test_set = {
            "Idx": [0, 1, 2],
            "Utterance": ["test_0", "test_1", "test_2"],
            "Label": [1, 1, 0]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = MultimodalEmotionLinesSentimentDataset(seed=0)
        yield dataset


def test_silicone_meld_s_sentiment_dataset_get_train_datasets(silicone_meld_s_sentiment_dataset):
    dataset = silicone_meld_s_sentiment_dataset
    train_dataset = dataset.get_train_datasets(0.3)
    assert len(train_dataset) == 2
    assert train_dataset[0]["id"] == 2 and \
        train_dataset[0]["text"] == "train_2" and train_dataset[0]["label"] == "positive"
    assert train_dataset[1]["id"] == 4 and \
        train_dataset[1]["text"] == "train_4" and train_dataset[1]["label"] == "neutral"


def test_silicone_meld_s_sentiment_dataset_get_dev_datasets(silicone_meld_s_sentiment_dataset):
    dataset = silicone_meld_s_sentiment_dataset
    dev_dataset = dataset.get_dev_datasets(0.7)
    assert len(dev_dataset) == 2
    assert dev_dataset[0]["id"] == 2 and \
        dev_dataset[0]["text"] == "val_2" and dev_dataset[0]["label"] == "neutral"
    assert dev_dataset[1]["id"] == 0 and \
        dev_dataset[1]["text"] == "val_0" and dev_dataset[1]["label"] == "negative"


def test_silicone_meld_s_sentiment_dataset_get_test_datasets(silicone_meld_s_sentiment_dataset):
    dataset = silicone_meld_s_sentiment_dataset
    test_dataset = dataset.get_test_datasets(0.5)
    assert len(test_dataset) == 1
    assert test_dataset[0]["id"] == 2 and \
        test_dataset[0]["text"] == "test_2" and test_dataset[0]["label"] == "negative"


def test_silicone_meld_s_sentiment_dataset_get_label_list(silicone_meld_s_sentiment_dataset):
    dataset = silicone_meld_s_sentiment_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 3
    assert label_list[0] == "negative"
    assert label_list[1] == "neutral"
    assert label_list[2] == "positive"


# Unit Testing for `SemaineSentimentDataset`
@pytest.fixture
def silicone_sem_sentiment_dataset():
    with patch("training.src.create_dataset.SemaineSentimentDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "Idx": Value(dtype="int32", id=None),
            "Utterance": Value(dtype="string", id=None),
            "Label": ClassLabel(num_classes=3, names=["negative", "neutral", "positive"], id=None),
        })
        train_set = {
            "Idx": [0, 1, 2, 3],
            "Utterance": ["train_0", "train_1", "train_2", "train_3"],
            "Label": [2, 0, 1, 1]
        }
        val_set = {
            "Idx": [0, 1],
            "Utterance": ["val_0", "val_1"],
            "Label": [1, 0]
        }
        test_set = {
            "Idx": [0, 1, 2],
            "Utterance": ["test_0", "test_1", "test_2"],
            "Label": [2, 2, 0]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = SemaineSentimentDataset(seed=0)
        yield dataset


def test_silicone_sem_sentiment_dataset_get_train_datasets(silicone_sem_sentiment_dataset):
    dataset = silicone_sem_sentiment_dataset
    train_dataset = dataset.get_train_datasets(0.8)
    assert len(train_dataset) == 3
    assert train_dataset[0]["id"] == 2 and \
        train_dataset[0]["text"] == "train_2" and train_dataset[0]["label"] == "neutral"
    assert train_dataset[1]["id"] == 0 and \
        train_dataset[1]["text"] == "train_0" and train_dataset[1]["label"] == "positive"
    assert train_dataset[2]["id"] == 1 and \
        train_dataset[2]["text"] == "train_1" and train_dataset[2]["label"] == "negative"


def test_silicone_sem_sentiment_dataset_get_dev_datasets(silicone_sem_sentiment_dataset):
    dataset = silicone_sem_sentiment_dataset
    dev_dataset = dataset.get_dev_datasets(0.5)
    assert len(dev_dataset) == 1
    assert dev_dataset[0]["id"] == 0 and \
        dev_dataset[0]["text"] == "val_0" and dev_dataset[0]["label"] == "neutral"


def test_silicone_sem_sentiment_dataset_get_test_datasets(silicone_sem_sentiment_dataset):
    dataset = silicone_sem_sentiment_dataset
    test_dataset = dataset.get_test_datasets(0.9)
    assert len(test_dataset) == 2
    assert test_dataset[0]["id"] == 2 and \
        test_dataset[0]["text"] == "test_2" and test_dataset[0]["label"] == "negative"
    assert test_dataset[1]["id"] == 0 and \
        test_dataset[1]["text"] == "test_0" and test_dataset[1]["label"] == "positive"


def test_silicone_sem_sentiment_dataset_get_label_list(silicone_sem_sentiment_dataset):
    dataset = silicone_sem_sentiment_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 3
    assert label_list[0] == "negative"
    assert label_list[1] == "neutral"
    assert label_list[2] == "positive"


# Unit Testing for `AmazonReviewsSentimentDataset`
@pytest.fixture
def amazon_reviews_multi_sentiment_dataset():
    with patch("training.src.create_dataset.AmazonReviewsSentimentDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "review_id": Value(dtype="string", id=None),
            "review_body": Value(dtype="string", id=None),
            "stars": Value(dtype="int32", id=None),
        })
        train_set = {
            "review_id": ["0", "1", "2", "3", "4", "5", "6", "7"],
            "review_body": ["train_0", "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_7"],
            "stars": [3, 2, 1, 2, 5, 4, 5, 3]
        }
        val_set = {
            "review_id": ["0", "1", "2", "3"],
            "review_body": ["val_0", "val_1", "val_2", "val_3"],
            "stars": [2, 3, 4, 4]
        }
        test_set = {
            "review_id": ["0", "1", "2", "3", "4"],
            "review_body": ["test_0", "test_1", "test_2", "test_3", "test_4"],
            "stars": [2, 5, 3, 1, 1]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = AmazonReviewsSentimentDataset(seed=0)
        yield dataset


def test_amazon_reviews_multi_sentiment_dataset_get_train_datasets(amazon_reviews_multi_sentiment_dataset):
    dataset = amazon_reviews_multi_sentiment_dataset
    train_dataset = dataset.get_train_datasets(0.4)
    assert len(train_dataset) == 3
    assert train_dataset[0]["id"] == "2" and \
        train_dataset[0]["text"] == "train_2" and train_dataset[0]["label"] == "extreme_negative"
    assert train_dataset[1]["id"] == "4" and \
        train_dataset[1]["text"] == "train_4" and train_dataset[1]["label"] == "extreme_positive"
    assert train_dataset[2]["id"] == "3" and \
        train_dataset[2]["text"] == "train_3" and train_dataset[2]["label"] == "negative"


def test_amazon_reviews_multi_sentiment_dataset_get_dev_datasets(amazon_reviews_multi_sentiment_dataset):
    dataset = amazon_reviews_multi_sentiment_dataset
    dev_dataset = dataset.get_dev_datasets(0.7)
    assert len(dev_dataset) == 2
    assert dev_dataset[0]["id"] == "2" and \
        dev_dataset[0]["text"] == "val_2" and dev_dataset[0]["label"] == "positive"
    assert dev_dataset[1]["id"] == "0" and \
        dev_dataset[1]["text"] == "val_0" and dev_dataset[1]["label"] == "negative"


def test_amazon_reviews_multi_sentiment_dataset_get_test_datasets(amazon_reviews_multi_sentiment_dataset):
    dataset = amazon_reviews_multi_sentiment_dataset
    test_dataset = dataset.get_test_datasets(0.3)
    assert len(test_dataset) == 1
    assert test_dataset[0]["id"] == "2" and \
        test_dataset[0]["text"] == "test_2" and test_dataset[0]["label"] == "neutral"


def test_amazon_reviews_multi_sentiment_dataset_get_label_list(amazon_reviews_multi_sentiment_dataset):
    dataset = amazon_reviews_multi_sentiment_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 5
    assert label_list[0] == "extreme_negative"
    assert label_list[1] == "negative"
    assert label_list[2] == "neutral"
    assert label_list[3] == "positive"
    assert label_list[4] == "extreme_positive"


# Unit Testing for `AmazonReviewsTopicDataset`
@pytest.fixture
def amazon_reviews_multi_topic_dataset():
    with patch("training.src.create_dataset.AmazonReviewsTopicDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "review_id": Value(dtype="string", id=None),
            "review_body": Value(dtype="string", id=None),
            "product_category": Value(dtype="string", id=None),
        })
        train_set = {
            "review_id": ["0", "1", "2", "3"],
            "review_body": ["train_0", "train_1", "train_2", "train_3"],
            "product_category": ["topic_0", "topic_1", "topic_0", "topic_1"]
        }
        val_set = {
            "review_id": ["0", "1"],
            "review_body": ["val_0", "val_1"],
            "product_category": ["topic_0", "topic_1"]
        }
        test_set = {
            "review_id": ["0", "1", "2"],
            "review_body": ["test_0", "test_1", "test_2"],
            "product_category": ["topic_1", "topic_0", "topic_1"]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = AmazonReviewsTopicDataset(seed=0)
        yield dataset


def test_amazon_reviews_multi_topic_dataset_get_train_datasets(amazon_reviews_multi_topic_dataset):
    dataset = amazon_reviews_multi_topic_dataset
    train_dataset = dataset.get_train_datasets(0.8)
    assert len(train_dataset) == 3
    assert train_dataset[0]["id"] == "2" and \
        train_dataset[0]["text"] == "train_2" and train_dataset[0]["label"] == "topic_0"
    assert train_dataset[1]["id"] == "0" and \
        train_dataset[1]["text"] == "train_0" and train_dataset[1]["label"] == "topic_0"
    assert train_dataset[2]["id"] == "1" and \
        train_dataset[2]["text"] == "train_1" and train_dataset[2]["label"] == "topic_1"


def test_amazon_reviews_multi_topic_dataset_get_dev_datasets(amazon_reviews_multi_topic_dataset):
    dataset = amazon_reviews_multi_topic_dataset
    dev_dataset = dataset.get_dev_datasets(0.6)
    assert len(dev_dataset) == 1
    assert dev_dataset[0]["id"] == "0" and \
        dev_dataset[0]["text"] == "val_0" and dev_dataset[0]["label"] == "topic_0"


def test_amazon_reviews_multi_topic_dataset_get_test_datasets(amazon_reviews_multi_topic_dataset):
    dataset = amazon_reviews_multi_topic_dataset
    test_dataset = dataset.get_test_datasets(0.7)
    assert len(test_dataset) == 2
    assert test_dataset[0]["id"] == "2" and \
        test_dataset[0]["text"] == "test_2" and test_dataset[0]["label"] == "topic_1"
    assert test_dataset[1]["id"] == "0" and \
        test_dataset[1]["text"] == "test_0" and test_dataset[1]["label"] == "topic_1"


def test_amazon_reviews_multi_topic_dataset_get_label_list(amazon_reviews_multi_topic_dataset):
    dataset = amazon_reviews_multi_topic_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 2
    assert label_list[0] == "topic_0"
    assert label_list[1] == "topic_1"


# Unit Testing for `SwitchboardTopicDataset`
@pytest.fixture
def silicone_swda_topic_dataset():
    with patch("training.src.create_dataset.SwitchboardTopicDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "Idx": Value(dtype="int32", id=None),
            "Utterance": Value(dtype="string", id=None),
            "Topic": Value(dtype="string", id=None),
        })
        train_set = {
            "Idx": [0, 1, 2, 3, 4, 5, 6],
            "Utterance": ["train_0", "train_1", "train_2", "train_3", "train_4", "train_5", "train_6"],
            "Topic": ["topic_0", "topic_1", "topic_0", "topic_2", "topic_1", "topic_2", "topic_1"]
        }
        val_set = {
            "Idx": [0, 1, 2, 3],
            "Utterance": ["val_0", "val_1", "val_2", "val_3"],
            "Topic": ["topic_0", "topic_1", "topic_2", "topic_1"]
        }
        test_set = {
            "Idx": [0, 1, 2, 3],
            "Utterance": ["test_0", "test_1", "test_2", "test_3"],
            "Topic": ["topic_1", "topic_0", "topic_0", "topic_2"]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = SwitchboardTopicDataset(seed=0)
        yield dataset


def test_silicone_swda_topic_dataset_get_train_datasets(silicone_swda_topic_dataset):
    dataset = silicone_swda_topic_dataset
    train_dataset = dataset.get_train_datasets(0.4)
    assert len(train_dataset) == 2
    assert train_dataset[0]["id"] == 2 and \
        train_dataset[0]["text"] == "train_2" and train_dataset[0]["label"] == "topic_0"
    assert train_dataset[1]["id"] == 4 and \
        train_dataset[1]["text"] == "train_4" and train_dataset[1]["label"] == "topic_1"


def test_silicone_swda_topic_dataset_get_dev_datasets(silicone_swda_topic_dataset):
    dataset = silicone_swda_topic_dataset
    dev_dataset = dataset.get_dev_datasets(0.3)
    assert len(dev_dataset) == 1
    assert dev_dataset[0]["id"] == 2 and \
        dev_dataset[0]["text"] == "val_2" and dev_dataset[0]["label"] == "topic_2"


def test_silicone_swda_topic_dataset_get_test_datasets(silicone_swda_topic_dataset):
    dataset = silicone_swda_topic_dataset
    test_dataset = dataset.get_test_datasets(0.6)
    assert len(test_dataset) == 2
    assert test_dataset[0]["id"] == 2 and \
        test_dataset[0]["text"] == "test_2" and test_dataset[0]["label"] == "topic_0"
    assert test_dataset[1]["id"] == 0 and \
        test_dataset[1]["text"] == "test_0" and test_dataset[1]["label"] == "topic_1"


def test_silicone_swda_topic_dataset_get_label_list(silicone_swda_topic_dataset):
    dataset = silicone_swda_topic_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 3
    assert label_list[0] == "topic_0"
    assert label_list[1] == "topic_1"
    assert label_list[2] == "topic_2"


# Unit Testing for `DailyDialogQuestionDataset`
@pytest.fixture
def daily_dialog_question_dataset():
    with patch("training.src.create_dataset.DailyDialogQuestionDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "dialog": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "act": Sequence(feature=ClassLabel(num_classes=2,
                            names=["question_0", "question_1"], id=None), length=-1, id=None),
        })
        train_set = {
            "dialog": [
                ["train_0", "train_1"],
                ["train_2", "train_3"],
                ["train_4", "train_5", "train_6"],
                ["train_7", "train_8"],
                ["train_9", "train_10"],
            ],
            "act": [
                [0, 1],
                [0, 0],
                [1, 0, 0],
                [1, 0],
                [0, 1],
            ]
        }
        val_set = {
            "dialog": [
                ["val_0", "val_1"],
                ["val_2", "val_3"],
                ["val_4", "val_5"],
            ],
            "act": [
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        }
        test_set = {
            "dialog": [
                ["test_0", "test_1"],
                ["test_2", "test_3"],
            ],
            "act": [
                [1, 0],
                [0, 0, 1],
            ]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = DailyDialogQuestionDataset(seed=0)
        yield dataset


def test_daily_dialog_question_dataset_get_train_datasets(daily_dialog_question_dataset):
    dataset = daily_dialog_question_dataset
    train_dataset = dataset.get_train_datasets(0.4)
    assert len(train_dataset) == 5
    assert train_dataset[0]["d_id"] == 0 and train_dataset[0]["u_id"] == 0 and \
        train_dataset[0]["text"] == "train_4" and train_dataset[0]["label"] == "question_1"
    assert train_dataset[1]["d_id"] == 0 and train_dataset[1]["u_id"] == 1 and \
        train_dataset[1]["text"] == "train_5" and train_dataset[1]["label"] == "question_0"
    assert train_dataset[2]["d_id"] == 0 and train_dataset[2]["u_id"] == 2 and \
        train_dataset[2]["text"] == "train_6" and train_dataset[2]["label"] == "question_0"
    assert train_dataset[3]["d_id"] == 1 and train_dataset[3]["u_id"] == 0 and \
        train_dataset[3]["text"] == "train_9" and train_dataset[3]["label"] == "question_0"
    assert train_dataset[4]["d_id"] == 1 and train_dataset[4]["u_id"] == 1 and \
        train_dataset[4]["text"] == "train_10" and train_dataset[4]["label"] == "question_1"


def test_daily_dialog_question_dataset_get_dev_datasets(daily_dialog_question_dataset):
    dataset = daily_dialog_question_dataset
    dev_dataset = dataset.get_dev_datasets(0.6)
    assert len(dev_dataset) == 2
    assert dev_dataset[0]["d_id"] == 0 and dev_dataset[0]["u_id"] == 0 and \
        dev_dataset[0]["text"] == "val_4" and dev_dataset[0]["label"] == "question_0"
    assert dev_dataset[1]["d_id"] == 0 and dev_dataset[1]["u_id"] == 1 and \
        dev_dataset[1]["text"] == "val_5" and dev_dataset[1]["label"] == "question_1"


def test_daily_dialog_question_dataset_get_test_datasets(daily_dialog_question_dataset):
    dataset = daily_dialog_question_dataset
    test_dataset = dataset.get_test_datasets(0.5)
    assert len(test_dataset) == 2
    assert test_dataset[0]["d_id"] == 0 and test_dataset[0]["u_id"] == 0 and \
        test_dataset[0]["text"] == "test_0" and test_dataset[0]["label"] == "question_1"
    assert test_dataset[1]["d_id"] == 0 and test_dataset[1]["u_id"] == 1 and \
        test_dataset[1]["text"] == "test_1" and test_dataset[1]["label"] == "question_0"


def test_daily_dialog_question_dataset_get_label_list(daily_dialog_question_dataset):
    dataset = daily_dialog_question_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 2
    assert label_list[0] == "question_0"
    assert label_list[1] == "question_1"


# Unit Testing for `IcsiMrdaQuestionDataset`
@pytest.fixture
def silicone_mrda_question_dataset():
    with patch("training.src.create_dataset.IcsiMrdaQuestionDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "Utterance_ID": Value(dtype="string", id=None),
            "Utterance": Value(dtype="string", id=None),
            "Label": ClassLabel(num_classes=3, names=["question_0", "question_1", "question_2"], id=None),
        })
        train_set = {
            "Utterance_ID": ["0", "1", "2", "3", "4", "5", "6"],
            "Utterance": ["train_0", "train_1", "train_2", "train_3", "train_4", "train_5", "train_6"],
            "Label": [0, 1, 2, 2, 1, 0, 0]
        }
        val_set = {
            "Utterance_ID": ["0", "1", "2", "3"],
            "Utterance": ["val_0", "val_1", "val_2", "val_3"],
            "Label": [1, 1, 2, 0]
        }
        test_set = {
            "Utterance_ID": ["0", "1", "2"],
            "Utterance": ["test_0", "test_1", "test_2"],
            "Label": [2, 1, 0]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = IcsiMrdaQuestionDataset(seed=0)
        yield dataset


def test_silicone_mrda_question_dataset_get_train_datasets(silicone_mrda_question_dataset):
    dataset = silicone_mrda_question_dataset
    train_dataset = dataset.get_train_datasets(0.7)
    assert len(train_dataset) == 4
    assert train_dataset[0]["id"] == "2" and \
        train_dataset[0]["text"] == "train_2" and train_dataset[0]["label"] == "question_2"
    assert train_dataset[1]["id"] == "4" and \
        train_dataset[1]["text"] == "train_4" and train_dataset[1]["label"] == "question_1"
    assert train_dataset[2]["id"] == "3" and \
        train_dataset[2]["text"] == "train_3" and train_dataset[2]["label"] == "question_2"
    assert train_dataset[3]["id"] == "6" and \
        train_dataset[3]["text"] == "train_6" and train_dataset[3]["label"] == "question_0"


def test_silicone_mrda_question_dataset_get_dev_datasets(silicone_mrda_question_dataset):
    dataset = silicone_mrda_question_dataset
    dev_dataset = dataset.get_dev_datasets(0.5)
    assert len(dev_dataset) == 2
    assert dev_dataset[0]["id"] == "2" and \
        dev_dataset[0]["text"] == "val_2" and dev_dataset[0]["label"] == "question_2"
    assert dev_dataset[1]["id"] == "0" and \
        dev_dataset[1]["text"] == "val_0" and dev_dataset[1]["label"] == "question_1"


def test_silicone_mrda_question_dataset_get_test_datasets(silicone_mrda_question_dataset):
    dataset = silicone_mrda_question_dataset
    test_dataset = dataset.get_test_datasets(0.4)
    assert len(test_dataset) == 1
    assert test_dataset[0]["id"] == "2" and \
        test_dataset[0]["text"] == "test_2" and test_dataset[0]["label"] == "question_0"


def test_silicone_mrda_question_dataset_get_label_list(silicone_mrda_question_dataset):
    dataset = silicone_mrda_question_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 3
    assert label_list[0] == "question_0"
    assert label_list[1] == "question_1"
    assert label_list[2] == "question_2"


# Unit Testing for `SwitchboardQuestionDataset`
@pytest.fixture
def silicone_swda_question_dataset():
    with patch("training.src.create_dataset.SwitchboardQuestionDataset.dataset",
               new_callable=PropertyMock) as mock_dataset:
        features = Features({
            "Idx": Value(dtype="int32", id=None),
            "Utterance": Value(dtype="string", id=None),
            "Label": ClassLabel(num_classes=4, names=["question_0", "question_1", "question_2", "question_3"], id=None),
        })
        train_set = {
            "Idx": [0, 1, 2, 3, 4, 5, 6, 7],
            "Utterance": ["train_0", "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_7"],
            "Label": [0, 0, 1, 2, 3, 2, 0, 1]
        }
        val_set = {
            "Idx": [0, 1, 2, 3],
            "Utterance": ["val_0", "val_1", "val_2", "val_3"],
            "Label": [0, 1, 2, 3]
        }
        test_set = {
            "Idx": [0, 1, 2, 3],
            "Utterance": ["test_0", "test_1", "test_2", "test_3"],
            "Label": [0, 1, 2, 3]
        }
        mock_dataset.return_value = DatasetDict({
            "train": Dataset.from_dict(train_set, features=features),
            "validation": Dataset.from_dict(val_set, features=features),
            "test": Dataset.from_dict(test_set, features=features),
        })
        dataset = SwitchboardQuestionDataset(seed=0)
        yield dataset


def test_silicone_swda_question_dataset_get_train_datasets(silicone_swda_question_dataset):
    dataset = silicone_swda_question_dataset
    train_dataset = dataset.get_train_datasets(0.4)
    assert len(train_dataset) == 3
    assert train_dataset[0]["id"] == 2 and \
        train_dataset[0]["text"] == "train_2" and train_dataset[0]["label"] == "question_1"
    assert train_dataset[1]["id"] == 4 and \
        train_dataset[1]["text"] == "train_4" and train_dataset[1]["label"] == "question_3"
    assert train_dataset[2]["id"] == 3 and \
        train_dataset[2]["text"] == "train_3" and train_dataset[2]["label"] == "question_2"


def test_silicone_swda_question_dataset_get_dev_datasets(silicone_swda_question_dataset):
    dataset = silicone_swda_question_dataset
    dev_dataset = dataset.get_dev_datasets(0.3)
    assert len(dev_dataset) == 1
    assert dev_dataset[0]["id"] == 2 and \
        dev_dataset[0]["text"] == "val_2" and dev_dataset[0]["label"] == "question_2"


def test_silicone_swda_question_dataset_get_test_datasets(silicone_swda_question_dataset):
    dataset = silicone_swda_question_dataset
    test_dataset = dataset.get_test_datasets(0.6)
    assert len(test_dataset) == 2
    assert test_dataset[0]["id"] == 2 and \
        test_dataset[0]["text"] == "test_2" and test_dataset[0]["label"] == "question_2"
    assert test_dataset[1]["id"] == 0 and \
        test_dataset[1]["text"] == "test_0" and test_dataset[1]["label"] == "question_0"


def test_silicone_swda_question_dataset_get_label_list(silicone_swda_question_dataset):
    dataset = silicone_swda_question_dataset
    label_list = dataset.get_label_list()
    assert len(label_list) == 4
    assert label_list[0] == "question_0"
    assert label_list[1] == "question_1"
    assert label_list[2] == "question_2"
    assert label_list[3] == "question_3"
