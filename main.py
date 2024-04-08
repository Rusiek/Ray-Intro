import ray
import logging

from utils import IMDBData, IMDBTransformer

logger_handler = logging.StreamHandler()
logger_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s")
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logger_handler)


ray.init()


if __name__ == "__main__":
    train_ds = IMDBData(
        paths="/home/rusiek/Studia/vi_sem/rozprochy/lab3/Ray-Intro-IMDB/benchmark/data/imdb_train.csv",
        data_transform=True,
        data_transformer=IMDBTransformer(),
    )
    test_ds = IMDBData(
        paths="/home/rusiek/Studia/vi_sem/rozprochy/lab3/Ray-Intro-IMDB/benchmark/data/imdb_test.csv",
        data_transform=True,
        data_transformer=IMDBTransformer(),
    )