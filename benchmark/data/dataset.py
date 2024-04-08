import ray
import logging

from typing import List

logger_handler = logging.StreamHandler()
logger_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s")
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logger_handler)


class BaseDataset:    
    def __init__(
            self,
            paths: List[str],
            seed: int = -1,
            ignore_missing_paths: bool = True,
            data_transform: bool = False,
            data_transformer: callable = None,
            batch_transform: bool = False,
            batch_transformer: callable = None,
            override_num_blocks: int = 1
        ):
        logger.info(f"Loading data from {paths}")
        self._dataset = ray.data.read_csv(
            paths=paths,
            ignore_missing_paths=ignore_missing_paths,
            override_num_blocks=override_num_blocks
        ).materialize()
        self._num_of_blocks = self._dataset.num_blocks()
        logger.info(f"Data loaded")
        logger.info(f"Data summary:\n{self._dataset.schema()}\n")

        if data_transform and not batch_transform:
            logger.info(f"Transforming data")
            self._dataset = self._dataset.map(
                data_transformer,
                concurrency=self._num_of_blocks
            )
            logger.info(f"Data transformed")
            logger.info(f"Data summary:\n{self._dataset.schema()}\n")
        
        if batch_transform and not data_transform:
            logger.info(f"Transforming data - batch version")
            self._dataset = self._dataset.map_batches(
                batch_transformer,
                concurrency=self._num_of_blocks)
            logger.info(f"Data transformed - batch version")
            logger.info(f"Data summary:\n{self._dataset.schema()}\n")
        
        if data_transform and batch_transform:
            logger.error(f"Both data and batch transformation set to True. Choose only one.")
            raise ValueError("Both data and batch transformation set to True. Choose only one.")

        if seed != -1:
            logger.info(f"Shuffling data")
            self._dataset.random_shuffle(seed=seed)
            logger.info(f"Data shuffled")


    def __len__(self):
        return self._dataset.count()
