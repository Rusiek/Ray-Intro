import logging
import torch

from typing import Dict

logger_handler = logging.StreamHandler()
logger_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s")
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logger_handler)


class BaseTransformer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, data: Dict) -> Dict:
        ...
