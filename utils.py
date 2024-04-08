from benchmark.data.dataset import BaseDataset
from benchmark.data.transformer import BaseTransformer

from typing import Dict


class IMDBData(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class IMDBTransformer(BaseTransformer):
    def __init__(self):
        super().__init__()

    def __call__(self, data: Dict) -> Dict:
        return {"names": data["names"], "budget_x": data["budget_x"]}
