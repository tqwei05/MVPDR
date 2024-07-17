from .plantwild import PlantWild
from .plantdoc import PlantDoc
from .plantvillage import PlantVillage


dataset_list = {
                "plantwild": PlantWild,
                "plantdoc": PlantDoc,
                "plantvillage": PlantVillage
                }


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)