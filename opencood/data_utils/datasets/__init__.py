from opencood.data_utils.datasets.late_fusion_dataset import getLateFusionDataset
from opencood.data_utils.datasets.late_heter_fusion_dataset import getLateheterFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import getEarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import getIntermediateFusionDataset
from opencood.data_utils.datasets.intermediate_2stage_fusion_dataset import getIntermediate2stageFusionDataset
from opencood.data_utils.datasets.intermediate_heter_fusion_dataset import getIntermediateheterFusionDataset
from opencood.data_utils.datasets.intermediate_heter_fusion_3class_dataset import getIntermediateheter3classFusionDataset

from opencood.data_utils.datasets.heter_infer.intermediate_heter_infer_fusion_dataset import getIntermediateheterinferFusionDataset

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from opencood.data_utils.datasets.basedataset.v2xsim_basedataset import V2XSIMBaseDataset
from opencood.data_utils.datasets.basedataset.dairv2x_basedataset import DAIRV2XBaseDataset
from opencood.data_utils.datasets.basedataset.v2xset_basedataset import V2XSETBaseDataset
from opencood.data_utils.datasets.basedataset.v2xreal_basedataset import V2XREALBaseDataset

from opencood.data_utils.datasets.intermediate_fusion_dataset_dair import IntermediateFusionDatasetDAIR


# the final range for evaluation
# GT_RANGE = [-102.4, -102.4, -6, 102.4, 102.4, 3]
# GT_RANGE = [-202.4, -202.4, -7, 202.4, 202.4, 2]
# GT_RANGE = [-100, -40, -15, 100, 40, 15]
GT_RANGE = [-140, -40, -3, 140, 40, 1]
# GT_RANGE = [-70, -70, -7, 70, 70, 4]
# GT_RANGE = [-51.2, -51.2, -5, 51.2, 51.2, 2]
# The communication range for cavs
COM_RANGE = 70 #50


def build_dataset(dataset_cfg, visualize=False, train=True, calibrate=False):
    fusion_name = dataset_cfg['fusion']['core_method']
    dataset_name = dataset_cfg['fusion']['dataset']

    # Special handling for DAIR-V2X intermediate fusion (legacy standalone implementation)
    # IntermediateFusionDatasetDAIR is a complete implementation that doesn't inherit from base
    if fusion_name.lower() == 'intermediate' and dataset_name.lower() == 'dairv2x' and 'dair_data_dir' in dataset_cfg:
        dataset = IntermediateFusionDatasetDAIR(
            params=dataset_cfg,
            visualize=visualize,
            train=train
        )
    else:
        # Standard factory pattern for all other combinations
        assert fusion_name in ['late', 'lateheter', 'intermediate', 'intermediate2stage', 'intermediateheter', 'early', 'intermediateheterinfer', 'intermediateheter3class'], \
            f"Invalid fusion method: {fusion_name}"
        assert dataset_name in ['opv2v', 'v2xsim', 'dairv2x', 'v2xset', 'v2xreal'], \
            f"Invalid dataset name: {dataset_name}"

        fusion_dataset_func = "get" + fusion_name.capitalize() + "FusionDataset"
        fusion_dataset_func = eval(fusion_dataset_func)
        base_dataset_cls = dataset_name.upper() + "BaseDataset"
        base_dataset_cls = eval(base_dataset_cls)

        dataset = fusion_dataset_func(base_dataset_cls)(
            params=dataset_cfg,
            visualize=visualize,
            train=train,
            calibrate=calibrate
        )

    return dataset
