import numpy as np

from smhi_dataloader import ClimateDataBuilder, parse_config


config_path = '/users/haackesk/Desktop/sirius/src/configs/sample_config.toml'
date_config = parse_config(config_path=config_path, config_keyword="dates")
preprocessing_config = parse_config(config_path=config_path, config_keyword="preprocessing")
predictor_config = parse_config(config_path=config_path, config_keyword="predictors")
static_features_config = parse_config(config_path=config_path, config_keyword="static_features")
target_config = parse_config(config_path=config_path, config_keyword="targets")

data_builder = ClimateDataBuilder(
    date_config,
    preprocessing_config,
    predictor_config, 
    static_features_config, 
    target_config
)

data_builder.build_training_set()

indices = np.sort(np.concatenate([data_builder.train_idx, data_builder.val_idx]))

data_builder.write_preprocessed_data(indices)