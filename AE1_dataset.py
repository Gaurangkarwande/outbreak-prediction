import numpy as np

def create_dataset(spatio_temporal_dataset, temporal_dataset, train_regions, cv_regions, test_regions ):
    train_dataset = np.transpose(np.vstack([spatio_temporal_dataset[region] for region in train_regions]), (0,3,1,2))
    train_labels = np.vstack([temporal_dataset[region] for region in train_regions])

    cv_dataset = np.transpose(np.vstack([spatio_temporal_dataset[region] for region in cv_regions]), (0,3,1,2))
    cv_labels = np.vstack([temporal_dataset[region] for region in cv_regions])

    test_dataset = np.transpose(np.vstack([spatio_temporal_dataset[region] for region in test_regions]), (0,3,1,2))
    test_labels = np.vstack([temporal_dataset[region] for region in test_regions])
    return train_dataset, train_labels, cv_dataset, cv_labels, test_dataset, test_labels