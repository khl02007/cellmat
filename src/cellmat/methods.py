from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import IncrementalPCA
import sklearn.metrics.pairwise
from scipy.stats import pearsonr

def _compute_contamination(pcs_target_unit, pcs_other_unit, n_neighbors: int):
    # get lengths
    n_spikes_target = pcs_target_unit.shape[0]
    n_spikes_other = pcs_other_unit.shape[0]

    # concatenate
    pcs_concat = np.concatenate((pcs_target_unit, pcs_other_unit), axis=0)
    label_concat = np.concatenate((np.zeros(n_spikes_target), np.ones(n_spikes_other)))

    # if n_neighbors is greater than the number of spikes in both clusters, set it to max possible
    if n_neighbors > len(label_concat):
        n_neighbors_adjusted = len(label_concat) - 1
    else:
        n_neighbors_adjusted = n_neighbors

    _, membership_ind = (
        NearestNeighbors(n_neighbors=n_neighbors_adjusted, algorithm="auto").fit(pcs_concat).kneighbors()
    )
    target_nn_in_other = np.sum(label_concat[membership_ind[:n_spikes_target]] == 1)

    # target_nn_in_target = np.sum(label_concat[membership_ind[:n_spikes_target]] == 0)
    # other_nn_in_other = np.sum(label_concat[membership_ind[n_spikes_target:]] == 1)

    contamination = target_nn_in_other / (n_spikes_target) / n_neighbors_adjusted

    return contamination

def compute_nn_distance(waveform_extractor1, waveform_extractor2, radius_um=50, seed=47, n_neighbors=5, n_spikes=1000, n_components=5):
    
    rng = np.random.default_rng(seed=seed)
    
    sparsity1 = si.compute_sparsity(waveform_extractor1, method='radius', radius_um=radius_um)
    sparsity2 = si.compute_sparsity(waveform_extractor2, method='radius', radius_um=radius_um)
    
    output = np.zeros((waveform_extractor1.sorting.get_num_units(),
                       waveform_extractor2.sorting.get_num_units()))
    
    all_channels = waveform_extractor1.recording.get_channel_ids()
    # print(all_channels)
    assert np.array_equal(waveform_extractor1.recording.get_channel_ids(), waveform_extractor2.recording.get_channel_ids()), "the waveform extractors are not from the same shanks"
    
    for unit_id1 in waveform_extractor1.sorting.get_unit_ids():
        waveforms_unit1 = waveform_extractor1.get_waveforms(unit_id=unit_id1)

        for unit_id2 in waveform_extractor2.sorting.get_unit_ids():
            waveforms_unit2 = waveform_extractor2.get_waveforms(unit_id=unit_id2)
            
            n_spikes = np.min([waveforms_unit1.shape[0], waveforms_unit2.shape[0], n_spikes])
            
            channels_with_signal = np.concatenate((sparsity1.unit_id_to_channel_ids[unit_id1], sparsity2.unit_id_to_channel_ids[unit_id2]))
            channels_with_signal = np.unique(channels_with_signal)
            # print(sparsity1.unit_id_to_channel_ids[unit_id1])
            # print(sparsity2.unit_id_to_channel_ids[unit_id2])
            
            indices = np.where(np.isin(all_channels, channels_with_signal))[0]
            # print(indices)
            
            waveforms_unit1_sparse = waveforms_unit1[:,:,indices]
            waveforms_unit2_sparse = waveforms_unit2[:,:,indices]

            spike_ind1 = rng.choice(waveforms_unit1_sparse.shape[0], size=n_spikes, replace=False)
            spike_ind2 = rng.choice(waveforms_unit2_sparse.shape[0], size=n_spikes, replace=False)

            waveforms_unit1_sparse = waveforms_unit1_sparse[spike_ind1]
            waveforms_unit2_sparse = waveforms_unit2_sparse[spike_ind2]

            waveforms_unit1_sparse = np.reshape(waveforms_unit1_sparse, (n_spikes,-1))
            waveforms_unit2_sparse = np.reshape(waveforms_unit2_sparse, (n_spikes,-1))

            # concatenate
            waveforms_concat = np.concatenate((waveforms_unit1_sparse, waveforms_unit2_sparse), axis=0)
            
            pca = IncrementalPCA(n_components=n_components)
            pca.partial_fit(waveforms_concat)
            projected_waveforms = pca.transform(waveforms_concat)
            
            output[waveform_extractor1.sorting.get_unit_ids()==unit_id1,
                   waveform_extractor2.sorting.get_unit_ids()==unit_id2] = _compute_contamination(projected_waveforms[:n_spikes], projected_waveforms[n_spikes:], n_neighbors)
    return output

import sklearn.metrics.pairwise

def compute_cosine_similarity(waveform_extractor1, waveform_extractor2, radius_um=50):
    sparsity1 = si.compute_sparsity(waveform_extractor1, method='radius', radius_um=radius_um)
    sparsity2 = si.compute_sparsity(waveform_extractor2, method='radius', radius_um=radius_um)

    output = np.zeros((waveform_extractor1.sorting.get_num_units(),
                       waveform_extractor2.sorting.get_num_units()))
    
    all_channels = waveform_extractor1.recording.get_channel_ids()
    assert np.array_equal(waveform_extractor1.recording.get_channel_ids(), waveform_extractor2.recording.get_channel_ids()), "the waveform extractors are not from the same shanks"
    for unit_id1 in waveform_extractor1.sorting.get_unit_ids():
        
        for unit_id2 in waveform_extractor2.sorting.get_unit_ids():
            
            channels_with_signal = np.concatenate((sparsity1.unit_id_to_channel_ids[unit_id1], sparsity2.unit_id_to_channel_ids[unit_id2]))
            channels_with_signal = np.unique(channels_with_signal)
            indices = np.where(np.isin(all_channels, channels_with_signal))[0]

            template_unit1 = waveform_extractor1.get_template(unit_id=unit_id1)
            template_unit1 = template_unit1[:,indices]
            template_unit1 = template_unit1.flatten()
            template_unit1 = np.expand_dims(template_unit1, axis=0)
            
            template_unit2 = waveform_extractor2.get_template(unit_id=unit_id2)
            template_unit2 = template_unit2[:,indices]
            template_unit2 = template_unit2.flatten()
            template_unit2 = np.expand_dims(template_unit2, axis=0)

            output[waveform_extractor1.sorting.get_unit_ids()==unit_id1,
                   waveform_extractor2.sorting.get_unit_ids()==unit_id2] = sklearn.metrics.pairwise.cosine_similarity(template_unit1, template_unit2)
    return output


def compute_acc_similarity(sorting1, sorting2, window_ms=100, bin_ms=2):
    
    corr1, _ = si.compute_correlograms(sorting1, window_ms=window_ms, bin_ms=bin_ms, method="numpy")
    ac1 = np.zeros((corr1.shape[0], corr1.shape[-1]))
    for i in range(corr1.shape[0]):
        ac1[i] = corr1[i,i,:]
        
    corr2, _ = si.compute_correlograms(sorting2, window_ms=window_ms, bin_ms=bin_ms, method="numpy")
    ac2 = np.zeros((corr2.shape[0], corr2.shape[-1]))
    for i in range(corr2.shape[0]):
        ac2[i] = corr2[i,i,:]
        
    output = np.zeros((sorting1.get_num_units(), sorting2.get_num_units()))
    for i in range(ac1.shape[0]):
        for j in range(ac2.shape[0]):
            k, _ = pearsonr(ac1[i], ac2[j])
            output[i,j]=k
    return output