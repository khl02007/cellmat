from .utils import (get_recording, get_sorting,
                    get_curation_labels, curate_sorting, count_unique_ones, find_unique_ones)
from .methods import compute_nn_distance
import spikeinterface.full as si
import subprocess
import numpy as np

def run_git_command(command):
    try:
        # Run the git command and capture the output
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        # Handle any errors that occurred during the command execution
        print(f"Error executing command: {e.output}")
        return None

def get_matched_cells(nwb_file_name1, nwb_file_name2, shank_id,
                      path_to_sorting_curation_repo='/home/kyu/repos/sorting-curations/'):

    recording1 = get_recording(nwb_file_name1, shank_id)
    recording2 = get_recording(nwb_file_name2, shank_id)


    sorting1 = get_sorting(nwb_file_name1, shank_id)
    sorting2 = get_sorting(nwb_file_name2, shank_id)
    
    # Example usage
    result = run_git_command(f"cd {path_to_sorting_curation_repo} && git pull origin main")
    if result is not None:
        print(result)
        
    curation_labels1 = get_curation_labels(nwb_file_name1, shank_id, path_to_sorting_curation_repo)
    curation_labels2 = get_curation_labels(nwb_file_name2, shank_id, path_to_sorting_curation_repo)
    
    sorting1 = curate_sorting(sorting1, curation_labels1)
    sorting2 = curate_sorting(sorting2, curation_labels2)
    
    waveform_extractor1 = si.extract_waveforms(recording1, sorting1, folder="/tmp/we1",
                                               ms_before=1.0, ms_after=1.0,
                                               max_spikes_per_unit=5000,
                                               overwrite=True,
                                               seed=47,
                                               dtype=np.float32
                                              )
    
    waveform_extractor2 = si.extract_waveforms(recording2, sorting2, folder="/tmp/we2",
                                               ms_before=1.0, ms_after=1.0,
                                               max_spikes_per_unit=5000,
                                               overwrite=True,
                                               seed=47,
                                               dtype=np.float32
                                              )
    
    nn_distance = compute_nn_distance(waveform_extractor1,
                                      waveform_extractor2,
                                      radius_um=50,
                                      n_neighbors=5,
                                      n_spikes=2000, n_components=4)
    
    nn_distance_threshold = 0.3
    k = find_unique_ones(nn_distance>nn_distance_threshold)
    output = {}
    for i in k:
        output[waveform_extractor1.sorting.get_unit_ids()[i[0]]] = waveform_extractor2.sorting.get_unit_ids()[i[1]]
    return output