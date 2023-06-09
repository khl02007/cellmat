import spikeinterface.full as si
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def get_recording(nwb_file_name, shank_id):
    recording_path = (sgs.SpikeSortingRecording & {"nwb_file_name":nwb_file_name,
                                                   "sort_group_id":shank_id}).fetch("recording_path")[0]
    recording = si.load_extractor(recording_path)
    if recording.get_num_segments()>1:
        recording = si.concatenate_recordings([recording])
    return recording

def get_sorting(nwb_file_name, shank_id):
    sorting_path = (sgs.SpikeSorting & {"nwb_file_name":nwb_file_name,
                                        "sort_group_id":shank_id}).fetch("sorting_path")[0]
    sorting = si.load_extractor(sorting_path)

    return sorting

def open_json(path):
    with open(path, 'r') as f:
        c = json.load(f)
    return c

def get_curation_labels(nwb_file_name, shank_id, path_to_sorting_curation_repo):
    curation_uri = (sgs.CurationFigurl & {"nwb_file_name": nwb_file_name,
                                          "sort_group_id":shank_id}).fetch("new_curation_uri")[0]
    local_path = curation_uri.replace('gh://LorenFrankLab/sorting-curations/main/', path_to_sorting_curation_repo)
    curation_labels = open_json(local_path)
    return curation_labels

def get_units_to_remove(curation_labels):
    exclude_labels = ['noise', 'reject', 'mua']
    units_to_remove = []
    for unit_id, labels in curation_labels['labelsByUnit'].items():
        if np.sum([i in exclude_labels for i in labels])>0:
            units_to_remove.append(int(unit_id))
    return units_to_remove

def curate_sorting(sorting, curation_labels, skip_merge=True):
    if curation_labels['labelsByUnit']:
        units_to_remove = get_units_to_remove(curation_labels)
        sorting = sorting.remove_units(units_to_remove)
    if curation_labels['mergeGroups'] and skip_merge==False:
        sorting = si.MergeUnitsSorting(parent_sorting=sorting,
                                       units_to_merge=curation_labels['mergeGroups'],
                                      )
    return sorting

def get_recording_list(nwb_file_name):
    r0 = get_recording(nwb_file_name, shank_id=0)
    r1 = get_recording(nwb_file_name, shank_id=1)
    r2 = get_recording(nwb_file_name, shank_id=2)
    r3 = get_recording(nwb_file_name, shank_id=3)

    return [r0,r1,r2,r3]

def get_sorting_list(nwb_file_name):
    s0 = get_sorting(nwb_file_name, shank_id=0)
    s1 = get_sorting(nwb_file_name, shank_id=1)
    s2 = get_sorting(nwb_file_name, shank_id=2)
    s3 = get_sorting(nwb_file_name, shank_id=3)

    return [s0,s1,s2,s3]

def plot_circles(ax, recording_list, start_frame, end_frame, edgecolor):
    locations = []
    values = []
    for recording in recording_list:
        x = recording.get_traces(start_frame=start_frame, end_frame=end_frame,
                                 return_scaled=True)
        values.append(np.std(x, axis=0))
        locations.append(recording.get_channel_locations())
    locations = np.concatenate(locations)
    values = np.concatenate(values)
    f = 0.8
    locations[:,0] = locations[:,0]*0.25
    for i in range(len(locations)):
        x, y = locations[i]
        value = values[i]

        circle = plt.Circle((x, y), radius=value*f, alpha=1, fill=False,
                           edgecolor=edgecolor, linewidth=1.2)
        ax.add_patch(circle)
        # ax.plot(x,y,'k.')

    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.axis('off')
    return values
    # plt.show()