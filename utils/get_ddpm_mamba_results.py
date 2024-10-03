import csv
import os
import wandb2numpy
import matplotlib.pyplot as plt
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


config = {
    "local": {
        'entity': 'tiger_or_cat',
        'project': '3_seed',
        'groups': ['bc_mamba_goal'],
        'fields': '',
        'runs': ["all"],
        'config': '',
        'output_path': '',
    }}

if __name__ == '__main__':

    clip = False

    sv_dir = '3seed_results'
    os.makedirs(sv_dir, exist_ok=True)

    task_suite = ['libero_goal', 'libero_90']#["libero_object", "libero_spatial", "libero_10"]

    diff_steps = [16]
    n_layers = [10, 16]

    d_states = [8, 16]
    d_convs = [2, 4]

    obs_seqs = [1, 5]

    action_seq = [5]
    store = ['max', 'mean']

    # fields = ['epoch39_average_success', 'epoch49_average_success', 'epoch59_average_success']

    fields = ['epoch49_average_success']

    csv_file = open(sv_dir + '/bc_mamba_goal.csv', 'w', newline='')
    writer = csv.writer(csv_file)

    for task in task_suite:

        for obs in obs_seqs:

            for diff in diff_steps:

                for n_layer in n_layers:

                    for state in d_states:

                        for conv in d_convs:

                            try:

                                writer.writerow([f'{task}_obs{obs}_diff{diff}_layer{n_layer}_state{state}_conv{conv}'])
                                writer.writerow(['mean', 'max'])

                                success_list = []
                                for evaluation in fields:
                                    config['local']['config'] = {
                                        'task_suite': {
                                            'values': [task]
                                        },
                                        'obs_seq': {
                                            'values': [obs]
                                        },
                                        'diff_steps': {
                                            'values': [diff]
                                        },
                                        'n_layers': {
                                            'values': [n_layer]
                                        },
                                        'mamba_ssm_cfg.d_state': {
                                            'values': [state]
                                        },
                                        'mamba_ssm_cfg.d_conv': {
                                            'values': [conv]
                                        },
                                    }

                                    config['local']['fields'] = [evaluation]

                                    data_dict, config_list = wandb2numpy.export_data(config)

                                    success = data_dict['local'][evaluation]

                                    success_list.append(success)

                                success_list = np.concatenate(success_list, axis=-1)

                                metric_means = success_list.mean(axis=-1)
                                # metric_maxes = success_list.max(axis=-1)

                                if clip:
                                    metric_means = np.sort(metric_means, axis=-1)
                                    metric_means = metric_means[1:4]
                                    metric_maxes = np.sort(metric_maxes)
                                    metric_maxes = metric_maxes[1:4]

                                writer.writerow([str(round(metric_means.mean(), 3)) + '+-' + str(round(metric_means.std(), 3))])
                                                 # str(round(metric_maxes.mean(), 3)) + '+-' + str(round(metric_maxes.std(), 3))])

                                writer.writerow([''])
                                writer.writerow([''])

                            except:
                                print(bcolors.WARNING + f'Error' + bcolors.ENDC)
