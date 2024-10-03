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
        'entity': 'davidjia',
        'project': 'Libero_qian',
        'groups': ['libero_spatial_5_history_pre_5_d_state_256_4_2'],
        'fields': '',
        'runs': ["all"],
        'config': '',
        'output_path': '',
    }}

if __name__ == '__main__':

    n_layers = [4, 8]
    mamba_d_state = [256]
    # mamba_d_conv = [2, 4]

    fields = ['average_success_rate_in_5_prediction_horizon']

    values = []

    csv_file = open('libero_spatial_transformamba_with_causal_mask_5history_5prediction_256state_4conv_2expand.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([''] + n_layers)

    for layer in n_layers:

        columns = []

        for d_state in mamba_d_state:
            config['local']['config'] = {
                'n_layers': {
                    'values': [layer]
                             },
                'mamba_ssm_cfg.d_state': {
                    'values': [d_state]
                }
            }

            config['local']['fields'] = fields

            data_dict, config_list = wandb2numpy.export_data(config)

            success = data_dict['local'][fields[0]]
            success_mean = success.mean()
            success_std = success.std()

            columns.append(str(round(success_mean, 3)) + '+-' + str(round(success_std, 3)))

        writer.writerow([d_state] + columns)
            # print(bcolors.WARNING + f"& ${round(success_mean, 3)} \scriptstyle \pm {round(success_std, 3)}$" + bcolors.ENDC)
        values.append(columns)

