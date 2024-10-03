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
        'project': 'libero_benchmark',
        'groups': ['ddpm_encdec_benchmark'],
        'fields': '',
        'runs': ["all"],
        'config': '',
        'output_path': '',
    }}

if __name__ == '__main__':

    sv_dir = 'ddpm_encdec_benchmark'
    os.makedirs(sv_dir, exist_ok=True)

    task_suite = ["libero_spatial", "libero_10"]
    diff_steps = [8, 16]
    obs_seq = [1, 5]
    action_seq = [5]
    store = ['max', 'mean']

    fields = ['epoch39_average_success', 'epoch49_average_success', 'epoch59_average_success']

    csv_file = open(sv_dir + '/output_matrix.csv', 'w', newline='')
    writer = csv.writer(csv_file)


    for task in task_suite:

        field = fields

        writer.writerow([f'{task}'])

        for seq in obs_seq:

            writer.writerow([f'obs_seq_{seq}'])

            for store_type in store:

                writer.writerow([''] + [''] + [f'{store_type}_{evaluation}' for evaluation in fields])

                for step in diff_steps:

                    columns = []

                    for evaluation in fields:
                        config['local']['config'] = {
                            'task_suite': {
                                'values': [task]
                            },
                            'diff_steps': {
                                'values': [step]
                            },
                            'obs_seq': {
                                'values': [seq]
                            },
                        }

                        config['local']['fields'] = [evaluation]

                        data_dict, config_list = wandb2numpy.export_data(config)

                        success = data_dict['local'][evaluation]
                        if store_type == 'max':
                            success_rate = success.max()
                        else:
                            success_rate = success.mean()

                        success_std = success.std()

                        columns.append(str(round(success_rate, 3)) + '+-' + str(round(success_std, 3)))

                    writer.writerow([''] + [f'diff_steps_{step}'] + columns)
                    columns.clear()

                writer.writerow([''])
                writer.writerow([''])



    # n_layer_encoder = [4, 6]
    # n_layer_decoder = [4, 6]
    # # n_layers = [6]
    # mamba_d_state_encoder = [8, 16]
    # mamba_d_state_decoder = [8, 16]
    # # mamba_ssm_d_state = [4, 8, 16]
    # mamba_d_conv = [2, 4]
    # mamba_expand = [2]
    # obs_seq = [5]
    # action_seq = [5]
    #
    # fields = {20: ['average_success_rate_in_20_prediction_horizon'],
    #           10: ['average_success_rate_in_10_prediction_horizon'],
    #           5: ['average_success_rate_in_5_prediction_horizon']}
    #
    # csv_file = open(sv_dir + '/output_matrix1.csv', 'w', newline='')
    # writer = csv.writer(csv_file)
    #
    # for e_layer in n_layer_encoder:
    #     for d_layer in n_layer_decoder:
    #
    #         field = fields[5]
    #
    #         for conv in mamba_d_conv:
    #
    #             writer.writerow([f'obs_5_action_5_conv_{conv}_expand_2'])
    #             writer.writerow([''] + [f'encoder_layer_{e_layer}_decoder_layer_{d_layer}'])
    #             writer.writerow([''] + [''] + [f'encoder_state_{encoder_state}' for encoder_state in mamba_d_state_encoder])
    #
    #             for decoder_state in mamba_d_state_decoder:
    #
    #                 columns = []
    #
    #                 for encoder_state in mamba_d_state_encoder:
    #                     config['local']['config'] = {
    #                         'n_layer_encoder': {
    #                             'values': [e_layer]
    #                         },
    #                         'n_layer_decoder': {
    #                             'values': [d_layer]
    #                         },
    #                         'mamba_encoder_cfg.d_state': {
    #                             'values': [encoder_state]
    #                         },
    #                         'mamba_decoder_cfg.d_state': {
    #                             'values': [decoder_state]
    #                         },
    #                         'd_conv': {
    #                             'values': [conv]
    #                         },
    #                         # 'obs_seq': {
    #                         #     'values': [obs]
    #                         # }
    #                     }
    #
    #                     config['local']['fields'] = field
    #
    #                     data_dict, config_list = wandb2numpy.export_data(config)
    #
    #                     success = data_dict['local'][field[0]]
    #                     success_mean = success.mean()
    #                     success_std = success.std()
    #
    #                     columns.append(str(round(success_mean, 3)) + '+-' + str(round(success_std, 3)))
    #
    #                 writer.writerow([''] + [f'decoder_state_{decoder_state}'] + columns)
    #                 # print(bcolors.WARNING + f"& ${round(success_mean, 3)} \scriptstyle \pm {round(success_std, 3)}$" + bcolors.ENDC)
    #
    #             writer.writerow([''])
    #             writer.writerow([''])

    # sv_dir = 'libero_spatial_5_obs_128_dstate_ddpmamba_suite'
    # os.makedirs(sv_dir, exist_ok=True)
    #
    # n_layers = [6, 10, 16]
    # mamba_d_conv = [8, 16]
    # mamba_expand = [4, 8]
    # action_seq = [10, 5]
    #
    # fields = {10: ['average_success_rate_in_10_prediction_horizon'],
    #           5: ['average_success_rate_in_5_prediction_horizon']}
    #
    # csv_file = open(sv_dir + '/output_matrix.csv', 'w', newline='')
    # writer = csv.writer(csv_file)


    # for seq in obs_seq:
    #
    #     field = fields[5]
    #
    #     for conv in mamba_d_conv:
    #
    #         writer.writerow([f'obs_{seq}_action_5_conv_{conv}_expand_2'])
    #         writer.writerow([''] + [f'd_state_{state}' for state in mamba_ssm_d_state])
    #
    #         for n_layer in n_layers:
    #
    #             columns = []
    #
    #             for state in mamba_ssm_d_state:
    #                 config['local']['config'] = {
    #                     'n_layers': {
    #                         'values': [n_layer]
    #                     },
    #                     'mamba_ssm_cfg.d_conv': {
    #                         'values': [conv]
    #                     },
    #                     'mamba_ssm_cfg.d_state': {
    #                         'values': [state]
    #                     },
    #                     'obs_seq': {
    #                         'values': [seq]
    #                     }
    #                 }
    #
    #                 config['local']['fields'] = field
    #
    #                 data_dict, config_list = wandb2numpy.export_data(config)
    #
    #                 success = data_dict['local'][field[0]]
    #                 success_mean = success.mean()
    #                 success_std = success.std()
    #
    #                 columns.append(str(round(success_mean, 3)) + '+-' + str(round(success_std, 3)))
    #
    #             writer.writerow([f'layer_{n_layer}'] + columns)
    #             # print(bcolors.WARNING + f"& ${round(success_mean, 3)} \scriptstyle \pm {round(success_std, 3)}$" + bcolors.ENDC)
    #
    #         writer.writerow([''])
    #         writer.writerow([''])