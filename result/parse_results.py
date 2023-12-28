import argparse
import os

import pandas as pd

from compute_dso_all_metrics import compute_dso_all_metrics


#     return r
def read_until_line_starts_with(inp, line):
    l = inp.readline()
    while l != "" and not l.startswith(line):
        l = inp.readline()
    return l


def create_all_metrics_dict(inp):
    l = inp.readline()
    # print(l)
    l = inp.readline()
    # print(l)
    val_dict = {}
    while l != "" and not l.startswith("%%%%%"):
        spl = l.split(" ")
        val_dict[spl[0]] = float(spl[1].strip())
        l = inp.readline()
        # print(l)
    print(val_dict)
    return val_dict


def parse_gp_file(filename, verbose=False):
    # print('filename=', filename)
    inp = open(filename, 'r')
    l = read_until_line_starts_with(inp, 'final hof')
    # print('l=', l)
    rs = []
    l = read_until_line_starts_with(inp, 'validate r=')
    while l != "":
        # print('l=', l.strip())
        tt = l[:-1].split()
        val_dict = create_all_metrics_dict(inp)
        rs.append([float(tt[2]), val_dict])
        l = read_until_line_starts_with(inp, 'validate r=')

    inp.close()
    # print(rs)
    rs.sort(key=lambda x: x[0], reverse=True)  # changes the list in-place (and returns None)
    # print(rs[0])
    r = rs[0]
    return r

def parse_dso_file(dso_log_filename, true_program_file, basepath, noise_std=0.0):
    if not os.path.isfile(dso_log_filename):
        print(dso_log_filename + ' does not exist')
    if not os.path.isfile(true_program_file):
        print(true_program_file + ' does not exisits')
    # print(dso_log_filename)
    inp = open(dso_log_filename, 'r')
    l = read_until_line_starts_with(inp, 'Source path______')

    data_frame_basepath = l.strip().split('______./')[-1]
    if len(data_frame_basepath) == 0:
        return None
    print(data_frame_basepath)
    csv_expr_dir = os.path.join(basepath,  data_frame_basepath)

    if not os.path.isdir(csv_expr_dir):
        print(csv_expr_dir, 'does not exists')
        return None
    else:
        print(csv_expr_dir, 'exists')
    csv_expr_path = ""
    for root, dirs, files in os.walk(csv_expr_dir):
        for name in files:
            if name.endswith("hof.csv"):
                csv_expr_path = os.path.join(root, name)
                break
    if not os.path.isfile(csv_expr_path):
        print("cannot find hof file")
        return None
    print(csv_expr_path)
    return compute_dso_all_metrics(true_program_file, 'normal', 0.0, csv_expr_path, testset_size=256)


def parse_exp_set(file_prefix, metric_name, noise_type, noise_scale, true_program_basepath, dso_basepath, keyword="Korns"):
    all_dso_r, all_gp_r, all_egp_r = {}, {}, {}
    gp_output_files, egp_output_files = {}, {}
    dso_output_files = {}
    for key in ['VPG', 'PQT', 'DSR', 'GPMELD']:
        dso_output_files[key] = {}
    for root, dirs, files in os.walk(file_prefix, topdown=False):
        for name in files:
            if keyword and keyword not in name:
                continue
            # if metric_name in name and noise_type in name and noise_scale in name:
            for key in ['VPG', 'PQT', 'DSR', 'GPMELD']:
                    if key in name:
                        dso_output_files[key][name.split('.')[0]] = os.path.join(root, name)
    print(dso_output_files)
    for prog in gp_output_files:
        try:
            filename = gp_output_files[prog]
            if not os.path.isfile(filename):
                raise FileExistsError(filename, 'does not exists!')
            gp_r = parse_gp_file(filename, verbose=True)
            all_gp_r[prog] = gp_r[-1]
            #print('gp', gp_r)
        except:
            print(f'cannot parse GP {filename}')

    for prog in egp_output_files:
        try:
            filename = egp_output_files[prog]
            print(f"egp file: {filename}")
            if not os.path.isfile(filename):
                raise FileExistsError(filename, 'does not exists!')
            egp_r = parse_gp_file(filename, verbose=False)
            all_egp_r[prog] = egp_r[-1]
            #print('egp', egp_r)
        except:
            print(f'cannot parse EGP {filename}')

    if dso_basepath == None:
        return all_dso_r, all_gp_r, all_egp_r

    for baseline_name in ['PQT', 'VPG', 'DSR', 'GPMELD']:
        all_dso_r[baseline_name] = {}
        for prog in dso_output_files[baseline_name]:
            filename = dso_output_files[baseline_name][prog]
            if not os.path.isfile(filename):
                print(filename, ' does not exists!')
            dso_r = parse_dso_file(filename, true_program_basepath + prog + '.in', dso_basepath)
            if dso_r != None:
                all_dso_r[baseline_name][prog] = dso_r

    return all_dso_r, all_gp_r, all_egp_r


def pretty_print_dso_family(all_rs, is_numbered=1):
    for key in ['neg_nmse', 'neg_mse', 'neg_rmse', 'neg_nrmse']:# 'inv_nrmse', 'inv_nmse', 'neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:
        print('{}, DSR, PQT, VPG, GPMELD'.format(key))
        if is_numbered == 1:
            for idx in range(10):
                print(idx, end=", ")
                for baseline_name in ['DSR', 'PQT', 'VPG', 'GPMELD']:
                    if str(idx) in all_rs[baseline_name]:
                        print(all_rs[baseline_name][str(idx)][key], end=", ")
                    else:
                        print(",", end=" ")
                print()
        elif is_numbered == 0:
            for prog in all_rs['DSR'].keys():
                print(prog, end=", ")
                for baseline_name in ['DSR', 'PQT', 'VPG', 'GPMELD']:
                    if prog not in all_rs[baseline_name]:
                        print(",", end=" ")
                    elif key in all_rs[baseline_name][prog]:
                        print(all_rs[baseline_name][prog][key], end=", ")
                    else:
                        print(",", end=" ")
                print()
            print()


def pretty_print_pair(all_gp_rs, all_egp_rs, metric_name, is_numbered=True):
    for key in ['neg_nmse']:# 'neg_nrmse', 'inv_nrmse', 'inv_nmse', 'neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:
        print(f"{key} GP, EGP")
        if is_numbered:
            for key in range(10):
                key = 'prog_' + str(key)
                if key in all_gp_rs:
                    print(all_gp_rs[key], end=", ")
                else:
                    print(",", end=" ")
                if key in all_egp_rs:
                    print(all_egp_rs[key])
                else:
                    print()
        else:
            for prog in all_gp_rs:
                print(prog, end=", ")
                if prog in all_gp_rs:
                    print(all_gp_rs[prog][key], end=", ")
                else:
                    print(",", end=" ")
                if prog in all_egp_rs:
                    print(all_egp_rs[prog][key])
                else:
                    print()


def pretty_print_eureqa(all_eureqa_rs):
    for key in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse', 'neg_mse', 'neg_rmse', 'inv_mse']:
        # print('{}\ndata idx, gp, expand_gp, dso'.format(key))
        print(key, ", EUREQA")
        for idx in range(10):
            print(idx, end=", ")
            if idx in all_eureqa_rs:
                print(all_eureqa_rs[idx][key])
            else:
                print()
        print()


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--fp', type=str, required=True)
    parser.add_argument('--metric', type=str, default='neg_mse', required=False)
    parser.add_argument('--dso_basepath', type=str, required=False, default='None')
    parser.add_argument("--keyword", type=str, default=None)
    parser.add_argument('--noise_type', type=str, required=True, default="None")
    parser.add_argument('--noise_scale', type=str, default='0.0')
    parser.add_argument('--is_numbered', type=int, default=0)
    parser.add_argument('--true_program_basepath', type=str,
                        default="./scibench/data/unencrypted/equations_feynman/")

    # Parse the argument
    args = parser.parse_args()

    all_dso_r, all_gp_r, all_egp_r = parse_exp_set(args.fp, args.metric, args.noise_type, args.noise_scale,
                                                   args.true_program_basepath, args.dso_basepath, args.keyword)
    # print(all_gp_r)
    # print(all_egp_r)
    print(all_dso_r)

    if len(all_dso_r) != 0:
        pretty_print_dso_family(all_dso_r, is_numbered=args.is_numbered)
    print("GP & EGP")
    if len(all_gp_r) != 0 or len(all_egp_r):
        pretty_print_pair(all_gp_r, all_egp_r, args.metric, is_numbered=args.is_numbered)
