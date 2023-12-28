import argparse
import os


def read_until_line_starts_with(inp, pattern):
    l = inp.readline()
    while l != "" and not pattern in l:
        l = inp.readline()
    return l


def create_all_metrics_dict(inp):
    l = inp.readline()
    # print(l)
    l = inp.readline()
    # print(l)
    val_dict = {}
    while l != "" and not l.startswith("------------------------------") and not l.startswith("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"):
        spl = l.split(" ")
        val_dict[spl[0]] = float(spl[1].strip())
        l = inp.readline()
        # print(l)
    # print(val_dict)
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


def parse_exp_set(file_prefix, noise_type, noise_scale, keyword="Korns"):
    all_randgp_r = {}
    randgp_output_files = {}
    for root, dirs, files in os.walk(file_prefix, topdown=False):
        for name in files:
            
            if keyword and keyword not in name:
                continue
            if noise_type in name and noise_scale in name:
                randgp_output_files[name.split('.')[0]] = os.path.join(root, name)
    print(randgp_output_files)
    for prog in randgp_output_files:
        print(prog,"....")
        try:
            filename = randgp_output_files[prog]
            if not os.path.isfile(filename):
                raise FileExistsError(filename, 'does not exists!')
            gp_r = parse_gp_file(filename, verbose=True)
            all_randgp_r[prog] = gp_r[-1]
            # print('gp', gp_r)
        except Exception as e: 
            print(e)
            print(f'cannot parse {filename}')

    return all_randgp_r


def pretty_print_pair(all_gp_rs, max_prog=10, is_numbered=True):
    for key in ['neg_nmse', 'neg_mse', 'neg_rmse', 'neg_nrmse']: #'inv_mse', 'inv_nmse', 'inv_nrmse'
        print(f"{key} ")
        if is_numbered:
            for i in range(max_prog):
                print(i, end=", ")
                progi = 'prog_' + str(i)
                if progi in all_gp_rs:
                    print(all_gp_rs[progi][key])
                else:
                    print()
        else:
            keys = sorted(list(all_gp_rs.keys()))
            for prog in keys:  # all_gp_rs:
                print(prog, end=", ")
                if prog in all_gp_rs:
                    print(all_gp_rs[prog][key])
                else:
                    print()


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--fp', type=str, required=True)
    # parser.add_argument('--metric', type=str, default='neg_nmse', required=True)
    parser.add_argument("--keyword", type=str, default=None)
    parser.add_argument('--noise_type', type=str, required=True, default="None")
    parser.add_argument('--noise_scale', type=str, default='0.0')
    parser.add_argument('--is_numbered', type=int, default=0)
    parser.add_argument('--max_prog', type=int, default=10)
    # Parse the argument
    args = parser.parse_args()
    all_randgp_r = parse_exp_set(args.fp, args.noise_type, args.noise_scale, args.keyword)

    print(args.keyword)
    if len(all_randgp_r) != 0:
        pretty_print_pair(all_randgp_r, max_prog=args.max_prog, is_numbered=args.is_numbered)
