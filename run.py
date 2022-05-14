import sys
import argparse
import os
# from subprocess import call
from subprocess import check_output
# TODO: verify requirements installed, establish cuda capability and adjust accordingly


def prepare_bash_call():
    script_call = ['./compare.sh', '-t', str(args.typerun), "-r", args.resultsfile, "-p", args.pythoncall]

    if args.imagefile is not None:
        script_call.append("-i")
        script_call.append(args.imagefile)

    return script_call

def set_graph_file():
    if args.graphfile is not None:
        return args.graphfile

    split_file = args.resultsfile.split('.')
    save_to = [x for x in split_file[:-1]]
    return ".".join(save_to) + '.png'

def parse_results(single_string):
    res_strings = single_string.splitlines()
    res = []

    for x in res_strings:
        if len(x) == 0 or x==' ':
            continue
        temp = []
        for y in x.split(';'):
            if '{' in y:
                y = y.replace('{', '')
                y = y.replace('}', '')
                y = y.split(',')
                y = [float(z) for z in y]
                temp.append(y)
            else:
                temp.append(float(y))
        res.append(temp)
    
    return res

def split_c_py(all_results, c_list, py_list):
    c_seq_set = False
    py_seq_set = False
    
    for x in all_results:
        if x[0] == 0 and not c_seq_set:
            c_seq_set = True
        elif x[0] == 0 and not py_seq_set:
            py_seq_set = True

        if c_seq_set and not py_seq_set:
            c_list.append(x)
        else:
            py_list.append(x)
    
    return


parser = argparse.ArgumentParser(description='Run compare.sh benchmarking and parse results')
parser.add_argument("-t", "--typerun", required=True, type=int, choices=[1, 2, 3], help="Required to specify whether to run CUDA C (1), Python Numba CUDA (2), or both (3)")
parser.add_argument("-r", "--resultsfile", default="results.txt", help="Specify file for numerical results")
parser.add_argument("-g", "--graphfile", help="Specify file for graph results")
parser.add_argument("-p", "--pythoncall", default="python3", help="Specify command to run Python scripts")
parser.add_argument("-i", "--imagefile", help="Specify specific image to use.")


args = parser.parse_args()
plot_file = set_graph_file()
data_headers = ["Block Dimensions", "Thread Dimensions", "Pixels Per Thread", "Pixels Per Block", "Time to Transfer Data", "Total Pixels Per ms", "GPU Pixels per ms"]



if __name__ == '__main__':
    res_string = check_output(prepare_bash_call()).decode()
    res_parsed = parse_results(res_string)
    c_results = res_parsed if args.typerun == 1 else []
    py_results = res_parsed if args.typerun == 2 else []

    if args.typerun == 3:
        split_c_py(res_parsed, c_results, py_results)
