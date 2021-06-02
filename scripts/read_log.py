import glob
import re
import os


def read_macro_results(result_dir, save_dir):
    filelist = glob.glob(result_dir)
    w = open(save_dir, 'w')
    for file in filelist:
        filename = os.path.basename(file)
        patt    = re.search(r'(\d+)-(\w+).log', filename)
        r = open(file, 'r')
        mse = re.search(r'(\d+\.\d+).*', r.readlines()[-1]).group(1)
        w.write(f'stock: {patt.group(1)}, model: {patt.group(2)}, MSE: {mse}\n')
        r.close()
    w.close()


if __name__ == '__main__':
    read_macro_results('./results/logs/vwap/m2t/macro/*.log', './results/summary/macro.txt')