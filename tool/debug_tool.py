#coding=utf8

import json

def parameter_reporter(model):
    parameter_log = {} 
    for name, para in model.named_parameters():
        if para.grad is None:
            continue
        parameter_log[name] = {}
        grad_mean = para.grad.mean()
        grad_std = para.grad.std()
        grad_min = para.grad.min()
        grad_max = para.grad.max()
        parameter_log[name]['Grad-mean'] = float(grad_mean)
        parameter_log[name]['Grad-min'] = float(grad_min)
        parameter_log[name]['Grad-max'] = float(grad_max)
        parameter_log[name]['Grad-std'] = float(grad_std)
        para_mean = para.mean()
        para_std = para.std()
        para_min = para.min()
        para_max = para.max()
        parameter_log[name]['Para-mean'] = float(para_mean)
        parameter_log[name]['Para-min'] = float(para_min)
        parameter_log[name]['Para-max'] = float(para_max)
        parameter_log[name]['Para-std'] = float(para_std)
    return parameter_log


def read_gradient_file(filename, output_file):
    fpout_grad_mean = open(output_file+'.grad.mean.txt', 'w')
    fpout_grad_min = open(output_file+'.grad.min.txt', 'w')
    fpout_grad_max = open(output_file+'.grad.max.txt', 'w')
    fpout_grad_std = open(output_file+'.grad.std.txt', 'w')
    fpout_para_mean = open(output_file+'.para.mean.txt', 'w')
    fpout_para_min = open(output_file+'.para.min.txt', 'w')
    fpout_para_max = open(output_file+'.para.max.txt', 'w')
    fpout_para_std = open(output_file+'.para.std.txt', 'w')
    parameter_log = {}
    parameter_log['Grad-mean'] = {}
    parameter_log['Grad-min'] = {}
    parameter_log['Grad-max'] = {}
    parameter_log['Grad-std'] = {}
    parameter_log['Para-mean'] = {}
    parameter_log['Para-min'] = {}
    parameter_log['Para-max'] = {}
    parameter_log['Para-std'] = {}
    for line in open(filename):
        json_obj = json.loads(line.strip())
        for para in json_obj:
            for key in json_obj[para]:
                if para not in parameter_log[key]:
                    parameter_log[key][para] = []
                parameter_log[key][para].append(json_obj[para][key])
    for key in parameter_log:
        sort_info = {}

        if key.endswith('-max') or key.endswith('-std') or key.endswith('-mean'):
            for para in parameter_log[key]:
                sort_info[para] = max(parameter_log[key][para])
            print (key+':')
            for item in sorted(sort_info.items(), key = lambda d:d[1], reverse = True)[:5]:
                print (item[0], parameter_log[key][item[0]][0], parameter_log[key][item[0]][-1])

        if key.endswith('-min'):
            for para in parameter_log[key]:
                sort_info[para] = min(parameter_log[key][para])
            print (key+':')
            for item in sorted(sort_info.items(), key = lambda d:d[1])[:5]:
                print (item[0], parameter_log[key][item[0]][0], parameter_log[key][item[0]][-1])

        print ('\n')

if __name__ == '__main__':
    #read_gradient_file('./outputs.webnlg/logs.plan.777/gradient.log', './outputs.webnlg/logs.plan.777/gradient')
    read_gradient_file('./outputs.webnlg/logs.base/gradient.log', './outputs.webnlg/logs.base/gradient')
