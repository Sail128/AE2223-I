from __future__ import print_function, unicode_literals
from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt, Separator
from PyInquirer import Validator, ValidationError
import regex

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class checkboxValidator(Validator):
    def validate(self, document):
        ok = len(document) != 0
        if not ok:
            raise ValidationError(
                message='Please select atleast one dataset')


class RangeValidator(Validator):
    def validate(self, document):
        ok = regex.match('^\\d+(-\\d+)?(?:,\\d+(?:-\\d+)?)*+$', document.text)
        if not ok:
            raise ValidationError(
                message='Please enter a valid range',
                cursor_position=len(document.text))  # Move cursor to end


questions = [
    {
        'type': 'checkbox',
        'qmark': '😃',
        'message': 'Select datasets to plot',
        'name': 'datasets',
        'choices': [],  # populate in main method
        'validate':lambda answer: False if len(answer) == 0 else True
    },
    {
        'type': 'list',
        'name': 'x-axis',
        'message': 'on the x-axis: ',
        'choices': [
            'N',
            'K'
        ]
    },
    {
        'type': 'list',
        'name': 'y-axis',
        'message': 'on the y-axis: ',
        'choices': []
    },
    {
        'type': 'input',
        'name': 'series',
        'message': 'What N to plot (p,n-m): ',
        'default': '2-5',
        'validate': RangeValidator,
        'when': lambda answers: answers['x-axis'] == 'K'
    },
    {
        'type': 'input',
        'name': 'series',
        'message': 'What K to plot (p,n-m): ',
        'default': '1-5',
        'validate': RangeValidator,
        'when': lambda answers: answers['x-axis'] == 'N'
    },
    {
        'type': 'input',
        'name': 'range',
        'message': 'What K range to plot (n-m): ',
        'default': '1-20',
        'validate': RangeValidator,
        'when': lambda answers: answers['x-axis'] == 'K'
    },
    {
        'type': 'input',
        'name': 'range',
        'message': 'What N range to plot (n-m): ',
        'default': '2-20',
        'validate': RangeValidator,
        'when': lambda answers: answers['x-axis'] == 'N'
    },
    {
        'type': 'confirm',
        'message': 'use-latex?',
        'name': 'use-tex',
        'default': True,
    },
    {
        'type': 'input',
        'name': 'y-label',
        'message': 'y-label: ',
        'default': '$L_{2}-error$'
    },
]

markers = [".", "o", "v", "^", "1", "2",
           "s", "p", "P", "h", "+", "x", "d", "D"]


def genTitle(colname):
    # ,K,N,l2phi,relL2phi,l1phi,relL1phi,l2u,relL2u,l1u,relL1u
    if colname == "K":
        return "K"
    elif colname == "N":
        return "N"
    elif colname == "l2phi":
        return "$L_{2}-error$ for $\\phi$"
    elif colname == "relL2phi":
        return "relative $L_{2}-error$ for $\\phi$"
    elif colname == "l1phi":
        return "$L_{1}-error$ for $\\phi$"
    elif colname == "relL1phi":
        return "relative $L_{1}-error$ for $\\phi$"
    elif colname == "l2u":
        return "$L_{2}-error$ for $u$"
    elif colname == "relL2u":
        return "relative $L_{2}-error$ for $u$"
    elif colname == "l1u":
        return "$L_{1}-error$ for $u$"
    elif colname == "relL1u":
        return "relative $L_{1}-error$ for $u$"
    else:
        return "Lol this shouldn't happen"


def plot(data, selection):
    """[summary]

    Arguments:
        data {pd.Panel} -- the data Panel from which to plot
        selection {dict} --  {'datasets': ['primal_dual_errors_c0', 'primal_dual_errors_c0_3'],
                                'series': '2-20',
                                'x-axis': 'K',
                                'y-axis': 'relL2phi'}
    """
    lines = []
    series = 'K' if selection['x-axis'] == 'N' else 'N'
    for a in selection['series'].split(","):
        if "-" in a:
            n, m = a.split("-")
            for i in range(int(n), int(m)+1):
                lines.append(i)
        else:
            lines.append(int(a))
    print(lines)
    # plot layout and axis setup
    fig, ax = plt.subplots(figsize=(10, 7))
    for item in selection['datasets']:
        df = data[item]
        dfName = " ".join(item.split("_")[0:2])
        c = ".".join(item.split("_")[3:]).replace("c", "").replace("C","")
        for line in lines:
            df[(df[series] == line)].plot(
                x=selection["x-axis"],
                y=selection["y-axis"],
                ax=ax,
                label="{} c={} ({}={})".format(dfName, c, series, line),
                marker=np.random.choice(markers)
            )
    x_range = selection['range'].split("-")
    plt.xlim(int(x_range[0]), int(x_range[1]))
    plt.rc('text', usetex=selection['use-tex'])
    plt.ylabel(selection['y-label'])
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.title(genTitle(selection["y-axis"]))
    fig.tight_layout()
    plt.show()


def main():
    parent_dir = "errors"
    infiles = []
    datapanel = {}
    for file in os.listdir(parent_dir):
        if file.endswith(".dat"):
            infiles.append(file)
            datapanel[file.split(".")[0]] = pd.read_csv(
                "{}/{}".format(parent_dir, file))

    print(infiles)
    data = pd.Panel(data=datapanel)

    print(data.shape)

    for set in list(data.items):
        questions[0]["choices"].append({'name': set})
    questions[0]["choices"][0]["checked"] = True
    questions[2]["choices"] = data[list(data.items)[0]].columns

    running = True
    while(running):
        answers = prompt(questions)  # , style=custom_style_2)
        if len(answers['datasets']) == 0:
            pprint("Please select atleast one dataset")
            continue
        pprint(answers)

        plot(data, answers)

        qexit = prompt([{
            'type': 'confirm',
            'message': 'Do you want to continue?',
            'name': 'continue',
            'default': True,
        }, ])
        if not qexit['continue']:
            running = False


if __name__ == "__main__":
    main()
