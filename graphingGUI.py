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
        ok = len(document)!=0
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
        'default':'2-20',
        'validate': RangeValidator,
        'when': lambda answers: answers['x-axis'] == 'K'
    },
    {
        'type': 'input',
        'name': 'series',
        'message': 'What K to plot (p,n-m): ',
        'default':'1-20',
        'validate': RangeValidator,
        'when': lambda answers: answers['x-axis'] == 'N'
    },
]

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
    series = 'K' if selection['x-axis'] =='N' else 'N'
    for a in selection['series'].split(","):
        if "-" in a:
            n,m = a.split("-")
            for i in range(int(n),int(m)+1):
                lines.append(i)
        else: lines.append(int(a))
    print(lines)
    # plot layout and axis setup
    fig, ax = plt.subplots(figsize=(10,7))
    fig.tight_layout()
    for item in selection['datasets']:
        df = data[item]
        for line in lines:
            df[(df[series]==line)].plot(x=selection["x-axis"],y=selection["y-axis"],ax=ax,label="{}={}".format(series,line))
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
    questions[0]["choices"][0]["checked"] =True
    questions[2]["choices"] = data[list(data.items)[0]].columns

    running = True
    while(running):
        answers = prompt(questions)  # , style=custom_style_2)
        if len(answers['datasets'])==0:
            pprint("Please select atleast one dataset")
            continue
        pprint(answers)

        plot(data,answers)

        qexit = prompt([    {
        'type': 'confirm',
        'message': 'Do you want to continue?',
        'name': 'continue',
        'default': True,
        },])
        if not qexit['continue']:
            running=False
    # print(data["primal_primal_errors_c0"][(data["primal_primal_errors_c0"]["K"]==1)])
    # data["primal_primal_errors_c0"][(data["primal_primal_errors_c0"]["K"]==1)].plot(x="N",y="relL2phi")
    # # plt.plot(data["primal_primal_errors_c0"][("K"==1)]["N","l2phi"])
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
