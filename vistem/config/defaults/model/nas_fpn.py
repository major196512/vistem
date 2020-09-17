from yacs.config import CfgNode as CN

_NAS_FPN = CN()
_NAS_FPN.CELL_INPUTS = [['p6', 'p4'],['rcb1', 'p4'],['rcb2', 'p3'],['rcb3', 'rcb2'],['rcb3', 'rcb4', 'p5'],['rcb2', 'rcb5', 'p7'],['rcb5', 'rcb6']]
_NAS_FPN.CELL_OUTPUTS = ['p4', 'p4', 'p3', 'p4', 'p5', 'p7', 'p6']
_NAS_FPN.CELL_OPS = ['GP', 'SUM', 'SUM', 'SUM', 'GP_SUM', 'GP_SUM', 'GP']
_NAS_FPN.NAS_OUTPUTS = ['rcb3','rcb4','rcb5','rcb7','rcb6']