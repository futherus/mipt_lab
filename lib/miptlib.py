import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sp
import scipy.optimize as op
from scipy.interpolate import UnivariateSpline
import copy
import itertools

def errcalc(func, at, delta):
    '''
    Calculate error using boundary method.

    :param func: function of n-arguments
    :param at: point to calculate error
    :param delta: errors of arguments

    Example:
        
        def func(x):
            return x[1]**2 + x[0]

        err, grad = errcalc(func, [1, 1], [0.1, 0.1])
    '''
    __at = np.column_stack(at)
    __delta = np.column_stack(delta)
    

    assert __delta.shape[1] == __at.shape[1], 'Errors and point must have the same length'
    __dim = __delta.shape[1]
    
    __delta = np.eye(__dim) * __delta[:, np.newaxis, :]
    __at = np.ones((__dim, __dim)) * __at[:, np.newaxis, :]
    
    __gradient = (np.apply_along_axis(func, 2, __at + __delta) - \
                  np.apply_along_axis(func, 2, __at - __delta)) / 2

    return np.apply_along_axis(np.linalg.norm, 1, __gradient), __gradient

def excel_cols2nums(cols):
    '''
    Converts list of excel columns to list of corresponding indices.
    Skips column if is already a number.

    :param cols: columns names as strings
    '''
    indices = []

    for col in cols:
        num = 0
        if type(col) == str:
            for c in col:
                num = num * 26 + (ord(c.upper()) - ord('A')) + 1
            num -= 1
        elif type(col) == int:
            num = col
        else:
            print("Wrong type of column. Should be str or int")
            num = col

        indices.append(num)

    return indices

def read_excel(filename, usecols, header = None, nrows = None, sheet_name = 0):
    '''
    Creates pandas.DataFrame from excel file with pandas.MultiIndex.

    :param filename:   name of excel file
    :param usecols:    int list with indices of columns to read
    :param header:     int list with indices of header rows
    :param sheet_name: index or name of sheet to read
    '''
    usecols = excel_cols2nums(list(usecols))

    if header == None or header == 0:
        return pd.read_excel(filename, usecols = usecols, nrows = nrows, sheet_name = sheet_name)

    # Read header rows
    __header_df = pd.read_excel(filename, nrows = 0, header = header, sheet_name = sheet_name)
    __header_mi = pd.MultiIndex.from_tuples(__header_df.columns.to_list())

    # Read data and set multiindex
    __data = pd.read_excel(filename, skiprows = header, header = None, usecols = usecols, nrows = nrows, sheet_name = sheet_name)
    __data.columns = __header_mi[usecols]

    return __data

def map_excel(data, fmt):
    '''
    Renames columns in pandas.DataFrame.

    :param __data: pandas.DataFrame to rename columns in
    :param __fmt:  map old_names->new_names

    Example:
        fmt = {
            'B, Wb' : 'B',
            'Webermeter' : 'Wm',
            'L, m' : 'L'
        }

        data = map_excel(data, fmt)
    '''

    return data.rename(columns = fmt)

def interp_linear(x, y):
    coeffs = np.polyfit(x, y, 1)
    return lambda x: coeffs[0] * x + coeffs[1]

PLOT_MARKER = itertools.cycle(['.', 'v', '^', '<', '>', '*', 'o', '+', '1', '2', '3', '4'])
def plot(x, y, label = None, color = None, xerr = 0, yerr = 0,
         begin = 0, end = None, exclude = [],
         x_min = None, x_max = None, marker_size = 6,
         linestyle = 'solid', linewidth = None, func = interp_linear, unique_marker='.'):
    '''
    Creates plot with approximating line.

    :param x:                   x coordinates of points
    :param y:                   y coordinates of points
    :param label:               label of plot in legend
    :param color:               color of plot
    :param xerr:                x errors of points
    :param yerr:                y errors of points
    :param begin:               index of first point used for approximating line
    :param end:                 index of last point used for approximating line
    :param exclude:             indices of 'error' points
    :param x_min:               left end of approximating line
    :param x_max:               right end of approximating line
    :param func:                function for approximating, None -> no approximating line
    :param marker_size:         points size
    :param unique_marker:       True -> use internal unique marker, otherwise use unique_marker as marker itself

    :return x_clean, y_clean: pd.Series of values used for approximating line
    '''
    
    assert len(x) == len(y), "x and y must have same length"
    
    end = (len(x) - 1) if (end == None) else end
    
    x_clean = []
    y_clean = []
    for i in range(begin, end + 1):
        if i in exclude:
            continue
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
        x_clean.append(x[i])
        y_clean.append(y[i])
    
    x_min = min(x_clean) if (x_min == None) else x_min
    x_max = max(x_clean) if (x_max == None) else x_max

    x_space = np.linspace(x_min, x_max, 100)
    
    if unique_marker == True:
        unique_marker = next(PLOT_MARKER)
    
    equ = None
    p = None
    # At least two points and function for approximating.
    if (func != None and end - begin + 1 >= 2):
        equ = func(x_clean, y_clean)
        p = plt.plot(x_space, equ(x_space), label = label, c = color, linestyle = linestyle, linewidth = linewidth)
    else:
        p = plt.plot([], [], label = label, c = color, linestyle = linestyle, linewidth = linewidth)

    plt.errorbar(x, y, xerr = xerr, yerr = yerr,
                 ms = marker_size, fmt = unique_marker,
                 c = p[-1].get_color())

    for i in exclude:
        plt.scatter(x[i], y[i], s = 60, marker = 'x', c = 'red')

    return pd.Series(x_clean), pd.Series(y_clean), equ

def rename(columns, fmt, inplace = False):
    __init_columns = columns
    __columns = __init_columns 
    
    for key in fmt.keys():
        __renamed = []
        
        for i in range(len(__init_columns)):
            col = __columns[i]

            if type(key) is tuple:
                if type(col) is tuple and __init_columns[i][:len(key)] == key:
                    col = __columns[i][:len(key) - 1] + (fmt[key],) + __columns[i][len(key):]
            elif type(key) is str:
                if type(col) is str and col == key:
                    col = fmt[key]
                elif type(col) is tuple and col[0] == key:
                    col = (fmt[key],) + __columns[i][1:]
            else:
                assert 0, 'Format index must be str or tuple'
                
            __renamed.append(col) 
        
        __columns = __renamed
                    
    return __columns #pd.Index(__columns, tupleize_cols = True)

class table:
    data = pd.DataFrame()
    latex_fmt = {
        # key    name      precision
        # 'I' : ['$I$, A', '{:.1f}'],
        # ('Group', 'I') : ['$I$, A', '{:.1f}'],
    }

    DEFAULT_PRECISION = '{:.1f}'
    DEFAULT_NAME = '!unnamed!'

    def __init__(self, data, fmt = {}):
        self.data = data
        self.latex_fmt = fmt
        
    def get_names(self):
        __fmt = self.latex_fmt
        return {__i:__fmt[__i][0] for __i in __fmt.keys() if len(__fmt[__i]) > 0}

    def get_precisions(self):
        __fmt = self.latex_fmt
        return {__i:__fmt[__i][1] for __i in __fmt.keys() if len(__fmt[__i]) > 1}

    def get_scales(self):
        __fmt = self.latex_fmt
        return {__i:__fmt[__i][2] for __i in __fmt.keys() if len(__fmt[__i]) > 2}

    def get_fmt(self):
        return self.latex_fmt
    
    def get_data(self):
        return self.data
    
    def set_name(self, key, name):
        if key in self.latex_fmt:
            __tmp = self.latex_fmt[key]
            __tmp[0] = name
        else:
            __tmp = [name, DEFAULT_PRECISION]

        self.latex_fmt[key] = __tmp

    def set_precision(self, key, precision):
        if key in self.latex_fmt:
            __tmp = self.latex_fmt[key]
            __tmp[1] = precision
        else:
            __tmp = [DEFAULT_NAME, precision]
        
        self.latex_fmt[key] = __tmp

    def set_fmt(self, fmt):
        self.latex_fmt = fmt
    
    def scale(self):
        __tab = table(self.data.copy(), self.latex_fmt.copy())
        
        __scales = self.get_scales()
        for key in list(__scales.keys()):
            if key in list(__tab.data.columns.values):
                __tab.data[key] *= 10**(__scales[key])

        return __tab
    
    def rename(self):
        __tab = table(self.data.copy(), self.latex_fmt.copy())
        
        __tab.data.columns = pd.Index(rename(__tab.data.columns.values, __tab.get_names()), tupleize_cols = True)
        __tab.latex_fmt = dict(zip(rename(list(__tab.latex_fmt.keys()), __tab.get_names()), list(__tab.latex_fmt.values())))
            
        return __tab  
        
    def insert(self, loc, column, value, fmt = []):
        assert not (column in self.data), "Column " + str(column) + " already exists"
        self.data.insert(loc = loc, column = column, value = value, allow_duplicates = False)
        
        if fmt:
            self.latex_fmt[column] = fmt
        
    def delete(self, column):
        assert (column in self.data), "Column " + str(column) + " does not exist"
        del self.data[column]
            
    def to_latex(self, file = None):
        '''
        Applies LaTeX formatting and saves table to file.
        
        :param filename: output file
        
        Example:
        
        fmt = {
            'x': ['$x$, дел.',   '{:.1f}'],
            'x0': ['$x_0$, дел.', '{:.3f}'],
            
            # It supports multi index renaming
            ('RC', 'R'): ['$R$, дел.',   '{:.2f}'], 
        }
        
        my_table.to_latex('my_table.tex')
        '''
        
        # Rename also changes latex_fmt. 
        # That is why we need to assign rename() to local variable
        __tab = self.rename() 
        return __tab.scale().data.style.format(__tab.get_precisions()) \
                    .hide(level=0, axis=0)                                 \
                    .to_latex( 
                        buf = file,
                        column_format="c" * len(__tab.data.columns.values),
                        hrules=True,
                        multicol_align = 'c',
                        environment = ''
                    )

def mnk(x, y, fmt = None, file = None, precision = 2):
    if fmt == None:
        fmt = {
                '<x>':    ['$\overline{x}$', '{:.' + str(precision) + 'e}'],
                'sx':     ['$\sigma_x^2$',   '{:.' + str(precision) + 'e}'],
                '<y>':    ['$\overline{y}$', '{:.' + str(precision) + 'e}'],
                'sy':     ['$\sigma_y^2$',   '{:.' + str(precision) + 'e}'],
                'rxy':    ['$r_{xy}$',       '{:.' + str(precision) + 'e}'],
                'a':      ['$a$',            '{:.' + str(precision) + 'f}'],
                'da':     ['$\Delta a$',     '{:.' + str(precision) + 'f}'],
                'b':      ['$b$',            '{:.' + str(precision) + 'f}'],
                'db':     ['$\Delta b$',     '{:.' + str(precision) + 'f}'],
        }
    
    __sx = (x**2).mean() - (x.mean())**2
    __sy = (y**2).mean() - (y.mean())**2
    
    __rxy = (y*x).mean() - (y.mean() * x.mean())
    
    __a  = __rxy / __sx
    __da = (1/(len(x) - 2) * (__sy/__sx - __a**2))**(0.5)
    
    __b  = y.mean() - __a * x.mean()
    __db = __da*(__sx + (x.mean())**2)**(1/2)

    __data = pd.DataFrame({
        '<x>':   [x.mean()],
        'sx':    [__sx],
        '<y>':   [y.mean()],
        'sy':    [__sy],
        'rxy':   [__rxy],
        'a':     [__a],
        'da':    [__da],
        'b':     [__b],
        'db':    [__db],
    })
    
    __tab = table(__data, fmt)
    if file:
        __tab.to_latex(file)
    
    return __tab
