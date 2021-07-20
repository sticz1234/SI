import numpy as np

def productN(args, op):    #algebraiczna
    return np.product(args, axis=0)

def zadeh_t(args,op):
    return np.minimum(args[0],args[1])

def lukasiewicz(args, op):
    zeros_array = np.zeros((np.size(args[0], axis = 0), np.size(args[0], axis = 1)))
    ones_array = np.ones((np.size(args[0], axis = 0), np.size(args[0], axis = 1)))
    return np.maximum(zeros_array, np.subtract(np.add(args[0], args[1]), ones_array))

def fodor(args, op):
    zeros_array = np.zeros((np.size(args[0], axis = 0), np.size(args[0], axis = 1)))
    ones_array = np.ones((np.size(args[0], axis = 0), np.size(args[0], axis = 1)))
    mask = np.add(args[0], args[1]) > ones_array
    return mask * np.minimum(args[0], args[1])

def drastic(args, op):
    zeros_array = np.zeros((np.size(args[0], axis = 0), np.size(args[0], axis = 1)))
    ones_array = np.ones((np.size(args[0], axis = 0), np.size(args[0], axis = 1)))
    mask_1 = np.equal(args[0], ones_array)
    mask_2 = np.equal(args[1], ones_array)
    a = mask_1 == 0
    a = a.astype(int)
    a_2 = a*mask_2
    b = mask_2 ==1
    b = b.astype(int)
    b_2 = b*mask_1
    wynik = b_2*args[0] + mask_1*args[1]
    return wynik


def einstein(args, op):
    twos_array = np.full((np.size(args[0], axis = 0), np.size(args[0], axis = 1)), 2)
    return np.divide(np.product(args, axis=0), np.subtract(twos_array, np.subtract(np.add(args[0], args[1]), np.product(args, axis=0))))

def t_norm_param(args, op):
    ones_array = np.ones(len(op))
    min_val = np.minimum(args[0], args[1])
    max_val = np.maximum(args[0], args[1])
    return np.multiply(ones_array, min_val) + np.multiply((ones_array - op), max_val)

