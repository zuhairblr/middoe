import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc
from scipy.optimize import root_scalar

#------------------------------------------ Model 05-----------------------------------------

def thetadic05():
    theta05 = [1.29e-10, 26927]
    theta05min = [1.00e-18, 10000]
    theta05max = [1.00e-2, 32000]
    theta05maxs = [max_val / theta for max_val, theta in zip(theta05max, theta05)]
    theta05mins = [min_val / theta for min_val, theta in zip(theta05min, theta05)]

    return theta05, theta05maxs, theta05mins


def f05(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    aps = phi['aps']
    k, Ea = theta[0], theta[1]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
    phia = np.log(k) - (Ea / (R * 333.15))
    phib = np.log(Ea / R)
    ki = np.exp(phia + np.exp(phib) * ((1 / 333.15) - (1 / T)))
    ka = (0.088 * cm * ki) / (rho * aps)
    dy1dt = -2*ka*(np.clip(y1, 0.001, 0.999))**(-1/2)
    output = [dy1dt]
    return output

#------------------------------------------ Model 06-----------------------------------------

def thetadic06():
    theta06 = [1.29e-5, 26927]
    theta06min = [1.00e-8, 10000]
    theta06max = [1.00e-2, 32000]
    theta06maxs = [max_val / theta for max_val, theta in zip(theta06max, theta06)]
    theta06mins = [min_val / theta for min_val, theta in zip(theta06min, theta06)]

    return theta06, theta06maxs, theta06mins


def f06(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    aps = phi['aps']
    k, Ea = theta[0], theta[1]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
    phia = np.log(k) - (Ea / (R * 333.15))
    phib = np.log(Ea / R)
    ki = np.exp(phia + np.exp(phib) * ((1 / 333.15) - (1 / T)))
    ka = (0.088 * cm * ki) / (rho * aps)
    dy1dt = 3*ka*(1-np.clip(y1, 0.001, 0.999))**(2/3)
    output = [dy1dt]
    return output

#------------------------------------------ Model 07-----------------------------------------

def thetadic07():
    theta07 = [1.29e-5, 26927]
    theta07min = [1.00e-8, 18000]
    theta07max = [1.00e-2, 32000]
    theta07maxs = [max_val / theta for max_val, theta in zip(theta07max, theta07)]
    theta07mins = [min_val / theta for min_val, theta in zip(theta07min, theta07)]

    return theta07, theta07maxs, theta07mins


def f07(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    aps = phi['aps']
    k, Ea = theta[0], theta[1]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
    phia = np.log(k) - (Ea / (R * 333.15))
    phib = np.log(Ea / R)
    ki = np.exp(phia + np.exp(phib) * ((1 / 333.15) - (1 / T)))
    ka = (0.088 * cm * ki) / (rho * (aps ** 2))
    dy1dt = ka
    output = [dy1dt]
    return output

# #our paper
#
# #my data-orig models
# #------------------------------------------ Model 08-----------------------------------------
#
# def thetadic08():
#     theta08 = [6.048450283701883e-07, 20191.94152486254]
#     theta08min = [1.00e-9, 18000]
#     theta08max = [1.00e-5, 28000]
#     theta08maxs = [max_val / theta for max_val, theta in zip(theta08max, theta08)]
#     theta08mins = [min_val / theta for min_val, theta in zip(theta08min, theta08)]
#
#     return theta08, theta08maxs, theta08mins
#
#
# def f08(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     aps = phi['aps']
#     k, Ea = theta[0], theta[1]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#
#     ka = (0.704 * cm * ki) / (rho * (aps ** 2))
#     dy1dt = -ka / (np.log(1 - np.clip(y1, 0.001, 0.999)))
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 09-----------------------------------------
#
# def thetadic09():
#     theta09 = [3.056226327814133e-07, 20197.265107932555]
#     theta09min = [1.00e-9, 18000]
#     theta09max = [1.00e-5, 28000]
#     theta09maxs = [max_val / theta for max_val, theta in zip(theta09max, theta09)]
#     theta09mins = [min_val / theta for min_val, theta in zip(theta09min, theta09)]
#
#     return theta09, theta09maxs, theta09mins
#
# def f09(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     aps = phi['aps']
#     k, Ea = theta[0], theta[1]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (1.056 * cm * ki) / (rho * (aps ** 2))
#     dy1dt = ka / (-2 + 2 * (1 - np.clip(y1, 0.001, 0.999)) ** (-1 / 3))
#     output = [dy1dt]
#     return output
#
#
# #------------------------------------------ Model 11-----------------------------------------
#
def thetadic11():
    # theta11 = [2.707812554014968e-08, 24144.427994246784, 1.931273198572943]
    theta11 = [3.1650075431484574e-07, 20239.398294836665, 1.2133668813455163]
    theta11min = [1.00e-9, 18000, 0.1]
    theta11max = [1.00e-5, 30000, 10]
    theta11maxs = [max_val / theta for max_val, theta in zip(theta11max, theta11)]
    theta11mins = [min_val / theta for min_val, theta in zip(theta11min, theta11)]

    return theta11, theta11maxs, theta11mins

def f11(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    cac = phi['cac']
    aps = phi['aps']
    k, Ea, AK = theta[0], theta[1], theta[2]
    R = 8.314  # Gas constant in J/(mol*K)
    am = 0.056
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))  # [mol.m-3]

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
    dy1dt = ka * np.exp(-AK * ka * t) / (2 * np.clip(y1, 0.001, 0.999))
    output = [dy1dt]
    return output
#
# #------------------------------------------ Model 12-----------------------------------------
#
# def thetadic12():
#     theta12 = [8.657783497064711e-08, 20190.20009631406, 2.194284154090643]
#     theta12min = [1.00e-10, 18000, 0.1]
#     theta12max=[1.00e-6, 30000, 10]
#     theta12maxs = [max_val / theta for max_val, theta in zip(theta12max, theta12)]
#     theta12mins = [min_val / theta for min_val, theta in zip(theta12min, theta12)]
#
#     return theta12, theta12maxs, theta12mins
#
# def f12(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     cac = phi['cac']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     am = 0.056
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
#     dy1dt = ka * 2 * np.exp(-AK * ka * t) / -(np.log(1 - np.clip(y1, 0.001, 0.999)))
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 13-----------------------------------------
#
# def thetadic13():
#     theta13 = [3.74989949780187e-08, 20213.603669155094, 6.949990334765804]
#     theta13min = [1.00e-10, 18000, 0.1]
#     theta13max=[1.00e-6, 30000, 10]
#     theta13maxs = [max_val / theta for max_val, theta in zip(theta13max, theta13)]
#     theta13mins = [min_val / theta for min_val, theta in zip(theta13min, theta13)]
#
#     return theta13, theta13maxs, theta13mins
#
#
# def f13(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     cac = phi['cac']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     am = 0.056
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
#     dy1dt = (3 / 2) * ka * np.exp(-AK * ka * t) / -((1 - np.clip(y1, 0.001, 0.999)) ** (1 / 3) - 1)
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 14-----------------------------------------
#
# def thetadic14():
#     theta14 = [3.822470068711253e-07, 20198.045600335983, 0.7671885604344969]
#     theta14min = [1.00e-9, 18000, 0.1]
#     theta14max=[1.00e-5, 30000, 10]
#     theta14maxs = [max_val / theta for max_val, theta in zip(theta14max, theta14)]
#     theta14mins = [min_val / theta for min_val, theta in zip(theta14min, theta14)]
#
#     return theta14, theta14maxs, theta14mins
#
# def f14(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     mld = phi['mld']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (8 * cm * ki) / (mld * (aps ** 2))
#     dy1dt = (ka * np.exp(-AK * ka * t)) / ((1 - np.clip(y1, 0.001, 0.999)) ** (-0.5) - 1)
#     output = [dy1dt]
#     return output
#
#
# def thetadic14p():
#     theta14p = [1.3341960133302085e-06, 20239.41387867327, 2.426734975569295]
#     theta14pmin = [1.00e-8, 18000, 0.1]
#     theta14pmax=[1.00e-4, 32000, 10]
#     theta14pmaxs = [max_val / theta for max_val, theta in zip(theta14pmax, theta14p)]
#     theta14pmins = [min_val / theta for min_val, theta in zip(theta14pmin, theta14p)]
#
#     return theta14p, theta14pmaxs, theta14pmins
#
# def f14p(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     mld = phi['mld']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (4 * cm * ki) / (mld * (aps ** 2))
#     dy1dt = (ka * np.exp(-AK * ka * t)) / np.clip(y1, 0.001, 0.999)
#     output = [dy1dt]
#     return output
#
# def thetadic14s():
#     theta14s = [1.8333977056274745e-07, 20170.507458025906, 2.5666991259436363e-07]
#     theta14smin = [1.00e-9, 18000, 1e-8]
#     theta14smax=[1.00e-5, 30000, 1e-6]
#     theta14smaxs = [max_val / theta for max_val, theta in zip(theta14smax, theta14s)]
#     theta14smins = [min_val / theta for min_val, theta in zip(theta14smin, theta14s)]
#
#     return theta14s, theta14smaxs, theta14smins
#
# def f14s(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     mld = phi['mld']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (12*cm * ki) / (mld * (aps ** 2))
#     dy1dt = (ka * np.exp(-AK * ka * t)) / ((1-(1-np.clip(y1, 0.001, 0.999))**(1/3))*((1-np.clip(y1, 0.001, 0.999))**(-2/3)))
#     output = [dy1dt]
#     return output













# #our paper
#
# #my data-reduced models
# #------------------------------------------ Model 08-----------------------------------------
#
# def thetadic08():
#     theta08 = [ 20191.94152486254]
#     theta08min = [ 18000]
#     theta08max = [ 28000]
#     theta08maxs = [max_val / theta for max_val, theta in zip(theta08max, theta08)]
#     theta08mins = [min_val / theta for min_val, theta in zip(theta08min, theta08)]
#
#     return theta08, theta08maxs, theta08mins
#
#
# def f08(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     aps = phi['aps']
#     k, Ea = 6.093787157470513e-07, theta[0]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#
#     ka = (0.704 * cm * ki) / (rho * (aps ** 2))
#     dy1dt = -ka / (np.log(1 - np.clip(y1, 0.001, 0.999)))
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 09-----------------------------------------
#
# def thetadic09():
#     theta09 = [20197.265107932555]
#     theta09min = [18000]
#     theta09max = [28000]
#     theta09maxs = [max_val / theta for max_val, theta in zip(theta09max, theta09)]
#     theta09mins = [min_val / theta for min_val, theta in zip(theta09min, theta09)]
#
#     return theta09, theta09maxs, theta09mins
#
# def f09(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     aps = phi['aps']
#     k, Ea = 3.0709296079313854e-07, theta[0]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (1.056 * cm * ki) / (rho * (aps ** 2))
#     dy1dt = ka / (-2 + 2 * (1 - np.clip(y1, 0.001, 0.999)) ** (-1 / 3))
#     output = [dy1dt]
#     return output
#
#
# #------------------------------------------ Model 11-----------------------------------------
#
# def thetadic11():
#     # theta11 = [2.707812554014968e-08, 24144.427994246784, 1.931273198572943]
#     theta11 = [20239.398294836665]
#     theta11min = [18000]
#     theta11max = [30000]
#     theta11maxs = [max_val / theta for max_val, theta in zip(theta11max, theta11)]
#     theta11mins = [min_val / theta for min_val, theta in zip(theta11min, theta11)]
#
#     return theta11, theta11maxs, theta11mins
#
# def f11(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     cac = phi['cac']
#     aps = phi['aps']
#     k, Ea, AK = 3.1836716396085585e-07, theta[0], 1.2140496411789563
#     R = 8.314  # Gas constant in J/(mol*K)
#     am = 0.056
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))  # [mol.m-3]
#
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
#     dy1dt = ka * np.exp(-AK * ka * t) / (2 * np.clip(y1, 0.001, 0.999))
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 12-----------------------------------------
#
# def thetadic12():
#     theta12 = [20190.20009631406]
#     theta12min = [ 18000]
#     theta12max=[30000]
#     theta12maxs = [max_val / theta for max_val, theta in zip(theta12max, theta12)]
#     theta12mins = [min_val / theta for min_val, theta in zip(theta12min, theta12)]
#
#     return theta12, theta12maxs, theta12mins
#
# def f12(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     cac = phi['cac']
#     aps = phi['aps']
#     k, Ea, AK = 8.754161236965817e-08, theta[0], 2.199508882068956
#     R = 8.314  # Gas constant in J/(mol*K)
#     am = 0.056
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
#     dy1dt = ka * 2 * np.exp(-AK * ka * t) / -(np.log(1 - np.clip(y1, 0.001, 0.999)))
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 13-----------------------------------------
#
# def thetadic13():
#     theta13 = [20213.603669155094]
#     theta13min = [ 18000]
#     theta13max=[ 30000]
#     theta13maxs = [max_val / theta for max_val, theta in zip(theta13max, theta13)]
#     theta13mins = [min_val / theta for min_val, theta in zip(theta13min, theta13)]
#
#     return theta13, theta13maxs, theta13mins
#
#
# def f13(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     cac = phi['cac']
#     aps = phi['aps']
#     k, Ea, AK = 3.763304658276041e-08, theta[0], 6.951277345053975
#     R = 8.314  # Gas constant in J/(mol*K)
#     am = 0.056
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
#     dy1dt = (3 / 2) * ka * np.exp(-AK * ka * t) / -((1 - np.clip(y1, 0.001, 0.999)) ** (1 / 3) - 1)
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 14-----------------------------------------
#
# def thetadic14():
#     theta14 = [20198.045600335983]
#     theta14min = [ 18000]
#     theta14max=[30000]
#     theta14maxs = [max_val / theta for max_val, theta in zip(theta14max, theta14)]
#     theta14mins = [min_val / theta for min_val, theta in zip(theta14min, theta14)]
#
#     return theta14, theta14maxs, theta14mins
#
# def f14(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     mld = phi['mld']
#     aps = phi['aps']
#     k, Ea, AK = 3.853056974365147e-07, theta[0], 0.8469922192821213
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (8 * cm * ki) / (mld * (aps ** 2))
#     dy1dt = (ka * np.exp(-AK * ka * t)) / ((1 - np.clip(y1, 0.001, 0.999)) ** (-0.5) - 1)
#     output = [dy1dt]
#     return output
#
#
# def thetadic14p():
#     theta14p = [20239.41387867327, 2.426734975569295]
#     theta14pmin = [ 18000, 0.1]
#     theta14pmax=[ 32000, 10]
#     theta14pmaxs = [max_val / theta for max_val, theta in zip(theta14pmax, theta14p)]
#     theta14pmins = [min_val / theta for min_val, theta in zip(theta14pmin, theta14p)]
#
#     return theta14p, theta14pmaxs, theta14pmins
#
# def f14p(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     mld = phi['mld']
#     aps = phi['aps']
#     k, Ea, AK = 1.341394433057882e-06, theta[0], theta[1]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (4 * cm * ki) / (mld * (aps ** 2))
#     dy1dt = (ka * np.exp(-AK * ka * t)) / np.clip(y1, 0.001, 0.999)
#     output = [dy1dt]
#     return output
#
# def thetadic14s():
#     theta14s = [ 20170.507458025906]
#     theta14smin = [ 18000]
#     theta14smax=[ 30000]
#     theta14smaxs = [max_val / theta for max_val, theta in zip(theta14smax, theta14s)]
#     theta14smins = [min_val / theta for min_val, theta in zip(theta14smin, theta14s)]
#
#     return theta14s, theta14smaxs, theta14smins
#
# def f14s(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     mld = phi['mld']
#     aps = phi['aps']
#     k, Ea, AK = 1.845517392105026e-07, theta[0], 1.862677128831102e-07
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # # 1. base
#     # ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # 4. modified 3
#     ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (12*cm * ki) / (mld * (aps ** 2))
#     dy1dt = (ka * np.exp(-AK * ka * t)) / ((1-(1-np.clip(y1, 0.001, 0.999))**(1/3))*((1-np.clip(y1, 0.001, 0.999))**(-2/3)))
#     output = [dy1dt]
#     return output






#
#
# #our paper
#
# #my data-orig models
# #------------------------------------------ Model 08-----------------------------------------
#
# def thetadic08():
#     theta08 = [6.123634259442429e-09, 22327.84145134412]
#     theta08min = [1.00e-9, 18000]
#     theta08max = [1.00e-5, 28000]
#     theta08maxs = [max_val / theta for max_val, theta in zip(theta08max, theta08)]
#     theta08mins = [min_val / theta for min_val, theta in zip(theta08min, theta08)]
#
#     return theta08, theta08maxs, theta08mins
#
#
# def f08(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     aps = phi['aps']
#     k, Ea = theta[0], theta[1]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
#     # 1. base
#     ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # # 4. modified 3
#     # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#
#     ka = (0.704 * cm * ki) / (rho * (aps ** 2))
#     dy1dt = -ka / (np.log(1 - np.clip(y1, 0.001, 0.999)))
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 09-----------------------------------------
#
# def thetadic09():
#     theta09 = [3.4270989918108122e-09, 22593.290494972498]
#     theta09min = [1.00e-9, 18000]
#     theta09max = [1.00e-5, 28000]
#     theta09maxs = [max_val / theta for max_val, theta in zip(theta09max, theta09)]
#     theta09mins = [min_val / theta for min_val, theta in zip(theta09min, theta09)]
#
#     return theta09, theta09maxs, theta09mins
#
# def f09(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     aps = phi['aps']
#     k, Ea = theta[0], theta[1]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#     # 1. base
#     ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # # 4. modified 3
#     # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (1.056 * cm * ki) / (rho * (aps ** 2))
#     dy1dt = ka / (-2 + 2 * (1 - np.clip(y1, 0.001, 0.999)) ** (-1 / 3))
#     output = [dy1dt]
#     return output
#
#
# #------------------------------------------ Model 11-----------------------------------------
#
# def thetadic11():
#     # theta11 = [2.707812554014968e-08, 24144.427994246784, 1.931273198572943]
#     theta11 = [2.7085852160960226e-08, 24145.261403540826, 1.9312105955376377]
#     theta11min = [1.00e-9, 18000, 0.1]
#     theta11max = [1.00e-5, 30000, 10]
#     theta11maxs = [max_val / theta for max_val, theta in zip(theta11max, theta11)]
#     theta11mins = [min_val / theta for min_val, theta in zip(theta11min, theta11)]
#
#     return theta11, theta11maxs, theta11mins
#
# def f11(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     cac = phi['cac']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     am = 0.056
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))  # [mol.m-3]
#
#     # 1. base
#     ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # # 4. modified 3
#     # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
#     dy1dt = ka * np.exp(-AK * ka * t) / (2 * np.clip(y1, 0.001, 0.999))
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 12-----------------------------------------
#
# def thetadic12():
#     theta12 = [7.787480228557358e-09, 24175.499957199194, 5.256598817693932]
#     theta12min = [1.00e-10, 18000, 0.1]
#     theta12max=[1.00e-6, 30000, 10]
#     theta12maxs = [max_val / theta for max_val, theta in zip(theta12max, theta12)]
#     theta12mins = [min_val / theta for min_val, theta in zip(theta12min, theta12)]
#
#     return theta12, theta12maxs, theta12mins
#
# def f12(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     cac = phi['cac']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     am = 0.056
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
#     # 1. base
#     ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # # 4. modified 3
#     # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
#     dy1dt = ka * 2 * np.exp(-AK * ka * t) / -(np.log(1 - np.clip(y1, 0.001, 0.999)))
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 13-----------------------------------------
#
# def thetadic13():
#     theta13 = [3.3041934285078867e-09, 24169.134790464035, 13.62202981032264]
#     theta13min = [1.00e-10, 18000, 0.1]
#     theta13max=[1.00e-6, 30000, 30]
#     theta13maxs = [max_val / theta for max_val, theta in zip(theta13max, theta13)]
#     theta13mins = [min_val / theta for min_val, theta in zip(theta13min, theta13)]
#
#     return theta13, theta13maxs, theta13mins
#
#
# def f13(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     rho = phi['rho']
#     cac = phi['cac']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     am = 0.056
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # 1. base
#     ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # # 4. modified 3
#     # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
#     dy1dt = (3 / 2) * ka * np.exp(-AK * ka * t) / -((1 - np.clip(y1, 0.001, 0.999)) ** (1 / 3) - 1)
#     output = [dy1dt]
#     return output
#
# #------------------------------------------ Model 14-----------------------------------------
#
# def thetadic14():
#     theta14 = [1.4897971676833285e-08, 24193.33328469753, 4.120548809910462]
#     theta14min = [1.00e-9, 18000, 0.1]
#     theta14max=[1.00e-5, 30000, 10]
#     theta14maxs = [max_val / theta for max_val, theta in zip(theta14max, theta14)]
#     theta14mins = [min_val / theta for min_val, theta in zip(theta14min, theta14)]
#
#     return theta14, theta14maxs, theta14mins
#
# def f14(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     mld = phi['mld']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # 1. base
#     ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # # 4. modified 3
#     # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (8 * cm * ki) / (mld * (aps ** 2))
#     dy1dt = (ka * np.exp(-AK * ka * t)) / ((1 - np.clip(y1, 0.001, 0.999)) ** (-0.5) - 1)
#     output = [dy1dt]
#     return output
#
#
# def thetadic14p():
#     theta14p = [4.801721849949891e-08, 24145.079391593325, 3.8624629595306534]
#     theta14pmin = [1.00e-8, 18000, 0.1]
#     theta14pmax=[1.00e-4, 32000, 10]
#     theta14pmaxs = [max_val / theta for max_val, theta in zip(theta14pmax, theta14p)]
#     theta14pmins = [min_val / theta for min_val, theta in zip(theta14pmin, theta14p)]
#
#     return theta14p, theta14pmaxs, theta14pmins
#
# def f14p(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     mld = phi['mld']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # 1. base
#     ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # # 4. modified 3
#     # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (4 * cm * ki) / (mld * (aps ** 2))
#     dy1dt = (ka * np.exp(-AK * ka * t)) / np.clip(y1, 0.001, 0.999)
#     output = [dy1dt]
#     return output
#
# def thetadic14s():
#     theta14s = [2.9389515163416562e-09, 22964.17486273399, 9.57546639072975e-07]
#     theta14smin = [1.00e-9, 18000, 1e-8]
#     theta14smax=[1.00e-5, 30000, 1e-6]
#     theta14smaxs = [max_val / theta for max_val, theta in zip(theta14smax, theta14s)]
#     theta14smins = [min_val / theta for min_val, theta in zip(theta14smin, theta14s)]
#
#     return theta14s, theta14smaxs, theta14smins
#
# def f14s(t, y, phi, phit, theta, te):
#     te_array = np.array(te) if not isinstance(te, np.ndarray) else te
#     y1 = y
#     P = np.interp(t, te_array, phit['P'])
#     T = np.interp(t, te_array, phit['T'])
#     mld = phi['mld']
#     aps = phi['aps']
#     k, Ea, AK = theta[0], theta[1], theta[2]
#     R = 8.314  # Gas constant in J/(mol*K)
#     # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
#     cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
#
#     # 1. base
#     ki = k * np.exp((-Ea / (R*T)))
#     # # 2. modified 1
#     # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
#     # # 3. modified 2
#     # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
#     # # 4. modified 3
#     # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
#     ka = (12*cm * ki) / (mld * (aps ** 2))
#     dy1dt = (ka * np.exp(-AK * ka * t)) / ((1-(1-np.clip(y1, 0.001, 0.999))**(1/3))*((1-np.clip(y1, 0.001, 0.999))**(-2/3)))
#     output = [dy1dt]
#     return output




#our paper

#my data-orig models
#------------------------------------------ Model 08-----------------------------------------

def thetadic08():
    theta08 = [6.123634259442429e-09, 22327.84145134412]
    theta08min = [1.00e-9, 18000]
    theta08max = [1.00e-5, 28000]
    theta08maxs = [max_val / theta for max_val, theta in zip(theta08max, theta08)]
    theta08mins = [min_val / theta for min_val, theta in zip(theta08min, theta08)]

    return theta08, theta08maxs, theta08mins


def f08(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    aps = phi['aps']
    k, Ea = theta[0], theta[1]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
    # 1. base
    ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # # 4. modified 3
    # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    ka = (0.704 * cm * ki) / (rho * (aps ** 2))
    dy1dt = -ka / (np.log(1 - np.clip(y1, 0.001, 0.999)))
    output = [dy1dt]
    return output

#------------------------------------------ Model 09-----------------------------------------

def thetadic09():
    theta09 = [3.4270989918108122e-09, 22593.290494972498]
    theta09min = [1.00e-9, 18000]
    theta09max = [1.00e-5, 28000]
    theta09maxs = [max_val / theta for max_val, theta in zip(theta09max, theta09)]
    theta09mins = [min_val / theta for min_val, theta in zip(theta09min, theta09)]

    return theta09, theta09maxs, theta09mins

def f09(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    aps = phi['aps']
    k, Ea = theta[0], theta[1]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))
    # 1. base
    ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # # 4. modified 3
    # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (1.056 * cm * ki) / (rho * (aps ** 2))
    dy1dt = ka / (-2 + 2 * (1 - np.clip(y1, 0.001, 0.999)) ** (-1 / 3))
    output = [dy1dt]
    return output


#------------------------------------------ Model 11-----------------------------------------

def thetadic11():
    # theta11 = [2.707812554014968e-08, 24144.427994246784, 1.931273198572943]
    theta11 = [ 1.931249398412675]
    theta11min = [ 0.1]
    theta11max = [ 10]
    theta11maxs = [max_val / theta for max_val, theta in zip(theta11max, theta11)]
    theta11mins = [min_val / theta for min_val, theta in zip(theta11min, theta11)]

    return theta11, theta11maxs, theta11mins

def f11(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    cac = phi['cac']
    aps = phi['aps']
    k, Ea, AK = 2.7082983737865782e-08, 24144.940854636305, theta[0]
    R = 8.314  # Gas constant in J/(mol*K)
    am = 0.056
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))  # [mol.m-3]

    # 1. base
    ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # # 4. modified 3
    # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
    dy1dt = ka * np.exp(-AK * ka * t) / (2 * np.clip(y1, 0.001, 0.999))
    output = [dy1dt]
    return output

#------------------------------------------ Model 12-----------------------------------------

def thetadic12():
    theta12 = [ 5.256432132408076]
    theta12min = [ 0.1]
    theta12max=[ 10]
    theta12maxs = [max_val / theta for max_val, theta in zip(theta12max, theta12)]
    theta12mins = [min_val / theta for min_val, theta in zip(theta12min, theta12)]

    return theta12, theta12maxs, theta12mins

def f12(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    cac = phi['cac']
    aps = phi['aps']
    k, Ea, AK = 7.784315597484832e-09, 24174.47459078301, theta[0]
    R = 8.314  # Gas constant in J/(mol*K)
    am = 0.056
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
    # 1. base
    ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # # 4. modified 3
    # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
    dy1dt = ka * 2 * np.exp(-AK * ka * t) / -(np.log(1 - np.clip(y1, 0.001, 0.999)))
    output = [dy1dt]
    return output

#------------------------------------------ Model 13-----------------------------------------

def thetadic13():
    theta13 = [3.305407834054765e-09,  13.621581302396294]
    theta13min = [1.00e-10,  0.1]
    theta13max=[1.00e-6,  30]
    theta13maxs = [max_val / theta for max_val, theta in zip(theta13max, theta13)]
    theta13mins = [min_val / theta for min_val, theta in zip(theta13min, theta13)]

    return theta13, theta13maxs, theta13mins


def f13(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    cac = phi['cac']
    aps = phi['aps']
    k, Ea, AK = theta[0], 24170.141842277128, theta[1]
    R = 8.314  # Gas constant in J/(mol*K)
    am = 0.056
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # 1. base
    ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # # 4. modified 3
    # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
    dy1dt = (3 / 2) * ka * np.exp(-AK * ka * t) / -((1 - np.clip(y1, 0.001, 0.999)) ** (1 / 3) - 1)
    output = [dy1dt]
    return output

#------------------------------------------ Model 14-----------------------------------------

def thetadic14():
    theta14 = [ 4.1205952870775775]
    theta14min = [ 0.1]
    theta14max=[ 10]
    theta14maxs = [max_val / theta for max_val, theta in zip(theta14max, theta14)]
    theta14mins = [min_val / theta for min_val, theta in zip(theta14min, theta14)]

    return theta14, theta14maxs, theta14mins

def f14(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    mld = phi['mld']
    aps = phi['aps']
    k, Ea, AK = 1.4890118791931668e-08, 24191.925871030264, theta[0]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # 1. base
    ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # # 4. modified 3
    # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (8 * cm * ki) / (mld * (aps ** 2))
    dy1dt = (ka * np.exp(-AK * ka * t)) / ((1 - np.clip(y1, 0.001, 0.999)) ** (-0.5) - 1)
    output = [dy1dt]
    return output


def thetadic14p():
    theta14p = [3.862418207089512]
    theta14pmin = [0.1]
    theta14pmax=[ 10]
    theta14pmaxs = [max_val / theta for max_val, theta in zip(theta14pmax, theta14p)]
    theta14pmins = [min_val / theta for min_val, theta in zip(theta14pmin, theta14p)]

    return theta14p, theta14pmaxs, theta14pmins

def f14p(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    mld = phi['mld']
    aps = phi['aps']
    k, Ea, AK = 4.8011826586961365e-08, 24144.805624598048, theta[0]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # 1. base
    ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # # 4. modified 3
    # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (4 * cm * ki) / (mld * (aps ** 2))
    dy1dt = (ka * np.exp(-AK * ka * t)) / np.clip(y1, 0.001, 0.999)
    output = [dy1dt]
    return output

def thetadic14s():
    theta14s = [2.9390288881519947e-09, 22964.24459636733]
    theta14smin = [1.00e-9, 18000]
    theta14smax=[1.00e-5, 30000]
    theta14smaxs = [max_val / theta for max_val, theta in zip(theta14smax, theta14s)]
    theta14smins = [min_val / theta for min_val, theta in zip(theta14smin, theta14s)]

    return theta14s, theta14smaxs, theta14smins

def f14s(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    mld = phi['mld']
    aps = phi['aps']
    k, Ea, AK = theta[0], theta[1], 9.985300665060775e-07
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # 1. base
    ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # # 4. modified 3
    # ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (12*cm * ki) / (mld * (aps ** 2))
    dy1dt = (ka * np.exp(-AK * ka * t)) / ((1-(1-np.clip(y1, 0.001, 0.999))**(1/3))*((1-np.clip(y1, 0.001, 0.999))**(-2/3)))
    output = [dy1dt]
    return output





#------------------------------------------ Model 15-----------------------------------------


def thetadic15():
    theta15 = [4.04e-6, 114.19]
    theta15min = [1.00e-9, 10]
    theta15max=[1.00e-3, 500]
    theta15maxs = [max_val / theta for max_val, theta in zip(theta15max, theta15)]
    theta15mins = [min_val / theta for min_val, theta in zip(theta15min, theta15)]

    return theta15, theta15maxs, theta15mins

def f15(t, y, phi, phit, theta, te):
    y1 = y
    sg = phi['sg']
    m = phi['m']
    ks, kp = theta[0], theta[1]
    # Compute reaction rates
    dy1dt = sg * m * ks * np.exp(- ks * kp * t)
    output = [dy1dt]
    return output

#------------------------------------------ Model 16-----------------------------------------
#joint my made model for ash layer diffusion and surface reaction

def thetadic16():
    theta16 = [7.2544e-7, 28005, 7.2544e-7, 28005]
    theta16min = [1.00e-9, 10000, 1.00e-9, 10000]
    theta16max=[1.00e-3, 34000, 1.00e-3, 34000]
    theta16maxs = [max_val / theta for max_val, theta in zip(theta16max, theta16)]
    theta16mins = [min_val / theta for min_val, theta in zip(theta16min, theta16)]

    return theta16, theta16maxs, theta16mins


def f16(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    rho = phi['rho']
    cac = phi['cac']
    aps = phi['aps']
    # k1, Ea1, k2, Ea2, AK = theta[0], theta[1], theta[2], theta[3], theta[4]
    k1, Ea1, k2, Ea2 = theta[0], theta[1], theta[2], theta[3]
    R = 8.314  # Gas constant in J/(mol*K)
    am = 0.056
    cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
    N=47095628
    # N=1
    # d0=2.35e-6*np.exp(-2119/(T-273.15))
    phia1 = np.log(k1) - (Ea1 / (R * 333.15))
    phib1 = np.log(Ea1 / R)
    k1 = np.exp(phia1 + np.exp(phib1) * ((1 / 333.15) - (1 / T)))
    phia2 = np.log(k2) - (Ea2 / (R * 333.15))
    phib2 = np.log(Ea2 / R)
    di = np.exp(phia2 + np.exp(phib2) * ((1 / 333.15) - (1 / T)))
    ka= (400*am*cm*(N))/(rho*cac*(aps))
    dy1dt = ka/((2/k1)+((aps*np.clip(y1, 0.001, 0.999))/(di)))
    output = [dy1dt]
    return output


#------------------------------------------ Model 17-----------------------------------------
#Ginstling-Brounshtein’s equation

def thetadic17():
    theta17 = [7.2544e-7, 28005]
    theta17min = [1.00e-9, 10000]
    theta17max=[1.00e-3, 34000]
    theta17maxs = [max_val / theta for max_val, theta in zip(theta17max, theta17)]
    theta17mins = [min_val / theta for min_val, theta in zip(theta17min, theta17)]

    return theta17, theta17maxs, theta17mins

from scipy.optimize import minimize_scalar


def f17(t, y, phi, phit, theta, te):
    # Ensure te is a numpy array
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te

    # Extract y1 from y (conversion G)
    y1 = y

    # Interpolate P and T based on time t
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Extract parameters
    aps = phi['aps']  # Particle size (m)
    k, Ea = theta[0], theta[1]  # Nominal kinetic constant and activation energy
    R = 8.314  # Universal gas constant in J/(mol*K)

    # Calculate CO2 concentration cm (mol/m^3)
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Calculate effective diffusivity De (m^2/s)
    phia = np.log(k) - (Ea / (R * 333.15))
    phib = Ea / R
    De = np.exp(phia + phib * ((1 / 333.15) - (1 / T)))  # Effective diffusivity formula

    # Check for valid De and cm
    if De <= 0 or cm <= 0:
        raise ValueError(f"Invalid De or cm: De={De}, cm={cm}")

    # Define the equation for ka
    def ka_equation(ka):
        try:
            sqrt_term = np.sqrt(ka) / (2 * np.sqrt(De))
            if sqrt_term > 100:  # Prevent overflow in erfc
                return np.inf
            left_side = np.sqrt(ka * np.pi) / (2 * cm * np.sqrt(De))
            right_side = np.exp(-ka / (4 * De)) / erfc(sqrt_term)
            return (left_side - right_side) ** 2  # Squared difference for optimization
        except Exception as e:
            raise ValueError(f"Error in ka_equation calculation: {e}")

    # Solve for ka using minimize_scalar
    try:
        result = minimize_scalar(ka_equation, bounds=(1e-10, 1e2), method='bounded')
        if result.success and result.fun < 1e-6:  # Check if the optimization succeeded
            ka = result.x
        else:
            raise ValueError("Optimization for ka did not converge.")
    except Exception as e:
        # Log debugging information
        print(f"Error solving for ka: {e}")
        print(f"De: {De}, cm: {cm}")
        raise RuntimeError(f"Error solving for ka: {e}")

    # Calculate dy1/dt using GB model equation
    denominator = 1 - (2 / 3) * y1 - (1 - y1) ** (2 / 3)  # Matches Equation (14)
    if abs(denominator) < 1e-10:
        raise ValueError("Denominator is too close to zero, check your parameters.")

    dy1dt = (12 * ka / aps ** 2) / denominator

    # Return the derivative as a list for compatibility with ODE solvers
    return [dy1dt]

#------------------------------------------ Model 22-----------------------------------------

def thetadic22():
    theta22 = [0.8, 5.0, 20.0, 0.3, 0.25, 0.5, 2.0, 100]
    theta22max=[1.0, 8.0, 25.0, 0.4, 0.4, 0.75, 3.0, 150]
    theta22min=[0.6, 3.0, 15.0, 0.2, 0.15, 0.25, 1.5, 70]
    theta22maxs = [max_val / theta for max_val, theta in zip(theta22max, theta22)]
    theta22mins = [min_val / theta for min_val, theta in zip(theta22min, theta22)]

    return theta22, theta22maxs, theta22mins


def f22(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1, y2, y3, y4 = y

    # Interpolate Temperature (T) and Heart Rate (HR) for the current time t
    T = np.interp(t, te_array, phit['T'])  # Interpolated temperature effect
    HR = np.interp(t, te_array, phit['HR'])  # Interpolated heart rate effect

    # Parameters
    ka = theta[0]  # Absorption rate constant
    Cl = theta[1] # Clearance from central compartment
    Vd = theta[2]   # Volume of distribution in central compartment
    k12 = theta[3]  # Rate from central to peripheral
    k21 = theta[4]  # Rate from peripheral to central
    kE = theta[5]   # Rate from central to effect compartment
    EC50 = theta[6]   # Drug concentration causing half-maximal effect
    Emax = theta[7]   # Maximum drug effect

    # Temperature and heart rate effect on clearance (optional)
    Cl_T = Cl * (1 + 0.01 * (T - 37))  # Clearance modification by temperature
    Cl_HR = Cl_T * (1 + 0.01 * (HR - 70))  # Clearance modification by heart rate

    # Differential equations for the pharmacokinetics and pharmacodynamics
    dy1dt = -ka * y1  # Drug absorption into the central compartment
    dy2dt = ka * y1 - (Cl_HR / Vd) * y2 + k21 * y3 - k12 * y2 - kE * y2
    dy3dt = k12 * y2 - k21 * y3
    dy4dt = (Emax * y2 / (EC50 + y2)) - y4  # PD effect using Emax model

    return [dy1dt, dy2dt, dy3dt, dy4dt]

