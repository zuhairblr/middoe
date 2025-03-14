import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc
from scipy.optimize import root_scalar

#------------------------------------------ Model 05-----------------------------------------
#shrinking core, De:const, surface reaction based, cylindrical particle

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
#shrinking core, De:const, surface reaction based, spherical particle

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
#shrinking core, De:const, Ash layer diffusion based, plate shape particle

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


# #------------------------------------------ Model 08-----------------------------------------
#shrinking core, De:const, Ash layer diffusion based, cylindrical particle

def thetadic08():
    theta08 = [6.048450283701883e-07, 20191.94152486254]
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
    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    ka = (0.704 * cm * ki) / (rho * (aps ** 2))
    dy1dt = -ka / (np.log(1 - np.clip(y1, 0.001, 0.999)))
    output = [dy1dt]
    return output

# #------------------------------------------ Model 09-----------------------------------------
#shrinking core, De:const, Ash layer diffusion based, spherical particle

def thetadic09():
    theta09 = [3.056226327814133e-07, 20197.265107932555]
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
    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (1.056 * cm * ki) / (rho * (aps ** 2))
    dy1dt = ka / (-2 + 2 * (1 - np.clip(y1, 0.001, 0.999)) ** (-1 / 3))
    output = [dy1dt]
    return output


# #------------------------------------------ Model 11-----------------------------------------
#shrinking core, De:dynamic, Ash layer diffusion based, plate shape particle
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
#shrinking core, De:dynamic, Ash layer diffusion based, cylindrical particle

def thetadic12():
    theta12 = [8.657783497064711e-08, 20190.20009631406, 2.194284154090643]
    theta12min = [1.00e-10, 18000, 0.1]
    theta12max=[1.00e-6, 30000, 10]
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
    k, Ea, AK = theta[0], theta[1], theta[2]
    R = 8.314  # Gas constant in J/(mol*K)
    am = 0.056
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400*(1/T-1/298.15))
    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
    dy1dt = ka * 2 * np.exp(-AK * ka * t) / -(np.log(1 - np.clip(y1, 0.001, 0.999)))
    output = [dy1dt]
    return output
#
# #------------------------------------------ Model 13-----------------------------------------
#shrinking core, De:dynamic, Ash layer diffusion based, spherical particle

def thetadic13():
    theta13 = [3.74989949780187e-08, 20213.603669155094, 6.949990334765804]
    theta13min = [1.00e-10, 18000, 0.1]
    theta13max=[1.00e-6, 30000, 10]
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
    k, Ea, AK = theta[0], theta[1], theta[2]
    R = 8.314  # Gas constant in J/(mol*K)
    am = 0.056
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))
    dy1dt = (3 / 2) * ka * np.exp(-AK * ka * t) / -((1 - np.clip(y1, 0.001, 0.999)) ** (1 / 3) - 1)
    output = [dy1dt]
    return output

#
# #------------------------------------------ Model 14-----------------------------------------
#shrinking core, De:dynamic and parabolic, Ash layer diffusion based, plate shape particle

def thetadic14():
    theta14 = [1.3341960133302085e-06, 20239.41387867327, 2.426734975569295]
    theta14min = [1.00e-8, 18000, 0.1]
    theta14max=[1.00e-4, 32000, 10]
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
    k, Ea, AK = theta[0], theta[1], theta[2]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (4 * cm * ki) / (mld * (aps ** 2))
    dy1dt = (ka * np.exp(-AK * ka * t)) / np.clip(y1, 0.001, 0.999)
    output = [dy1dt]
    return output


#
# #------------------------------------------ Model 15-----------------------------------------
#shrinking core, De:dynamic and parabolic, Ash layer diffusion based, cylindrical particle

def thetadic15():
    theta15 = [3.822470068711253e-07, 20198.045600335983, 0.7671885604344969]
    theta15min = [1.00e-9, 18000, 0.1]
    theta15max=[1.00e-5, 30000, 10]
    theta15maxs = [max_val / theta for max_val, theta in zip(theta15max, theta15)]
    theta15mins = [min_val / theta for min_val, theta in zip(theta15min, theta15)]

    return theta15, theta15maxs, theta15mins

def f15(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    mld = phi['mld']
    aps = phi['aps']
    k, Ea, AK = theta[0], theta[1], theta[2]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (8 * cm * ki) / (mld * (aps ** 2))
    dy1dt = (ka * np.exp(-AK * ka * t)) / ((1 - np.clip(y1, 0.001, 0.999)) ** (-0.5) - 1)
    output = [dy1dt]
    return output

#
# #------------------------------------------ Model 16-----------------------------------------
#shrinking core, De:dynamic and parabolic, Ash layer diffusion based, spherical particle

def thetadic16():
    theta16 = [1.8333977056274745e-07, 20170.507458025906, 2.5666991259436363e-07]
    theta16min = [1.00e-9, 18000, 1e-8]
    theta16max=[1.00e-5, 30000, 1e-6]
    theta16maxs = [max_val / theta for max_val, theta in zip(theta16max, theta16)]
    theta16mins = [min_val / theta for min_val, theta in zip(theta16min, theta16)]

    return theta16, theta16maxs, theta16mins

def f16(t, y, phi, phit, theta, te):
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])
    mld = phi['mld']
    aps = phi['aps']
    k, Ea, AK = theta[0], theta[1], theta[2]
    R = 8.314  # Gas constant in J/(mol*K)
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))
    ka = (12*cm * ki) / (mld * (aps ** 2))
    dy1dt = (ka * np.exp(-AK * ka * t)) / ((1-(1-np.clip(y1, 0.001, 0.999))**(1/3))*((1-np.clip(y1, 0.001, 0.999))**(-2/3)))
    output = [dy1dt]
    return output


