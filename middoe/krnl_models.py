import numpy as np
from scipy.special import erfc


# ------------------------------------------ Model 05-----------------------------------------
# Shrinking core model: De = const, Surface reaction controlled, Cylindrical particle

def thetadic05():
    # Parameters: [k (rate constant), Ea (activation energy)]
    theta05 = [1.29e-10, 26927]  # Nominal values
    theta05min = [1.00e-18, 10000]  # Minimum bounds
    theta05max = [1.00e-2, 32000]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta05maxs = [max_val / theta for max_val, theta in zip(theta05max, theta05)]
    theta05mins = [min_val / theta for min_val, theta in zip(theta05min, theta05)]

    return theta05, theta05maxs, theta05mins


def f05(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    rho = phi['rho']  # Density [kg/m^3]
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea = theta[0], theta[1]  # pre-exponential factor and activation energy
    R = 8.314  # Universal gas constant (J/mol·K)

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    #2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with differnet reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # apparent rate constant (ka)
    ka = (0.088 * cm * ki) / (rho * aps)

    # Reaction rate equation
    dy1dt = -2 * ka * (np.clip(y1, 0.001, 0.999)) ** (-1 / 2)

    return [dy1dt]

#------------------------------------------ Model 06-----------------------------------------
# Shrinking core model: De = const, Surface reaction controlled, Spherical particle

def thetadic06():
    # Parameters: [k (rate constant), Ea (activation energy)]
    theta06 = [1.29e-5, 26927]  # Nominal values
    theta06min = [1.00e-8, 10000]  # Minimum bounds
    theta06max = [1.00e-2, 32000]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta06maxs = [max_val / theta for max_val, theta in zip(theta06max, theta06)]
    theta06mins = [min_val / theta for min_val, theta in zip(theta06min, theta06)]

    return theta06, theta06maxs, theta06mins


def f06(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    rho = phi['rho']  # Density [kg/m^3]
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea = theta[0], theta[1]  # pre-exponential factor and activation energy
    R = 8.314  # Universal gas constant (J/mol·K)

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # apparent rate constant (ka)
    ka = (0.088 * cm * ki) / (rho * aps)

    # Reaction rate equation
    dy1dt = 3 * ka * (1 - np.clip(y1, 0.001, 0.999)) ** (2/3)

    return [dy1dt]


#------------------------------------------ Model 07-----------------------------------------
# Shrinking core model: De = const, Ash layer diffusion controlled, Plate shape particle

def thetadic07():
    # Parameters: [k (rate constant), Ea (activation energy)]
    theta07 = [1.29e-5, 26927]  # Nominal values
    theta07min = [1.00e-8, 18000]  # Minimum bounds
    theta07max = [1.00e-2, 32000]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta07maxs = [max_val / theta for max_val, theta in zip(theta07max, theta07)]
    theta07mins = [min_val / theta for min_val, theta in zip(theta07min, theta07)]

    return theta07, theta07maxs, theta07mins


def f07(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    rho = phi['rho']  # Density [kg/m^3]
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea = theta[0], theta[1]  # pre-exponential factor and activation energy
    R = 8.314  # Universal gas constant (J/mol·K)

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # Apparent rate constant (ka) adjusted for diffusion-controlled system
    ka = (0.088 * cm * ki) / (rho * (aps ** 2))

    # Reaction rate equation (diffusion controlled)
    dy1dt = ka

    return [dy1dt]

# #------------------------------------------ Model 08-----------------------------------------
# Shrinking core model: De = const, Ash layer diffusion controlled, Cylindrical particle

def thetadic08():
    # Parameters: [k (rate constant), Ea (activation energy)]
    theta08 = [6.048450283701883e-07, 20191.94152486254]  # Nominal values
    theta08min = [1.00e-9, 18000]  # Minimum bounds
    theta08max = [1.00e-5, 28000]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta08maxs = [max_val / theta for max_val, theta in zip(theta08max, theta08)]
    theta08mins = [min_val / theta for min_val, theta in zip(theta08min, theta08)]

    return theta08, theta08maxs, theta08mins


def f08(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    rho = phi['rho']  # Density [kg/m^3]
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea = theta[0], theta[1]  # pre-exponential factor and activation energy
    R = 8.314  # Universal gas constant (J/mol·K)

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # Apparent rate constant (ka) adjusted for diffusion-controlled system
    ka = (0.704 * cm * ki) / (rho * (aps ** 2))

    # Reaction rate equation (diffusion controlled)
    dy1dt = -ka / (np.log(1 - np.clip(y1, 0.001, 0.999)))

    return [dy1dt]

# #------------------------------------------ Model 09-----------------------------------------
# Shrinking core model: De = const, Ash layer diffusion controlled, Spherical particle

def thetadic09():
    # Parameters: [k (rate constant), Ea (activation energy)]
    theta09 = [3.056226327814133e-07, 20197.265107932555]  # Nominal values
    theta09min = [1.00e-9, 18000]  # Minimum bounds
    theta09max = [1.00e-5, 28000]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta09maxs = [max_val / theta for max_val, theta in zip(theta09max, theta09)]
    theta09mins = [min_val / theta for min_val, theta in zip(theta09min, theta09)]

    return theta09, theta09maxs, theta09mins


def f09(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    rho = phi['rho']  # Density [kg/m^3]
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea = theta[0], theta[1]  # pre-exponential factor and activation energy
    R = 8.314  # Universal gas constant (J/mol·K)

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # Apparent rate constant (ka) adjusted for diffusion-controlled system
    ka = (1.056 * cm * ki) / (rho * (aps ** 2))

    # Reaction rate equation (diffusion controlled)
    dy1dt = ka / (-2 + 2 * (1 - np.clip(y1, 0.001, 0.999)) ** (-1 / 3))

    return [dy1dt]


# #------------------------------------------ Model 11-----------------------------------------
# Shrinking core model: De = dynamic, Ash layer diffusion controlled, Plate shape particle

def thetadic11():
    # Parameters definitions
    theta11 = [3.1650075431484574e-07, 20239.398294836665, 1.2133668813455163]  # Nominal values
    theta11min = [1.00e-9, 18000, 0.1]  # Minimum bounds
    theta11max = [1.00e-5, 30000, 10]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta11maxs = [max_val / theta for max_val, theta in zip(theta11max, theta11)]
    theta11mins = [min_val / theta for min_val, theta in zip(theta11min, theta11)]

    return theta11, theta11maxs, theta11mins


def f11(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    rho = 2468.3  # Density [kg/m^3]
    cac = 14.58  # Fraction of active CaO in mineral wt%
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea, AK = theta[0], theta[1], theta[2]  # k (rate constant), Ea (activation energy), AK (decay ratio)
    R = 8.314  # Universal gas constant (J/mol·K)
    am = 0.056  # Molar mass of CaO [kg.mol-1]

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # Apparent rate constant (ka)
    ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))

    # Reaction rate equation
    dy1dt = ka * np.exp(-AK * ka * t) / (2 * np.clip(y1, 0.001, 0.999))

    return [dy1dt]

# #------------------------------------------ Model 12-----------------------------------------
# Shrinking core model: De = dynamic, Ash layer diffusion controlled, Cylindrical particle

def thetadic12():
    # Parameters definitions
    theta12 = [8.657783497064711e-08, 20190.20009631406, 2.194284154090643]  # Nominal values
    theta12min = [1.00e-10, 18000, 0.1]  # Minimum bounds
    theta12max = [1.00e-6, 30000, 10]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta12maxs = [max_val / theta for max_val, theta in zip(theta12max, theta12)]
    theta12mins = [min_val / theta for min_val, theta in zip(theta12min, theta12)]

    return theta12, theta12maxs, theta12mins


def f12(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    rho = 2468.3  # Density [kg/m^3]
    cac = 14.58  # Fraction of active CaO in mineral wt%
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea, AK = theta[0], theta[1], theta[2]  # k (rate constant), Ea (activation energy), AK (decay ratio)
    R = 8.314  # Universal gas constant (J/mol·K)
    am = 0.056  # Molar mass of CaO [kg.mol-1]

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # Apparent rate constant (ka)
    ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))

    # Reaction rate equation
    dy1dt = ka * 2 * np.exp(-AK * ka * t) / -(np.log(1 - np.clip(y1, 0.001, 0.999)))

    return [dy1dt]


# #------------------------------------------ Model 13-----------------------------------------
# Shrinking core model: De = dynamic, Ash layer diffusion controlled, Spherical particle

def thetadic13():
    # Parameters definitions
    theta13 = [3.74989949780187e-08, 20213.603669155094, 6.949990334765804]  # Nominal values
    theta13min = [1.00e-10, 18000, 0.1]  # Minimum bounds
    theta13max = [1.00e-6, 30000, 10]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta13maxs = [max_val / theta for max_val, theta in zip(theta13max, theta13)]
    theta13mins = [min_val / theta for min_val, theta in zip(theta13min, theta13)]

    return theta13, theta13maxs, theta13mins


def f13(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    rho = 2468.3  # Density [kg/m^3]
    cac = 14.58  # Fraction of active CaO in mineral wt%
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea, AK = theta[0], theta[1], theta[2]  # k (rate constant), Ea (activation energy), AK (decay ratio)
    R = 8.314  # Universal gas constant (J/mol·K)
    am = 0.056  # Molar mass of CaO [kg.mol-1]

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # Apparent rate constant (ka)
    ka = (200 * 4 * am * cm * ki) / (rho * cac * (aps ** 2))

    # Reaction rate equation
    dy1dt = (3 / 2) * ka * np.exp(-AK * ka * t) / -((1 - np.clip(y1, 0.001, 0.999)) ** (1 / 3) - 1)

    return [dy1dt]

# #------------------------------------------ Model 14-----------------------------------------
# Shrinking core model: De = dynamic and parabolic, Ash layer diffusion controlled, Plate shape particle

def thetadic14():
    # Parameters definitions
    theta14 = [1.3341960133302085e-06, 20239.41387867327, 2.426734975569295]  # Nominal values
    theta14min = [1.00e-8, 18000, 0.1]  # Minimum bounds
    theta14max = [1.00e-4, 32000, 10]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta14maxs = [max_val / theta for max_val, theta in zip(theta14max, theta14)]
    theta14mins = [min_val / theta for min_val, theta in zip(theta14min, theta14)]

    return theta14, theta14maxs, theta14mins


def f14(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    mld = phi['mld']  # Molar density of reaction product [mol/m^3]
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea, AK = theta[0], theta[1], theta[2]  # k (rate constant), Ea (activation energy), AK (decay ratio)
    R = 8.314  # Universal gas constant (J/mol·K)

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # Apparent rate constant (ka)
    ka = (4 * cm * ki) / (mld * (aps ** 2))

    # Reaction rate equation
    dy1dt = (ka * np.exp(-AK * ka * t)) / np.clip(y1, 0.001, 0.999)

    return [dy1dt]

# #------------------------------------------ Model 15-----------------------------------------
# Shrinking core model: De = dynamic and parabolic, Ash layer diffusion controlled, Cylindrical particle
# Parameters: k (rate constant), Ea (activation energy), AK (decay ratio)

def thetadic15():
    # Parameters definitions
    theta15 = [3.822470068711253e-07, 20198.045600335983, 0.7671885604344969]  # Nominal values
    theta15min = [1.00e-9, 18000, 0.1]  # Minimum bounds
    theta15max = [1.00e-5, 30000, 10]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta15maxs = [max_val / theta for max_val, theta in zip(theta15max, theta15)]
    theta15mins = [min_val / theta for min_val, theta in zip(theta15min, theta15)]

    return theta15, theta15maxs, theta15mins


def f15(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    mld = phi['mld']  # Molar density of reaction product [mol/m^3]
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea, AK = theta[0], theta[1], theta[2]  # k (rate constant), Ea (activation energy), AK (decay ratio)
    R = 8.314  # Universal gas constant (J/mol·K)

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # Apparent rate constant (ka)
    ka = (8 * cm * ki) / (mld * (aps ** 2))

    # Reaction rate equation
    dy1dt = (ka * np.exp(-AK * ka * t)) / ((1 - np.clip(y1, 0.001, 0.999)) ** (-0.5) - 1)

    return [dy1dt]


# #------------------------------------------ Model 16-----------------------------------------
# Shrinking core model: De = dynamic and parabolic, Ash layer diffusion controlled, Spherical particle
# Parameters: k (rate constant), Ea (activation energy), AK (decay ratio)

def thetadic16():
    # Parameters definitions
    theta16 = [1.8333977056274745e-07, 20170.507458025906, 2.5666991259436363e-07]  # Nominal values
    theta16min = [1.00e-9, 18000, 1e-8]  # Minimum bounds
    theta16max = [1.00e-5, 30000, 1e-6]  # Maximum bounds

    # Normalized values for parameter scaling during estimation
    theta16maxs = [max_val / theta for max_val, theta in zip(theta16max, theta16)]
    theta16mins = [min_val / theta for min_val, theta in zip(theta16min, theta16)]

    return theta16, theta16maxs, theta16mins


def f19(t, y, phi, phit, theta, te):
    # Extract relevant variables
    te_array = np.array(te) if not isinstance(te, np.ndarray) else te
    y1 = y  # Carbonation Efficiency (y1)

    # Interpolate time-dependent variables (pressure [bar] & temperature [K])
    P = np.interp(t, te_array, phit['P'])
    T = np.interp(t, te_array, phit['T'])

    # Physical constants
    mld = phi['mld']  # Molar density of reaction product [mol/m^3]
    aps = phi['aps']  # Particle average diameter [m]
    k, Ea, AK = theta[0], theta[1], theta[2]  # k (rate constant), Ea (activation energy), AK (decay ratio)
    R = 8.314  # Universal gas constant (J/mol·K)

    # CO2 concentration calculation based on Henry law
    # 1. Miao et. al, Kinetic analysis on CO2 sequestration from flue gas through direct aqueous
    # mineral carbonation of circulating fluidized bed combustion fly ash, Fuel, 2023 based calculation
    # cm = (P * 100000) / (2.82 * 10 ** 6 * np.exp(-2044 / T))
    # 2. NIST based calculation
    cm = P * 1000 * 0.035 * np.exp(2400 * (1 / T - 1 / 298.15))

    # Arrhenius decomposition (with different reparameterisation methods)

    # # 1. base
    # ki = k * np.exp((-Ea / (R*T)))
    # # 2. modified 1
    # ki = (k * np.exp(-Ea / (R * 333.15))) * np.exp((Ea / R) * ((1 / 333.15) - (1 / T)))
    # # 3. modified 2
    # ki = np.exp((np.log(k) - (Ea/(R*333.15)) + (Ea/R) * ((1/333.15) - (1/T))))
    # 4. modified 3
    ki = np.exp((np.log(k) - (Ea / (R * 333.15))) + np.exp((np.log(Ea / R))) * ((1 / 333.15) - (1 / T)))

    # Apparent rate constant (ka)
    ka = (12 * cm * ki) / (mld * (aps ** 2))

    # Reaction rate equation
    dy1dt = (ka * np.exp(-AK * ka * t)) / ((1 - (1 - np.clip(y1, 0.001, 0.999)) ** (1 / 3)) * ((1 - np.clip(y1, 0.001, 0.999)) ** (-2 / 3)))

    return [dy1dt]

