"""High-resolution ionic-state model for mineral carbonation.

This script implements the research model documented in SM 6.0,
``High-resolution ionic-state kinetic modelling of mineral carbonation
systems for optimised experimentation``.  The implementation resolves
particle-size-distribution bins, radial pore-solution transport, C-H/C-S-H
dissolution, carbonate speciation, non-ideal electrolyte activities, and
particle-to-bulk coupling.

The code is intentionally kept as a publication companion script: numerical
settings, material constants, and plotting/profiling hooks remain in-line so
that the executable model follows the supplementary material without an
additional configuration layer.

Notes
-----
The inline section comments reference the corresponding parts of SM 6.0:
S1 covers particle-domain transport, S2 initial phase inventories, S3 solid
phase dissolution, S4 carbonate chemistry, S5 activity corrections, S6
particle-to-bulk coupling, and S7 numerical solution and mass-balance checks.
"""
import time
import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cProfile, pstats
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

profiler = cProfile.Profile()
profiler.enable()


MICRON_LABEL = r'$\mu$m'
MW_CA = 40.078e-3
MW_OH = 17.007e-3
MW_CAOH2 = 74e-3
MW_CSH_EFFECTIVE = 145e-3
MW_SIO2_EQUIV = 60.0843e-3
CSH_OH_PER_CA_RELEASE = 2.00
SI_GEL_CA_SI_THRESHOLD = 0.6
SI_GEL_STRICT_CA_SI_THRESHOLD = 0.1

def calculate_bulk_csh_metrics(radial_profiles, shell_masses_per_bin, n_bins, bin_data=None,
                               ca_si_threshold=SI_GEL_CA_SI_THRESHOLD):
    if radial_profiles is None or shell_masses_per_bin is None or n_bins <= 0:
        return {}

    n_time = None
    for b in range(n_bins):
        bin_key = f'bin_{b}'
        if bin_key in radial_profiles:
            n_time = radial_profiles[bin_key]['Ca_CSH_moles'].shape[1]
            break

    if n_time is None:
        return {}

    total_csh_ca_mol = np.zeros(n_time)
    total_csh_si_mol = np.zeros(n_time)
    total_si_gel_mol = np.zeros(n_time)
    total_si_gel_region_mass_kg = np.zeros(n_time)
    total_solid_shell_mass_kg = 0.0

    for b in range(n_bins):
        bin_key = f'bin_{b}'
        if bin_key not in radial_profiles:
            continue

        profiles = radial_profiles[bin_key]
        ca_csh_moles = np.maximum(np.asarray(profiles['Ca_CSH_moles'], dtype=float), 0.0)
        si_csh_moles = np.maximum(np.asarray(profiles['Si_CSH_moles'], dtype=float), 0.0)
        mass_shells = np.asarray(shell_masses_per_bin[b], dtype=float)
        if bin_data is not None:
            mass_shells = mass_shells * float(bin_data[b].get('N_particles', 1.0))
        mass_shells_2d = mass_shells[:, np.newaxis]

        ca_si = ca_csh_moles / np.maximum(si_csh_moles, 1e-30)
        si_gel_mask = ca_si <= ca_si_threshold

        total_csh_ca_mol += np.sum(ca_csh_moles, axis=0)
        total_csh_si_mol += np.sum(si_csh_moles, axis=0)
        total_si_gel_mol += np.sum(np.where(si_gel_mask, si_csh_moles, 0.0), axis=0)
        total_si_gel_region_mass_kg += np.sum(np.where(si_gel_mask, mass_shells_2d, 0.0), axis=0)
        total_solid_shell_mass_kg += float(np.sum(mass_shells))

    bulk_csh_ca_si = total_csh_ca_mol / np.maximum(total_csh_si_mol, 1e-30)
    bulk_si_gel_conc = total_si_gel_mol / max(total_solid_shell_mass_kg, 1e-30)
    bulk_si_gel_region_conc = total_si_gel_mol / np.maximum(total_si_gel_region_mass_kg, 1e-30)
    bulk_si_gel_fraction = total_si_gel_mol / np.maximum(total_csh_si_mol, 1e-30)

    return {
        'bulk_CSH_Ca_mol': total_csh_ca_mol,
        'bulk_CSH_Si_mol': total_csh_si_mol,
        'bulk_CSH_Ca_Si': bulk_csh_ca_si,
        'bulk_Si_gel_mol': total_si_gel_mol,
        'bulk_Si_gel_conc': bulk_si_gel_conc,
        'bulk_Si_gel_region_conc': bulk_si_gel_region_conc,
        'bulk_Si_gel_fraction': bulk_si_gel_fraction,
        'Si_gel_Ca_Si_threshold': ca_si_threshold,
    }


def calculate_csh_si_gel_weight_concentrations(result,
                                               ca_si_threshold=SI_GEL_CA_SI_THRESHOLD):
    """
    Convert whole-particle C-S-H reservoir changes to slurry-basis g/L values.

    The model stores C-S-H as solid Ca and fixed solid Si reservoirs in every
    radial shell. C-S-H mass loss is therefore estimated from the Ca removed
    from the C-S-H reservoir plus its associated OH removal. Si-gel mass is
    reported as SiO2-equivalent mass because the model tracks Si moles, not a
    hydrated gel formula.
    """
    radial_profiles = result.get('radial_profiles')
    n_bins = int(result.get('n_bins', 0))
    if radial_profiles is None or n_bins <= 0:
        return {}

    n_time = None
    for b in range(n_bins):
        bin_key = f'bin_{b}'
        if bin_key in radial_profiles:
            n_time = radial_profiles[bin_key]['Ca_CSH_moles'].shape[1]
            break

    if n_time is None:
        return {}

    total_csh_ca_mol = np.zeros(n_time)
    total_csh_si_mol = np.zeros(n_time)
    strict_si_gel_mol = np.zeros(n_time)

    for b in range(n_bins):
        bin_key = f'bin_{b}'
        if bin_key not in radial_profiles:
            continue

        profiles = radial_profiles[bin_key]
        ca_csh_moles = np.maximum(np.asarray(profiles['Ca_CSH_moles'], dtype=float), 0.0)
        si_csh_moles = np.maximum(np.asarray(profiles['Si_CSH_moles'], dtype=float), 0.0)

        ca_si = ca_csh_moles / np.maximum(si_csh_moles, 1e-30)
        si_gel_mask = (si_csh_moles > 1e-30) & (ca_si <= ca_si_threshold)

        total_csh_ca_mol += np.sum(ca_csh_moles, axis=0)
        total_csh_si_mol += np.sum(si_csh_moles, axis=0)
        strict_si_gel_mol += np.sum(np.where(si_gel_mask, si_csh_moles, 0.0), axis=0)

    slurry_volume_l = float(result.get('slurry_volume_l', result.get('slurry_volume', 1.0)))
    slurry_volume_l = max(slurry_volume_l, 1e-30)

    initial_csh_ca_mol = total_csh_ca_mol[0] if total_csh_ca_mol.size else 0.0
    initial_csh_si_mol = total_csh_si_mol[0] if total_csh_si_mol.size else 0.0

    csh_ca_removed_mol = np.maximum(initial_csh_ca_mol - total_csh_ca_mol, 0.0)
    csh_oh_released_mol = csh_ca_removed_mol * CSH_OH_PER_CA_RELEASE
    csh_ca_released_mass_kg = csh_ca_removed_mol * MW_CA
    csh_oh_released_mass_kg = csh_oh_released_mol * MW_OH
    csh_mass_loss_kg = csh_ca_released_mass_kg + csh_oh_released_mass_kg
    initial_csh_mass_kg = initial_csh_si_mol * MW_CSH_EFFECTIVE
    current_csh_mass_kg = np.maximum(initial_csh_mass_kg - csh_mass_loss_kg, 0.0)

    strict_si_gel_formed_mol = np.maximum(strict_si_gel_mol - strict_si_gel_mol[0], 0.0)
    strict_si_gel_mass_kg = strict_si_gel_formed_mol * MW_SIO2_EQUIV
    stacked_csh_mass_kg = np.maximum(
        initial_csh_mass_kg
        - strict_si_gel_mass_kg
        - csh_ca_released_mass_kg
        - csh_oh_released_mass_kg,
        0.0,
    )

    return {
        'total_CSH_Ca_mol_for_mass': total_csh_ca_mol,
        'total_CSH_Si_mol_for_mass': total_csh_si_mol,
        'CSH_mass_concentration_g_l': current_csh_mass_kg * 1000.0 / slurry_volume_l,
        'CSH_stacked_remaining_g_l': stacked_csh_mass_kg * 1000.0 / slurry_volume_l,
        'CSH_mass_change_g_l': -csh_mass_loss_kg * 1000.0 / slurry_volume_l,
        'CSH_mass_loss_g_l': csh_mass_loss_kg * 1000.0 / slurry_volume_l,
        'CSH_released_Ca_mol': csh_ca_removed_mol,
        'CSH_released_OH_mol': csh_oh_released_mol,
        'CSH_released_Ca_g_l': csh_ca_released_mass_kg * 1000.0 / slurry_volume_l,
        'CSH_released_OH_g_l': csh_oh_released_mass_kg * 1000.0 / slurry_volume_l,
        'Si_gel_strict_mol': strict_si_gel_mol,
        'Si_gel_strict_formation_g_l': strict_si_gel_mass_kg * 1000.0 / slurry_volume_l,
        'Si_gel_strict_Ca_Si_threshold': ca_si_threshold,
        'slurry_volume_l_for_mass_conc': slurry_volume_l,
    }



def solve_model(t, y0, tii, tvi, theta):  # final
    """Simulate the coupled particle-pore and bulk ionic carbonation model.

    The model follows SM 6.0 by combining a particle-size-distribution
    representation (S1.1), finite-volume radial transport in spherical shells
    (S1.2-S1.4), initial C-H/C-S-H inventories (S2), undersaturation-driven
    solid dissolution (S3), bulk carbonate chemistry (S4), Debye-Huckel
    activity corrections (S5), particle-to-bulk flux coupling (S6), and BDF
    time integration with calcium mass-balance diagnostics (S7).

    Parameters
    ----------
    t : array_like
        Time grid used for solver output and interpolation of time-varying
        profiles, in seconds.
    y0 : array_like or None
        Compatibility argument retained for the MIDDoE model-call signature.
        The current script constructs the initial state internally from
        ``tii``, ``tvi``, and the in-line material constants.
    tii : dict
        Time-independent inputs. Required keys include ``"SLR"`` and optional
        ``"t_CO2_inject"`` for the switch from pre-carbonation to CO2 exposure.
    tvi : dict
        Time-varying profiles with ``"T"`` [K] and ``"P"`` [bar] arrays defined
        on ``t``.
    theta : array_like
        Parameter vector ``[kCH, kCSH, D_CH, kLa]`` for C-H dissolution,
        C-S-H decalcification, effective pore diffusivity, and liquid-side CO2
        mass transfer.

    Returns
    -------
    dict
        Model output dictionary with the legacy MIDDoE keys ``"t"``, ``"y"``,
        ``"diameter"``, ``"ddiameter_dt"``, and ``"radial_profiles"``, plus
        diagnostic entries for pH, bulk ionic strength, reaction rates, solver
        statistics, and calcium/carbonation post-processing.

    Notes
    -----
    The outer radial shell is treated as the bulk-connected boundary layer
    described in SM S1.4 and S6.1. Its pore volume is excluded from the
    interior pore inventory, its Ca/OH concentrations are locked to the bulk
    state, and dissolution sources in that shell are added directly to the
    bulk phase.
    """

    import time
    import numpy as np
    from assimulo.problem import Explicit_Problem
    from assimulo.solvers import CVode

    t_start = time.perf_counter()

    # ================================================================
    # Model inputs and operating profiles
    # ================================================================
    t = np.array(t, dtype=np.float64)
    theta = np.array(theta, dtype=np.float64)
    T_profile = np.array(tvi['T'], dtype=np.float64)
    P_profile = np.array(tvi['P'], dtype=np.float64)

    R_gas = 8.314
    t_CO2_inject = tii.get('t_CO2_inject', 100.0)

    # ================================================================
    # Particle size distribution bins (SM S1.1)
    # ================================================================
    n_bins = 5
    # bin_fractions = np.array([0.013372, 0.133512, 0.479668, 0.240563, 0.132886], dtype=np.float64)
    # bin_diameters = np.array([1.102931e-05, 4.919846e-05, 7.176340e-05, 9.765320e-05, 1.291534e-04], dtype=np.float64)

    # # 3-bin PSD with equal mass fractions (each = 1/3)
    # bin_diameters2 = np.array([60.288948, 74.879131, 110.210994], dtype=float)  # um
    # bin_fractions = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)  # [-]
    # 5-bin PSD with equal mass fractions (each = 1/5)
    bin_diameters2 = np.array([56.54, 74.28, 84.73, 96.73, 119.55], dtype=float)  # um
    bin_fractions = np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5], dtype=float)  # [-]
    # # # 10-bin PSD with equal mass fractions (each = 1/10)
    # bin_diameters2 = np.array([31.3172, 66.6383, 74.7904, 81.5791, 87.8899,
    #          94.3885, 101.1522, 109.1284, 119.4332, 153.5589], dtype=float)  # um
    # bin_fractions = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)  # [-]

    # Convert representative bin diameters to SI units for the radial grid.
    bin_diameters = bin_diameters2 * 1e-6  # m

    # bin_diameters_um = np.array([
    #     32.870687, 46.950993, 55.101221, 61.356398, 67.633246,
    #     74.936454, 83.958083, 94.346641, 107.220607, 134.982610
    # ], dtype=np.float64)
    #
    # bin_diameters = bin_diameters_um * 1e-6  # convert um to m
    #
    # bin_fractions = np.array([
    #     0.000952, 0.018875, 0.031827, 0.044932, 0.061274,
    #     0.089064, 0.147493, 0.205444, 0.231624, 0.169515
    # ], dtype=np.float64)

    uncvar = {
        'CH': 0.02, 'CSH': 0.23, 'porosity': 0.3, 'CaCO3_porosity': 0.3,
        'rho_solid_material': 2468, 'rho_solid_CaCO3': 2710,
        'slurry_volume': 0.2, 'LOI': 0.22, 'initCa_Si': 1.0,
    }

    const = {
        'Mw_CaO': 56e-3, 'Mw_CaOH2': 74e-3, 'Mw_CaCO3': 100.1e-3,
        'Mw_CSH': 145e-3, 'Mw_CO2': 44e-3,
        'k_R4f': 1.4e11, 'k_R1f': 2.4e-2, 'k_R2f': 6e9,
        'k_R5f': 8.4e3, 'k_R3f': 1.88e6,
        'T_ref': 298.15,
        'Keq_R1_ref': 4.21e-7, 'Keq_R2_ref': 5000.0,
        'Keq_R3_ref': 10 ** (9.108), 'Keq_R4_ref': 1.077e14,
        'Keq_R5_ref': 4.2e7,
        'deltaH_R1': -20000.0, 'deltaH_R2': 14400.0,
        'deltaH_R3': -12100.0, 'deltaH_R4': 55800.0, 'deltaH_R5': -41400.0,
    }

    Keq_R4_init = float(const['Keq_R4_ref']) * np.exp(
        -float(const['deltaH_R4']) / R_gas * (1.0 / T_profile[0] - 1.0 / float(const['T_ref']))
    )
    Kw_init = 1.0 / Keq_R4_init
    pKw = -np.log10(Kw_init)

    pH_initial = 7
    C_proton_mol_l = 10 ** (-pH_initial)
    C_OH_mol_l = 10 ** (-(pKw - pH_initial))

    # ================================================================
    # Material constants and kinetic parameters (SM S2-S4)
    # ================================================================
    SLR = float(tii['SLR'])
    CH = float(uncvar['CH'])
    CSH = float(uncvar['CSH'])
    LOI = float(uncvar['LOI'])
    initCa_Si = float(uncvar['initCa_Si'])
    porosity = float(uncvar['porosity'])
    slurry_volume = float(uncvar['slurry_volume'])

    rho_solid_material = float(uncvar['rho_solid_material']) * 1e-3
    rho_solid_CaCO3 = float(uncvar['rho_solid_CaCO3']) * 1e-3

    Mw_CaO = float(const['Mw_CaO'])
    Mw_CaCO3 = float(const['Mw_CaCO3'])
    Mw_CO2 = float(const['Mw_CO2'])
    MW_CaOH2 = float(const['Mw_CaOH2'])
    MW_CSH = float(const['Mw_CSH'])

    k_R1f = float(const['k_R1f'])
    k_R2f = float(const['k_R2f'])
    k_R3f = float(const['k_R3f'])
    k_R5f = float(const['k_R5f'])
    T_ref = float(const['T_ref'])

    Keq_R1_ref = float(const['Keq_R1_ref'])
    Keq_R2_ref = float(const['Keq_R2_ref'])
    Keq_R3_ref = float(const['Keq_R3_ref'])
    Keq_R4_ref = float(const['Keq_R4_ref'])
    Keq_R5_ref = float(const['Keq_R5_ref'])

    deltaH_R1 = float(const['deltaH_R1'])
    deltaH_R2 = float(const['deltaH_R2'])
    deltaH_R3 = float(const['deltaH_R3'])
    deltaH_R4 = float(const['deltaH_R4'])
    deltaH_R5 = float(const['deltaH_R5'])

    molar_vol_CaCO3 = Mw_CaCO3 / rho_solid_CaCO3
    total_mass_kg = SLR * slurry_volume
    dry_mass_kg = total_mass_kg * (1.0 - LOI)

    # ================================================================
    # Radial shell discretization (SM S1.4)
    # ================================================================
    Nr = 5
    eps = 1e-50

    # ================================================================
    # Precompute per-bin geometry and inventories (SM S1.1, S1.4, S2.2)
    #   - vol_shells_L: per-particle shell volumes from radial edges.
    #   - vol_pore_shells_L_total: pore volumes aggregated over all particles.
    #   - shells 0..Nr-2 form the conservative interior pore domain.
    #   - shell Nr-1 is the bulk-connected boundary layer excluded from
    #     interior pore mass accounting.
    # ================================================================
    bin_data = []

    for b in range(n_bins):
        db = bin_diameters[b]
        mass_frac = bin_fractions[b]

        V_particle_L = (4.0 / 3.0) * np.pi * (db / 2.0) ** 3 * 1000.0
        V_solid_particle_L = (1.0 - porosity) * V_particle_L
        mass_particle_kg = rho_solid_material * V_solid_particle_L

        total_mass_bin = total_mass_kg * mass_frac
        N_particles_b = total_mass_bin / mass_particle_kg

        CH_solid_mol_b = total_mass_bin * CH / MW_CaOH2
        CSH_Si_mol_b = total_mass_bin * CSH / MW_CSH
        CSH_Ca_mol_b = CSH_Si_mol_b * initCa_Si

        r_particle_b = db / 2.0
        r_edges_b = np.linspace(0.0, r_particle_b, Nr + 1)
        r_centers_b = 0.5 * (r_edges_b[:-1] + r_edges_b[1:])

        v_inner = r_edges_b[:-1] ** 3
        v_outer = r_edges_b[1:] ** 3
        vol_shells_L = (4.0 / 3.0) * np.pi * (v_outer - v_inner) * 1000.0  # per particle
        vol_pore_shells_L_particle = vol_shells_L * porosity
        vol_pore_shells_L_total = vol_pore_shells_L_particle * N_particles_b

        vol_solid_shells_L = vol_shells_L * (1.0 - porosity)
        Si_profile_const_b = CSH_Si_mol_b * (vol_solid_shells_L / np.sum(vol_solid_shells_L))
        Si_profile_const_b = np.maximum(Si_profile_const_b, eps)

        # Precompute face areas for FV fluxes between interior shells
        # Faces between shell i and i+1 are at r_edges[i+1].
        # We'll use center-to-center distances for interior faces (i=0..Nr-3),
        # and a half-cell distance to the bulk-connected layer for the last interface (shell Nr-2 -> bulk).
        r_faces_internal = r_edges_b[1:Nr - 1]  # faces between shells 0..Nr-3 and 1..Nr-2
        A_faces_total_internal = 4.0 * np.pi * (r_faces_internal ** 2) * N_particles_b  # m^2 over all particles
        dr_centers = np.diff(r_centers_b)  # length Nr-1 (center-to-center)

        r_face_bulk = r_edges_b[Nr - 1]  # interface between shell Nr-2 and bulk-layer (shell Nr-1)
        A_face_bulk_total = 4.0 * np.pi * (r_face_bulk ** 2) * N_particles_b
        dr_to_bulk = r_face_bulk - r_centers_b[Nr - 2]  # half-cell distance

        bin_data.append({
            'diameter': db,
            'mass_fraction': mass_frac,
            'N_particles': N_particles_b,
            'CH_solid_mol': CH_solid_mol_b,
            'CSH_Ca_mol': CSH_Ca_mol_b,
            'CSH_Si_mol': CSH_Si_mol_b,
            'Si_profile_const': Si_profile_const_b,
            'r_particle': r_particle_b,
            'r_edges': r_edges_b,
            'r_centers': r_centers_b,
            'vol_shells_L': vol_shells_L,
            'vol_pore_shells_L_total': vol_pore_shells_L_total,
            'A_faces_total_internal': A_faces_total_internal,
            'dr_centers': dr_centers,
            'A_face_bulk_total': A_face_bulk_total,
            'dr_to_bulk': max(dr_to_bulk, 1e-30),
        })

    # ================================================================
    # Thermodynamic and activity-coefficient helpers (SM S3-S5)
    # ================================================================
    def calcGamma_vectorized(I_array, z, a_i, A, B):
        """Return vectorized activity coefficients using SM Eq. S47."""
        I_p = np.clip(I_array, 1e-12, 20.0)

        sqrt_Ip = np.sqrt(I_p)
        denominator = 1.0 + a_i * B * sqrt_Ip
        lnGamma = (
                -A * z ** 2 * sqrt_Ip / denominator +
                (0.2 - 4.17e-5 * I_p) * A * z ** 2 * I_p
        )
        return np.exp(np.clip(lnGamma, np.log(0.01), np.log(10.0)))

    def calc_Ksp_CSH_vectorized1(Ca_Si_array):
        """Alternative C-S-H solubility correlation retained for comparison.

        This inactive helper represents an empirical log-linear C-S-H
        solubility fit over the low-Ca/Si carbonation-gel range.
        """
        Ca_Si = np.atleast_1d(Ca_Si_array)
        Ca_Si = np.clip(Ca_Si, 0.05, 1.6)  # Carbonation gel range

        # Empirical log-linear fit over the retained Ca/Si interval.
        logKsp = -8.2 + 1.4 * Ca_Si
        Ksp = 10 ** logKsp

        return Ksp

    import numpy as np

    def calc_Ksp_CSH_vectorized2(Ca_Si_array):
        """Alternative endpoint interpolation for C-S-H solubility.

        This inactive helper interpolates log10(Ksp) between tobermorite-like
        and jennite-like endmembers at 25 degC. It is retained as a documented
        comparison path, while the active model uses the Bullard-Scherer
        correlation described in SM S3.4.
        """

        Ca_Si = np.atleast_1d(Ca_Si_array).astype(float)

        # Clamp Ca/Si to the range covered by the two endmembers
        # Ca_Si_eff = np.clip(Ca_Si, 0.83, 1.67)

        # Linear fit through (0.83, -8.0) and (1.67, -13.17) from Matschei et al. (2007)
        # Slope and intercept in log10-space:
        b = -6.1547619  # ≈ (-13.17 + 8.0) / (1.67 - 0.83)
        a = -2.8904762  # chosen so that log10Ksp(0.83) = -8.0

        log10_Ksp = a + b * Ca_Si
        Ksp = 10.0 ** log10_Ksp

        return Ksp

    def calc_Ksp_CSH_vectorized(Ca_Si_array):
        """Compute the local C-S-H solubility product from Ca/Si.

        Parameters
        ----------
        Ca_Si_array : array_like
            Local C-S-H calcium-to-silicon molar ratio in each radial shell.

        Returns
        -------
        ndarray
            Composition-dependent C-S-H solubility product.

        Notes
        -----
        Implements the Bullard-Scherer log-linear relation used in SM Eq. S26:
        ``log10(Ksp) = -2.039 - 6.466 * Ca/Si``. The active code clips the
        evaluated ratio to retain numerical continuity for highly decalcified
        shells.
        """
        Ca_Si = np.atleast_1d(Ca_Si_array).astype(float)
        # Maintain continuity for highly decalcified shell states (SM S3.4).
        Ca_Si_eff = np.clip(Ca_Si, 0.05, 2.0)
        log10_Ksp = -2.039 - 6.466 * Ca_Si_eff
        return 10.0 ** log10_Ksp

    def calc_OH_per_CSH_vectorized(Ca_Si_array):
        """Return the fixed OH release factor used for C-S-H decalcification."""
        return np.ones_like(np.atleast_1d(Ca_Si_array)) * 2.05

    # ================================================================
    # Numerical state-vector layout (SM S7)
    # ================================================================
    n_states_per_bin_full = 5 * Nr
    n_bulk_states = 7
    total_n_states_full = n_bins * n_states_per_bin_full + n_bulk_states
    idx_bulk_full = n_bins * n_states_per_bin_full

    n_states_per_bin_red = 4 * Nr
    total_n_states_red = n_bins * n_states_per_bin_red + n_bulk_states
    idx_bulk_red = n_bins * n_states_per_bin_red

    # ================================================================
    # Initial phase inventories and reduced-state projection (SM S2, S7)
    # ================================================================
    y0_full = np.zeros(total_n_states_full, dtype=np.float64)

    for b in range(n_bins):
        vol_shells_L = bin_data[b]['vol_shells_L']
        vol_solid_shells_L = vol_shells_L * (1.0 - porosity)

        CH_profile_b = bin_data[b]['CH_solid_mol'] * (vol_solid_shells_L / np.sum(vol_solid_shells_L))
        CSH_Ca_profile_b = bin_data[b]['CSH_Ca_mol'] * (vol_solid_shells_L / np.sum(vol_solid_shells_L))
        CSH_Si_profile_b = bin_data[b]['Si_profile_const']

        idx_bin_full = b * n_states_per_bin_full
        y0_full[idx_bin_full + 0 * Nr: idx_bin_full + 1 * Nr] = eps
        y0_full[idx_bin_full + 1 * Nr: idx_bin_full + 2 * Nr] = C_OH_mol_l
        y0_full[idx_bin_full + 2 * Nr: idx_bin_full + 3 * Nr] = CH_profile_b
        y0_full[idx_bin_full + 3 * Nr: idx_bin_full + 4 * Nr] = CSH_Ca_profile_b
        y0_full[idx_bin_full + 4 * Nr: idx_bin_full + 5 * Nr] = CSH_Si_profile_b

    y0_full[idx_bulk_full:idx_bulk_full + 7] = [eps, C_OH_mol_l, eps, C_proton_mol_l, eps, eps, eps]

    y0_red = np.zeros(total_n_states_red, dtype=np.float64)
    for b in range(n_bins):
        idx_bin_full = b * n_states_per_bin_full
        idx_bin_red = b * n_states_per_bin_red

        y0_red[idx_bin_red + 0 * Nr: idx_bin_red + 1 * Nr] = y0_full[idx_bin_full + 0 * Nr: idx_bin_full + 1 * Nr]  # Ca
        y0_red[idx_bin_red + 1 * Nr: idx_bin_red + 2 * Nr] = y0_full[idx_bin_full + 1 * Nr: idx_bin_full + 2 * Nr]  # OH
        y0_red[idx_bin_red + 2 * Nr: idx_bin_red + 3 * Nr] = y0_full[idx_bin_full + 2 * Nr: idx_bin_full + 3 * Nr]  # CH
        y0_red[idx_bin_red + 3 * Nr: idx_bin_red + 4 * Nr] = y0_full[
            idx_bin_full + 3 * Nr: idx_bin_full + 4 * Nr]  # Ca_CSH

    y0_red[idx_bulk_red:idx_bulk_red + 7] = y0_full[idx_bulk_full:idx_bulk_full + 7]

    # ================================================================
    # Equilibrium, solubility, and profile interpolation helpers (SM S3-S5)
    # ================================================================
    def vanthoff(K_eq_ref, deltaH, T):
        """Temperature-correct an equilibrium constant using SM Eq. S43."""
        exponent = -deltaH / R_gas * (1.0 / T - 1.0 / T_ref)
        return K_eq_ref * np.exp(np.clip(exponent, -100, 100))

    def ksps2(T):
        """Alternative C-H solubility estimate retained for comparison."""
        deltaH = -16730.0
        deltaS = -157.86
        lnKsp_T = (-deltaH / R_gas) * (1.0 / T) + (deltaS / R_gas)
        return np.exp(np.clip(lnKsp_T, -100, 10))

    def ksps(T):
        """Compute the C-H solubility product from formation Gibbs energies."""
        R = 8.314  # J/mol/K

        # Standard formation Gibbs energies at 298.15 K (SM S3.2).
        Gf_Ca2 = -553540.0
        Gf_OH = -157240.0
        Gf_CH = -897490.0

        # Reaction Gibbs energy for Ca(OH)2(s) -> Ca2+ + 2OH-.
        deltaG = (Gf_Ca2 + 2 * Gf_OH - Gf_CH)

        lnK = -deltaG / (R * T)

        return np.exp(np.clip(lnK, -100, 100))

    def T_at_time(tt):
        """Interpolate the imposed temperature profile at solver time ``tt``."""
        return np.interp(tt, t, T_profile)

    def P_at_time(tt):
        """Interpolate the imposed CO2 pressure profile at solver time ``tt``."""
        return np.interp(tt, t, P_profile)

    def bulk_pH_from_OH(tt, C_OH_bulk):
        """Recover bulk pH from OH- using the water-equilibrium constraint."""
        T_loc = T_at_time(tt)
        Keq_R4_loc = vanthoff(Keq_R4_ref, deltaH_R4, T_loc)
        Kw_loc = 1.0 / max(Keq_R4_loc, 1e-30)
        C_H_from_Kw = Kw_loc / max(C_OH_bulk, 1e-30)
        return -np.log10(max(C_H_from_Kw, 1e-30))

    def calcGamma(I, z, a_i, A, B):
        """Return a scalar Debye-Huckel activity coefficient (SM S5.2)."""
        Ip = float(np.clip(I, 1e-12, 20.0))

        sqrt_Ip = np.sqrt(Ip)
        lnGamma = (
                -A * z ** 2 * sqrt_Ip / (1.0 + a_i * B * sqrt_Ip) +
                (0.2 - 4.17e-5 * Ip) * A * z ** 2 * Ip
        )
        return float(np.exp(np.clip(lnGamma, np.log(0.01), np.log(10.0))))

    def kspCaCO3(T):
        """Return the temperature-dependent CaCO3 solubility product."""
        result = 10.0 ** (-(0.01183 * (T - 273.15) + 8.03))
        return max(result, 1e-15)

    # Extended Debye-Huckel ion parameters (SM S5.2)
    aCa = 6.3e-10
    aOH = 3.5e-10
    aH = 9.0e-11
    aHCO3 = 4.0e-10
    aCO3 = 4.0e-10
    A_DH = 0.0162
    B_DH = 0.508
    r_Ca = 6.3e-10
    r_OH = 3.5e-10
    D_OH_factor = r_Ca / r_OH

    # ================================================================
    # Bulk/interior liquid volume partition (SM S1.4, S6.2)
    # ================================================================
    total_interior_pore_volume = 0.0
    for b in range(n_bins):
        Vp = bin_data[b]['vol_pore_shells_L_total']
        total_interior_pore_volume += float(np.sum(Vp[:-1]))

    bulk_liquid_volume = slurry_volume - total_interior_pore_volume

    # Initial calcium inventory for SM S7 mass-balance verification.
    total_Ca_initial_mol = sum(
        bin_data[b]['CH_solid_mol'] + bin_data[b]['CSH_Ca_mol']
        for b in range(n_bins)
    )

    # ================================================================
    # Calcium mass-balance diagnostic on the full state (SM S7)
    # ================================================================
    def print_mass_balance(tt, y_full_state, phase_name):
        """Print the calcium and carbonate inventories used in SM Eq. S68."""
        C_Ca_bulk = max(y_full_state[idx_bulk_full + 0], 0.0)
        C_OH_bulk = max(y_full_state[idx_bulk_full + 1], 0.0)
        C_HCO3_bulk = max(y_full_state[idx_bulk_full + 2], 0.0)
        C_H_bulk = max(y_full_state[idx_bulk_full + 3], 1e-20)
        C_CO3_bulk = max(y_full_state[idx_bulk_full + 4], 0.0)
        C_CaCO3_bulk = max(y_full_state[idx_bulk_full + 5], 0.0)
        C_CO2_aq = max(y_full_state[idx_bulk_full + 6], 0.0)

        Ca_bulk_mol = C_Ca_bulk * bulk_liquid_volume
        OH_bulk_mol = C_OH_bulk * bulk_liquid_volume

        CO2_aq_mol = C_CO2_aq * bulk_liquid_volume
        HCO3_mol = C_HCO3_bulk * bulk_liquid_volume
        CO3_mol = C_CO3_bulk * bulk_liquid_volume
        CaCO3_mol = C_CaCO3_bulk * bulk_liquid_volume

        Ca_CH_solid_mol = 0.0
        Ca_CSH_solid_mol = 0.0
        Ca_pore_aq_mol = 0.0
        OH_pore_mol = 0.0

        for b in range(n_bins):
            idx_bin = b * n_states_per_bin_full
            Ca_prof = np.maximum(y_full_state[idx_bin + 0 * Nr: idx_bin + 1 * Nr], 0.0)
            OH_prof = np.maximum(y_full_state[idx_bin + 1 * Nr: idx_bin + 2 * Nr], 0.0)

            CH_moles_b = float(np.sum(np.maximum(y_full_state[idx_bin + 2 * Nr: idx_bin + 3 * Nr], 0.0)))
            CSH_Ca_moles_b = float(np.sum(np.maximum(y_full_state[idx_bin + 3 * Nr: idx_bin + 4 * Nr], 0.0)))

            Vp = bin_data[b]['vol_pore_shells_L_total']

            # Interior pores only (exclude last shell)
            Ca_pore_aq_mol += float(np.sum(Ca_prof[:-1] * Vp[:-1]))
            OH_pore_mol += float(np.sum(OH_prof[:-1] * Vp[:-1]))

            Ca_CH_solid_mol += CH_moles_b
            Ca_CSH_solid_mol += CSH_Ca_moles_b

        Ca_total_current = Ca_bulk_mol + Ca_pore_aq_mol + Ca_CH_solid_mol + Ca_CSH_solid_mol + CaCO3_mol
        Ca_error = (Ca_total_current - total_Ca_initial_mol) / total_Ca_initial_mol * 100.0

        OH_total_current = OH_bulk_mol + OH_pore_mol

        CO2_total_mol = CO2_aq_mol + HCO3_mol + CO3_mol + CaCO3_mol
        carbonation_alpha = (CaCO3_mol * Mw_CO2) / (total_Ca_initial_mol * Mw_CaO) * 100.0
        pH_bulk = bulk_pH_from_OH(tt, C_OH_bulk)

        print(f"\n{'=' * 75}")
        print(f"  MASS BALANCE @ t={tt:.2f}s ({phase_name})")
        print(f"{'=' * 75}")

        print("Ca Distribution:")
        print(
            f"  Bulk liquid (aq):   {Ca_bulk_mol:12.6e} mol  ({Ca_bulk_mol / max(Ca_total_current, 1e-30) * 100:5.2f}%)")
        print(
            f"  Pore liquid (aq):   {Ca_pore_aq_mol:12.6e} mol  ({Ca_pore_aq_mol / max(Ca_total_current, 1e-30) * 100:5.2f}%)")
        print(
            f"  CH solid (as Ca):   {Ca_CH_solid_mol:12.6e} mol  ({Ca_CH_solid_mol / max(Ca_total_current, 1e-30) * 100:5.2f}%)")
        print(
            f"  CSH solid (as Ca):  {Ca_CSH_solid_mol:12.6e} mol  ({Ca_CSH_solid_mol / max(Ca_total_current, 1e-30) * 100:5.2f}%)")
        print(f"  CaCO3 precipitated: {CaCO3_mol:12.6e} mol  ({CaCO3_mol / max(Ca_total_current, 1e-30) * 100:5.2f}%)")
        print("  ───────────────────────────────────────────────────────────")
        print(f"  TOTAL Ca current:   {Ca_total_current:12.6e} mol")
        print(f"  TOTAL Ca initial:   {total_Ca_initial_mol:12.6e} mol")
        print(f"  ERROR:              {Ca_error:+.6f}%  {'✓' if abs(Ca_error) < 0.5 else '⚠'}")

        print("\nOH Distribution:")
        print(
            f"  Bulk liquid:        {OH_bulk_mol:12.6e} mol  ({OH_bulk_mol / max(OH_total_current, 1e-30) * 100:5.2f}%)")
        print(
            f"  Particle pores:     {OH_pore_mol:12.6e} mol  ({OH_pore_mol / max(OH_total_current, 1e-30) * 100:5.2f}%)")
        print(f"  TOTAL OH:           {OH_total_current:12.6e} mol")

        print("\nCO2 Species:")
        print(f"  CO2(aq):            {CO2_aq_mol:12.6e} mol  ({CO2_aq_mol / max(CO2_total_mol, 1e-20) * 100:5.2f}%)")
        print(f"  HCO3⁻:              {HCO3_mol:12.6e} mol  ({HCO3_mol / max(CO2_total_mol, 1e-20) * 100:5.2f}%)")
        print(f"  CO3²⁻:              {CO3_mol:12.6e} mol  ({CO3_mol / max(CO2_total_mol, 1e-20) * 100:5.2f}%)")
        print(f"  CaCO3 (solid):      {CaCO3_mol:12.6e} mol  ({CaCO3_mol / max(CO2_total_mol, 1e-20) * 100:5.2f}%)")
        print(f"  TOTAL CO2:          {CO2_total_mol:12.6e} mol")

        print("\nSolution State:")
        print(f"  [Ca²⁺]_bulk:        {C_Ca_bulk:12.6e} mol/L")
        print(f"  [OH⁻]_bulk:         {C_OH_bulk:12.6e} mol/L")
        print(f"  pH_bulk:            {pH_bulk:6.3f}")
        print(f"  Carbonation α:      {carbonation_alpha:6.3f}%")
        print(f"{'=' * 75}\n")

        return Ca_total_current, Ca_error

    # ================================================================
    # Initial condition report (SM S2)
    # ================================================================
    print(f"\n{'#' * 75}")
    print(f"# INITIAL SYSTEM COMPOSITION")
    print(f"{'#' * 75}")

    total_CH_mol = sum(bin_data[b]['CH_solid_mol'] for b in range(n_bins))
    total_CSH_Ca_mol = sum(bin_data[b]['CSH_Ca_mol'] for b in range(n_bins))
    total_CSH_Si_mol = sum(bin_data[b]['CSH_Si_mol'] for b in range(n_bins))

    print("Solid Phase:")
    print(f"  CH (as Ca):         {total_CH_mol:.6e} mol")
    print(f"  CSH (as Ca):        {total_CSH_Ca_mol:.6e} mol")
    print(f"  CSH (as Si):        {total_CSH_Si_mol:.6e} mol")
    print(f"  Ca/Si ratio:        {total_CSH_Ca_mol / max(total_CSH_Si_mol, 1e-30):.3f}")
    print("  ───────────────────────────────────────────────────────────")
    print(f"  TOTAL Ca:           {total_Ca_initial_mol:.6e} mol")

    print("\nLiquid Phase:")
    print(f"  Bulk volume:        {bulk_liquid_volume:.6e} L")
    print(f"  Interior pores:     {total_interior_pore_volume:.6e} L")
    print(f"  Total liquid:       {(bulk_liquid_volume + total_interior_pore_volume):.6e} L")
    print(f"  Initial pH:         {pH_initial:.3f}")

    print("\nDiscretization:")
    print(f"  Nr (radial shells): {Nr}")
    print(f"  PSD bins:           {n_bins}")
    print(f"  Total DOFs (FULL):  {total_n_states_full}")
    print(f"  Total DOFs (ODE):   {total_n_states_red}")
    print(f"{'#' * 75}\n")

    # ================================================================
    # Reduced right-hand side factory (SM S7)
    # ================================================================

    # ================================================================
    # Diagnostic traces for ionic strength and carbonate reaction rates.
    # ================================================================
    I_bulk_history = []
    rates_history = {'R1': [], 'R2': [], 'R3': [], 'R5': []}
    time_history_rhs = []

    def make_ode_function(phase):
        """Create the ODE right-hand side for pre-carbonation or carbonation."""
        def rhs(tt, y):
            """Evaluate coupled pore-domain and bulk-domain derivatives."""
            nonlocal I_bulk_history, rates_history, time_history_rhs
            y_safe = np.maximum(y, 0.0)

            (C_Ca_bulk_mol_l, C_OH_bulk_mol_l,
             C_bicarbonate_mol_l, C_proton_mol_l_state, C_carbonate_mol_l,
             C_CaCO3_mol_l, C_CO2_aq_mol_l) = y_safe[idx_bulk_red:idx_bulk_red + 7]

            C_Ca_bulk_mol_l = max(C_Ca_bulk_mol_l, eps)
            C_OH_bulk_mol_l = max(C_OH_bulk_mol_l, eps)
            C_bicarbonate_mol_l = max(C_bicarbonate_mol_l, eps)
            C_carbonate_mol_l = max(C_carbonate_mol_l, eps)
            C_CO2_aq_mol_l = max(C_CO2_aq_mol_l, eps)

            # Allow CaCO3 inventory to be exactly zero, but never negative.
            C_CaCO3_mol_l = max(C_CaCO3_mol_l, 0.0)

            T_loc = T_at_time(tt)
            P_loc = P_at_time(tt)

            k1 = float(theta[0])
            k2 = float(theta[1])
            D_CH = float(theta[2])
            kLa = float(theta[3])
            # Ambipolar transport is represented with the fitted effective D_CH.
            # D_OH = D_CH * D_OH_factor
            D_OH = D_CH

            Keq_R1 = vanthoff(Keq_R1_ref, deltaH_R1, T_loc)
            Keq_R2 = vanthoff(Keq_R2_ref, deltaH_R2, T_loc)
            Keq_R3 = vanthoff(Keq_R3_ref, deltaH_R3, T_loc)
            Keq_R4 = vanthoff(Keq_R4_ref, deltaH_R4, T_loc)
            Keq_R5 = vanthoff(Keq_R5_ref, deltaH_R5, T_loc)

            Ksp_CaCO3 = kspCaCO3(T_loc)
            Ksp_CH = ksps(T_loc)
            Kw = 1.0 / Keq_R4

            # Algebraic water-equilibrium closure for bulk H+ (SM S7).
            C_proton_bulk = Kw / max(C_OH_bulk_mol_l, 1e-20)
            C_proton_bulk = max(C_proton_bulk, eps)

            I_bulk = 0.5 * (
                    4.0 * C_Ca_bulk_mol_l +
                    1.0 * C_OH_bulk_mol_l +
                    1.0 * C_bicarbonate_mol_l +
                    1.0 * C_proton_bulk +
                    4.0 * C_carbonate_mol_l
            )
            I_bulk = max(I_bulk, 1e-12)

            gamma_Ca_bulk = calcGamma(I_bulk, 2, aCa, A_DH, B_DH)
            gamma_OH_bulk = calcGamma(I_bulk, 1, aOH, A_DH, B_DH)
            gamma_CO3_bulk = calcGamma(I_bulk, 2, aCO3, A_DH, B_DH)
            gamma_H_bulk = calcGamma(I_bulk, 1, aH, A_DH, B_DH)
            gamma_HCO3_bulk = calcGamma(I_bulk, 1, aHCO3, A_DH, B_DH)

            aa_Ca = np.clip(C_Ca_bulk_mol_l * gamma_Ca_bulk, eps, 1e3)
            aa_CO3 = np.clip(C_carbonate_mol_l * gamma_CO3_bulk, eps, 1e3)
            aa_OH = np.clip(C_OH_bulk_mol_l * gamma_OH_bulk, eps, 10.0)
            aa_HCO3 = np.clip(C_bicarbonate_mol_l * gamma_HCO3_bulk, eps, 1e3)
            aa_H = np.clip(C_proton_bulk * gamma_H_bulk, eps, 1.0)
            c_std = 1.0

            if phase == 1:
                r_R1 = r_R2 = r_R3 = r_R5 = 0.0
                dCO2_aq_dt = 0.0
            else:
                # Bulk carbonate reaction network R1-R5 (SM S4.2).
                r_R1_forward = k_R1f * C_CO2_aq_mol_l
                r_R1_reverse = (k_R1f / Keq_R1) * (aa_HCO3 / c_std) * (aa_H / c_std)
                r_R1 = r_R1_forward - r_R1_reverse

                r_R2_forward = k_R2f * (aa_HCO3 / c_std) * (aa_OH / c_std)
                r_R2_reverse = (k_R2f / Keq_R2) * (aa_CO3 / c_std)
                r_R2 = r_R2_forward - r_R2_reverse

                Q = (aa_Ca / c_std) * (aa_CO3 / c_std)
                r_R3 = k_R3f * (Q - Ksp_CaCO3)

                # Prevent reverse precipitation from consuming absent CaCO3.
                if C_CaCO3_mol_l <= 1e-30 and r_R3 < 0.0:
                    r_R3 = 0.0

                r_R5_forward = k_R5f * C_CO2_aq_mol_l * (aa_OH / c_std)
                r_R5_reverse = (k_R5f / Keq_R5) * (aa_HCO3 / c_std)
                r_R5 = r_R5_forward - r_R5_reverse

                CO2_sat = P_loc * 35.0 * np.exp(
                    np.clip(2400.0 * ((1.0 / T_loc) - (1.0 / T_ref)), -20.0, 20.0)
                ) * 1e-3
                dCO2_aq_dt = kLa * (CO2_sat - C_CO2_aq_mol_l) - r_R1 - r_R5

            dy = np.zeros_like(y)

            # Conservative transport contributions to bulk come from FV
            # interface fluxes at the particle/bulk boundary (SM S6.1).
            bulk_flux_Ca_mol_s = 0.0
            bulk_flux_OH_mol_s = 0.0

            # Sources in the bulk-connected boundary shell enter the bulk
            # inventory directly, as described in SM S1.4 and S6.1.
            bulk_source_Ca_mol_s = 0.0
            bulk_source_OH_mol_s = 0.0

            # Track surface-constrained indices so their derivatives remain
            # consistent with the Dirichlet surface condition (SM Eq. S5).
            surf_idx_Ca = []
            surf_idx_OH = []

            for b in range(n_bins):
                idx_bin = b * n_states_per_bin_red

                Ca_prof = np.maximum(y_safe[idx_bin + 0 * Nr: idx_bin + 1 * Nr], eps)
                OH_prof = np.maximum(y_safe[idx_bin + 1 * Nr: idx_bin + 2 * Nr], eps)
                n_CH = np.maximum(y_safe[idx_bin + 2 * Nr: idx_bin + 3 * Nr], 0.0)
                n_CaCSH = np.maximum(y_safe[idx_bin + 3 * Nr: idx_bin + 4 * Nr], 0.0)

                # Bulk-connected layer: enforce the surface concentration
                # continuity condition used in SM Eq. S5.
                Ca_prof[-1] = C_Ca_bulk_mol_l
                OH_prof[-1] = C_OH_bulk_mol_l

                n_SiCSH_const = bin_data[b]['Si_profile_const']
                V_pore = bin_data[b]['vol_pore_shells_L_total']  # (Nr,)
                Vp_safe = np.maximum(V_pore, eps)

                # Local pore-solution equilibria and ionic strength (SM S5.1).
                C_proton_local = Kw / np.maximum(OH_prof, eps)
                I_local = 0.5 * (4.0 * Ca_prof + 1.0 * OH_prof + 1.0 * C_proton_local)
                I_local = np.maximum(I_local, 1e-12)

                gamma_Ca = calcGamma_vectorized(I_local, 2, aCa, A_DH, B_DH)
                gamma_OH = calcGamma_vectorized(I_local, 1, aOH, A_DH, B_DH)

                Ca_Si = n_CaCSH / np.maximum(n_SiCSH_const, eps)
                Ca_Si = np.clip(Ca_Si, 0.05, 1.8)

                Ksp_CSH = calc_Ksp_CSH_vectorized(Ca_Si)
                OH_per_CSH = calc_OH_per_CSH_vectorized(Ca_Si)

                aa_Ca_local = Ca_prof * gamma_Ca
                aa_OH_local = OH_prof * gamma_OH

                S_CH = (aa_Ca_local * (aa_OH_local ** 2)) / max(Ksp_CH, eps)
                S_CSH = (aa_Ca_local * (aa_OH_local ** OH_per_CSH)) / np.maximum(Ksp_CSH, eps)

                r_CH = np.where(
                    (S_CH < 1.0) & (n_CH > 1e-30),
                    k1 * np.clip(1.0 - S_CH, 0.0, 10.0) * n_CH,
                    0.0
                )
                r_CSH = np.where(
                    (S_CSH < 1.0) & (n_CaCSH > 1e-30),
                    k2 * np.clip(1.0 - S_CSH, 0.0, 10.0) * n_CaCSH,
                    0.0
                )

                # =========================
                # Finite-volume radial diffusion (SM S1.2, S7)
                # =========================
                dCa_dt = np.zeros(Nr)
                dOH_dt = np.zeros(Nr)

                # Interior faces between shells i and i+1 (i=0..Nr-3).
                # Flux sign convention: F > 0 means outward toward larger r.
                A_faces = bin_data[b]['A_faces_total_internal']  # faces at r_edges[1..Nr-2], length Nr-2
                dr_cent = bin_data[b]['dr_centers']  # length Nr-1
                # dr_cent[i] is the distance between shell centers i and i+1.

                # Face k lies between shells k and k+1 at r_edges[k+1].
                if Nr >= 3:
                    dC_Ca = (Ca_prof[1:Nr - 1] - Ca_prof[0:Nr - 2])  # between 0..Nr-2
                    dC_OH = (OH_prof[1:Nr - 1] - OH_prof[0:Nr - 2])
                    dr_face = dr_cent[0:Nr - 2]  # distances for pairs 0-1,1-2,...,(Nr-3)-(Nr-2)

                    grad_Ca = 1000.0 * (dC_Ca / np.maximum(dr_face, 1e-30))
                    grad_OH = 1000.0 * (dC_OH / np.maximum(dr_face, 1e-30))

                    F_Ca_internal = -D_CH * porosity * A_faces * grad_Ca  # mol/s across all particles
                    F_OH_internal = -D_OH * porosity * A_faces * grad_OH

                else:
                    F_Ca_internal = np.zeros(0)
                    F_OH_internal = np.zeros(0)

                # Boundary flux from shell Nr-2 to the bulk-connected layer
                # implements SM Eq. S51.
                A_bulk = bin_data[b]['A_face_bulk_total']
                dr_bulk = bin_data[b]['dr_to_bulk']
                grad_Ca_bulk = 1000.0 * (C_Ca_bulk_mol_l - Ca_prof[Nr - 2]) / dr_bulk
                grad_OH_bulk = 1000.0 * (C_OH_bulk_mol_l - OH_prof[Nr - 2]) / dr_bulk

                F_Ca_bulk = -D_CH * porosity * A_bulk * grad_Ca_bulk
                F_OH_bulk = -D_OH * porosity * A_bulk * grad_OH_bulk

                # Update only the interior pore-shell concentrations.
                for i in range(Nr - 1):
                    F_in_Ca = 0.0 if i == 0 else float(F_Ca_internal[i - 1])
                    F_out_Ca = float(F_Ca_bulk) if i == (Nr - 2) else float(F_Ca_internal[i])
                    dCa_dt[i] += (F_in_Ca - F_out_Ca) / max(V_pore[i], eps)

                    F_in_OH = 0.0 if i == 0 else float(F_OH_internal[i - 1])
                    F_out_OH = float(F_OH_bulk) if i == (Nr - 2) else float(F_OH_internal[i])
                    dOH_dt[i] += (F_in_OH - F_out_OH) / max(V_pore[i], eps)

                # =========================
                # Solid dissolution source terms (SM S3)
                # =========================
                src_Ca = (r_CH + r_CSH) / Vp_safe
                src_OH = (2.0 * r_CH + OH_per_CSH * r_CSH) / Vp_safe

                # Add sources into the conservative interior pore shells.
                dCa_dt[0:Nr - 1] += src_Ca[0:Nr - 1]
                dOH_dt[0:Nr - 1] += src_OH[0:Nr - 1]

                # Sources in the bulk-connected shell are already in the bulk
                # control volume.
                bulk_source_Ca_mol_s += float(r_CH[-1] + r_CSH[-1])
                bulk_source_OH_mol_s += float(2.0 * r_CH[-1] + OH_per_CSH[-1] * r_CSH[-1])

                # C-H/C-S-H solid inventories are tracked in every shell.
                dy[idx_bin + 2 * Nr: idx_bin + 3 * Nr] = -r_CH
                dy[idx_bin + 3 * Nr: idx_bin + 4 * Nr] = -r_CSH

                # Pore concentration derivatives.
                dy[idx_bin + 0 * Nr: idx_bin + 1 * Nr] = dCa_dt
                dy[idx_bin + 1 * Nr: idx_bin + 2 * Nr] = dOH_dt

                # Track boundary-shell entries to keep them aligned with bulk.
                surf_idx_Ca.append(idx_bin + 0 * Nr + (Nr - 1))
                surf_idx_OH.append(idx_bin + 1 * Nr + (Nr - 1))

                # Outward positive boundary flux increases the bulk inventory.
                bulk_flux_Ca_mol_s += float(F_Ca_bulk)
                bulk_flux_OH_mol_s += float(F_OH_bulk)

            # Bulk phase mass balances (SM S6.2), in mol/L/s.
            dC_Ca_bulk_dt = (bulk_flux_Ca_mol_s + bulk_source_Ca_mol_s) / bulk_liquid_volume - r_R3
            dC_OH_bulk_dt = (bulk_flux_OH_mol_s + bulk_source_OH_mol_s) / bulk_liquid_volume - r_R5 - r_R2 - r_R1
            dC_HCO3_dt = r_R1 + r_R5 - r_R2
            dC_CO3_dt = r_R2 - r_R3
            dC_CaCO3_dt = r_R3
            dC_H_dt = -(C_proton_bulk / max(C_OH_bulk_mol_l, 1e-20)) * dC_OH_bulk_dt

            dy[idx_bulk_red + 0] = dC_Ca_bulk_dt
            dy[idx_bulk_red + 1] = dC_OH_bulk_dt
            dy[idx_bulk_red + 2] = dC_HCO3_dt
            dy[idx_bulk_red + 3] = dC_H_dt
            dy[idx_bulk_red + 4] = dC_CO3_dt
            dy[idx_bulk_red + 5] = dC_CaCO3_dt
            dy[idx_bulk_red + 6] = dCO2_aq_dt

            # Keep the bulk-connected layer locked to bulk in time; output is
            # forced exactly in expand_red_to_full().
            for idx_s in surf_idx_Ca:
                dy[idx_s] = dC_Ca_bulk_dt
            for idx_s in surf_idx_OH:
                dy[idx_s] = dC_OH_bulk_dt

            # Store diagnostic traces for output interpolation.
            I_bulk_history.append(float(I_bulk))
            rates_history['R1'].append(float(r_R1))
            rates_history['R2'].append(float(r_R2))
            rates_history['R3'].append(float(r_R3))
            rates_history['R5'].append(float(r_R5))
            time_history_rhs.append(float(tt))

            return np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)

        return rhs

    # ================================================================
    # Expand reduced state history to the legacy full output layout.
    # Surface Ca/OH entries are forced to bulk values to preserve SM Eq. S5.
    # ================================================================
    def expand_red_to_full(y_red_matrix):
        """Map the reduced ODE state back to the full diagnostic state layout."""
        n_time = y_red_matrix.shape[0]
        y_full_matrix = np.zeros((n_time, total_n_states_full), dtype=np.float64)

        for b in range(n_bins):
            idx_bin_red = b * n_states_per_bin_red
            idx_bin_full = b * n_states_per_bin_full

            y_full_matrix[:, idx_bin_full + 0 * Nr: idx_bin_full + 1 * Nr] = y_red_matrix[
                :, idx_bin_red + 0 * Nr: idx_bin_red + 1 * Nr]  # Ca
            y_full_matrix[:, idx_bin_full + 1 * Nr: idx_bin_full + 2 * Nr] = y_red_matrix[
                :, idx_bin_red + 1 * Nr: idx_bin_red + 2 * Nr]  # OH
            y_full_matrix[:, idx_bin_full + 2 * Nr: idx_bin_full + 3 * Nr] = y_red_matrix[
                :, idx_bin_red + 2 * Nr: idx_bin_red + 3 * Nr]  # CH
            y_full_matrix[:, idx_bin_full + 3 * Nr: idx_bin_full + 4 * Nr] = y_red_matrix[
                :, idx_bin_red + 3 * Nr: idx_bin_red + 4 * Nr]  # CaCSH
            y_full_matrix[:, idx_bin_full + 4 * Nr: idx_bin_full + 5 * Nr] = bin_data[b][
                'Si_profile_const']  # SiCSH const

        y_full_matrix[:, idx_bulk_full:idx_bulk_full + 7] = y_red_matrix[:, idx_bulk_red:idx_bulk_red + 7]

        # Enforce exact surface=bulk in outputs for each PSD bin.
        for b in range(n_bins):
            idx_bin_full = b * n_states_per_bin_full
            y_full_matrix[:, idx_bin_full + (Nr - 1)] = y_full_matrix[:, idx_bulk_full + 0]  # Ca_surf = Ca_bulk
            y_full_matrix[:, idx_bin_full + Nr + (Nr - 1)] = y_full_matrix[:, idx_bulk_full + 1]  # OH_surf = OH_bulk

        return y_full_matrix

    # ================================================================
    # BDF solver controls (SM S7)
    # ================================================================
    rtol_phase1, atol_phase1 = 1e-3, 1e-9
    rtol_phase2, atol_phase2 = 1e-5, 1e-12

    # ================================================================
    # Phase 1: pre-carbonation hydration/leaching
    # ================================================================
    print(">>> Starting Phase 1 (Hydration, no CO2)...")
    problem1 = Explicit_Problem(make_ode_function(phase=1), y0_red, t0=0.0)
    sim1 = CVode(problem1)
    sim1.discr = 'BDF'
    sim1.iter = 'Newton'
    if hasattr(sim1, "linearsolver"):
        sim1.linearsolver = 'DENSE'
    sim1.maxsteps = 1000
    sim1.maxord = 3
    sim1.rtol = rtol_phase1
    sim1.atol = atol_phase1
    sim1.verbosity = 0

    t_eval_phase1 = t[t <= t_CO2_inject]
    ncp1 = max(1, len(t_eval_phase1) - 1)
    t_phase1, y_phase1_red = sim1.simulate(tfinal=t_CO2_inject, ncp=ncp1)
    nfev_phase1 = sim1.get_statistics()['nfcns']

    y_phase1_full = expand_red_to_full(y_phase1_red)
    print(f"✓ Phase 1 complete: {len(t_phase1)} points, {nfev_phase1} evaluations")
    print_mass_balance(t_phase1[-1], y_phase1_full[-1, :], "Phase 1 End")

    # ================================================================
    # Phase 2: carbonation after CO2 injection
    # ================================================================
    print("\n>>> Starting Phase 2 (Carbonation with CO2)...")
    y0_phase2_red = y_phase1_red[-1, :]

    problem2 = Explicit_Problem(make_ode_function(phase=2), y0_phase2_red, t0=t_CO2_inject)
    sim2 = CVode(problem2)
    sim2.discr = 'BDF'
    sim2.iter = 'Newton'
    if hasattr(sim2, "linearsolver"):
        sim2.linearsolver = 'DENSE'
    sim2.maxsteps = 1000
    sim2.maxord = 3
    sim2.rtol = rtol_phase2
    sim2.atol = atol_phase2
    sim2.verbosity = 0

    t_eval_phase2 = t[t > t_CO2_inject]
    ncp2 = max(1, len(t_eval_phase2) - 1)
    t_phase2, y_phase2_red = sim2.simulate(tfinal=t[-1], ncp=ncp2)
    nfev_phase2 = sim2.get_statistics()['nfcns']

    y_phase2_full = expand_red_to_full(y_phase2_red)
    print(f"✓ Phase 2 complete: {len(t_phase2)} points, {nfev_phase2} evaluations")

    phase2_duration = t[-1] - t_CO2_inject
    for frac, name in zip([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"]):
        t_check = t_CO2_inject + frac * phase2_duration
        idx = int(np.argmin(np.abs(t_phase2 - t_check)))
        print_mass_balance(t_phase2[idx], y_phase2_full[idx, :], f"Phase 2 @ {name}")

    # ================================================================
    # Combine phase trajectories
    # ================================================================
    t_combined = np.concatenate([t_phase1, t_phase2[1:]])
    y_combined_full = np.vstack([y_phase1_full, y_phase2_full[1:, :]])

    t_total = time.perf_counter() - t_start

    class CombinedSolution:
        """Small solution adapter preserving the previous solver-like API."""
        def __init__(self, nfev1, nfev2):
            self.t = t_combined
            self.y = y_combined_full.T  # FULL layout
            self.success = True
            self.message = "Completed"
            self.nfev = int(nfev1 + nfev2)
            self.njev = 0
            self.nlu = 0

    sol = CombinedSolution(nfev_phase1, nfev_phase2)

    print(f"\n✓ Total solve time: {t_total:.2f}s | Nr={Nr} | nfev={sol.nfev}\n")

    # ================================================================
    # Post-processing outputs using the legacy MIDDoE interface
    # ================================================================
    y_output = np.zeros((5 * n_bins + 7, len(sol.t)))

    for b in range(n_bins):
        idx_bin_base = b * n_states_per_bin_full
        y_output[5 * b + 0, :] = sol.y[idx_bin_base + (Nr - 1), :]  # Ca surface (forced to bulk)
        y_output[5 * b + 1, :] = sol.y[idx_bin_base + Nr + (Nr - 1), :]  # OH surface (forced to bulk)
        y_output[5 * b + 2, :] = sol.y[idx_bin_base + 2 * Nr, :]  # CH (shell 0)
        y_output[5 * b + 3, :] = sol.y[idx_bin_base + 3 * Nr, :]  # CaCSH (shell 0)
        y_output[5 * b + 4, :] = sol.y[idx_bin_base + 4 * Nr, :]  # SiCSH (shell 0 const)

    y_output[5 * n_bins + 0, :] = sol.y[idx_bulk_full + 0, :]
    y_output[5 * n_bins + 1, :] = sol.y[idx_bulk_full + 1, :]
    y_output[5 * n_bins + 2, :] = sol.y[idx_bulk_full + 2, :]
    y_output[5 * n_bins + 3, :] = sol.y[idx_bulk_full + 3, :]
    y_output[5 * n_bins + 4, :] = sol.y[idx_bulk_full + 4, :]
    y_output[5 * n_bins + 5, :] = sol.y[idx_bulk_full + 5, :]
    y_output[5 * n_bins + 6, :] = sol.y[idx_bulk_full + 6, :]

    diameter_series = np.tile(bin_diameters, (len(sol.t), 1)).T
    ddiameter_dt_series = np.zeros_like(diameter_series)

    # Radial profiles for particle-domain diagnostics (SM S1-S3)
    radial_profiles = {}
    shell_masses_per_bin = {}

    for b in range(n_bins):
        idx_bin_base = b * n_states_per_bin_full

        CH_moles = sol.y[idx_bin_base + 2 * Nr: idx_bin_base + 3 * Nr, :]
        Ca_CSH_moles = sol.y[idx_bin_base + 3 * Nr: idx_bin_base + 4 * Nr, :]
        Si_CSH_moles = sol.y[idx_bin_base + 4 * Nr: idx_bin_base + 5 * Nr, :]

        Ca_Si_profile = Ca_CSH_moles / np.maximum(Si_CSH_moles, 1e-30)

        vol_shells_L = bin_data[b]['vol_shells_L']
        vol_solid_shells_L = vol_shells_L * (1.0 - porosity)
        mass_solid_shells_kg = vol_solid_shells_L * rho_solid_material
        shell_masses_per_bin[b] = mass_solid_shells_kg
        total_mass_solid_shells_kg = mass_solid_shells_kg * bin_data[b]['N_particles']

        CH_conc = CH_moles / total_mass_solid_shells_kg[:, np.newaxis]
        CH_percent = portlandite_percent_from_mol_per_kg(CH_conc)
        CSH_conc = Si_CSH_moles / total_mass_solid_shells_kg[:, np.newaxis]

        radial_profiles[f'bin_{b}'] = {
            'diameter': bin_diameters[b],
            'Ca_profile': sol.y[idx_bin_base + 0 * Nr: idx_bin_base + 1 * Nr, :],
            'OH_profile': sol.y[idx_bin_base + 1 * Nr: idx_bin_base + 2 * Nr, :],
            'CH_moles': CH_moles,
            'CH_profile': CH_conc,
            'CH_percent': CH_percent,
            'CSH_profile': CSH_conc,
            'Ca_Si_profile': Ca_Si_profile,
            'Ca_CSH_moles': Ca_CSH_moles,
            'Si_CSH_moles': Si_CSH_moles,
        }

    bulk_csh_metrics = calculate_bulk_csh_metrics(
        radial_profiles,
        shell_masses_per_bin,
        n_bins,
        bin_data,
        SI_GEL_CA_SI_THRESHOLD,
    )

    total_CaCO3_mol = sol.y[idx_bulk_full + 5, :] * bulk_liquid_volume
    Ca_unreacted_sum_mol = total_Ca_initial_mol - total_CaCO3_mol
    X = (total_CaCO3_mol * Mw_CO2) * 100 / (dry_mass_kg * 11.66)

    # ================================================================
    # Reconstruct pH from OH- using the water-equilibrium constraint (SM S7)
    # ================================================================
    # Extract bulk OH- concentration over time.
    OH_bulk = np.maximum(y_output[5 * n_bins + 1, :], 1e-30)

    # Evaluate T(t) on the solver output grid.
    T_model = np.interp(sol.t, t, T_profile)

    # Compute Kw(T) using the van't Hoff correction.
    Keq_R4_t = Keq_R4_ref * np.exp(-deltaH_R4 / R_gas * (1.0 / T_model - 1.0 / T_ref))
    Kw_t = 1.0 / Keq_R4_t

    # Compute H+ from water equilibrium: [H+] = Kw / [OH-].
    H_from_Kw = Kw_t / OH_bulk

    # Compute pH = -log10([H+]).
    pH_calc = -np.log10(np.maximum(H_from_Kw, 1e-30))

    # ================================================================
    # Interpolate ionic strength and reaction-rate diagnostics to output times.
    # ================================================================
    time_rhs_array = np.array(time_history_rhs)
    I_bulk_array = np.array(I_bulk_history)

    # Interpolate to match sol.t.
    I_bulk_output = np.interp(sol.t, time_rhs_array, I_bulk_array)

    rates_output = {}
    for key in ['R1', 'R2', 'R3', 'R5']:
        rates_array = np.array(rates_history[key])
        rates_output[key] = np.interp(sol.t, time_rhs_array, rates_array)

    result = {
        't': sol.t,
        'y': y_output,
        'diameter': diameter_series,
        'ddiameter_dt': ddiameter_dt_series,
        'radial_profiles': radial_profiles,
        'Ca_unreacted_sum_mol': Ca_unreacted_sum_mol,
        'X': X,
        'sol_full': sol,
        'Nr': Nr,
        'n_bins': n_bins,
        'bin_data': bin_data,
        'shell_masses_per_bin': shell_masses_per_bin,
        'slurry_volume_l': slurry_volume,
        'dry_mass_kg': dry_mass_kg,
        'pH': pH_calc,
        'I_bulk': I_bulk_output,  # Ionic strength in bulk over time
        'rates': rates_output,  # Dictionary of reaction rates over time (R1, R2, R3, R5)
    }

    result.update(bulk_csh_metrics)
    return result



def run_model():
    """Run the SM 6.0 ionic carbonation example with profiling enabled.

    This entry point defines the demonstration time grid, operating profiles,
    fixed time-independent inputs, and fitted parameter vector used to call
    :func:`solve_model`. It is intended as a reproducible script-level example,
    not as part of the MIDDoE public API.
    """
    # ------------------------------------------------------------
    # Example operating conditions and parameter vector
    # ------------------------------------------------------------
    t = np.linspace(0, 10900, 300)
    T_profile = np.full_like(t, 298.15, dtype=float)
    P_profile = np.full_like(t, 0.15, dtype=float)

    tii = {
        'SLR': 0.1,
        't_CO2_inject': 100,
    }


    theta = [
        1.041123e-01,  # kCH
        6.298146e-04,  # kCSH
        6.207170e-11,  # D
        9.732595e-03,  # kLa
    ]

    tvi = {'T': T_profile, 'P': P_profile}

    # ------------------------------------------------------------
    # Execute model with cumulative profiler output
    # ------------------------------------------------------------
    profiler = cProfile.Profile()
    profiler.enable()

    result = solve_model(t, None, tii, tvi, theta)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(15)

    print(f"\nnfev: {result['sol_full'].nfev}")
    print(f"nlu: {result['sol_full'].nlu}")


if __name__ == "__main__":
    run_model()



