import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import matplotlib as mpl


def plot_separate_reactions_from_excel_all_black(file_path):
    mpl.rcParams['font.family'] = 'Gill Sans MT'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Gill Sans MT'
    mpl.rcParams['mathtext.it'] = 'Gill Sans MT:italic'

    df = pd.read_excel(file_path)
    plt.rcParams['font.family'] = 'Gill Sans MT'

    x_min, x_max = df['reaction time'].min(), df['reaction time'].max()
    x_margin = 0.00 * (x_max - x_min)
    x_lim = (x_min - x_margin, x_max + x_margin)

    # Plot for RCF all black
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    y1_min = min(df['RCF-dp0'].min(), df['RCF-T'].min(), df['RCF-PCO2'].min())
    y1_max = max(df['RCF-dp0'].max(), df['RCF-T'].max(), df['RCF-PCO2'].max())
    y1_margin = 0.05 * (y1_max - y1_min)
    y1_lim = (y1_min - y1_margin, y1_max + y1_margin)

    line_rcf_dp0, = ax1.plot(df['reaction time'], df['RCF-dp0'], 'k-', label=r'$d_{p0}$', linewidth=2.5)
    line_rcf_T, = ax1.plot(df['reaction time'], df['RCF-T'], 'k--', label=r'$T$', linewidth=2.5)
    line_rcf_pco2, = ax1.plot(df['reaction time'], df['RCF-PCO2'], 'k:', label=r'$P_{CO_{2}}$', linewidth=2.5)

    ax1.set_xlabel(r'Time ($t$) [min]', fontsize=16)
    ax1.set_ylabel(r"Total Sobol Index - RCF's carbonation controls", fontsize=16)
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y1_lim)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    ax1.set_xticks([0, 60, 120, 180])
    ax1.xaxis.set_minor_locator(ticker.NullLocator())
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax1.tick_params(axis='y', which='minor', length=4, color='gray')

    line_rcf_T.set_dashes([2, 2, 2])

    lines1 = [line_rcf_dp0, line_rcf_T, line_rcf_pco2]
    labels1 = [line.get_label() for line in lines1]
    ax1.legend(lines1, labels1, loc='best', frameon=False, facecolor='white', fontsize=16)

    base_path, _ = os.path.splitext(file_path)
    svg_path1 = base_path + "_RCF_all_blackgill.svg"
    fig1.savefig(svg_path1, format='svg')
    print(f"RCF plot saved as SVG: {svg_path1}")

    plt.tight_layout()
    plt.show()

    # Plot for RCP all black
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    y2_min = min(df['RCP-dp0'].min(), df['RCP-T'].min(), df['RCP-PCO2'].min())
    y2_max = max(df['RCP-dp0'].max(), df['RCP-T'].max(), df['RCP-PCO2'].max())
    y2_margin = 0.05 * (y2_max - y2_min)
    y2_lim = (y2_min - y2_margin, y2_max + y2_margin)

    line_rcp_dp0, = ax2.plot(df['reaction time'], df['RCP-dp0'], 'k-', label=r'$d_{p0}$', linewidth=2.5)
    line_rcp_T, = ax2.plot(df['reaction time'], df['RCP-T'], 'k--', label=r'$T$', linewidth=2.5)
    line_rcp_pco2, = ax2.plot(df['reaction time'], df['RCP-PCO2'], 'k:', label=r'$P_{CO_{2}}$', linewidth=2.5)

    ax2.set_xlabel(r'Time ($t$) [min]', fontsize=16)
    ax2.set_ylabel(r"Total Sobol Index - RCP's carbonation controls", fontsize=16)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y2_lim)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    ax2.set_xticks([0, 60, 120, 180])
    ax2.xaxis.set_minor_locator(ticker.NullLocator())
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax2.tick_params(axis='y', which='minor', length=4, color='gray')

    line_rcp_T.set_dashes([2, 2, 2])

    lines2 = [line_rcp_dp0, line_rcp_T, line_rcp_pco2]
    labels2 = [line.get_label() for line in lines2]
    ax2.legend(lines2, labels2, loc='best', frameon=False, facecolor='white', fontsize=16)

    svg_path2 = base_path + "_RCP_all_blackgill.svg"
    fig2.savefig(svg_path2, format='svg')
    print(f"RCP plot saved as SVG: {svg_path2}")

    plt.tight_layout()
    plt.show()


# Example usage:
plot_separate_reactions_from_excel_all_black(r"C:\Users\bolosey94542\Desktop\to transfer\CVAL\reports\papers\rcf wet\sens var\sensa.xlsx")
