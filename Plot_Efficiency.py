#!/usr/bin/env python3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import glob
import numpy as np
from astropy.io import fits
import os
import re
import sys
import astropy.table as at
# from astLib import astCoords
import astropy.coordinates as coord
from astropy import units as u
import time
from astropy.table import Table
import csv
import logging
from collections import OrderedDict
from scipy.special import erf, erfinv
from scipy.optimize import minimize
from astropy.stats import binom_conf_interval
from lmfit import Parameters
from lmfit import minimize as lm_minimize
import copy
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings('ignore')

def chi_square_s_curve(params, mag_bins=None, obs_efficiency=None):
        x0, a, n = params
        model_efficiency = self.limiting_mag_model(mag_bins, x0, a, n)
        chi_square = np.sum(((model_efficiency - obs_efficiency)) ** 2.0)
        return chi_square

def chi_square_s_curve2(params, mag_bins, obs_efficiency):
        x0 = params['x0']
        a = params['a']
        n = params['n']

        model_efficiency = n * (1.0 - (erf(a * (mag_bins - x0)) + 1.0) / 2.0)
        residual = (model_efficiency - obs_efficiency)**2.0

        return residual

def compute_efficiency(data, model_mags, perform_fit=True):

    mag_cent = []
    mag_effi = []
    for i in np.arange(len(model_mags)-1):
            mag_lo = model_mags[i]
            mag_hi = model_mags[i+1]
            mag_cent.append((mag_lo+mag_hi)/2.0)
            mask = (data['sim_mag']<mag_hi) & (data['sim_mag']>mag_lo)

            total = len(data[mask])
            if total==0:
                mag_effi.append(np.nan)
            else:
                detected = 0
                for row in data[mask]:
                    if row['snr']>3.0:
                        detected += 1
                mag_effi.append(1.0*detected/total)

            print(mag_lo, mag_hi, mag_effi[i])

    params = Parameters()
    params.add('x0', value=21.5, min=10.0, max=25.0)
    params.add('a', value=1.0, min=0.0, max=np.inf)
    params.add('n', value=0.9, min=0.0, max=1.0)

    unweighted_minimize_result = lm_minimize(chi_square_s_curve2,
            params, args=(mag_cent, mag_effi), nan_policy='omit')

    if unweighted_minimize_result:
        print("UNWEIGHTED lm fit params:\n\tx0: %0.3f +/- %0.3f\n\ta: %0.3f +/- %0.3f\n\tn: %0.3f +/- %0.3f" %
                      (unweighted_minimize_result.params['x0'].value,
                       unweighted_minimize_result.params['x0'].stderr,
                       unweighted_minimize_result.params['a'].value,
                       unweighted_minimize_result.params['a'].stderr,
                       unweighted_minimize_result.params['n'].value,
                       unweighted_minimize_result.params['n'].stderr))


    return mag_cent, mag_effi, unweighted_minimize_result

def limiting_mag_model(input_mags, x0, a, n):

    y = n * (1.0 - (erf(a * (input_mags - x0)) + 1.0) / 2.0)
    return y


def crossmatch_fake_stars(fakemag_file, dcmp_files, masking=True):

    injected_data = Table.read(fakemag_file, format='ascii')

    outdata = Table([[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],
        [0.],[0],['X'*20],['X'*20],['X'*100]],
        names=('x','y','sim_mag','det_mag','det_magerr','snr',
            'sim_flux','sim_psf_flux','sim_zpt',
            'det_flux','det_fluxerr','det_zpt','extendedness','Nmask','mask',
            'flag','image')).copy()[:0]

    for i,file in enumerate(dcmp_files):

        print(f'File {i}')

        hdu = fits.open(file, mode='readonly')
        print('Got header')
        test=np.loadtxt(file, skiprows=1, dtype=str)
        dcmp_data = Table(test)
        print(f'Loaded file {file}')

        image = hdu[0].header['IMNAME']
        zptmag = float(hdu[0].header['ZPTMAGAV'])

        submask = np.array([image in f for f in injected_data['imagefile']])
        subdata = injected_data[submask]

        for row in dcmp_data:
            crossmatch = (subdata['x']-float(row['col0']))**2+\
                (subdata['y']-float(row['col1']))**2<1.0
            if not len(subdata[crossmatch])==1:
                print(f'ERROR WITH CROSSMATCH {file}')
                x=float(row['col1']) ; y=float(row['col2'])
                n=len(subdata[crossmatch])
                print(f'{image}')
                print(f'x={x}, y={y}')
                print(f'n={n}')
                print(subdata[crossmatch]['x','y','mag'])
                raise Exception('STOP!')

            data = subdata[crossmatch][0]

            sim_mag = float('%2.3f'%float(data['mag']))
            sim_flux = float('%6.3f'%float(data['flux']))
            sim_psf_flux = float('%6.3f'%float(data['psf_flux']))
            sim_zpt = float('%2.3f'%float(data['zpt']))
            det_mag = float('%2.3f'%(float(row['col2'])+zptmag))
            if float(row['col5'])!=0.0:
                snr = float('%5.4f'%(float(row['col4'])/float(row['col5'])))
            else:
                snr = 0.0

            if masking:
                if np.isnan(float(row['col3'])):
                    continue
                if np.isinf(float(row['col3'])):
                    continue
                if float(row['col3'])==0.0:
                    continue

            print([float(row['col0']), float(row['col1']), sim_mag,
                det_mag,float(row['col3']),
                snr,
                sim_flux,sim_psf_flux,sim_zpt,
                float(row['col4']),float(row['col5']),
                zptmag,float(row['col18']),int(row['col21']),
                str(row['col20']),str(row['col19']),file])

            outdata.add_row([float(row['col0']), float(row['col1']), sim_mag,
                det_mag,float(row['col3']),
                snr,
                sim_flux,sim_psf_flux,sim_zpt,
                float(row['col4']),float(row['col5']),
                zptmag,float(row['col18']),int(row['col21']),
                str(row['col20']),str(row['col19']),file])

    if masking:
        for file in np.unique(outdata['image']):
            mask = outdata['image']==file
            subdata = outdata[mask]
            submask = subdata['snr']>3.0
            if len(subdata[submask])==0:
                outdata = outdata[~mask]

    return(outdata)

def calculate_and_plot_efficiency(base_path, uniform_subdirs, snr, outimg,
    outdatafile, dcmp_type="fake.dcmp", bright=19.0, dim=24.0, bin_size=0.2,
    eff_target=0.5):

    model_mags = bright + bin_size*np.arange(int((dim-bright)/bin_size)+1)

    for sd in uniform_subdirs:

        gl = "%s/%s/%s" % (base_path, sd, "*.%s" % dcmp_type)
        dcmp_files = glob.glob(gl)
        print("Processing sub dir: %s...\n%s dcmp files" % (sd,len(dcmp_files)))
        print("Glob: ",gl)

        basedir = sd.replace('_tmpl','')
        logdir = base_path.replace('workspace','logs')

        fakemag_file = os.path.join(logdir, basedir, 'fakemags.txt')
        print(f'{fakemag_file}')
        if not os.path.exists(fakemag_file):
            print(f'ERROR: {fakemag_file} does not exist')
            raise Exception('STOP!')

        outdata = crossmatch_fake_stars(fakemag_file, dcmp_files)
        outdata.write(outdatafile, overwrite=True, format='ascii')

        mag_cent, mag_effi, unweighted_minimize_result=compute_efficiency(
            outdata, model_mags)

        params = (unweighted_minimize_result.params['x0'].value,
                  unweighted_minimize_result.params['a'].value,
                  unweighted_minimize_result.params['n'].value
            )

        def efficiency(params, mag_bins):
            x0, a, n = params
            model_efficiency = n*(1.0-(erf(a*(mag_bins-x0))+1.0)/2.0)
            return(model_efficiency)

        # Given a set of params, compute magnitude where we get input efficiency
        def compute_mag_limit(params, eff_target):
            x0, a, n = params
            mag_limit = x0 + erfinv(1.0 - 2.0*eff_target)/a
            return(mag_limit)

        fig, ax = plt.subplots()

        plot_mags = np.linspace(bright, dim, 4000)

        ax.bar(mag_cent, mag_effi, bin_size, zorder=1, edgecolor='black')
        model_eff = efficiency(params, plot_mags)
        ax.plot(plot_mags, model_eff, zorder=10, color='red')

        x0=compute_mag_limit(params, eff_target)

        ax.text(dim-1.5, 0.95, r'$m_{\rm limit}$='+'%2.2f'%float(x0))
        ax.text(dim-1.5, 0.9, f'SNR={snr}')
        ax.text(dim-1.5, 0.85, f'Efficiecny={eff_target}')

        ax.vlines(x0, 0, eff_target, linestyle='dashed', color='red')

        ax.set_xlabel('Apparent Brightness (AB mag)')
        ax.set_ylabel('Recovery Fraction per magnitude bin')

        plt.savefig(outimg)

        print("Done.")

        return(x0)

if __name__ == "__main__":

    # Can call from command line on output data file to recompute efficiency
    if len(sys.argv)<2:
        print('Usage: Plot_Efficiency.py datafile')
        sys.exit()

    # Default parameters
    eff_target=0.8
    snr=3.0

    bright = 18.0
    dim = 24.0
    bin_size = 0.2

    model_mags = bright + bin_size*np.arange(int((dim-bright)/bin_size)+1)

    filename = sys.argv[1]
    if len(sys.argv)>2:
        outimg = sys.argv[2]
    else:
        outimg = filename.replace('.dat','.png')

    outdata = Table.read(sys.argv[1], format='ascii')

    mag_cent, mag_effi, unweighted_minimize_result=compute_efficiency(
            outdata, model_mags)

    params = (unweighted_minimize_result.params['x0'].value,
                  unweighted_minimize_result.params['a'].value,
                  unweighted_minimize_result.params['n'].value
            )

    def efficiency(params, mag_bins):
            x0, a, n = params
            model_efficiency = n*(1.0-(erf(a*(mag_bins-x0))+1.0)/2.0)
            return(model_efficiency)

    # Given a set of params, compute magnitude where we get input efficiency
    def compute_mag_limit(params, eff_target):
            x0, a, n = params
            mag_limit = x0 + erfinv(1.0 - 2.0*eff_target)/a
            return(mag_limit)

    fig, ax = plt.subplots()

    plot_mags = np.linspace(bright, dim, 4000)

    ax.bar(mag_cent, mag_effi, bin_size, zorder=1, edgecolor='black')
    model_eff = efficiency(params, plot_mags)
    ax.plot(plot_mags, model_eff, zorder=10, color='red')

    x0=compute_mag_limit(params, eff_target)

    ax.text(dim-1.5, 0.95, r'$m_{\rm limit}$='+'%2.2f'%float(x0))
    ax.text(dim-1.5, 0.9, f'SNR={snr}')
    ax.text(dim-1.5, 0.85, f'Efficiecny={eff_target}')

    ax.vlines(x0, 0, eff_target, linestyle='dashed', color='red')

    ax.set_xlabel('Apparent Brightness (AB mag)')
    ax.set_ylabel('Recovery Fraction per magnitude bin')

    plt.savefig(outimg)

    print("Done.")
