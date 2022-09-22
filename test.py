from __future__ import print_function
import shutil
import time
import random
import glob
import copy
import numpy as np
from astropy.io import fits
import os
from txtobj import txtobj
import re
import sys
from astropy import wcs
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

import Plot_Efficiency

image = '2020fhs.g.200525_2236723_1735.081.sw.fits'

subdir = os.path.join('2020fhs_fake_1_tmpl', 'g')
work_path = os.path.join('/lustre/hpc/storage/dark/YSE/data', 
    'workspace')
bright = 19
dim = 24

print(f'subdir: {subdir}')
print(f'work_path: {work_path}')
print(f'Magnitude range: {bright} {dim}')

if not os.path.exists('Plots'):
    os.makedirs('Plots')

outfile = os.path.join('Plots', image.replace('.fits','_eff.png'))

print(f'out plot: {outfile}')

limit=Plot_Efficiency.calculate_and_plot_efficiency(work_path, [subdir],
            3.0, outfile, bright=bright, dim=dim)

print(limit)
