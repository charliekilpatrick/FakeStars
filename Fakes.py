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

import warnings
warnings.filterwarnings('ignore')

# This is specific to GPC1.  See forcedophot method
photcode_map = {'g':'0x2011613',
                'r':'0x2011614',
                'i':'0x2011615',
                'z':'0x2011616',
                'y':'0x2011617',}

def is_number(num):
    try:
        num = float(num)
    except ValueError:
        return(False)
    return(True)

def parse_coord(ra, dec):
    if (not (is_number(ra) and is_number(dec)) and
        (':' not in ra and ':' not in dec)):
        error = 'ERROR: cannot interpret: {ra} {dec}'
        print(error.format(ra=ra, dec=dec))
        return(None)

    if (':' in ra and ':' in dec):
        # Input RA/DEC are sexagesimal
        unit = (u.hourangle, u.deg)
    else:
        unit = (u.deg, u.deg)

    try:
        coord = SkyCoord(ra, dec, frame='icrs', unit=unit)
        return(coord)
    except ValueError:
        error = 'ERROR: Cannot parse coordinates: {ra} {dec}'
        print(error.format(ra=ra,dec=dec))
        return(None)

class FileAssociation():
    def __init__(self, image_file,
                 image_dcmp_file,
                 image_mask_file,
                 image_noise_file,
                 image_log_dir,
                 fake_image_file,
                 fake_image_dcmp_file,
                 fake_image_mask_file,
                 fake_image_noise_file,
                 fake_log_dir,
                 template_file,
                 template_dcmp_file,
                 template_mask_file,
                 template_log_dir,
                 diff_dcmp_file):

        self.image_file = image_file
        self.image_dcmp_file = image_dcmp_file
        self.image_mask_file = image_mask_file
        self.image_noise_file = image_noise_file
        self.image_log_dir = image_log_dir
        self.fake_image_file = fake_image_file
        self.fake_image_dcmp_file = fake_image_dcmp_file
        self.fake_image_mask_file = fake_image_mask_file
        self.fake_image_noise_file = fake_image_noise_file
        self.fake_log_dir = fake_log_dir
        self.template_file = template_file
        self.template_dcmp_file = template_dcmp_file
        self.template_mask_file = template_mask_file
        self.template_log_dir = template_log_dir
        self.diff_dcmp_file = diff_dcmp_file

    def __repr__(self):
        return str(self.__dict__)

class DetermineEfficiencies():

    def __init__(self, root_path, log_base, work_base,
        image_dir, template_dir, image_list, template_list, filt,
        save_img=False):

        self.root_path = root_path
        self.image_dir = image_dir
        self.template_dir = template_dir

        self.log_base = log_base
        self.work_base = work_base

        self.filter = filt

        self.coord_list = ''
        self.coord_xy_list = ''

        self.filter = ''

        self.image_path = "{0}/{1}/{2}/{3}".format(self.root_path,
            self.work_base, self.image_dir, filt)
        self.template_path = "{0}/{1}/{2}/tmpl/{3}".format(self.root_path,
            self.work_base, self.image_dir, filt)

        self.log_path = self.image_path.replace(self.work_base, self.log_base)

        self.image_list = image_list
        self.template_list = template_list

        # Unpack image list
        self.image_names = []
        with open(image_list, 'r') as input_file:
            self.image_names += input_file.read().splitlines()

        self.image_files = []
        for i in self.image_names:
            self.image_files.append("{0}/{1}".format(self.image_path, i))

        # Unpack image list
        self.template_names = []
        with open(template_list, 'r') as input_file:
            self.template_names += input_file.read().splitlines()

        self.template_files = []
        for t in self.template_names:
            self.template_files.append("{0}/{1}".format(self.template_path, t))

        self.forced_dcmps = []

        self.save_img = save_img

    def initialize(self, filt):

        # These directories depend on the iteration #
        self.fake_image_dir = "{0}_fake".format(self.image_dir)
        self.fake_image_path = "{0}/{1}/{2}/{3}".format(self.root_path,
            self.work_base, self.fake_image_dir, filt)
        self.fake_diff_image_path = self.fake_image_path.replace(
            self.fake_image_dir, self.fake_image_dir+'_tmpl')

        # Delete workspace, rawdata, logs for fake_image_dir and fake_diff_image
        for directory in [self.work_base, self.log_base, 'rawdata']:
            fake_path = self.fake_image_path.replace(self.work_base, directory)
            fake_diff_path = self.fake_diff_image_path.replace(self.work_base, directory)

            if os.path.exists(fake_path):
                print(f'Deleting {fake_path}')
                shutil.rmtree(fake_path, ignore_errors=True)
            if os.path.exists(fake_diff_path):
                print(f'Deleting {fake_diff_path}')
                shutil.rmtree(fake_diff_path, ignore_errors=True)

        self.fake_log_path = self.fake_image_path.replace(self.work_base,
            self.log_base)
        self.fake_stitched_path = self.fake_log_path.replace(self.log_base,
            "rawdata")
        self.diff_dir_name = "%s_fake_%s" % (self.image_dir, self.template_dir)
        self.diff_dir_path = "%s/%s/%s/%s" % (self.root_path, self.work_base,
            self.diff_dir_name, filt)
        self.fake_mag_file = self.fake_log_path + "/fakemags.txt"

        print('Fake image directory:',self.fake_image_dir)
        print('Fake image path:',self.fake_image_path)
        print('Fake log path:',self.fake_log_path)
        print('Fake raw path:',self.fake_stitched_path)
        print('Diff dir name:',self.diff_dir_name)
        print('Diff dir path:',self.diff_dir_path)
        print('Fake magnitude file:',self.fake_mag_file)

        self.initialize_dirs_and_logs()
        self.file_associations = self.build_file_associations()

        for image in self.image_names:
            self.copy_logs(self.file_associations[image])
            self.copy_templates(self.file_associations[image])

    def copy_logs(self, file_associations):
        # Copy over logs from main image directory to fake directory
        # also copy over template logs
        image_log_dir = file_associations.image_log_dir
        tmpl_log_dir = file_associations.template_log_dir
        fake_log_dir = file_associations.fake_log_dir

        if os.path.exists(image_log_dir):
            print(f'Copying {image_log_dir}->{fake_log_dir}')
            for file in glob.glob(os.path.join(image_log_dir, '*')):
                if 'ABSPHOT' not in file: continue
                if 'errlist' in file: continue
                if 'nosuccess' in file: continue
                basefile = os.path.basename(file)
                basefile = basefile.replace(self.image_dir, self.fake_image_dir)
                if not os.path.exists(os.path.join(fake_log_dir, basefile)):
                    shutil.copyfile(file, os.path.join(fake_log_dir, basefile))

        if os.path.exists(tmpl_log_dir):
            fake_tmpl_log_dir = fake_log_dir.replace(self.fake_image_dir,
                os.path.join(self.fake_image_dir, 'tmpl'))
            if not os.path.exists(fake_tmpl_log_dir):
                print(f'Making fake template dir: {fake_tmpl_log_dir}')
                os.makedirs(fake_tmpl_log_dir)
            print(f'Copying {tmpl_log_dir}->{fake_tmpl_log_dir}')
            for file in glob.glob(os.path.join(tmpl_log_dir, '*')):
                if 'ABSPHOT' not in file: continue
                if 'errlist' in file: continue
                if 'nosuccess' in file: continue
                basefile = os.path.basename(file)
                basefile = basefile.replace(self.image_dir, self.fake_image_dir)
                if not os.path.exists(os.path.join(fake_tmpl_log_dir, basefile)):
                    shutil.copyfile(file, os.path.join(fake_tmpl_log_dir, basefile))

    def copy_templates(self, file_associations):
        # Copy template file over to fake directory
        template_file = file_associations.template_file
        # Need to copy template dcmp, mask, and nosie file as well
        template_dcmp = template_file.replace('.fits','.dcmp')
        template_mask = template_file.replace('.fits','.mask.fits')
        template_noise = template_file.replace('.fits','.noise.fits')

        fake_image_dir = self.fake_image_path
        fake_tmpl_dir = fake_image_dir.replace(self.fake_image_dir,
            os.path.join(self.fake_image_dir, 'tmpl'))

        for file in [template_file, template_dcmp, template_mask,
            template_noise]:
            if os.path.exists(file):
                if not os.path.exists(fake_tmpl_dir):
                    print(f'Making {fake_tmpl_dir}')
                    os.makedirs(fake_tmpl_dir)
                print(f'Copying {file}->{fake_tmpl_dir}')
                basefile = os.path.basename(file)
                shutil.copyfile(file, os.path.join(fake_tmpl_dir, basefile))


    def edit_logs(self, image, fake_list):
        image = os.path.basename(image)
        base_image = image.replace('.fits','').replace('.sw','')

        file_association = self.file_associations[image]
        fake_log_dir = file_association.fake_log_dir

        fake_tmpl_log_dir = fake_log_dir.replace(self.fake_image_dir,
            os.path.join(self.fake_image_dir, 'tmpl'))

        absphot_log = self.fake_image_dir+'_'+self.filter+'.ABSPHOT.outlist'
        absphot_log = os.path.join(fake_log_dir, absphot_log)

        absphot_tmpl_log = self.fake_image_dir+'_tmpl_'+self.filter+'.ABSPHOT.outlist'
        absphot_tmpl_log = os.path.join(fake_tmpl_log_dir, absphot_tmpl_log)

        if not os.path.exists(absphot_log):
            print(f'Could not find {absphot_log}')
            raise Exception("Stop!")

        if not os.path.exists(absphot_tmpl_log):
            print(f'Could not find {absphot_tmpl_log}')
            raise Exception("Stop!")

        with open(absphot_log, 'r') as absphot:
            line = None
            for line in absphot.readlines():
                if base_image in line:
                    break

        if not line or base_image not in line:
            print(f'Could not find {base_image} in {absphot_log}')
            raise Exception("Stop!")

        fake_lines = []
        for fake in fake_list:
            fake_basename = os.path.basename(fake)
            fake_basename = fake_basename.replace('.fits','').replace('.sw','')

            fake_line = line
            fake_line = fake_line.replace(self.image_dir+'/',
                self.fake_image_dir+'/')
            fake_line = fake_line.replace(base_image, fake_basename)

            fake_lines.append(fake_line)

        with open(absphot_log, 'w') as absphot:
            for line in fake_lines:
                absphot.write(line)

        tmpl_lines = []
        with open(absphot_tmpl_log, 'r') as absphot:
            for line in absphot.readlines():
                line = line.replace(self.image_dir+'/',
                    self.fake_image_dir+'/')
                tmpl_lines.append(line)

        with open(absphot_tmpl_log, 'w') as absphot:
            for line in tmpl_lines:
                absphot.write(line)


    def initialize_dirs_and_logs(self):

        if not os.path.exists(self.fake_image_path):
            os.makedirs(self.fake_image_path)

        if not os.path.exists(self.fake_log_path):
            os.makedirs(self.fake_log_path)

        if not os.path.exists(self.fake_stitched_path):
            os.makedirs(self.fake_stitched_path)

        # Create file to hold fake magnitudes
        if not os.path.isfile(self.fake_mag_file):
            with open(self.fake_mag_file, 'w') as fout:
                print('# dcmpfile imagefile x y mag flux psf_flux zpt',file=fout)

    def build_file_associations(self, nfakes=1000):

        file_associations = {}
        image_fields = [i.split('.')[0] for i in self.image_names]
        template_fields = [t.split('.')[0] for t in self.template_names]

        for i, image_name in enumerate(self.image_names):

            image_file = self.image_files[i]
            image_dcmp_file = image_file.replace(".fits", ".dcmp")

            image_log_dir = os.path.split(image_file)[0]
            image_log_dir = image_log_dir.replace(self.work_base,
                self.log_base)

            template_log_dir = image_log_dir.replace(self.image_dir,
                os.path.join(self.image_dir, 'tmpl'))

            image_mask_file = None
            if os.path.exists(image_file.replace('.fits', '.mask.fits.gz')):
                image_mask_file = image_file.replace('.fits', '.mask.fits.gz')
            else:
                image_mask_file = image_file.replace('.fits', '.mask.fits')

            image_noise_file = None
            if os.path.exists(image_file.replace('.fits', '.noise.fits.gz')):
                image_noise_file = image_file.replace('.fits', '.noise.fits.gz')
            else:
                image_noise_file = image_file.replace('.fits', '.noise.fits')

            try:
                if len(template_fields)==1:
                    template_index=0
                else:
                    template_index = np.where(np.asarray(template_fields) == image_fields[i])[0][0]
            except:
                print("\n\n*** NO TEMPLATE ASSOCIATION FOR %s ***\n\n" % image_name)
                raise Exception("Stop!")

            template_name = self.template_names[template_index]
            template_file = self.template_files[template_index]
            template_dcmp_file = template_file.replace("fits", "dcmp")

            template_mask_file = None
            if os.path.exists(template_file.replace('.fits', '.mask.fits.gz')):
                template_mask_file = template_file.replace('.fits', '.mask.fits.gz')
            else:
                template_mask_file = template_file.replace('.fits', '.mask.fits')

            fake_log_dir = image_log_dir.replace(self.image_dir,
                self.fake_image_dir)

            fake_image_name = [image_name.replace('.sw.fits', "_fake{0}.sw.fits".format(
                str(i).zfill(3))) for i in np.arange(nfakes)]
            fake_image_file = ["%s/%s" % (self.fake_image_path, f)
                for f in fake_image_name]
            fake_image_dcmp_file = ["%s/%s" % (self.fake_image_path,
                f.replace(".fits", ".dcmp")) for f in fake_image_name]
            fake_image_mask_file = ["%s/%s" % (self.fake_image_path,
                f.replace('.fits', '.mask.fits.gz')) for f in fake_image_name]
            fake_image_noise_file = ["%s/%s" % (self.fake_image_path,
                f.replace('.fits', '.noise.fits.gz')) for f in fake_image_name]

            temp_field_match = template_name.split('.ut')[0]

            diff_dcmp_name = ["{0}_{1}.diff.dcmp".format(f[:-8],
                template_name.replace(temp_field_match[:-1], "")[:-8])
                for f in fake_image_name]

            diff_dcmp_file = ["%s/%s" % (self.diff_dir_path, d)
                for d in diff_dcmp_name]

            filemissing = False
            for file in [image_file,
                         image_dcmp_file,
                         image_mask_file,
                         image_noise_file,
                         image_log_dir,
                         template_file,
                         template_dcmp_file,
                         template_mask_file,
                         template_log_dir]:

                if not os.path.exists(file):
                    # filemissing = True
                    print('warning : file %s does not exist' % file)
                    raise Exception("Stop!")

            file_associations[image_name] = FileAssociation(image_file,
                                                            image_dcmp_file,
                                                            image_mask_file,
                                                            image_noise_file,
                                                            image_log_dir,
                                                            fake_image_file,
                                                            fake_image_dcmp_file,
                                                            fake_image_mask_file,
                                                            fake_image_noise_file,
                                                            fake_log_dir,
                                                            template_file,
                                                            template_dcmp_file,
                                                            template_mask_file,
                                                            template_log_dir,
                                                            diff_dcmp_file)

        return file_associations

    def append_fake_mag_file(self, line):
        if not os.path.isfile(self.fake_mag_file):
            with open(self.fake_mag_file, 'w') as fout:
               	print('# dcmpfile imagefile x y mag flux psf_flux zpt',file=fout)
        with open(self.fake_mag_file, 'a') as fout:
            print(line, file=fout)

    def generate_psf(self, psf_x, psf_xy, psf_y, psf_size):

        psf_model = np.zeros([psf_size, psf_size])
        for i in range(psf_size):
            for j in range(psf_size):
                x = i - psf_size/2
                y = j - psf_size/2
                zsq = 1/2.*(x**2./psf_x**2. + 2*psf_xy*x*y + y**2./psf_y**2.)
                psf_model[j, i] = (1 + zsq + 1/2.*zsq**2. + 1/6.*zsq**3.)**(-1)

        psf_model = psf_model / np.sum(psf_model)

        return psf_model

    def plant_fakes(self, img, fake_mag_range, coord_list=None, clobber=False,
        n_fake_images=None, fake_mags=[]):

        fake_mag_min, fake_mag_max, n_fake_mags = fake_mag_range

        # Get list of files to add fake stars to...
        psf_shape = 31

        # Plant fake stars
        fake_list = []
        if True:

            print("Opening: %s" % img)
            try:
                file_association = self.file_associations[img]
            except:
                # Could not find input file - check input list against work dir
                # return None and continue to next image
                print("Could not find file association for: %s" % img)
                return(None)

            image_hdu = fits.open(file_association.image_file)
            image_data = image_hdu[0].data.astype('float')
            dcmp_header = fits.getheader(file_association.image_dcmp_file)

            zpt = None
            try:
                zpt = dcmp_header['ZPTMAG']
            except KeyError:
                # Problem with this image, probably failed on ABSPHOT
                # return None and continue to next image
                print("'ZPTMAG' keyword doesn't exist. Exiting...")
                return(None)

            psf_model = self.generate_psf(dcmp_header['DPSIGX'],
                dcmp_header['DPSIGXY'], dcmp_header['DPSIGY'], psf_shape)

            w = wcs.WCS(dcmp_header)
            table = Table.read(coord_list, format='ascii', names=('ra','dec'))

            fwhm = dcmp_header['FWHM']

            fake_x = [] ; fake_y = []
            ra,dec = np.loadtxt(coord_list,unpack=True,dtype=str)
            try:
                ra,dec = ra.astype(float),dec.astype(float)
                sc = SkyCoord(ra,dec,unit=u.deg)
            except:
                sc = SkyCoord(ra,dec,unit=(u.hour,u.deg))
            xpos,ypos = w.wcs_world2pix(sc.ra.deg,sc.dec.deg,0)
            for x,y in zip(xpos, ypos):
                fake_x.append(x) ; fake_y.append(y)

            print("Opening: %s" % file_association.image_mask_file)
            mask_hdu = fits.open(file_association.image_mask_file)

            mask_data = mask_hdu[0].data.astype('float')

            imshapex, imshapey = np.shape(image_data)

            good_idx = []

            # Check for any masking in a NxN region
            region_size = 7
            half_size = (region_size-1)/2
            for j in range(len(fake_y)):
                if np.sum(mask_data[int(fake_y[j]-half_size):int(fake_y[j]+half_size),
                    int(fake_x[j]-half_size):int(fake_x[j]+half_size)])==0.0:
                    good_idx.append(j)

            good_idx = np.array(good_idx)
            fake_x = np.array(fake_x)[good_idx.astype(int)]
            fake_y = np.array(fake_y)[good_idx.astype(int)]

            if len(good_idx)<5:
                # Will be prohibitive to run this reduction, so skip
                return(None)

            # Generate coord_xy_list
            if not os.path.exists(self.fake_diff_image_path):
                os.makedirs(self.fake_diff_image_path)
            coord_xy_list = os.path.join(self.fake_diff_image_path,
                os.path.basename(coord_list).replace('.txt','.xy'))
            self.coord_xy_list = coord_xy_list
            with open(self.coord_xy_list, 'w') as coord_file:
                for x,y in zip(fake_x, fake_y):
                    coord_file.write('{0} {1} \n'.format(x,y))

            if not n_fake_images:
                n_fake_images = int(np.ceil(n_fake_mags/len(good_idx)))

            print('There are {0} good indices for image {1}'.format(
                len(good_idx), file_association.image_file))
            print(f'Need to create {n_fake_images} images')

            # Build PSFs
            for ii in np.arange(n_fake_images):
                fake_mags = np.random.uniform(fake_mag_min,
                    fake_mag_max, len(good_idx))

                fake_image_file = file_association.fake_image_file[ii]

                base_image_data = copy.copy(image_data)
                base_image_hdu = copy.copy(image_hdu)

                kk=0
                for m, x, y in zip(fake_mags, fake_x, fake_y):

                    psf_flux = 10 ** (-0.4 * (m - zpt))

                    max_size = np.shape(psf_model)[0]
                    dx = dy = int((max_size - 1) / 2)

                    # Add a jitter to the position of order 0.1 * fwhm size
                    mu, sigma = 0., 0.0005 * fwhm / 2.355
                    sx, sy = np.random.normal(mu, sigma, 2)

                    flux = np.sum(psf_model * psf_flux)
                    psf_norm = np.sum(psf_model)

                    # Append log file with new fake point
                    self.append_fake_mag_file('%s %s %s %s %s %s %s %s' % (
                        file_association.image_dcmp_file, fake_image_file,
                        x, y, m, flux, psf_flux, zpt))

                    message = f'mag={m}, x={x}, y={y}, flux={flux}, '
                    message += f'psf_flux={psf_flux}, zpt={zpt}, '
                    message += f'psf_norm={psf_norm}, sy={sy}, sx={sx}, '
                    message += f'dy={dy}, dx={dy}'
                    print(message)

                    # Add all of this data to the image header
                    if len(fake_mags)<100:
                        pref='FK'+str(kk).zfill(2)
                    else:
                        pref='FK'+str(kk).zfill(3)

                    base_image_hdu[0].header[pref+'X']=float(x)
                    base_image_hdu[0].header[pref+'Y']=float(y)
                    base_image_hdu[0].header[pref+'FLX']=float(flux)
                    base_image_hdu[0].header[pref+'MAG']=float(m)

                    img_coord = w.all_pix2world([[x, y]], 1)
                    ra = img_coord[0][0]
                    dec = img_coord[0][1]

                    base_image_hdu[0].header[pref+'RA']=ra
                    base_image_hdu[0].header[pref+'DEC']=dec

                    source_model = psf_model * psf_flux
                    source_model = np.array(source_model)

                    base_image_data[int(y+sy) - dy:int(y+sy) + dy + 1,
                               int(x+sx) - dx:int(x+sx) + dx + 1] \
                               += source_model

                base_image_hdu[0].data[:] = base_image_data
                print('Writing out to {0}'.format(fake_image_file))
                base_image_hdu.writeto(fake_image_file, clobber=True,
                    output_verify='ignore')

                # Copy the dcmp file to the fake directory
                shutil.copyfile(file_association.image_dcmp_file,
                    file_association.fake_image_dcmp_file[ii])

                # Copy the mask file to the fake directory
                shutil.copyfile(file_association.image_mask_file,
                    file_association.fake_image_mask_file[ii])

                # Copy the noise file to the fake directory
                shutil.copyfile(file_association.image_noise_file,
                    file_association.fake_image_noise_file[ii])

                fake_list.append(fake_image_file)

        return(fake_list)

    def run_fake_reductions(self, fake_image_dir, filt, redo='-redo',
        diffimstats=False):

        cmd = f'eventloop.pl -events {fake_image_dir} {filt} '
        cmd += f'-stage MATCHTEMPL,DIFFIM,DIFFIMSTATS {redo} -condor '
        cmd += '-k BATCH_OPTIONS \'-maxcpus 8\''

        if diffimstats:
            assert self.coord_list != ''
            assert os.path.exists(self.coord_list)
            coord_list = os.path.abspath(self.coord_list)
            cmd += ' -k DIS_OPTIONS \'--dMmax 0.03 --customlist '
            cmd += f' {coord_list} --jitter 0.0\''

        os.system(cmd)

    def diffim_check(self, fake_diff_image_path, fakes, it=1):

        message = f'Iteration {it} '
        check_list = []
        for exten in ['diff.fits','diff.kernel','diff.mask.fits',
            'diff.noise.fits']:

            num_done = [len(glob.glob(os.path.join(fake_diff_image_path,
                f+'*'+exten))) for f in fakes]
            check = all([n>0 for n in num_done])

            check_list.append(check)
            message += exten+'='+str(np.sum(num_done))+' '

        return(check_list, num_done, message)

    def diffimstats_check(self, fake_diff_image_path, fakes, it=1):

        message = f'Iteration {it} '
        check_list = []
        num_done = 0
        for f in fakes:
            gl=glob.glob(os.path.join(fake_diff_image_path,f+'*diff.fits'))
            if len(gl)!=1:
                check_list.append(False)
                continue

            try:
                hdu = fits.open(gl[0], mode='readonly')
            except OSError:
                check_list.append(False)
                continue

            if any([k not in hdu[0].header.keys() for k in ['DSGDAVG','DSGDX2N',
                'DSGDNPCP','DSGDN','DSGDPCP']]):
                check_list.append(False)
                continue

            check_list.append(True)
            num_done += 1

        message += 'num done='+str(num_done)

        return(check_list, num_done, message)


    def wait_for_fake_reductions(self, fake_diff_image_path, fake_list,
        last_stage='DIFFIMSTATS', max_it=100, time_sleep=40):

        t0 = time.time()

        fakes = [os.path.basename(f).replace('.fits','').replace('.sw','')
            for f in fake_list]

        print(f'Waiting for fake difference images in {fake_diff_image_path}')
        print('There are {0} being processed by photpipe'.format(
            len(fake_list)))

        it=1
        last_done = 0
        while True:
            if last_stage=='DIFFIM':
                check_list, num_done, message = self.diffim_check(
                    fake_diff_image_path, fakes, it=it)
            elif last_stage=='DIFFIMSTATS':
                check_list, num_done, message = self.diffimstats_check(
                    fake_diff_image_path, fakes, it=it)
            # Basic condition for terminating if something goes wrong
            if it>max_it:
                break


            t1 = time.time()
            dt = t1-t0

            dt = '%3.3f'%dt

            print(message+f', time on photpipe: {dt} seconds')
            if all(check_list):
                break
            else:
                if num_done > 0 and num_done==last_done:
                    self.run_fake_reductions(self.fake_image_dir, self.filter,
                        redo='-redobad')
                    time.sleep(time_sleep)
                    it=it+1
                    continue
                else:
                    it=it+1
                    time.sleep(time_sleep)

            last_done = num_done

    def run_forcedophot(self, fake_diff_image_path, fake_list, filt):

        fakes = [os.path.basename(f).replace('.fits','').replace('.sw','')
            for f in fake_list]
        fake_diffims = [glob.glob(os.path.join(fake_diff_image_path,
            f+'*.diff.fits')) for f in fakes]

        if any([len(f)!=1 for f in fake_diffims]):
            print(f'ERROR: missing difference images in {fake_diff_image_path}')
            for ii,f in enumerate(fakes):
                expr=os.path.join(fake_diff_image_path, f+'*.diff.fits')
                gl = glob.glob(expr)
                n = len(gl)
                print(expr, gl, n)
            raise Exception('STOP!')

        fake_diffims = [f[0] for f in fake_diffims]

        # Open coordinate file
        assert os.path.exists(self.coord_xy_list)

        cmd = 'forcedophot4image.pl {image} {tmp_dophot} {dcmp} {xylist} '
        cmd += '{photcode} 1.00000 {logfile}'

        for fake_file in fake_diffims:

            tmp_dophot = fake_file.replace('.diff.fits','.diff.dophotout')
            dcmp = fake_file.replace('.diff.fits','.diff.fake.dcmp')
            logfile = fake_file.replace('.diff.fits','.diff.forced.log')

            photcode = photcode_map[filt]

            do_cmd = cmd.format(image=fake_file, tmp_dophot=tmp_dophot,
                dcmp=dcmp, xylist=self.coord_xy_list, photcode=photcode,
                logfile=logfile)

            print(do_cmd)
            os.system(do_cmd)

            assert os.path.exists(dcmp)

            self.forced_dcmps.append(dcmp)


class AllStages():
    def __init__(self):
        pass

    def add_options(self, parser=None, usage=None):

        if parser == None:
            parser = optparse.OptionParser(usage=usage,
                conflict_handler="resolve")

        parser.add_option('--clobber', default=False, action="store_true",
            help='overwrite previous results if set')
        parser.add_option('--root_path', default='$PIPE_DATA/workstch',
            type='string', help='Root directory')
        parser.add_option('--log_base', default='logstch', type='string',
            help='Root directory')
        parser.add_option('--work_base', default='workstch', type='string',
            help='Image directory')
        parser.add_option('--image_dir', default='gw190425', type='string',
            help='Image directory')
        parser.add_option('--template_dir', default='gw190425tmpl',
            type='string', help='Template directory')
        parser.add_option('--image_list',
            default='gw190425/<change this to the one you want>.txt',
            type='string', help='File with all observations')
        parser.add_option('--template_list',
            default='gw190425/<change this to the one you want>.txt',
            type='string', help='File with all templates')
        parser.add_option('--fake_mag_range', default=(18, 25, 1500),
            nargs=3, type='float',
            help='Fake mag tuple: (min, max, # of stars)')
        parser.add_option('--coord_list', default='coord_list.txt', type=str,
            help='Input coordinate list for where fake stars should be planted')
        parser.add_option('--filter', default='g', type=str,
            help='Filter to reduce, needed for file and directory structure')
        parser.add_option('--save-img', default=False, action="store_true",
            help='save the fake star subtraction images to another dir.')
        parser.add_option('--target-efficiency', default=0.8, type=float,
            help='target efficiency for modeling the efficiency curve.')
        parser.add_option('--target-snr', default=3.0, type=float,
            help='target S/N for modeling the efficiency curve.')
        parser.add_option('--use-diffimstats', default=False,
            action='store_true', help='Use forced photometry from diffimstats')

        return(parser)


if __name__ == "__main__":

    import optparse

    usagestring='USAGE: DetEff.py'

    allstages = AllStages()
    parser = allstages.add_options(usage=usagestring)
    options,  args = parser.parse_args()

    root_path = os.path.expandvars(options.root_path)

    detEff = DetermineEfficiencies(root_path=root_path,
                                   log_base=options.log_base,
                                   work_base=options.work_base,
                                   image_dir=options.image_dir,
                                   template_dir=options.template_dir,
                                   image_list=options.image_list,
                                   template_list=options.template_list,
                                   filt=options.filter,
                                   save_img=options.save_img)

    paramfile = os.environ['PIPE_PARAMS']
    os.environ['EXTRAPARAMFILE']=paramfile+'.fakes.yse'

    detEff.options = options

    detEff.filter = options.filter

    if options.coord_list and os.path.exists(options.coord_list):
        detEff.coord_list = options.coord_list

    t0 = time.time()

    limit_outfile = open('{0}_{1}_limits.dat'.format(detEff.image_dir.strip(),
        options.filter.strip()), 'w')

    limit_outfile.write('# imagename mag_limit \n')

    for image in detEff.image_names:
        print(image)
        detEff.initialize(options.filter)

        fake_list=detEff.plant_fakes(image, options.fake_mag_range,
                coord_list=options.coord_list)
        if not fake_list:
            # Did not plant a sufficient number of fakes, so continue
            continue

        # Edit logs to contain fake images
        detEff.edit_logs(image, copy.copy(fake_list))

        # Run slurm job for fake images
        detEff.run_fake_reductions(detEff.fake_image_dir, options.filter,
            diffimstats=options.use_diffimstats)

        # Check for fake images done
        detEff.wait_for_fake_reductions(detEff.fake_diff_image_path,
            copy.copy(fake_list))

        if not options.use_diffimstats:
            detEff.run_forcedophot(detEff.fake_diff_image_path,
                copy.copy(fake_list), options.filter)

        fake_diff_dir = "{0}_fake_tmpl".format(detEff.image_dir)
        subdir = os.path.join(fake_diff_dir, options.filter)
        work_path = os.path.join(detEff.root_path, detEff.work_base)
        bright = options.fake_mag_range[0]
        dim = options.fake_mag_range[1]

        print(f'subdir: {subdir}')
        print(f'work_path: {work_path}')
        print(f'Magnitude range: {bright} {dim}')

        if not os.path.exists('Fakes'):
            os.makedirs('Fakes')

        outimg = os.path.join('Fakes', image.replace('.fits','_eff.png'))
        outdata = os.path.join('Fakes', image.replace('.fits','_eff.dat'))

        print(f'out plot: {outimg}')
        print(f'out data: {outdata}')

        limit=Plot_Efficiency.calculate_and_plot_efficiency(work_path, [subdir],
            options.target_snr, outimg, outdata, bright=bright, dim=dim,
            eff_target=options.target_efficiency,
            diffimstats=options.use_diffimstats)

        if not limit:
            print(f'WARNING: an error occurred with limit calculation.')
            continue

        fake_list=detEff.plant_fakes(image, options.fake_mag_range,
                coord_list=options.coord_list,
                n_fake_images=1, fake_mags=[limit])

        if not os.path.exists('FakeImg'):
            os.makedirs('FakeImg')
        outfakeimgfile = os.path.join('FakeImg', os.path.basename(fake_list[0]))
        shutil.copyfile(fake_list[0], outfakeimgfile)

        limit_outfile.write('{0} {1} \n'.format(image, limit))

        if detEff.save_img:
            # Grab all images in diff image directory
            full_work_dir = os.path.join(work_path, subdir)
            fake_files = glob.glob(os.path.join(full_work_dir, '*.diff.fits'))

            outdir = full_work_dir.replace('_fake_tmpl','_fake_tmpl_save')
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            for file in fake_files:
                imgfile = file.replace('.diff.fits','.diff.im.fits')

                basefile = os.path.basename(file)
                baseimgfile = os.path.basename(imgfile)

                outfile = os.path.join(outdir, basefile)
                outimgfile = os.path.join(outdir, baseimgfile)

                if os.path.exists(file):
                    shutil.copyfile(file, outfile)
                if os.path.exists(imgfile):
                    shutil.copyfile(imgfile, outimgfile)

    limit_outfile.close()

    t1 = time.time()
    total = t1 - t0

    print("\n*****************\nProcess time: %s\n*****************\n" % total)
