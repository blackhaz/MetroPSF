# -*- coding: utf-8 -*-
# MetroPSF (C) Copyright 2021, Maxym Usatov <maxim.usatov@bcsatellite.net>
# Thanks to Cliff Kotnik from AAVSO for contributing code and ideas.
# Please refer to metropsf.pdf for license information.

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os.path
import csv
import numbers
import requests
import sys
import argparse
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import SqrtStretch, LogStretch, AsinhStretch
from PIL import Image, ImageTk, ImageMath, ImageOps
from astroquery.astrometry_net import AstrometryNet
from astroquery.vizier import Vizier
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from astropy.nddata import Cutout2D

import tkinter as tk
from tkinter import filedialog as fd

# Photometry
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.psf import IterativelySubtractedPSFPhotometry
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter, SLSQPLSQFitter, SimplexLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from photutils.utils import calc_total_error

# Minor Planets
import pympc

# No imports beyond this line.

def save_background_image(stretch_min, stretch_max, zoom_level, image_data):
    global FITS_minimum
    global FITS_maximum
    global background_image
    background_image = Image.fromarray(image_data)
    width, height = background_image.size
    new_size = (int(width * zoom_level), int(height * zoom_level))
    background_image = background_image.resize(new_size, Image.ANTIALIAS)
    background_image = ImageMath.eval("(a + " + str(stretch_min / 100 * FITS_maximum) + ") * 255 / " + str(stretch_max / 100 * FITS_maximum), a=background_image)
    background_image = ImageMath.eval("convert(a, 'L')", a=background_image)
    background_image.save('background.jpg')

def save_image(stretch_min, stretch_max, zoom_level, image_data, filename):
    global FITS_minimum
    global FITS_maximum
    _image = Image.fromarray(image_data)
    width, height = _image.size
    new_size = (int(width * zoom_level), int(height * zoom_level))
    _image = _image.resize(new_size, Image.ANTIALIAS)
    _image = ImageMath.eval("(a + " + str(stretch_min / 100 * FITS_maximum) + ") * 255 / " + str(stretch_max / 100 * FITS_maximum), a=background_image)
    _image = ImageMath.eval("convert(a, 'L')", a=background_image)
    _image.save(filename)

def generate_FITS_thumbnail(stretch_min, stretch_max, zoom_level, stretching_stringvar):
    global generated_image
    global image_data
    global FITS_minimum
    global FITS_maximum
    converted_data = image_data.astype(float)
    if stretching_stringvar == "Square Root":
        stretch = SqrtStretch()
        converted_data = (converted_data - np.min(converted_data)) / np.ptp(converted_data)
        converted_data = stretch(converted_data)

    if stretching_stringvar == "Log":
        stretch = LogStretch()
        converted_data = (converted_data - np.min(converted_data)) / np.ptp(converted_data)
        converted_data = stretch(converted_data)

    if stretching_stringvar == "Asinh":
        stretch = AsinhStretch()
        converted_data = (converted_data - np.min(converted_data)) / np.ptp(converted_data)
        converted_data = stretch(converted_data)

    """
    CLKotnik 2021-08-11
    Altered the stretch low and high from percentage of histogram range to multiples of std dev
    from the mean.  With the low/min stretch set above the high/max the image is inverted.
    """
    imean = np.mean(converted_data)
    isd = np.std(converted_data)
    imin = np.min(converted_data)
    imax = np.max(converted_data)
    black = max(imean + stretch_min * isd, imin)
    white = min(imean + stretch_max * isd, imax)
    slope = 1.0 / (white - black)
    converted_data = np.clip((converted_data - black) * slope, 0.0, 1.0)

    generated_image = Image.fromarray(converted_data * 255.0)
    width, height = generated_image.size
    new_size = (int(width * zoom_level), int(height * zoom_level))
    generated_image = generated_image.resize(new_size, Image.ANTIALIAS)

image_file = ""
image_data = []

class MyGUI:

    zoom_level = 1
    linreg_error = 0
    zoom_step = 0.5
    photometry_results_plotted = False
    results_tab_df=pd.DataFrame()
    bkg_value = 0
    fit_shape = 21
    error_raised = False
    histogram_slider_low = 0
    histogram_slider_high = 5
    last_clicked_x = 0
    last_clicked_y = 0
    last_clicked_differential_magnitude = 0
    last_clicked_differential_uncertainty = 0
    ensemble_size = 0
    a = 0
    b = 0
    jd = 0
    image_file = ""
    photometry_circles = {}

    def console_msg(self, message):
        self.console.insert(tk.END, message+"\n")
        self.console.see(tk.END)
        self.window.update_idletasks()
        file = open("metropsf.log", "a+", encoding="utf-8")
        file.write(str(message) + "\n")
        file.close()

    def display_image(self):
        if len(image_data) > 0:
            self.canvas.delete("all")
            global generated_image
            generate_FITS_thumbnail(self.histogram_slider_low, self.histogram_slider_high, self.zoom_level,
                                    self.stretching_stringvar.get())
            self.image = ImageTk.PhotoImage(generated_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            self.canvas.bind("<Button-1>", self.mouse_click)
            if self.photometry_results_plotted:
                self.plot_photometry()

    def display_background(self):
        self.canvas.delete("all")
        self.console_msg("Displaying extracted background.")
        global background_image
        self.image = ImageTk.PhotoImage(background_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))


    def load_FITS(self, image_file):
        global image_figure
        global image_data
        global image
        global image_width
        global image_height
        global header
        global FITS_minimum
        global FITS_maximum
        global generated_image
        try:
            self.console_msg("Loading FITS: " + image_file)
            with fits.open(image_file) as image:
                self.image_file = image_file
                self.filename_label['text'] = "FITS: " + image_file
                self.canvas.delete("all")
                self.zoom_level = 1
                self.photometry_results_plotted = False
                self.results_tab_df = pd.DataFrame()
                header = image[0].header
                image_data = fits.getdata(image_file)
                image_width = image_data.shape[1]
                image_height = image_data.shape[0]
                self.wcs_header = WCS(image[0].header)

                if self.crop_fits_entry.get() != "100":
                    factor = 100 / int(self.crop_fits_entry.get())
                    new_width = int(image_width / factor)
                    new_height = int(image_height / factor)
                    x0 = int((image_width - new_width) / 2)
                    y0 = int((image_height - new_height) / 2)
                    x1 = x0 + new_width
                    y1 = y0 + new_width
                    image_data = image_data[y0:y1, x0:x1]
                    image_width = new_width
                    image_height = new_height

                #self.console_msg("WCS Header: " + str(self.wcs_header))
                FITS_minimum = np.min(image_data)
                FITS_maximum = np.max(image_data)
                self.console_msg("Width: " + str(image_width) + " Height: " + str(image_height))
                self.console_msg("FITS Minimum: " + str(FITS_minimum) + " Maximum: " + str(FITS_maximum))
                if 'filter' in header:
                    self.filter = str(header['filter'])
                    self.set_entry_text(self.filter_entry, self.filter)
                    self.console_msg("Filter: " + self.filter)
                else:
                    self.console_msg("Filter name not in FITS header. Set filter manually.")
                if 'exptime' in header:
                    exptime = header['exptime']
                    self.set_entry_text(self.exposure_entry, exptime)
                    self.console_msg("Exposure: " + str(exptime))
                else:
                    self.console_msg("Exposure (EXPTIME) not in FITS header. Set exposure manually.")

                if 'gain' in header:
                    gain = header['gain']
                    self.set_entry_text(self.ccd_gain_entry, gain)
                    self.console_msg("Gain (e-/ADU): " + str(gain))
                else:
                    self.console_msg("Gain not in FITS header. Set gain manually for aperture photometry.")


                self.jd = 0

                if 'date-obs' in header:
                    try:
                        date_obs = Time(header['date-obs'])
                        self.jd = Time(date_obs, format='jd')
                        self.console_msg("Julian date at the start of exposure (from DATE-OBS): " + str(self.jd))
                        self.set_entry_text(self.exposure_start_entry, str(self.jd))
                    except Exception as e:
                        self.error_raised = True
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        self.console_msg(str(exc_tb.tb_lineno) + " " + str(e))
                        pass

                if 'jd' in header:
                    jd = header['jd']
                    self.console_msg("Julian date at the start of exposure (from JD): " + str(jd))
                    self.jd = jd
                    self.exposure_start_entry.delete(0, tk.END)
                    self.exposure_start_entry.insert(0, str(self.jd))

                self.bkg_value = np.median(image_data)
                self.console_msg("Median background level, ADU: " + str(self.bkg_value))
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))
            pass

    def initialize_debug(self):
        #self.set_entry_text(self.crop_fits_entry, "30")
        self.load_settings(file_name="t11 CG Dra.mpsf")
        image_file = "calibrated-T11-blackhaz-CG Dra-20210516-010321-V-BIN1-E-300-012.fit"
        #image_file = "AmericoWatkins/123.fits"

        if len(image_file) > 0:
            self.load_FITS(image_file)
            self.display_image()
        #self.get_MPCs_in_the_frame()

    def get_MPCs_in_the_frame(self):
        global image_width, image_height
        try:
            frame_center = self.wcs_header.pixel_to_world(int(image_width / 2), (image_height / 2))
            frame_edge = self.wcs_header.pixel_to_world(int(image_width), (image_height / 2))
            frame_radius = frame_edge.separation(frame_center)
            date_obs = Time(self.exposure_start_entry.get(), format="jd")
            self.console_msg(
               "Searching center α δ " + frame_center.to_string("hmsdms") + ", radius " + str(frame_radius) + ", date "+str(date_obs))

            import ephem
            #xephem_db = open("mpcorb_xephem.csv").readlines()
            xephem_db = open("stephengould.csv").readlines()
            for xephem_str in xephem_db:
                mp = ephem.readdb(xephem_str.strip())
                observer = ephem.Observer()
                observer.lon = '-84.31'
                observer.lat = '33.00'
                observer.date = date_obs.datetime
                mp.compute(observer)
                self.console_msg(str(mp.a_ra)+" "+str(mp.a_dec))

        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))



    def open_FITS_file(self):
        image_file = fd.askopenfilename()
        if len(image_file) > 0:
            self.load_FITS(image_file)
            self.display_image()

    def save_FITS_file_as(self):
        global image
        global image_data
        global header
        file_name = fd.asksaveasfile(defaultextension='.fits')
        try:
            if len(str(file_name)) > 0:
                self.console_msg("Saving FITS as " + str(file_name.name))
                fits.writeto(file_name.name, image_data, header, overwrite=True)
                self.console_msg("Saved.")
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))

    def save_FITS_file(self):
        global image
        global image_data
        global header
        file_name = self.image_file
        try:
            if len(str(file_name)) > 0:
                self.console_msg("Saving FITS as " + str(file_name))
                fits.writeto(file_name, image_data, header, overwrite=True)
                self.console_msg("Saved.")
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))

    def aperture_photometry(self):
        global header
        self.console_msg("Initiating aperture photometry..")
        try:
            self.fit_shape = int(self.photometry_aperture_entry.get())
            fwhm = int(self.fwhm_entry.get())
            star_detection_threshold = float(self.star_detection_threshold_entry.get())
            iterations = int(self.photometry_iterations_entry.get())
            bkgrms = MADStdBackgroundRMS()
            sharplo = float(self.sharplo_entry.get())
            bkg_filter_size = int(self.bkg_filter_size_entry.get())
            std = bkgrms(image_data)
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MedianBackground()
            self.console_msg("Estimating background..")
            bkg = Background2D(image_data, (self.fit_shape * 1, self.fit_shape * 1),
                               filter_size=(bkg_filter_size, bkg_filter_size), sigma_clip=sigma_clip,
                               bkg_estimator=bkg_estimator)
            clean_image = image_data-bkg.background
            save_background_image(self.histogram_slider_low, self.histogram_slider_high, self.zoom_level, bkg.background)
            self.console_msg("Estimating total error..")
            #effective_gain = int(float(self.exposure_entry.get()))
            effective_gain = float(self.ccd_gain_entry.get())
            error = calc_total_error(image_data, bkg.background_rms, effective_gain)

            self.console_msg("Finding stars..")
            iraffind = IRAFStarFinder(threshold=star_detection_threshold * std,
                                      fwhm=fwhm, roundhi=3.0, roundlo=-5.0,
                                       sharplo=sharplo, sharphi=2.0)
            positions_found = iraffind(clean_image)
            positions = np.array(positions_found[['xcentroid', 'ycentroid']])
            positions = [list(item) for item in positions] # Convert a list of tuples to 2D array
            aperture = CircularAperture(positions, r=self.fit_shape)
            self.console_msg("Performing photometry..")
            phot_table = aperture_photometry(clean_image, aperture, error=error)
            self.results_tab_df = phot_table.to_pandas()
            # Let's rename and add columns so the resulting dataframe will be consistent with that produced by the PSF photometry
            self.results_tab_df.rename(columns={"xcenter": "x_0",
                                                "ycenter": "y_0",
                                                "aperture_sum": "flux_fit",
                                                "aperture_sum_err": "flux_unc"}, inplace=True)
            self.results_tab_df["x_fit"] = self.results_tab_df["x_0"]
            self.results_tab_df["y_fit"] = self.results_tab_df["y_0"]
            self.results_tab_df["flux_0"] = self.results_tab_df["flux_fit"]
            self.results_tab_df["sigma_fit"] = 0

            self.results_tab_df["removed_from_ensemble"] = False
            self.results_tab_df.to_csv(self.image_file + ".phot", index=False)
            self.console_msg("Photometry saved to " + str(self.image_file + ".phot"))
            self.plot_photometry()
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))


    def perform_photometry(self):
        global header
        self.console_msg("Initiating interactively subtracted PSF photometry..")
        try:
            self.fit_shape = int(self.photometry_aperture_entry.get())
            fwhm = int(self.fwhm_entry.get())
            star_detection_threshold = float(self.star_detection_threshold_entry.get())
            iterations = int(self.photometry_iterations_entry.get())
            bkgrms = MADStdBackgroundRMS()
            sharplo = float(self.sharplo_entry.get())
            bkg_filter_size = int(self.bkg_filter_size_entry.get())
            std = bkgrms(image_data)
            self.console_msg("Finding stars..")
            iraffind = IRAFStarFinder(threshold=star_detection_threshold * std,
                                      fwhm=fwhm, roundhi=3.0, roundlo=-5.0,
                                       sharplo=sharplo, sharphi=2.0)

            daogroup = DAOGroup(2.0 * fwhm * gaussian_sigma_to_fwhm)
            #mmm_bkg = MMMBackground()

            sigma_clip = SigmaClip(sigma = 3.0)
            bkg_estimator = MedianBackground()
            self.console_msg("Estimating background..")
            bkg = Background2D(image_data, (self.fit_shape * 1, self.fit_shape * 1),
                               filter_size=(bkg_filter_size, bkg_filter_size), sigma_clip=sigma_clip,
                               bkg_estimator=bkg_estimator)
            clean_image = image_data-bkg.background
            save_background_image(self.histogram_slider_low, self.histogram_slider_high, self.zoom_level, bkg.background)

            if self.fitter_stringvar.get() == "Levenberg-Marquardt":
                self.console_msg("Setting fitter to Levenberg-Marquardt")
                fitter = LevMarLSQFitter(calc_uncertainties=True)

            if self.fitter_stringvar.get() == "Linear Least Square":
                self.console_msg("Setting fitter to Linear Least Square")
                fitter = LinearLSQFitter(calc_uncertainties=True)

            if self.fitter_stringvar.get() == "Sequential LS Programming":
                self.console_msg("Setting fitter to Sequential Least Squares Programming")
                fitter = SLSQPLSQFitter()

            if self.fitter_stringvar.get() == "Simplex LS":
                self.console_msg("Setting fitter to Simplex and Least Squares Statistic")
                fitter = SimplexLSQFitter()


            psf_model = IntegratedGaussianPRF(sigma=2)  # sigma=2 here is the initial guess
            psf_model.sigma.fixed = False   # This allows to fit Gaussian PRF sigma as well
            photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                            group_maker=daogroup,
                                                            psf_model=psf_model,
                                                            bkg_estimator=None,
                                                            fitter=LevMarLSQFitter(),
                                                            niters=iterations, fitshape=(self.fit_shape, self.fit_shape))
            self.console_msg("Performing photometry..")
            result_tab = photometry(image=clean_image)
            residual_image = photometry.get_residual_image()
            #fits.writeto("residuals.fits", residual_image, header, overwrite=True)
            #self.console_msg("Done. PSF fitter message(s): " + str(fitter.fit_info['message']))

            self.results_tab_df = result_tab.to_pandas()
            self.results_tab_df["removed_from_ensemble"] = False

            self.results_tab_df.to_csv(self.image_file + ".phot", index=False)
            self.console_msg("Photometry saved to " + str(self.image_file + ".phot"))
            self.plot_photometry()
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))


    def create_circle(self, x, y, r, canvas_name, outline_color="grey50"):  # center coordinates, radius
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        self.photometry_circles[str(x)+str(y)] = canvas_name.create_oval(x0, y0, x1, y1, outline=outline_color)

    def plot_photometry(self):
        try:
            matches_in_photometry_table = False
            if "match_id" in self.results_tab_df:
                matches_in_photometry_table = True
            vsx_ids_in_photometry_table = False
            if "vsx_id" in self.results_tab_df:
                vsx_ids_in_photometry_table = True
            exptime = float(self.exposure_entry.get())
            if os.path.isfile(self.image_file+".phot"):
                self.fit_shape = int(self.photometry_aperture_entry.get())
                self.results_tab_df = pd.read_csv(self.image_file + ".phot")
                if "removed_from_ensemble" not in self.results_tab_df:
                    self.results_tab_df["removed_from_ensemble"] = False   # This prefilling is required for backwards compatibility to read .phot files from older versions.
                # Calculate instrumental magnitudes
                self.results_tab_df["inst_mag"] = -2.5 * np.log(self.results_tab_df["flux_fit"] / exptime) + 25
                self.results_tab_df["inst_mag_min"] = -2.5 * np.log((self.results_tab_df["flux_fit"] - self.results_tab_df["flux_unc"]) / exptime) + 25
                self.results_tab_df["inst_mag_max"] = -2.5 * np.log((self.results_tab_df["flux_fit"] + self.results_tab_df["flux_unc"]) / exptime) + 25
                self.photometry_results_plotted = True
                for index, row in self.results_tab_df.iterrows():
                    outline_color = "grey50"
                    if matches_in_photometry_table:
                        if len(str(row["match_id"])) > 0 and str(row["match_id"]) != "nan":
                            outline_color="green"
                    if row["removed_from_ensemble"]:
                        outline_color="red"
                    if vsx_ids_in_photometry_table:
                        if len(str(row["vsx_id"])) > 0 and str(row["vsx_id"]) != "nan":
                            outline_color="yellow"
                    self.create_circle(x=row["x_fit"] * self.zoom_level, y=row["y_fit"] * self.zoom_level,
                                       r=self.fit_shape / 2 * self.zoom_level, canvas_name=self.canvas, outline_color=outline_color)
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))

    def hide_photometry(self):
        self.photometry_results_plotted = False
        self.display_image()

    def plot_sigma_heatmap(self):
        global image_width, image_height
        try:
            heatmap = np.empty((image_width, image_height))
            fig, ax = plt.subplots()
            if os.path.isfile(self.image_file+".phot"):
                self.results_tab_df = pd.read_csv(self.image_file + ".phot")
                for index, row in self.results_tab_df.iterrows():
                    heatmap[int(row["x_fit"]), int(row["y_fit"])] = row["sigma_fit"]
                im = ax.imshow(heatmap)
                plt.show()


        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))


    def match_photometry_table(self, x, y, r=5):
        x_criterion = self.results_tab_df['x_fit'] < (x + r)
        matched_objects = self.results_tab_df[x_criterion]
        x_criterion = self.results_tab_df['x_fit'] > (x - r)
        matched_objects = matched_objects[x_criterion]
        y_criterion = self.results_tab_df['y_fit'] < (y + r)
        matched_objects = matched_objects[y_criterion]
        y_criterion = self.results_tab_df['y_fit'] > (y - r)
        matched_objects = matched_objects[y_criterion]
        if len(matched_objects) > 0:
            return(matched_objects.iloc[0]["x_fit"],
                   matched_objects.iloc[0]["y_fit"],
                   matched_objects.iloc[0]["flux_fit"],
                   matched_objects.iloc[0]["sigma_fit"],
                   matched_objects.iloc[0]["inst_mag"],
                   matched_objects.iloc[0]["flux_unc"],
                   matched_objects.iloc[0]["inst_mag_min"],
                   matched_objects.iloc[0]["inst_mag_max"],
                   )
        else:
            return(0, 0, 0, 0, 0, 0, 0, 0)

    def mouse_click(self, event):
        global image_data
        self.last_clicked_differential_magnitude = 0
        self.last_clicked_differential_uncertainty = 0
        decimal_places = int(self.decimal_places_entry.get())
        vsx_ids_in_photometry_table = False
        if "vsx_id" in self.results_tab_df:
            vsx_ids_in_photometry_table = True
        self.display_image()
        self.console_msg("")
        x = int(self.canvas.canvasx(event.x) / self.zoom_level)
        y = int(self.canvas.canvasy(event.y) / self.zoom_level)
        self.last_clicked_x = x
        self.last_clicked_y = y
        ADU = image_data[y-1, x-1]
        sky = self.wcs_header.pixel_to_world(x, y)
        sky_coordinate_string = ""
        if hasattr(sky, 'ra'):
            c = SkyCoord(ra=sky.ra, dec=sky.dec)
            sky_coordinate_string = "α δ: "+c.to_string("hmsdms")
        self.console_msg("Position X: "+str(x)+"\t Y: "+str(y)+"\t ADU: "+str(ADU) + "\t\t\t" + sky_coordinate_string)
        psf_canvas_x = x
        psf_canvas_y = y
        if self.photometry_results_plotted:
            x_fit, y_fit, flux_fit, sigma_fit, inst_mag, flux_unc, inst_mag_min, inst_mag_max = self.match_photometry_table(x, y)
            sky = self.wcs_header.pixel_to_world(x_fit, y_fit)
            sky_coordinate_string = ""
            if hasattr(sky, 'ra'):
                c = SkyCoord(ra=sky.ra, dec=sky.dec)
                sky_coordinate_string = " α δ: " + c.to_string("hmsdms")
            if x_fit != 0 and y_fit != 0:
                psf_canvas_x = x_fit
                psf_canvas_y = y_fit
            if str(x_fit)+str(y_fit) in self.photometry_circles:
                self.canvas.delete(self.photometry_circles[str(x_fit)+str(y_fit)])
            self.canvas.create_line(x_fit*self.zoom_level, y_fit*self.zoom_level - 35*self.zoom_level, x_fit*self.zoom_level, y_fit*self.zoom_level - 10*self.zoom_level, fill="white") # Draw "target" lines
            self.canvas.create_line(x_fit*self.zoom_level+35*self.zoom_level, y_fit*self.zoom_level, x_fit*self.zoom_level + 10*self.zoom_level, y_fit*self.zoom_level, fill="white")
            self.console_msg("Photometry fits, X: " + str(round(x_fit,2)) + " Y: " + str(round(y_fit,2)) + " Flux (ADU): " + str(
                round(flux_fit,2)) + " Instrumental magnitude: " + str(round(inst_mag,2)) + " PSF σ: " + str(round(sigma_fit, 2)) +sky_coordinate_string)
            self.set_entry_text(self.object_name_entry, "")     # Reset object name field in the left panel to avoid user mistakes
            if "match_id" in self.results_tab_df:
                matching_star_criterion = (self.results_tab_df["x_fit"] == x_fit) & (self.results_tab_df["y_fit"] == y_fit)
                if len(self.results_tab_df[matching_star_criterion]) > 0:
                    matching_star = self.results_tab_df[matching_star_criterion].iloc[0]
                    if type(matching_star["match_id"]) in (str, int, np.float64):
                        self.console_msg(
                            "Matching catalog source ID: " + str(matching_star["match_id"]) + " magnitude: " + str(
                                matching_star["match_mag"]))
                        self.set_entry_text(self.object_name_entry, str(matching_star["match_id"]))
                    if vsx_ids_in_photometry_table:
                        if len(str(matching_star["vsx_id"])) > 0 and str(matching_star["vsx_id"]) != "nan":
                            self.console_msg("Matching VSX Source: " + str(matching_star["vsx_id"]))
                            self.set_entry_text(self.object_name_entry, str(matching_star["vsx_id"]))
                        if len(str(matching_star["nearby_vsx_id"])) > 0 and str(
                                matching_star["nearby_vsx_id"]) != "nan" and self.nearby_vsx_var.get():
                            self.console_msg(
                                "Nearby VSX Source: " + str(matching_star["nearby_vsx_id"]) + " separation: " + str(
                                    matching_star["nearby_vsx_separation"]))

            if self.a != 0 and self.b != 0 and x_fit != 0 and y_fit != 0:
                differential_magnitude = round(inst_mag * self.a + self.b, decimal_places)

                #poisson_noise_error = 0
                #if self.results_tab_df["flux_unc"].sum() == 0:
                    # Aperture photometry mode - use Poisson noise error
                    #snr = flux_fit/self.bkg_value
                    #poisson_noise_error = 1.0857 / snr

                photometry_error_lower = abs(inst_mag - inst_mag_min)
                photometry_error_upper = abs(inst_mag - inst_mag_max)
                photometry_error = photometry_error_lower
                if photometry_error_upper > photometry_error_lower:
                    photometry_error = photometry_error_upper

                total_error = np.nan
                if "mag_error" in self.results_tab_df:
                    if self.results_tab_df["flux_unc"].sum() != 0:
                        total_error = round(np.sqrt(self.linreg_error ** 2 + photometry_error ** 2), decimal_places)
                        self.console_msg("Linear Regression Error: " + str(round(self.linreg_error, decimal_places)) + " Photometry Error: " + str(round(photometry_error, decimal_places)))
                    #if self.results_tab_df["flux_unc"].sum() == 0:
                    #    total_error = round(np.sqrt(self.linreg_error ** 2 + poisson_noise_error ** 2), decimal_places)
                    #    self.console_msg("Linear Regression Error: " + str(round(self.linreg_error, decimal_places)) + " Poisson Noise Error: " + str(round(poisson_noise_error, decimal_places)))

                obj_name_string = ""
                if type(matching_star["match_id"]) in (str, int, np.float64) and str(matching_star["match_id"]) != "nan":
                    obj_name_string = matching_star["match_id"] + ", "
                if vsx_ids_in_photometry_table and len(str(matching_star["vsx_id"])) > 0 and str(matching_star["vsx_id"]) != "nan":
                    obj_name_string = str(matching_star["vsx_id"]) + ", "
                self.console_msg("Differential ensemble photometry magnitude:")
                jd_time = Time(self.jd, format='jd')
                self.console_msg(obj_name_string + str(jd_time.to_value('iso')) + " UTC, " + self.filter_entry.get() + " = " + str(round(differential_magnitude, decimal_places)) + " ± " + str(
                            total_error))

                self.last_clicked_differential_magnitude = round(differential_magnitude, decimal_places)
                self.last_clicked_differential_uncertainty = total_error
                # Checking PSF sigma
                median_psf_sigma = self.results_tab_df["sigma_fit"].median()
                if sigma_fit > median_psf_sigma*1.2:
                    self.console_msg("WARNING: This source has PSF σ exceeding median value by more than 20%. Check PSF shape - likely non-linear sensor regime or improperly subtracted sky background.")
        self.update_PSF_canvas(psf_canvas_x, psf_canvas_y)

    def update_PSF_canvas(self, x, y):
        global image_data
        global FITS_minimum
        global FITS_maximum
        try:
            if len(image_data) > 0:
                self.fit_shape = int(self.photometry_aperture_entry.get())
                x0 = int(x - (self.fit_shape - 1) / 2)
                y0 = int(y - (self.fit_shape - 1) / 2)
                x1 = int(x + (self.fit_shape - 1) / 2)
                y1 = int(y + (self.fit_shape - 1) / 2)
                position = (x, y)
                size = (self.fit_shape - 1, self.fit_shape - 1)
                data = Cutout2D(image_data, position, size).data
                x = np.arange(x0, x1, 1)
                y = np.arange(y0, y1, 1)
                x, y = np.meshgrid(x, y)
                self.psf_plot.clear()
                self.psf_plot.plot_surface(x, y, data, cmap=cm.jet)
                self.psf_plot_canvas.draw()
        except Exception as e:
            self.error_raised = True
            pass


    def update_PSF_canvas_2d(self, x, y):
        global image_data
        global FITS_minimum
        global FITS_maximum
        image_crop = Image.fromarray(image_data)
        self.fit_shape = int(self.photometry_aperture_entry.get())
        x0 = int(x - (self.fit_shape-1)/2)
        y0 = int(y - (self.fit_shape-1)/2)
        x1 = int(x + (self.fit_shape-1)/2)
        y1 = int(y + (self.fit_shape-1)/2)
        image_crop = image_crop.crop((x0, y0, x1, y1))
        image_crop = image_crop.resize((300, 300), resample=0)
        image_crop = ImageMath.eval("a * 255 / " + str(self.histogram_slider_high / 100 * FITS_maximum), a=image_crop)
        self.image_crop = ImageTk.PhotoImage(image_crop)
        self.psf_canvas.create_image(0, 0, anchor=tk.NW, image=self.image_crop)


    def zoom_in(self):
        self.canvas.scale("all", 0, 0, 1+self.zoom_step, 1+self.zoom_step)
        self.zoom_level = self.zoom_level * (1+self.zoom_step)
        self.console_msg("Zoom: "+str(self.zoom_level))
        self.display_image()

    def zoom_out(self):
        self.canvas.scale("all", 0, 0, 1-self.zoom_step, 1-self.zoom_step)
        self.zoom_level = self.zoom_level * (1-self.zoom_step)
        self.console_msg("Zoom: " + str(self.zoom_level))
        self.display_image()

    def zoom_100(self):
        self.canvas.scale("all", 0, 0, 1, 1)
        self.zoom_level = 1
        self.console_msg("Zoom: " + str(self.zoom_level))
        self.display_image()

    def solve_image(self):
        global generated_image
        global header
        self.console_msg("Solving via Astrometry.Net..")
        try:

            ast = AstrometryNet()
            ast.api_key = self.astrometrynet_key_entry.get()
            ast.URL = "http://" + self.astrometrynet_entry.get()
            ast.API_URL = "http://" + self.astrometrynet_entry.get() + "/api"
            sources_df = self.results_tab_df.sort_values("flux_fit", ascending=False)
            width, height = generated_image.size

            self.wcs_header = ast.solve_from_source_list(sources_df['x_fit'], sources_df['y_fit'],
                                                         width, height,
                                                         solve_timeout=360)

            self.console_msg(
                "Astrometry.Net solution reference point RA: " + str(self.wcs_header["CRVAL1"]) + " Dec: " + str(
                    self.wcs_header["CRVAL2"]))
            header = header + self.wcs_header
            self.wcs_header = WCS(header)
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))

    def get_comparison_stars(self):
        global image_width, image_height
        try:
            self.filter = self.filter_entry.get()

            frame_center = self.wcs_header.pixel_to_world(int(image_width / 2), (image_height / 2))
            frame_center_coordinates = SkyCoord(ra=frame_center.ra, dec=frame_center.dec)
            frame_edge = self.wcs_header.pixel_to_world(int(image_width), (image_height / 2))
            frame_edge_coordinates = SkyCoord(ra=frame_edge.ra, dec=frame_edge.dec)
            frame_radius = frame_edge.separation(frame_center)
            self.console_msg(
                "Inquiring VizieR, center α δ " + frame_center.to_string("hmsdms") + ", radius " + str(frame_radius))

            if self.catalog_stringvar.get() == "APASS DR9":
                catalog = "II/336"
                ra_column_name = "RAJ2000"
                dec_column_name = "DEJ2000"

            if self.catalog_stringvar.get() == "URAT1":
                catalog = "I/329"
                ra_column_name = "RAJ2000"
                dec_column_name = "DEJ2000"

            if self.catalog_stringvar.get() == "USNO-B1.0":
                catalog = "I/284"
                ra_column_name = "RAJ2000"
                dec_column_name = "DEJ2000"

            if self.catalog_stringvar.get() == "VizieR Catalog":
                catalog = self.vizier_catalog_entry.get()
                ra_column_name = "RAJ2000"
                dec_column_name = "DEJ2000"

            if self.catalog_stringvar.get() == "Gaia DR2":
                catalog = "I/345"
                ra_column_name = "RA_ICRS"
                dec_column_name = "DE_ICRS"

            vizier_mag_column = self.filter + "mag"

            comparison_stars = Vizier(catalog=catalog, row_limit=-1).query_region(frame_center, frame_radius)[0]
            self.console_msg("Found " + str(len(comparison_stars)) + " objects in the field.")
            # print(comparison_stars)
            if vizier_mag_column not in comparison_stars.colnames:
                self.console_msg(
                    "Catalog " + self.catalog_stringvar.get() + " does not list " + self.filter + " magnitudes.")
                return

            if True:
                self.console_msg("Updating photometry table with sky coordinates..")
                for index, row in self.results_tab_df.iterrows():
                    sky = self.wcs_header.pixel_to_world(row["x_fit"], row["y_fit"])
                    c = SkyCoord(ra=sky.ra, dec=sky.dec)
                    self.results_tab_df.loc[index, "ra_fit"] = c.ra / u.deg
                    self.results_tab_df.loc[index, "dec_fit"] = c.dec / u.deg
                self.results_tab_df.to_csv(self.image_file + ".phot", index=False)

            if True:
                self.console_msg("Matching catalogs..")
                matching_radius = float(self.matching_radius_entry.get()) * 0.000277778  # arcsec to degrees
                catalog_comparison = SkyCoord(comparison_stars[ra_column_name], comparison_stars[dec_column_name])
                for index, row in self.results_tab_df.iterrows():
                    photometry_star_coordinates = SkyCoord(ra=row["ra_fit"] * u.deg, dec=row["dec_fit"] * u.deg)
                    match_index, d2d_match, d3d_match = photometry_star_coordinates.match_to_catalog_sky(
                        catalog_comparison)
                    #print(str(photometry_star_coordinates))
                    #print(match_index)
                    match_id = comparison_stars[match_index][0]  # Name of the catalog
                    match_ra = comparison_stars[match_index][ra_column_name]
                    match_dec = comparison_stars[match_index][dec_column_name]
                    match_mag = comparison_stars[match_index][vizier_mag_column]
                    match_coordinates = SkyCoord(ra=match_ra * u.deg, dec=match_dec * u.deg)
                    separation = photometry_star_coordinates.separation(match_coordinates)
                    if separation < matching_radius * u.deg:
                        self.results_tab_df.loc[index, "match_id"] = str(self.catalog_stringvar.get()) + " " + str(
                            match_id)
                        self.results_tab_df.loc[index, "match_ra"] = match_ra
                        self.results_tab_df.loc[index, "match_dec"] = match_dec
                        self.results_tab_df.loc[index, "match_mag"] = match_mag
                    else:
                        self.results_tab_df.loc[index, "match_id"] = ""
                        self.results_tab_df.loc[index, "match_ra"] = ""
                        self.results_tab_df.loc[index, "match_dec"] = ""
                        self.results_tab_df.loc[index, "match_mag"] = ""

                if self.remove_vsx_var.get():
                    self.console_msg("Inquiring VizieR for VSX variables in the field..")
                    vsx_result = Vizier(catalog="B/vsx/vsx", row_limit=-1).query_region(frame_center, frame_radius)
                    #print(vsx_result)
                    if len(vsx_result) > 0:
                        vsx_stars = vsx_result[0]
                        self.console_msg("Found " + str(len(vsx_stars)) + " VSX sources in the field. Matching..")
                        catalog_vsx = SkyCoord(vsx_stars["RAJ2000"], vsx_stars["DEJ2000"])
                        for index, row in self.results_tab_df.iterrows():
                            photometry_star_coordinates = SkyCoord(ra=row["ra_fit"] * u.deg, dec=row["dec_fit"] * u.deg)
                            match_index, d2d_match, d3d_match = photometry_star_coordinates.match_to_catalog_sky(
                                catalog_vsx)
                            match_id = vsx_stars[match_index]["Name"]  # Name of the catalog
                            match_ra = vsx_stars[match_index]["RAJ2000"]
                            match_dec = vsx_stars[match_index]["DEJ2000"]
                            match_coordinates = SkyCoord(ra=match_ra * u.deg, dec=match_dec * u.deg)
                            separation = photometry_star_coordinates.separation(match_coordinates)
                            if separation < matching_radius * u.deg:
                                self.results_tab_df.loc[index, "vsx_id"] = str(match_id)
                            else:
                                self.results_tab_df.loc[index, "vsx_id"] = ""
                            # Check 30 arcsec vicinity for nearby VSX sources:
                            if separation < 0.00833333 * u.deg:
                                self.results_tab_df.loc[index, "nearby_vsx_id"] = str(match_id)
                                self.results_tab_df.loc[index, "nearby_vsx_separation"] = separation
                            else:
                                self.results_tab_df.loc[index, "nearby_vsx_id"] = ""
                                self.results_tab_df.loc[index, "nearby_vsx_separation"] = ""
                    else:
                        self.console_msg("Found no VSX sources in the field.")
                self.results_tab_df.to_csv(self.image_file + ".phot", index=False)
                self.console_msg("Photometry table saved to " + str(self.image_file + ".phot"))
                self.display_image()
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno) + " " + str(e))

    def set_entry_text(self, entry, text):
        entry.delete(0, tk.END)
        entry.insert(0, text)

    def update_display(self):
        self.display_image()
        self.update_PSF_canvas(self.last_clicked_x, self.last_clicked_y)

    def plot_photometry_menu_action(self):
        self.plot_photometry()
        self.update_display()

    def safe_float_convert(self, x):
        try:
            z = float(x)
            if np.isnan(z):
                return False    # Nan!
            return True  # numeric, success!
        except ValueError:
            return False  # not numeric
        except TypeError:
            return False  # null type

    def find_linear_regression_model(self):
        # Select stars with catalog matches with known magnitudes
        self.console_msg("Finding linear regression model with ensemble stars..")
        vsx_ids_in_photometry_table = False
        if "vsx_id" in self.results_tab_df:
            vsx_ids_in_photometry_table = True
        try:
            min_comparison_magnitude = int(self.min_ensemble_magnitude_entry.get())
            max_comparison_magnitude = int(self.max_ensemble_magnitude_entry.get())

            # Only stars with known magnitude from catalogs
            mask = self.results_tab_df['match_mag'].map(self.safe_float_convert)
            ensemble_stars = self.results_tab_df.loc[mask]

            # Only stars without VSX id
            if self.remove_vsx_var.get() and vsx_ids_in_photometry_table:
                self.console_msg("Ensemble size before removal of VSX sources: "+str(len(ensemble_stars)))
                mask = ensemble_stars['vsx_id'].isnull()
                ensemble_stars = ensemble_stars.loc[mask]

            # Convert magnitudes to floats
            ensemble_stars["inst_mag"] = ensemble_stars["inst_mag"].astype(float)
            ensemble_stars["match_mag"] = ensemble_stars["match_mag"].astype(float)

            # Filter by magnitude
            mask = ensemble_stars['match_mag'] < max_comparison_magnitude
            ensemble_stars = ensemble_stars.loc[mask]
            mask = ensemble_stars['match_mag'] > min_comparison_magnitude
            ensemble_stars = ensemble_stars.loc[mask]

            # Remove removed outliers
            mask = ensemble_stars['removed_from_ensemble']==False
            ensemble_stars = ensemble_stars.loc[mask]

            weights = None

            if self.weighting_stringvar.get() == "Raw Flux":
                ensemble_stars = ensemble_stars.sort_values(by=['flux_0'], ascending=False)
                ensemble_stars = ensemble_stars.head(n=int(self.ensemble_limit_entry.get()))
                weights = ensemble_stars["flux_0"]

            if self.weighting_stringvar.get() == "Instrumental Magnitude":
                ensemble_stars = ensemble_stars.sort_values(by=['inst_mag'], ascending=True)
                ensemble_stars = ensemble_stars.head(n=int(self.ensemble_limit_entry.get()))
                weights = 1/ensemble_stars["inst_mag"]

            if self.weighting_stringvar.get() == "PSF Sigma":
                ensemble_stars = ensemble_stars.sort_values(by=['sigma_fit'], ascending=True)
                ensemble_stars = ensemble_stars.head(n=int(self.ensemble_limit_entry.get()))
                weights = 1/ensemble_stars["sigma_fit"]

            self.console_msg("Using "+str(len(ensemble_stars))+" ensemble stars.")

            x = ensemble_stars["inst_mag"]
            y = ensemble_stars["match_mag"]

            self.a, self.b = np.polyfit(x, y, deg=1, w=weights)  # Fit a 1st degree polynomial
            fit_fn = np.poly1d((self.a, self.b))
            yhat = fit_fn(x)
            ybar = np.sum(y) / len(y)
            ssreg = np.sum((yhat - ybar) ** 2)
            sstot = np.sum((y - ybar) ** 2)
            r_squared = ssreg / sstot
            self.console_msg(
                "Linear regression fit: a = " + str(self.a) + " b = " + str(self.b) + " r^2 = " + str(r_squared))

            self.linreg_plot.clear()
            self.linreg_plot.plot(x, y, 'ro', ms=1)
            self.linreg_plot.plot(x, fit_fn(x), '--k', ms=1)
            self.linreg_plot.text(0.1, 0.85, "n = "+str(len(ensemble_stars)), transform=self.linreg_plot.transAxes)
            self.plot_canvas.draw()

            self.ensemble_size = len(ensemble_stars)

            for index, row in self.results_tab_df.iterrows():
                self.results_tab_df.loc[index, "differential_mag"] = round(float(row["inst_mag"]) * self.a + self.b, 3)

            for index, row in ensemble_stars.iterrows():
                ensemble_stars.loc[index, "differential_mag"] = round(float(row["inst_mag"]) * self.a + self.b, 3)

            for index, row in self.results_tab_df.iterrows():
                try:
                    self.results_tab_df.loc[index, "mag_error"] = abs(
                        float(row["differential_mag"]) - float(row["match_mag"]))
                except:
                    continue

            for index, row in ensemble_stars.iterrows():
                try:
                    #ensemble_stars.loc[index, "mag_error"] = abs(
                    #    float(row["differential_mag"]) - float(row["match_mag"]))
                    ensemble_stars.loc[index, "mag_error_squared"] = (float(row["differential_mag"]) - float(row["match_mag"]))**2
                except:
                    continue
            self.linreg_error = np.sqrt(ensemble_stars["mag_error_squared"].sum() / (len(ensemble_stars)-1))        # Standard deviation

            self.results_tab_df.to_csv(self.image_file + ".phot", index=False)
            self.console_msg("Photometry table with differential magnitudes saved to " + str(self.image_file + ".phot"))
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))

    def remove_fit_outlier(self):
        # Find not yet removed outlier within the ensemble magnitude range
        self.console_msg("Processing..")
        error = 0
        outlier_index = 0
        min_comparison_magnitude = int(self.min_ensemble_magnitude_entry.get())
        max_comparison_magnitude = int(self.max_ensemble_magnitude_entry.get())
        for index, row in self.results_tab_df.iterrows():
            if abs(row["mag_error"]) > error and row["removed_from_ensemble"] == False and float(row[
                "match_mag"]) > min_comparison_magnitude and float(row["match_mag"]) < max_comparison_magnitude:
                outlier_index = index
                error = abs(row["mag_error"])
        self.results_tab_df.loc[outlier_index, "removed_from_ensemble"] = True
        self.find_linear_regression_model()
        self.update_display()

    def remove_distant_fit_outliers(self):
        # Removing outliers further than X arcseconds from the target
        self.console_msg("Processing..")
        max_separation = int(self.max_outliers_separation_entry.get())
        target_coordinates = self.wcs_header.pixel_to_world(self.last_clicked_x, self.last_clicked_y)
        n = 0
        for index, row in self.results_tab_df.iterrows():
            comparison_coordinates = self.wcs_header.pixel_to_world(row["x_fit"], row["y_fit"])
            separation = comparison_coordinates.separation(target_coordinates)
            if separation > max_separation * 0.000277778 * u.deg:
                self.results_tab_df.loc[index, "removed_from_ensemble"] = True
                n = n + 1
        self.console_msg("Removed "+str(n)+" sources from the ensemble as distant outliers.")
        self.find_linear_regression_model()
        self.update_display()

    def remove_fit_outliers_until_ensemble_limit(self):
        while self.ensemble_size > int(self.ensemble_limit_entry.get()):
            self.remove_fit_outlier()

    def reset_fit_outliers(self):
        self.console_msg("Processing..")
        self.results_tab_df["removed_from_ensemble"] = False
        self.find_linear_regression_model()
        self.update_display()

    def delete_photometry_file(self):
        if os.path.isfile(self.image_file + ".phot"):
            os.remove(self.image_file + ".phot")
            self.console_msg("Photometry file deleted.")
            self.a = 0
            self.b = 0
            self.update_display()

    def update_histogram_low(self, value):
        self.histogram_slider_low = int(value)
        self.update_display()

    def update_histogram_high(self, value):
        self.histogram_slider_high = int(value)
        self.update_display()

    def save_settings_as(self):
        file_name = fd.asksaveasfile(defaultextension='.mpsf')
        try:
            if len(str(file_name)) > 0:
                self.console_msg("Saving settings as " + str(file_name.name))
                settings = {}
                settings.update({'photometry_aperture_entry': self.photometry_aperture_entry.get()})
                settings.update({'min_ensemble_magnitude_entry': self.min_ensemble_magnitude_entry.get()})
                settings.update({'max_ensemble_magnitude_entry': self.max_ensemble_magnitude_entry.get()})
                settings.update({'fwhm_entry': self.fwhm_entry.get()})
                settings.update({'star_detection_threshold_entry': self.star_detection_threshold_entry.get()})
                settings.update({'photometry_iterations_entry': self.photometry_iterations_entry.get()})
                settings.update({'sharplo_entry': self.sharplo_entry.get()})
                settings.update({'bkg_filter_size_entry': self.bkg_filter_size_entry.get()})
                settings.update({'matching_radius_entry': self.matching_radius_entry.get()})
                settings.update({'ensemble_limit_entry': self.ensemble_limit_entry.get()})
                settings.update({'decimal_places_entry': self.decimal_places_entry.get()})
                settings.update({'obscode_entry': self.obscode_entry.get()})
                settings.update({'aavso_obscode_entry': self.aavso_obscode_entry.get()})
                settings.update({'latitude_entry': self.latitude_entry.get()})
                settings.update({'longitude_entry': self.longitude_entry.get()})
                settings.update({'height_entry': self.height_entry.get()})
                settings.update({'telescope_entry': self.telescope_entry.get()})
                settings.update({'telescope_design_entry': self.accessory_entry.get()})
                settings.update({'ccd_entry': self.ccd_entry.get()})
                settings.update({'detector_entry': self.ccd_gain_entry.get()})
                settings.update({'weighting_stringvar': self.weighting_stringvar.get()})
                settings.update({'catalog_stringvar': self.catalog_stringvar.get()})
                settings.update({'vizier_catalog_entry': self.vizier_catalog_entry.get()})
                settings.update({'remove_vsx_var': self.remove_vsx_var.get()})
                settings.update({'nearby_vsx_var': self.nearby_vsx_var.get()})
                settings.update({'max_outliers_separation_entry': self.max_outliers_separation_entry.get()})
                settings.update({'fitter_stringvar': self.fitter_stringvar.get()})
                settings.update({'batch_psf_var': self.batch_psf_var.get()})
                settings.update({'crop_fits_entry': self.crop_fits_entry.get()})
                settings.update({'astrometrynet_entry': self.astrometrynet_entry.get()})
                settings.update({'astrometrynet_key_entry': self.astrometrynet_key_entry.get()})
                with open(str(file_name.name), 'w') as f:
                    w = csv.DictWriter(f, settings.keys())
                    w.writeheader()
                    w.writerow(settings)
                self.console_msg("Saved.")
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))

    def load_settings(self, file_name=""):
        try:
            if len(file_name) == 0:
                file_name = fd.askopenfilename()
            if len(str(file_name)) > 0:
                self.console_msg("Loading settings from " + str(file_name))
                settings = {}
                with open(str(file_name)) as f:
                    r = csv.DictReader(f)
                    for row in r:
                        row=dict(row)   # dict from OrderedDict, required for Python < 3.8 as DictReader behavior changed
                        settings.update(row)    # append settings dictionary with the read row
                    for key in settings:
                        if type(getattr(self, key)) == tk.Entry:
                            self.set_entry_text(getattr(self, key), settings[key])
                        if type(getattr(self, key)) == tk.StringVar:
                            getattr(self, key).set(settings[key])
                        if type(getattr(self, key)) == tk.BooleanVar:
                            getattr(self, key).set(settings[key])
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno)+" "+str(e))

    def generate_baa_report(self):
        global image_width, image_height
        report_dir = "baa_reports"
        if os.path.exists(os.path.dirname(self.image_file)):
            self.console_msg("Changing folder to " + os.path.dirname(self.image_file))
            os.chdir(os.path.dirname(self.image_file))
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        image_basename = os.path.basename(self.image_file)
        report_filename = os.path.join(report_dir, "BAA " + os.path.splitext(image_basename)[0] + " " + str(self.object_name_entry.get()) + ".txt")
        try:
            with open(report_filename, mode='w') as f:
                f.write("File Format\tCCD/DSLR v2.01\n")
                f.write("Observation Method\tCCD\n")
                f.write("Variable\t" + str(self.object_name_entry.get()) + "\n")
                # Generate Chart ID
                chart_id = self.catalog_stringvar.get()
                if self.catalog_stringvar.get() == "VizieR Catalog":
                    chart_id = "VizieR " + self.vizier_catalog_entry.get()
                frame_center = self.wcs_header.pixel_to_world(int(image_width / 2), (image_height / 2))
                frame_top_left = self.wcs_header.pixel_to_world(0, 0)
                frame_bottom_left = self.wcs_header.pixel_to_world(0, image_height)
                frame_top_right = self.wcs_header.pixel_to_world(image_width, 0)
                frame_center_string = frame_center.to_string("hmsdms", precision=0, sep="")
                fov_horizontal = frame_top_right.separation(frame_top_left).arcminute
                fov_vertical = frame_top_left.separation(frame_bottom_left).arcminute
                chart_id = chart_id + " RA Dec:" + frame_center_string + " FOV:" + str(
                    int(fov_horizontal)) + "'x" + str(int(fov_vertical)) + "'"
                if len(chart_id) > 50:
                    self.console_msg("Chart ID length is longer than 50 bytes - correct manually!")
                f.write("Chart ID\t" + chart_id + "\n")
                f.write("Observer code\t" + str(self.obscode_entry.get()) + "\n")
                f.write(
                    "Location\t" + str(self.latitude_entry.get()) + " " + str(self.longitude_entry.get()) + " H" + str(
                        self.height_entry.get()) + "m\n")
                f.write("Telescope\t" + str(self.telescope_entry.get()) + " " + str(self.accessory_entry.get()) + "\n")
                f.write("Camera\t" + str(self.ccd_entry.get()) + "\n")
                f.write("Magnitude type\tPreCalculated\n")
                f.write("Photometry software\t" + self.program_name + "\n")
                f.write("\n")
                f.write("JulianDate\tFilter\tVarAbsMag\tVarAbsErr\tExpLen\tCmpStar\n")
                f.write(
                    str(round(float(self.exposure_start_entry.get()), 5)) + "\t" + str(self.filter_entry.get()) + "\t" + str(
                        self.last_clicked_differential_magnitude) + "\t" + str(
                        self.last_clicked_differential_uncertainty) +
                    "\t" + str(self.exposure_entry.get()) + "\t" + "Ensemble\n")
            self.console_msg("BAA Photometry Database report saved to " + str(report_filename))
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno) + " " + str(e))

    def aavso_get_comparison_stars(self, frame_center, filter_band='V', field_of_view=18.5, maglimit=20):
        try:
            ra = frame_center.to_string("hmsdms").split()[0].replace("h", " ").replace("m", " ").replace("s", "")
            dec = frame_center.to_string("hmsdms").split()[1].replace("d", " ").replace("m", " ").replace("s",
                                                                                                          "").replace(
                "+", "")
            r = requests.get(
                'https://www.aavso.org/apps/vsp/api/chart/?format=json&fov=' + str(field_of_view) + '&ra=' + str(
                    ra) + '&dec=' + str(dec) + '&maglimit=' + str(maglimit))
            chart_id = r.json()['chartid']
            self.console_msg('Downloaded AAVSO Comparison Star Chart ID ' + str(chart_id))
            result = pd.DataFrame(columns=["AUID", "RA", "Dec", "Mag"])
            for star in r.json()['photometry']:
                auid = star['auid']
                ra = star['ra']
                dec = star['dec']
                for band in star['bands']:
                    if band['band'] == filter_band:
                        mag = band['mag']
                result = result.append({
                    "AUID": auid,
                    "RA": ra,
                    "Dec": dec,
                    "Mag": mag
                }, ignore_index=True)
            return result
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno) + " " + str(e))

    def generate_aavso_report(self):
        global image_width, image_height
        report_dir = "aavso_reports"
        if os.path.exists(os.path.dirname(self.image_file)):
            self.console_msg("Changing folder to " + os.path.dirname(self.image_file))
            os.chdir(os.path.dirname(self.image_file))
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        image_basename = os.path.basename(self.image_file)
        report_filename = os.path.join(report_dir, "AAVSO " + os.path.splitext(image_basename)[0] + " " + str(self.object_name_entry.get()) + ".txt")
        frame_center = self.wcs_header.pixel_to_world(int(image_width / 2), (image_height / 2))
        frame_top_left = self.wcs_header.pixel_to_world(0, 0)
        frame_bottom_left = self.wcs_header.pixel_to_world(0, image_height)
        frame_top_right = self.wcs_header.pixel_to_world(image_width, 0)
        fov_horizontal = frame_top_right.separation(frame_top_left).arcminute
        fov_vertical = frame_top_left.separation(frame_bottom_left).arcminute

        self.console_msg("Getting AAVSO Comparison Stars..")
        comparison_stars = self.aavso_get_comparison_stars(frame_center, filter_band=str(self.filter_entry.get()), field_of_view=fov_horizontal, maglimit=self.max_ensemble_magnitude_entry.get())

        self.console_msg("Matching comparison stars to the photometry table..")
        matching_radius = float(self.matching_radius_entry.get()) * 0.000277778  # arcsec to degrees
        catalog_comparison = SkyCoord(comparison_stars["RA"], comparison_stars["Dec"], unit=(u.hourangle, u.deg))
        for index, row in self.results_tab_df.iterrows():
            photometry_star_coordinates = SkyCoord(ra=row["ra_fit"] * u.deg, dec=row["dec_fit"] * u.deg)
            match_index, d2d_match, d3d_match = photometry_star_coordinates.match_to_catalog_sky(
                catalog_comparison)
            match_id = comparison_stars.iloc[match_index]["AUID"]
            match_ra = comparison_stars.iloc[match_index]["RA"]
            match_dec = comparison_stars.iloc[match_index]["Dec"]
            match_mag = comparison_stars.iloc[match_index]["Mag"]
            match_coordinates = SkyCoord(ra=match_ra, dec=match_dec, unit=(u.hourangle, u.deg))
            separation = photometry_star_coordinates.separation(match_coordinates)
            if separation < matching_radius * u.deg:
                self.results_tab_df.loc[index, "auid"] = str(match_id)
                self.results_tab_df.loc[index, "auid_mag"] = float(match_mag)
            else:
                self.results_tab_df.loc[index, "auid"] = ""
                self.results_tab_df.loc[index, "auid_mag"] = np.nan

        self.console_msg("Searching for the best check star for magnitude "+str(self.last_clicked_differential_magnitude))

        self.results_tab_df["difference_between_differential_mag_and_auid_mag"] = abs(self.results_tab_df["differential_mag"] - self.results_tab_df["auid_mag"])
        self.results_tab_df["difference_between_target_mag_and_auid_mag"] = abs(float(self.last_clicked_differential_magnitude) - self.results_tab_df["auid_mag"])

        # Find comparison star that has the least magntiude difference to target star
        comparison_star = self.results_tab_df[self.results_tab_df["difference_between_target_mag_and_auid_mag"].eq(self.results_tab_df["difference_between_target_mag_and_auid_mag"].min())].iloc[0]
        self.console_msg("Found check star "+str(comparison_star["auid"])+" of magnitude "+str(comparison_star["auid_mag"])+", difference with current photometry: "+str(comparison_star["difference_between_differential_mag_and_auid_mag"]))
        self.console_msg("Removing this comparison star from ensemble..")
        self.results_tab_df.loc[self.results_tab_df["auid"] == comparison_star["auid"], "removed_from_ensemble"] = True
        self.console_msg("Finding new fit and differential magntiude..")
        self.find_linear_regression_model()
        x = self.last_clicked_x
        y = self.last_clicked_y
        self.canvas.xview(tk.MOVETO, 0)     # Reset canvas position
        self.canvas.yview(tk.MOVETO, 0)
        self.canvas.event_generate('<Button-1>', x=x, y=y)  # Simulate a mouse click to redo photometry
        width, height = generated_image.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.canvas.xview(tk.MOVETO, (x-canvas_width/2)/width)  # Center canvas on the object
        self.canvas.yview(tk.MOVETO, (y-canvas_height/2)/height)
        # Updating comparison star with new photometry
        comparison_star = self.results_tab_df[self.results_tab_df["difference_between_target_mag_and_auid_mag"].eq(
            self.results_tab_df["difference_between_target_mag_and_auid_mag"].min())].iloc[0]

        try:
            with open(report_filename, mode='w') as f:
                f.write("#TYPE=Extended\n")
                f.write("#OBSCODE="+self.aavso_obscode_entry.get()+"\n")
                f.write("#SOFTWARE="+self.program_name+"\n")
                f.write("#DELIM=,\n")
                f.write("#DATE=JD\n")
                f.write("#OBSTYPE=CCD\n")
                f.write("#STARID,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES\n")

                starid = str(self.object_name_entry.get())
                date = str(round(float(self.exposure_start_entry.get()), 5))
                mag =  str(self.last_clicked_differential_magnitude)
                merr = str(self.last_clicked_differential_uncertainty)
                filt = str(self.filter_entry.get())
                trans = "NO"
                mtype = "STD"
                cname = "ENSEMBLE"
                cmag = "na"
                kname = str(comparison_star["auid"])
                kmag = str(comparison_star["differential_mag"])
                amass = "na"
                group = "na"
                chart = self.catalog_stringvar.get()
                if self.catalog_stringvar.get() == "VizieR Catalog":
                    chart = "VizieR " + self.vizier_catalog_entry.get()
                notes = "na"
                f.write(starid+","+date+","+mag+","+merr+","+filt+","+trans+","+mtype+","+cname+","+cmag+","+kname+","+kmag+","+amass+","+group+","+chart+","+notes)

            self.console_msg("AAVSO Photometry Database report saved to " + str(report_filename))
        except Exception as e:
            self.error_raised = True
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.console_msg(str(exc_tb.tb_lineno) + " " + str(e))

    def next_vsx_source(self):
        mask = self.results_tab_df['vsx_id'].notnull()
        vsx_sources = self.results_tab_df.loc[mask]
        next_vsx_source_id = ""
        next_row = False
        if self.object_name_entry.get() == "":
            next_vsx_source_id = vsx_sources.iloc[0]['vsx_id']      # Reset to first VSX source
        if len(vsx_sources) > 0 and self.object_name_entry.get() != "":
            for index, row in vsx_sources.iterrows():
                if next_row:
                    next_vsx_source_id = row['vsx_id']
                    break
                if row['vsx_id'] == self.object_name_entry.get():
                    next_row = True
            if next_vsx_source_id == "":
                next_vsx_source_id = vsx_sources.iloc[0]['vsx_id']  # Reset to first VSX source
        self.console_msg("Next VSX Source: "+next_vsx_source_id)
        x = int(self.results_tab_df.loc[self.results_tab_df['vsx_id'] == next_vsx_source_id]['x_0'] * self.zoom_level)
        y = int(self.results_tab_df.loc[self.results_tab_df['vsx_id'] == next_vsx_source_id]['y_0'] * self.zoom_level)
        self.canvas.xview(tk.MOVETO, 0)     # Reset canvas position
        self.canvas.yview(tk.MOVETO, 0)
        self.canvas.event_generate('<Button-1>',x=x,y=y)  # Simulate a mouse click
        width, height = generated_image.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.canvas.xview(tk.MOVETO, (x-canvas_width/2)/width)  # Center canvas on the object
        self.canvas.yview(tk.MOVETO, (y-canvas_height/2)/height)

    def report_on_all_vsx_sources(self):
        mask = self.results_tab_df['vsx_id'].notnull()
        vsx_sources = self.results_tab_df.loc[mask]
        n = len(vsx_sources)
        for i in range(0, n):
            self.next_vsx_source()
            self.reset_fit_outliers()
            self.remove_distant_fit_outliers()
            self.remove_fit_outliers_until_ensemble_limit()
            x = self.last_clicked_x
            y = self.last_clicked_y
            self.canvas.xview(tk.MOVETO, 0)  # Reset canvas position
            self.canvas.yview(tk.MOVETO, 0)
            self.canvas.event_generate('<Button-1>', x=x, y=y)  # Simulate a mouse click to redo photometry
            self.console_msg("Generating AAVSO report..")
            self.generate_aavso_report()
            self.console_msg("Generating BAA report..")
            self.generate_baa_report()
        self.console_msg("Report on all VSX sources finished for this frame.")

    def batch_report_on_all_vsx_sources(self, batch_dir=""):
        self.error_raised = False
        if len(batch_dir) == 0:
            batch_dir = fd.askdirectory()
        if len(batch_dir) > 0:
            self.console_msg("Changing working directory to "+batch_dir)
            os.chdir(batch_dir)
            dir_list = os.listdir(batch_dir)
            for filename in dir_list:
                if os.path.splitext(filename)[1].lower() == ".fits" or os.path.splitext(filename)[1].lower() == ".fit":
                    try:
                        if self.error_raised:
                            self.console_msg("Aborting batch processing due to an error.")
                            break
                        full_filename=os.path.join(batch_dir, filename)
                        self.console_msg("Processing " + full_filename)
                        self.load_FITS(full_filename)
                        self.display_image()
                        if self.batch_psf_var.get():
                            self.perform_photometry()
                        else:
                            self.aperture_photometry()
                        self.solve_image()
                        self.get_comparison_stars()
                        self.console_msg("Finding linear fit..")
                        self.find_linear_regression_model()
                        self.console_msg("Reporting on all VSX sources..")
                        self.report_on_all_vsx_sources()
                    except Exception as e:
                        self.error_raised = True
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        self.console_msg(str(exc_tb.tb_lineno) + " " + str(e))
                        continue
            self.console_msg("Batch processing finished.")

    def display_curve_from_baa_reports(self):
        if len(self.object_name_entry.get()) == 0:
            self.console_msg("Enter object name or click on a source with name.")
            return
        dir = fd.askdirectory()
        object_name = self.object_name_entry.get()
        filter = self.filter_entry.get()
        light_curve = pd.DataFrame()
        if len(dir) > 0:
            dir_list = os.listdir(dir)
            for filename in dir_list:
                if object_name + ".txt" in filename:
                    self.console_msg("Adding point from "+filename)
                    full_filename = os.path.join(dir, filename)
                    point = pd.read_csv(full_filename, header=10, sep='\t')    # Skip to the 12th line with data
                    try:
                        light_curve = light_curve.append(point, ignore_index=True)
                    except Exception as e:
                        self.error_raised = True
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        self.console_msg(str(exc_tb.tb_lineno) + " " + str(e))
                        continue
            fig, ax = plt.subplots(figsize=(12,7))
            plt.gca().invert_yaxis()
            ax.xaxis.set_tick_params(which='major', size=6, width=1, direction='in', top='on')
            ax.xaxis.set_tick_params(which='minor', size=6, width=1, direction='in', top='on')
            ax.yaxis.set_tick_params(which='major', size=6, width=1, direction='in', right='on')
            ax.yaxis.set_tick_params(which='minor', size=6, width=1, direction='in', right='on')
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize('medium')
                item.set_family('serif')
            ax.xaxis.label.set_fontsize('x-large')
            ax.yaxis.label.set_fontsize('x-large')
            formatter = ScalarFormatter(useOffset=False)
            ax.xaxis.set_major_formatter(formatter)
            title = object_name+"\n"
            if len(self.obscode_entry.get()) > 0:
                title = title + "Observer code "+self.obscode_entry.get()+", "
            if len(self.telescope_entry.get()) > 0:
                title = title + self.telescope_entry.get()+", "
            if len(self.accessory_entry.get()) > 0:
                title = title + self.accessory_entry.get()+", "
            if len(self.ccd_entry.get()) > 0:
                title = title + self.ccd_entry.get()
            ax.set_title(title)
            ax.title.set_fontsize('x-large')
            ax.errorbar(light_curve["JulianDate"]-2400000.5, light_curve["VarAbsMag"], yerr=light_curve["VarAbsErr"], fmt="o", color="black")
            ax.set_xlabel('MJD', labelpad=10)
            ax.set_ylabel(filter, labelpad=10)
            plt.show()


    def __init__(self):
        self.window = tk.Tk()
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        self.program_name = "MetroPSF 0.16"

        # Matplotlib settings
        matplotlib.rc('xtick', labelsize=7)
        matplotlib.rc('ytick', labelsize=7)

        # Maximize that works everywhere
        m = self.window.maxsize()
        self.window.geometry('{}x{}+0+0'.format(*m))

        self.window.title(self.program_name)

        self.menubar = tk.Menu(self.window)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open...", command=self.open_FITS_file)
        self.filemenu.add_command(label="Save", command=self.save_FITS_file)
        self.filemenu.add_command(label="Save As...", command=self.save_FITS_file_as)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Load Settings...", command=self.load_settings)
        self.filemenu.add_command(label="Save Settings As...", command=self.save_settings_as)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.viewmenu = tk.Menu(self.menubar, tearoff=0)
        self.viewmenu.add_command(label="Update", command=self.update_display)
        self.viewmenu.add_separator()
        self.viewmenu.add_command(label="Zoom In", command=self.zoom_in)
        self.viewmenu.add_command(label="Zoom Out", command=self.zoom_out)
        self.viewmenu.add_command(label="100% Zoom", command=self.zoom_100)
        self.viewmenu.add_separator()
        self.viewmenu.add_command(label="Next VSX Source", command=self.next_vsx_source)
        self.menubar.add_cascade(label="View", menu=self.viewmenu)

        self.photometrymenu = tk.Menu(self.menubar, tearoff=0)
        self.photometrymenu.add_command(label="Iteratively Subtracted PSF Photometry", command=self.perform_photometry)
        self.photometrymenu.add_command(label="Aperture Photometry", command=self.aperture_photometry)
        self.photometrymenu.add_separator()
        self.photometrymenu.add_command(label="Plot", command=self.plot_photometry_menu_action)
        self.photometrymenu.add_command(label="Hide", command=self.hide_photometry)
        self.photometrymenu.add_separator()
        self.photometrymenu.add_command(label="Solve Image", command=self.solve_image)
        self.photometrymenu.add_command(label="Get Comparison Stars", command=self.get_comparison_stars)
        self.photometrymenu.add_command(label="Find Regression Model", command=self.find_linear_regression_model)
        self.photometrymenu.add_separator()
        self.photometrymenu.add_command(label="Remove Fit Outlier", command=self.remove_fit_outlier)
        self.photometrymenu.add_command(label="Remove Fit Outliers Until Ensemble Limit", command=self.remove_fit_outliers_until_ensemble_limit)
        self.photometrymenu.add_command(label="Remove Fit Outliers Beyond Separation Limit", command=self.remove_distant_fit_outliers)
        self.photometrymenu.add_command(label="Reset Fit Outliers", command=self.reset_fit_outliers)
        self.photometrymenu.add_separator()
        self.photometrymenu.add_command(label="Delete Photometry File", command=self.delete_photometry_file)
        self.photometrymenu.add_command(label="Display Background", command=self.display_background)
        #self.photometrymenu.add_command(label="Plot Sigma Heatmap", command=self.plot_sigma_heatmap)
        self.menubar.add_cascade(label="Photometry", menu=self.photometrymenu)

        self.reportmenu = tk.Menu(self.menubar, tearoff=0)
        self.reportmenu.add_command(label="BAA: Generate Report", command=self.generate_baa_report)
        self.reportmenu.add_command(label="BAA: Light Curve from Reports..", command=self.display_curve_from_baa_reports)
        self.reportmenu.add_separator()
        self.reportmenu.add_command(label="AAVSO: Generate Report", command=self.generate_aavso_report)
        self.reportmenu.add_separator()
        self.reportmenu.add_command(label="BAA/AAVSO Reports on All VSX Sources", command=self.report_on_all_vsx_sources)
        self.reportmenu.add_command(label="BAA/AAVSO Batch Reports on All VSX Sources..", command=self.batch_report_on_all_vsx_sources)

        self.menubar.add_cascade(label="Report", menu=self.reportmenu)

        self.window.config(menu=self.menubar)

        self.left_half = tk.Frame(self.window)  # Left half of the window
        self.left_half.grid(row=0, column=0, sticky=tk.NSEW)
        self.center = tk.Frame(self.window)  # Center of the window
        self.center.grid(row=0, column=1, sticky=tk.NSEW)
        self.right_half = tk.Frame(self.window)  # Right half of the window
        self.right_half.grid(row=0, column=2, sticky=tk.NSEW)
        tk.Grid.columnconfigure(self.window, 1, weight=1) # Expand center horizontally
        tk.Grid.rowconfigure(self.window, 0, weight=1) # Expand everything vertically


        self.filename_label = tk.Label(self.center, text="FITS:" + image_file)
        self.filename_label.grid(row=0, column=0) # Place label

        self.canvas = tk.Canvas(self.center, bg='black') # Main canvas
        self.canvas.grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W) # Place main canvas, sticky to occupy entire
                                                                      # cell dimensions
        tk.Grid.columnconfigure(self.center, 0, weight=1) # Expand main canvas column to fit whole window
        tk.Grid.rowconfigure(self.center, 1, weight=1) # Expand main canvas row to fit whole window
        self.canvas_scrollbar_V = tk.Scrollbar(self.center, orient=tk.VERTICAL) # Main canvas scrollbars
        self.canvas_scrollbar_V.grid(row=1, column=1)
        self.canvas_scrollbar_V.grid(sticky=tk.N+tk.S+tk.E+tk.W, column=1, row=1)
        self.canvas_scrollbar_H = tk.Scrollbar(self.center, orient=tk.HORIZONTAL)
        self.canvas_scrollbar_H.grid(row=2, column=0)
        self.canvas_scrollbar_H.grid(sticky=tk.N + tk.S + tk.E + tk.W, column=0, row=2)
        self.canvas_scrollbar_H.config(command=self.canvas.xview)
        self.canvas_scrollbar_V.config(command=self.canvas.yview)
        self.canvas.config(xscrollcommand=self.canvas_scrollbar_H.set)
        self.canvas.config(yscrollcommand=self.canvas_scrollbar_V.set)

        self.right_frame = tk.Frame(self.right_half) # We will lay out interface things into the new right_frame grid
        self.right_frame.grid(row=1, column=2, sticky=tk.N) # Place right_frame into the top of the main canvas row, right next to it
        #self.psf_canvas = tk.Canvas(self.right_frame, bg='grey', width=300, height=300) # Small PSF canvas
        self.fig_psf = Figure()
        self.psf_plot = self.fig_psf.add_subplot(111, projection='3d')
        self.psf_plot_canvas = FigureCanvasTkAgg(self.fig_psf, self.right_frame) # PSF 3D plot canvas - Matplotlib wrapper for Tk
        self.psf_plot_canvas.draw()
        self.psf_canvas = self.psf_plot_canvas.get_tk_widget()
        self.psf_canvas.config(width=int(self.screen_width/8.5), height=int(self.screen_width/8.5))
        self.psf_canvas.grid(row=0, column=0) # Allocate small PSF canvas to a new grid inside the right_frame

        self.fig = Figure()
        self.linreg_plot = self.fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, self.right_frame) # Linear regression canvas - Matplotlib wrapper for Tk
        self.plot_canvas.draw()
        self.linreg_canvas = self.plot_canvas.get_tk_widget()
        self.linreg_canvas.config(width=int(self.screen_width/8.5), height=int(self.screen_width/12))
        self.linreg_canvas.grid(row=1, column=0) # Allocate small PSF canvas to a new grid inside the right_frame

        self.left_frame = tk.Frame(self.left_half) # We will lay out interface things into the new right_frame grid
        self.left_frame.grid(row=0, column=0, sticky=tk.N) # Place right_frame into the top of the main canvas row, right next to it

        self.settings_frame = tk.Frame(self.left_frame)  # Frame to hold settings grid
        self.settings_frame.grid(row=2, column=0, sticky=tk.NSEW)  # Settings_frame under the canvas in the right_frame
        tk.Grid.columnconfigure(self.settings_frame, 0, weight=1)  # Expand settings_frame column that holds labels

        settings_entry_width = 6
        extended_settings_entry_width = 30

        row = 0

        self.photometry_aperture_label = tk.Label(self.settings_frame, text="Fitting Width/Height, px:")
        self.photometry_aperture_label.grid(row=row, column=0, sticky=tk.W)
        self.photometry_aperture_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.photometry_aperture_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.photometry_aperture_entry, self.fit_shape)
        row = row + 1

        self.min_ensemble_magnitude_label = tk.Label(self.settings_frame, text="Minimum Ensemble Magnitude:")
        self.min_ensemble_magnitude_label.grid(row=row, column=0, sticky=tk.W)
        self.min_ensemble_magnitude_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.min_ensemble_magnitude_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.min_ensemble_magnitude_entry, "7")
        row = row + 1

        self.max_ensemble_magnitude_label = tk.Label(self.settings_frame, text="Maximum Ensemble Magnitude:")
        self.max_ensemble_magnitude_label.grid(row=row, column=0, sticky=tk.W)
        self.max_ensemble_magnitude_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.max_ensemble_magnitude_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.max_ensemble_magnitude_entry, "20")
        row = row + 1

        self.fwhm_label = tk.Label(self.settings_frame, text="FWHM, px:")
        self.fwhm_label.grid(row=row, column=0, sticky=tk.W)
        self.fwhm_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.fwhm_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.fwhm_entry, "4")
        row = row + 1

        self.star_detection_threshold_label = tk.Label(self.settings_frame, text="Star Detection Threshold, σ:")
        self.star_detection_threshold_label.grid(row=row, column=0, sticky=tk.W)
        self.star_detection_threshold_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.star_detection_threshold_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.star_detection_threshold_entry, "10")
        row = row + 1

        self.photometry_iterations_label = tk.Label(self.settings_frame, text="Photometry Iterations:")
        self.photometry_iterations_label.grid(row=row, column=0, sticky=tk.W)
        self.photometry_iterations_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.photometry_iterations_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.photometry_iterations_entry, "1")
        row = row + 1

        self.sharplo_label = tk.Label(self.settings_frame, text="Lower Bound for Sharpness:")
        self.sharplo_label.grid(row=row, column=0, sticky=tk.W)
        self.sharplo_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.sharplo_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.sharplo_entry, "0.5")
        row = row + 1

        self.bkg_filter_size_label = tk.Label(self.settings_frame, text="Background Median Filter, px:")
        self.bkg_filter_size_label.grid(row=row, column=0, sticky=tk.W)
        self.bkg_filter_size_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.bkg_filter_size_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.bkg_filter_size_entry, "1")
        row = row + 1

        self.filter_label = tk.Label(self.settings_frame, text="CCD Filter:")
        self.filter_label.grid(row=row, column=0, sticky=tk.W)
        self.filter_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.filter_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.filter_entry, "")
        row = row + 1

        self.exposure_label = tk.Label(self.settings_frame, text="Exposure Time:")
        self.exposure_label.grid(row=row, column=0, sticky=tk.W)
        self.exposure_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.exposure_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.exposure_entry, "0")
        row = row + 1

        self.matching_radius_label = tk.Label(self.settings_frame, text="Matching Radius, arcsec:")
        self.matching_radius_label.grid(row=row, column=0, sticky=tk.W)
        self.matching_radius_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.matching_radius_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.matching_radius_entry, "2")
        row = row + 1

        self.ensemble_limit_label = tk.Label(self.settings_frame, text="Limit Ensemble to:")
        self.ensemble_limit_label.grid(row=row, column=0, sticky=tk.W)
        self.ensemble_limit_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.ensemble_limit_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.ensemble_limit_entry, "1000")
        row = row + 1

        self.decimal_places_label = tk.Label(self.settings_frame, text="Decimal Places to Report:")
        self.decimal_places_label.grid(row=row, column=0, sticky=tk.W)
        self.decimal_places_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.decimal_places_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.decimal_places_entry, "2")
        row = row + 1

        self.max_outliers_separation_label = tk.Label(self.settings_frame, text="Ensemble Outliers Separation Limit, arcsec:")
        self.max_outliers_separation_label.grid(row=row, column=0, sticky=tk.W)
        self.max_outliers_separation_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.max_outliers_separation_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.max_outliers_separation_entry, "300")
        row = row + 1

        self.crop_fits_label = tk.Label(self.settings_frame, text="FITS Crop, %:")
        self.crop_fits_label.grid(row=row, column=0, sticky=tk.W)
        self.crop_fits_entry = tk.Entry(self.settings_frame, width=settings_entry_width)
        self.crop_fits_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.crop_fits_entry, "100")
        row = row + 1


        self.astrometrynet_label = tk.Label(self.settings_frame, text="Astrometry.net Server:")
        self.astrometrynet_label.grid(row=row, column=0, stick=tk.W)
        self.astrometrynet_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width)
        self.astrometrynet_entry.grid(row=row, column=1, sticky=tk.E)
        self.set_entry_text(self.astrometrynet_entry, "nova.astrometry.net")
        row = row + 1

        self.astrometrynet_key_label = tk.Label(self.settings_frame, text="Astrometry.net API Key:")
        self.astrometrynet_key_label.grid(row=row, column=0, stick=tk.W)
        self.astrometrynet_key_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width)
        self.astrometrynet_key_entry.grid(row=row, column=1, sticky=tk.E)
        self.astrometrynet_key_entry.config(show="*")
        self.set_entry_text(self.astrometrynet_key_entry, "pwjgdcpwaugkhkln")
        row = row + 1

        self.obscode_label = tk.Label(self.settings_frame, text="BAA Observer Code:")
        self.obscode_label.grid(row=row, column=0, stick=tk.W)
        self.obscode_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.obscode_entry.grid(row=row, column=1, sticky=tk.E)
        row = row + 1

        self.aavso_obscode_label = tk.Label(self.settings_frame, text="AAVSO Observer Code:")
        self.aavso_obscode_label.grid(row=row, column=0, stick=tk.W)
        self.aavso_obscode_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.aavso_obscode_entry.grid(row=row, column=1, sticky=tk.E)
        row = row + 1

        self.latitude_label = tk.Label(self.settings_frame, text="Observatory Latitude:")
        self.latitude_label.grid(row=row, column=0, stick=tk.W)
        self.latitude_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.latitude_entry.grid(row=row, column=1, sticky=tk.E)
        row = row + 1

        self.longitude_label = tk.Label(self.settings_frame, text="Observatory Longitude:")
        self.longitude_label.grid(row=row, column=0, stick=tk.W)
        self.longitude_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.longitude_entry.grid(row=row, column=1, sticky=tk.E)
        row = row + 1

        self.height_label = tk.Label(self.settings_frame, text="Observatory Height, m ASL:")
        self.height_label.grid(row=row, column=0, stick=tk.W)
        self.height_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.height_entry.grid(row=row, column=1, sticky=tk.E)
        row = row + 1

        self.cod_label = tk.Label(self.settings_frame, text="COD Observatory Code:")
        self.cod_label.grid(row=row, column=0, stick=tk.W)
        self.cod_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.cod_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.con_label = tk.Label(self.settings_frame, text="CON Contact:")
        self.con_label.grid(row=row, column=0, stick=tk.W)
        self.con_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.con_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.obs_label = tk.Label(self.settings_frame, text="OBS Observers:")
        self.obs_label.grid(row=row, column=0, stick=tk.W)
        self.obs_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.obs_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.mea_label = tk.Label(self.settings_frame, text="MEA Measurers:")
        self.mea_label.grid(row=row, column=0, stick=tk.W)
        self.mea_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.mea_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.com2_label = tk.Label(self.settings_frame, text="COM Address:")
        self.com2_label.grid(row=row, column=0, stick=tk.W)
        self.com2_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.com2_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.com3_label = tk.Label(self.settings_frame, text="COM Observatory and Site Name:")
        self.com3_label.grid(row=row, column=0, stick=tk.W)
        self.com3_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.com3_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.ac2_label = tk.Label(self.settings_frame, text="AC2 E-mail:")
        self.ac2_label.grid(row=row, column=0, stick=tk.W)
        self.ac2_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.ac2_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.telescope_label = tk.Label(self.settings_frame, text="Telescope:")
        self.telescope_label.grid(row=row, column=0, stick=tk.W)
        self.telescope_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.telescope_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.accessory_label = tk.Label(self.settings_frame, text="Accessory:")
        self.accessory_label.grid(row=row, column=0, stick=tk.W)
        self.accessory_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.accessory_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.ccd_label = tk.Label(self.settings_frame, text="Camera:")
        self.ccd_label.grid(row=row, column=0, stick=tk.W)
        self.ccd_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.ccd_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.ccd_gain_label = tk.Label(self.settings_frame, text="Gain, e-/ADU:")
        self.ccd_gain_label.grid(row=row, column=0, stick=tk.W)
        self.ccd_gain_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width)
        self.ccd_gain_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1


        self.exposure_start_label = tk.Label(self.settings_frame, text="Exposure Start, JD:")
        self.exposure_start_label.grid(row=row, column=0, stick=tk.W)
        self.exposure_start_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.exposure_start_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        self.object_name_label = tk.Label(self.settings_frame, text="Object Name:")
        self.object_name_label.grid(row=row, column=0, stick=tk.W)
        self.object_name_entry = tk.Entry(self.settings_frame, width=extended_settings_entry_width, background='pink')
        self.object_name_entry.grid(row=row, column=1, sticky=tk.EW)
        row = row + 1

        # Here we have full-width settings dropdowns in the right frame
        self.weighting_label = tk.Label(self.right_frame, text="Ensemble Fit Weighting:")
        self.weighting_label.grid(row=3, column=0, sticky=tk.W)
        self.weighting_stringvar = tk.StringVar()
        self.weighting_stringvar.set("None")
        self.weighting_dropdown = tk.OptionMenu(self.right_frame, self.weighting_stringvar, "None", "Raw Flux", "Instrumental Magnitude", "PSF Sigma")
        self.weighting_dropdown.grid(row=4, column=0, sticky=tk.EW)

        self.catalog_label = tk.Label(self.right_frame, text="Comparison Catalog:")
        self.catalog_label.grid(row=5, column=0, sticky=tk.W)
        self.catalog_stringvar = tk.StringVar()
        self.catalog_stringvar.set("APASS DR9")
        self.catalog_dropdown = tk.OptionMenu(self.right_frame, self.catalog_stringvar, "APASS DR9", "URAT1", "USNO-B1.0", "Gaia DR2", "VizieR Catalog")
        self.catalog_dropdown.grid(row=6, column=0, sticky=tk.EW)

        self.vizier_catalog_label = tk.Label(self.right_frame, text="VizieR Catalog Number:")
        self.vizier_catalog_label.grid(row=7, column=0, sticky=tk.W)
        self.vizier_catalog_entry = tk.Entry(self.right_frame)
        self.vizier_catalog_entry.grid(row=8, column=0, sticky=tk.EW)

        self.fitter_label = tk.Label(self.right_frame, text="PSF Fitter:")
        self.fitter_label.grid(row=9, column=0, sticky=tk.W)
        self.fitter_stringvar = tk.StringVar()
        self.fitter_stringvar.set("Levenberg-Marquardt")
        self.fitter_dropdown = tk.OptionMenu(self.right_frame, self.fitter_stringvar, "Levenberg-Marquardt", "Linear Least Square", "Sequential LS Programming", "Simplex LS")
        self.fitter_dropdown.grid(row=10, column=0, sticky=tk.EW)

        self.remove_vsx_var = tk.BooleanVar()
        self.remove_vsx_checkbox = tk.Checkbutton(self.right_frame, text="Ignore VSX Sources in Ensemble",variable=self.remove_vsx_var)
        self.remove_vsx_checkbox.grid(row=11, column=0, sticky=tk.EW)
        self.remove_vsx_var.set(True)

        self.nearby_vsx_var = tk.BooleanVar()
        self.nearby_vsx_checkbox = tk.Checkbutton(self.right_frame, text="Report VSX Sources Nearby",variable=self.nearby_vsx_var)
        self.nearby_vsx_checkbox.grid(row=12, column=0, sticky=tk.EW)
        self.nearby_vsx_var.set(True)

        self.batch_psf_var = tk.BooleanVar()
        self.batch_psf_checkbox = tk.Checkbutton(self.right_frame, text="PSF Photometry Batch Processing",variable=self.batch_psf_var)
        self.batch_psf_checkbox.grid(row=13, column=0, sticky=tk.EW)
        self.batch_psf_var.set(True)

        # Histogram stretch sliders
        self.stretch_label = tk.Label(self.right_frame, text="Histogram Stretch Low/High:")
        self.stretch_label.grid(row=14, column=0, sticky=tk.W)
        self.stretch_low = tk.Scale(self.right_frame, from_=-10, to=100, orient=tk.HORIZONTAL, command=self.update_histogram_low)
        self.stretch_low.grid(row=15, column=0, sticky=tk.EW)
        self.stretch_high = tk.Scale(self.right_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_histogram_high)
        self.stretch_high.set(5)
        self.stretch_high.grid(row=16, column=0, sticky=tk.EW)

        self.stretching_label = tk.Label(self.right_frame, text="Image Stretching:")
        self.stretching_label.grid(row=17, column=0, sticky=tk.W)
        self.stretching_stringvar = tk.StringVar()
        self.stretching_stringvar.set("None")
        self.stretching_dropdown = tk.OptionMenu(self.right_frame, self.stretching_stringvar, "None", "Square Root", "Log", "Asinh")
        self.stretching_dropdown.grid(row=18, column=0, sticky=tk.EW)
        self.stretching_stringvar.trace("w", lambda name, index, mode, sv=self.stretching_stringvar: self.update_display())

        # Console below
        self.console = tk.Text(self.center, height=10, bg='black', fg='white', width=200)
        self.console.grid(sticky=tk.N+tk.S+tk.E+tk.W, column=0, row=3)
        self.console_scrollbar = tk.Scrollbar(self.center)
        self.console_scrollbar.grid(sticky=tk.N + tk.S + tk.E + tk.W, column=1, row=3)

        self.console.config(yscrollcommand=self.console_scrollbar.set)
        self.console_scrollbar.config(command=self.console.yview)

        self.console_msg(self.program_name+" Ready.")

        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', help="FITS file or directory to process")
        parser.add_argument('-s', '--set', help="MetroPSF settings file to use for processing")
        parser.add_argument('-r', '--report', help="Generate BAA/AAVSO report on all VSX sources", action='store_true')

        args = parser.parse_args()

        if args.set:
            self.load_settings(args.set)

        if args.input and os.path.isfile(args.input):
            self.load_FITS(args.input)
            self.display_image()

        if args.input and os.path.isfile(args.input) and args.report:
            if self.batch_psf_var.get():
                self.perform_photometry()
            else:
                self.aperture_photometry()
            self.solve_image()
            self.get_comparison_stars()
            self.console_msg("Finding linear fit..")
            self.find_linear_regression_model()
            self.console_msg("Reporting on all VSX sources..")
            self.report_on_all_vsx_sources()
            quit()

        if args.input and os.path.isdir(args.input) and args.report:
            self.batch_report_on_all_vsx_sources(args.input)
            quit()

        #self.initialize_debug()
        #self.plot_photometry()

        tk.mainloop()


if __name__ == "__main__":
    myGUI = MyGUI()




