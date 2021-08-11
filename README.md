# MetroPSF
Astronomical Photometry Prgroam

MetroPSF is an open source astronomical photometry program. It’s goal is providing a convenient and intuitive graphical user interface to algorithms implemented in photutils (Bradley et al., 2020), in particular iteratively subtracted point spread function (PSF) photometry—a variant of the DAOPHOT algorithm by Stetson (1987) for PSF photometry in crowded ﬁelds, and aperture photometry.

MetroPSF can perform blind astrometric calibration of images via Astrometry.net service (Lang et al., 2010), request comparison photometry data from various catalogs in VizieR (Ochsenbein et al., 2000), match sources and perform diﬀerential photometry via linear regression ﬁts to a weighted ensemble of sources. Such diﬀerential ensemble photometry method is described by Paxson (2010). It can also generate photometry reports compliant with British Astronomy Association’s (BAA) Photometry Database and American Association of Variable Star Observers (AAVSO) submission guidelines, as well as process multiple FITS ﬁles in batch mode and report on all VSX sources found. It can generate light curves from individual report files produced.

The program is developed using Python’s standard Tkinter graphical user interface and relies on a rather conservative subset of astronomical and data processing libraries to work that are typically well-maintained. The program is currently in the stage of a working prototype that requires further development and code refactoring. Feedback and suggestions are very welcome. MetroPSF is conﬁrmed to work on Windows, Linux and FreeBSD operating systems and is, in principle, compatible with all operating systems capable of installing Python and the Astropy package for astronomy (Astropy Collaboration et al., 2013, 2018).

For full documentation, please refer to the included metropsf.pdf file. 
