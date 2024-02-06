# Using ePDFpy GUI
ePDFpy separated into two tabs; `Profile extraction` and `PDF analysis`. Users should process the image process, such as center fitting to extraction I(q) in `Profile extraction`. The extracted I(q) is used to calculate G(r) in `PDF analysis`, using autofit (advanced fitting). 

Following contents are the instructions for each process.  

# Profile extraction tab
## Image loading and preparation
1) In the `open` menu, click `open image file` to open single diffraction image or select a directory by clicking `open image stack` to open multiple diffraction images within selected directory.
   1) The data formats can be text files (`txt`, `csv`), image files (`tiff`, `jpg`), or standard TEM data files like `dm3`,`dm4` and `mrc`.
2) After loading the data, users can apply or adjust the beam stopper mask, using the dropbox menu in the `mask` control box.
   1) If users saved previous mask, they can easily load the pre-saved mask shape from dropbox
   2) Users can draw mask, by clicking `[edit]` and `new` button. From the mask drawing GUI module, move and add polygon to cover the adequate beam mask area.
   3) After completing the drawing, save polygon with typing name in the below box.
3) Single blank image can be uploaded to ePDFpy for background noise removal.
   1) Users can load the image by clicking `Open Blank image` and choose the image file.
   2) After loading the image, the noise can be subtracted by clicking `Remove noise`
      1) `Yes to all`: Subtract the loaded blank image from every loaded datasets
      2) `Yes`: Subtract the blank image only from choosed data
   3) If users want to cancel the noise subtraction, simply click `Revert noise`.
   4) Users can check the current status of noise removal in the text box below.

## Image process operation
1) `Find center`: Users can start the center fitting process by clicking the button.
   1) If the process is finished and have optimal center result, it will be on the `Center finding setting` spin box and each x and y axis will be shown in the colormap.
   2) Users can adjust the center coordinates by changing the numbers in the setting spin box. The changed result will instantly updated on the GUI (image panel, polar transformed image panel).
2) `Get azimuthal data`: The intensity profile ($\I(q)$) is calculated by clicking the button.
   1) After changing the center, update the I(q) by clickung the button,
3) `Calculate all data`: Both of the image process (`Find center` and `Get azimuthal data`) for all loaded data are applied iteratively.

## Saving the data
The extracted profile, I(q), can be saved with text file (`csv`,`txt`). Using `Save` in `Save and Load`, single or multiple profiles can be saved. The saved profiles can be reloaded, without doing the same image process. 

# PDF analysis
Users can do manual fitting by changing each parameters or use advanced fitting, which can set all 3 parameters with optimal conditions.
## Loading the data
Extracted I(q) is automatically used in the PDF analysis. If users want to load the pre-saved intensity profile or revisit previous result, simply load the data from `Open` in `Save and Load` menu. 
- Open azavg: Loading intensity profile data only. Loading up all data in the sample folder can be done by choosing folder in the 'stack'  menu.
- Open preset: Loading all previous analysis results, including I(q), G(r) and all parameteres. Similarly, multiple data can be load with 'stack' menu.
## Settings for the analysis
1) `Element`: Choose atomic number (Z) and ratio which consist the sample's atomic structure (i.e. Tantala -> (73.Ta, 2) (8.O, 5)).
   1) If users want to save the element sets, click `Save` and choose the name.
   2) Pre-saved element setting can be loaded through `load` dropdown menu.
2) Calibration factor: Insert number to convert pixel distance to scattering vector `q`.
3) Click `Apply to all` if all loaded datasets are using same calibration factor and same atomic compositions.

## Autofit (advanced fit)
1) After setting up elements and calibration factor, confirm the minimum q value (or pixel number). ePDFpy atutomatically choose the first saddle point, which assumes that the effect of diffraction signal exceeds the trasmitted beam. Users can adjust the value by draging the highlighted area or change number.
2) If the minimum q value is fixed, click `advanced fitting` button to open the GUI. Users should set 3 search range for the parameters in the GUI.
   1) `pixel range`: Pixel distance corredponding to maximum scattering vector `q`. For the convenience, it is set to input the pixel values. Users can convert pixel distance to q, using relation; $ calibration factor = (q/2\pi) * pixel $ (i.e. q = 22 is correponding to pixel = 1050, with 0.00334 calibration factor). It is expected that optimal q is around 20 ~ 22.
   2) `q_k range`: Cut-off `q` value. This value is written with `q` value, not pixel. 
   3) `Noise level (%)`: Threshold ratio for the noise peak value under 1 Angstrom. This ratio correspond to maximum G(r) peak value in less than 1 Angstrom range.
   4) `How many results`: The number of results to be displayed.
3) After setting all parameters, users can click `Autofit` button, to process the advanced fitting. 
4) The results are displayed in the below table. Each plot will be highlighted in the plot panel for corresponding parameter results. After choosing one result, click `Select` button to use that in `PDF Analysis` tab.

## File save
 
- `Save current preset`: If the data is loaded from previous result, users can overwrite the new result to original files.
- `Save current preset as`: If users want to save the data in new path, use this option.
- `Save all preset`: Same with `Save current preset`, but with every loaded files.
- `Save all preset as`: Same with `Save current preset as`, but with every loaded files.

All paramters and plots are saved in each preset or text files.
- json file: Contains all parameters: q range, q_k, N, damping factor, calibration factor, center coordinates, etc
- azavg.csv: Only contains intensity profile with arranged in pixel distance
- q.csv: Contains all q space related data: scattering vector q, Intensity profiel I(q), reduced intensity function ($\phi(q)$ ), damped $\phi(q)
- r.csv: Contains all real space related data: real space range r, reduced pair distribution function G(r)
