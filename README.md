Plant Pest Dashboard and Simulator
User Guide


# 1. Summary

### This folder contains two local applications

Both applications run locally on your own computer. They open in a web browser, but the browser is only used as the screen for the application.

The main purpose of the package is to estimate plant pest prevalence trends from real survey data, and explore survey and analysis designs that affect estimation reliability.

## 1. Plant Pest Dashboard
   This application is used with real survey data. It loads survey polygons and host coverage polygons, estimates prevalence over time, fits two trend methods, and creates a printable PDF report.

## 2. Plant Pest Simulator
   This is a decision support tool used with simulated data. It creates a fake plant pest epidemic. It then tests how well different survey designs and analysis methods recover that true epidemic state.

# 2. What the two applications are for

### Plant Pest Dashboard

Use this when you have real GIS survey polygons and host coverage polygon data.

### It answers questions such as

- What prevalence was estimated in each survey year?
- Does prevalence appear to be increasing or decreasing?
- Was the observed survey effort close to a required effort benchmark?

### Plant Pest Simulator

Use this when you want to test survey designs before using them in the real world.

### It answers questions such as

- What happens if infection is highly clustered?
- What happens if surveys are targeted toward suspected infected areas?
- What happens if the same places are surveyed again each year?
- How many samples are needed under different assumptions?
- Which survey design gives more accurate estimates under the same simulated epidemic?
- How well do the Change Method and Regression Method recover the true prevalence change?

The simulator is not a disease forecast. It is a controlled testing tool. It creates synthetic examples so that the true epidemic is known.


# 3. What you need before starting

## You need

### 1. A Windows computer.
### 2. Internet access for the first setup.
### 3. Permission to install Python if it is not already installed.
### 4. This full folder downloaded or copied onto your computer.

### Recommended folder location

C:\Users\YourName\Documents\plant_pest_dashboard

Avoid running the applications from inside a ZIP file. Always extract the folder first.

Avoid moving only individual files out of this folder. The applications expect the Python files, batch files, requirements file, and data folders to stay together.


# 4. How to install Python

You only need to do this once on each computer.

### 1. Open your web browser.
### 2. Go to:

   https://www.python.org/downloads/

### 3. Download Python 3.11 or later.
### 4. Open the downloaded installer.
### 5. On the first installer screen, tick the box called:

   Add Python to PATH

### 6. Click Install Now.
### 7. Wait for the installation to finish.
### 8. Close the installer.

### Important

The "Add Python to PATH" box is important. It lets the setup file find Python automatically.

If Python is already installed, you can try running Setup.bat first. If setup cannot find Python, install Python using the steps above.


# 5. How to download the package from GitHub

### If you are downloading from GitHub

### 1. Open the GitHub page for the project.
### 2. Click the green Code button.
### 3. Click Download ZIP.
### 4. Save the ZIP file.
### 5. Right-click the ZIP file.
### 6. Click Extract All.
### 7. Move the extracted folder to a simple location, such as Documents.

Do not run Setup.bat while the files are still inside the ZIP file.


# 6. First-time setup

You only need to run setup once after downloading the folder.

### 1. Open the plant_pest_dashboard folder.
### 2. Double-click Setup.bat.
### 3. A black command window will open.
### 4. The setup will create a local Python environment inside this folder.
### 5. The setup will install all Python packages listed in requirements.txt.
### 6. Wait until it finishes.

The first setup may take several minutes.

### When setup has finished, you should see

Setup complete.
Use "Run Dashboard.bat" or "Run Simulator.bat".

Then press any key to close the window.

If setup fails, read the message in the black command window. Common causes are:

- Python is not installed.
- Python was installed without "Add Python to PATH".
- The computer has no internet access.
- A firewall or proxy is blocking Python package downloads.
- The user does not have permission to install software.


# 7. How to open the Dashboard

### 1. Open the plant_pest_dashboard folder.
### 2. Double-click Run Dashboard.bat.
### 3. A black command window will open.
### 4. Your web browser should open automatically.
### 5. The Dashboard uses this local address:

   http://127.0.0.1:8000

If the browser does not open automatically, copy the address above into your browser.

Do not close the black command window while using the Dashboard. If you close it, the application stops.

### When you are finished

### 1. Close the browser tab.
### 2. Close the black command window.


# 8. How to open the Simulator

### 1. Open the plant_pest_dashboard folder.
### 2. Double-click Run Simulator.bat.
### 3. A black command window will open.
### 4. Your web browser should open automatically.
### 5. The Simulator uses this local address:

   http://127.0.0.1:8001

If the browser does not open automatically, copy the address above into your browser.

Do not close the black command window while using the Simulator. If you close it, the application stops.

The Dashboard and Simulator use different local addresses. This means they can both be open at the same time if needed.


# 9. What this folder contains

## Main files

### 1. Setup.bat
   Runs the first-time setup. It creates the local Python environment and installs the required packages.

### 2. Run Dashboard.bat
   Opens the Plant Pest Dashboard.

### 3. Run Simulator.bat
   Opens the Plant Pest Simulator.

### 4. requirements.txt
   Lists the Python packages needed by the applications.

### 5. plant_pest_dashboard.py
   The main dashboard application for real survey data.

### 6. plant_pest_simulator.py
   The simulator application for synthetic experiments.

### 7. plant_pest_backend.py
   Shared simulator and calculation code used by the applications.

### 8. DEFRA_data
   Default data folder. This may contain survey polygons, host coverage polygons, SPHN polygons, and other supporting files.

### 9. README.txt
   This guide.

### Important shapefile note

A shapefile is usually made of several files with the same name but different endings, such as:

- .shp
- .shx
- .dbf
- .prj
- .cpg

Keep these files together. Do not move only the .shp file on its own.


# 10. Dashboard user guide

## The Dashboard has three main windows

### 1. Home
### 2. Data Viewer
### 3. Results


## 10.1 Home window

### Purpose

The Home window loads the data used by the Dashboard.

### Typical steps

### 1. Open the Dashboard.
### 2. Check the file paths shown on the Home page.
### 3. Press Load Data.
### 4. Wait until loading finishes.
### 5. Go to Data Viewer to inspect the map, or go to Results to generate estimates.

### The Home page can load

### 1. Survey polygons
   These are the areas that were surveyed.

### 2. Host coverage polygons
   These are the areas where the host plant is present.

### 3. SPHN polygons
   These are optional polygons for the map for SPHNs that have been issued.

If a path is blank or incorrect, that layer may not load. The app should still open, but some map layers or results may be missing.


## 10.2 Data Viewer window

### Purpose

The Data Viewer lets you inspect the loaded polygon data on an interactive map.

### Use this window to check

- whether survey polygons loaded correctly,
- whether host coverage polygons loaded correctly,
- which survey year is being displayed,
- whether survey polygons overlap with host coverage polygons,
- whether the survey pattern looks broad or targeted.

### Typical steps

### 1. Load data on the Home page.
### 2. Open the Data Viewer window.
### 3. Use the year slider to choose a survey year.
### 4. Turn map layers on or off using the layer control.
### 5. Zoom in to inspect survey and host polygons.

This window is mainly for checking the data visually before producing results.


## 10.3 Results window

### Purpose

The Results window converts the loaded survey data into prevalence estimates and a PDF report.

### Typical steps

### 1. Load data on the Home page.
### 2. Open the Results window.
### 3. Press Generate Results.
### 4. Review the plots and tables.
### 5. If needed, open Settings and change how the data are interpreted.
### 6. Press Results to PDF to create the report.

### The Results window produces

- estimated prevalence over time,
- Change Method results,
- Regression Method results,
- yearly survey summaries,
- observed effort compared with indicative required effort,
- a printable PDF report.


## 10.4 Results settings

The settings menu controls how the real data are interpreted.

### Survey data settings

These settings decide which survey outcome labels count as disease-positive or disease-negative.

### Example

- "Confirmed infected" may count as positive.
- "No evidence of P ramorum" may count as negative.
- "Awaiting site visit" should usually be ignored because it does not say whether disease is present or absent.

### Host landscape settings

These settings decide which host polygons are included in the host population.

For simple host coverage files, use all host polygons.

If a future file contains several species or host types, a user can choose which column and values represent the host plant of interest.

### Host density

Host density converts mapped area into an estimated number of hosts.

### Example

If host density is 2,500 hosts per km2, then a 1 km2 host area is treated as containing about 2,500 hosts.

### Include unmatched survey polygons

Some survey polygons may not fall inside the host coverage polygons. This can happen if the host coverage file is incomplete.

If this option is switched on, those unmatched survey polygons are treated as additional host areas rather than being ignored.

### Shared statistical settings

These settings affect the indicative effort benchmarks. They are not proof that the historical survey design was random or ideal.

### Regression Method settings

These settings control the model used to fit the prevalence trend.


## 10.5 How the Dashboard estimates prevalence

### The Dashboard does the following

### 1. It loads the host coverage polygons.
### 2. It estimates how many hosts are in each host polygon using:

   polygon area x host density

### 3. It loads survey polygons for each year.
### 4. It checks which survey polygons overlap which host polygons.
### 5. It uses survey status labels to decide whether each survey polygon is positive, negative, or ignored.
### 6. It estimates how much host area was surveyed.
### 7. It converts surveyed area into estimated surveyed hosts.
### 8. It converts positive surveyed area into estimated infected hosts.
### 9. It estimates yearly prevalence as:

   estimated infected hosts / estimated surveyed hosts

This is an area-based approximation. It depends on the host density assumption and on the quality of the polygon data.


## 10.6 How to read the PDF report

The PDF report starts with a summary.

### The first page includes

- the overall trend,
- estimated change in prevalence,
- estimated final prevalence,
- prevalence change per year,
- key caveats.

### Later pages include

- prevalence trend plots,
- yearly survey tables,
- observed effort versus indicative required effort,
- effort summary tables.

### Important

If the survey footprint is very targeted, the estimated prevalence may be higher than the true prevalence across the wider host landscape. This is because targeted surveys are more likely to visit places where disease is suspected.


# 11. Simulator user guide

The Simulator is arranged as a step-by-step workflow.

### The main windows are

### 1. Host Landscape
### 2. True Prevalence
### 3. Infection Landscape
### 4. Sampling Effort
### 5. Survey Design
### 6. Results
### 7. Scenario Comparison


## 11.1 Recommended first run

Use this first to check that the Simulator works.

### 1. Open Run Simulator.bat.
### 2. Go to Host Landscape.
### 3. Choose Generate synthetic clusters.
### 4. Press Generate host landscape.
### 5. Go to True Prevalence.
### 6. Keep the default values.
### 7. Press Generate prevalence curve.
### 8. Go to Infection Landscape.
### 9. Keep infection clustering as Random.
### 10. Press Generate infection landscape.
### 11. Go to Sampling Effort.
### 12. Keep the default values.
### 13. Press Estimate sample sizes.
### 14. Go to Survey Design.
### 15. Keep method-specific sample sizes selected.
### 16. Press Simulate surveys.
### 17. Go to Results.
### 18. Press Fit Appendix C and E.
### 19. Review the plot and key results table.


## 11.2 Host Landscape window

### Purpose

This creates the host population that can become infected.

### Options

### 1. Load host polygons
   Uses a real host coverage polygon file. Each polygon becomes one host cluster.

### 2. Generate synthetic clusters
   Creates a fake landscape for controlled experiments.

### 3. Host density
   Converts area into host count.

### Outputs

- host landscape bubble plot,
- landscape summary table,
- cluster size distribution plot.

### Interpretation

Each bubble is a host cluster. Larger bubbles contain more hosts.


## 11.3 True Prevalence window

### Purpose

This defines the true disease prevalence over time.

The true prevalence curve is the hidden truth that the survey methods are trying to estimate.

### Options

### 1. Number of rounds
   Number of survey time points.

### 2. Curve shape
   Shape of the true prevalence curve.

### 3. Initial prevalence
   True prevalence at the first round.

### 4. Final prevalence
   True prevalence at the final round.

### Outputs

- plot of true prevalence over time.


## 11.4 Infection Landscape window

### Purpose

This places infection onto the host landscape while keeping the total prevalence close to the true prevalence curve.

### Clustering levels

### 1. Random
   Infection is spread broadly, as if each host has approximately equal chance of being infected.

### 2. Low
   Infection is slightly clustered.

### 3. Medium
   Infection is more concentrated around infected clusters and nearby clusters.

### 4. High
   Infection is strongly concentrated around hotspot clusters.

### Outputs

- infection bubble plot,
- round summary table,
- hotspot table.

### Interpretation

Each bubble is a host cluster. Larger bubbles contain more hosts. Darker red means a higher infected proportion in that cluster.


## 11.5 Sampling Effort window

### Purpose

This estimates how many samples are needed for the two methods.

### Methods

### 1. Change Method
   Compares the first and final survey.

### 2. Regression Method
   Uses all survey rounds to estimate a trend.

### Important settings

### 1. Significance level
   How strict the evidence threshold is. The default is 0.05.

### 2. Power
   How likely the design is to detect the chosen effect if it is really present. The default is 0.80.

### 3. Change Method detectable change
   The change in prevalence the Change Method is designed to detect.

### 4. Regression Method design prevalence
   The target or final prevalence used in the Regression Method sample size calculation.

### 5. Clustering inflation
   Increases sample size when clustered sampling is expected to give less information than simple random sampling.

### Outputs

- required sampling effort table,
- required sample size plot.


## 11.6 Survey Design window

### Purpose

This simulates how field surveys are carried out.

### Main options

### 1. Simple random sampling
   Every host has equal chance of being sampled.

### 2. Multistage sampling
   Clusters are selected first, then hosts are sampled within selected clusters.

### 3. Targeted surveying
   Surveys are more likely to sample infected or high-risk areas.

### 4. Survey overlap
   Some of the same hosts or sampling locations are reused across rounds.

### 5. Detection sensitivity
   Controls whether infected sampled hosts are always detected or sometimes missed.

### Outputs

- survey allocation plot,
- survey round summary table,
- targeted versus untargeted comparison.


## 11.7 Results window

### Purpose

This fits the two analysis methods to the simulated survey data.

### Outputs

- true prevalence curve,
- survey estimates,
- Change Method result,
- Regression Method fitted trend,
- key results table.

### Use this page to check

- whether the methods recovered the true trend,
- whether one method performed better than the other,
- whether estimates are biased upward or downward.


## 11.8 Scenario Comparison window

### Purpose

This compares two complete simulation configurations.

Scenario A and Scenario B start with the same baseline settings. You can then change one setting in Scenario B to test a "what if" question.

### Examples

- Scenario A uses random surveying; Scenario B uses targeted surveying.
- Scenario A has random infection; Scenario B has high infection clustering.
- Scenario A has no overlap; Scenario B revisits sampled hosts.
- Scenario A has no clustering inflation; Scenario B uses pilot-estimated inflation.

### Runs per scenario

### 1. If runs per scenario is 1:
   The app shows one fitted example for each scenario.

### 2. If runs per scenario is greater than 1:
   The app repeats the simulation many times and shows error boxplots.

### How to read the error plots

- Values close to zero are good.
- Positive values mean the method overestimated.
- Negative values mean the method underestimated.
- Wider boxes mean results are more variable.

### Recommended workflow

### 1. Keep Scenario A as the baseline.
### 2. Change one important setting in Scenario B.
### 3. Run 1 simulation first to visually inspect behaviour.
### 4. Increase to 100 or more runs to compare reliability.


# 12. Simulator logic

This section explains how the Simulator works internally.

## 1. Host landscape

The simulator represents the landscape as clusters.

A cluster can represent a forest, plantation, crop field, or host site.

### Each cluster has

- a location,
- an area,
- a host count.

Each host also has a host ID. This lets the simulator sample the same host again when overlap is used.


## 2. True prevalence curve

The user defines the true prevalence at each survey round.

This true curve is the answer that the survey methods are trying to recover.


## 3. Infection landscape

### For each round, the simulator calculates

true prevalence x total hosts = target infected hosts

If the target infected count increases, new infected host IDs are added.

If the target infected count decreases, infected host IDs are removed.

This keeps the whole simulated epidemic close to the true prevalence curve.


## 4. Infection clustering

Random infection means every host has roughly equal chance of infection.

Clustered infection means infection is more likely near:

- existing infected clusters,
- initial hotspot clusters,
- nearby infected clusters.

The simulator uses the five nearest neighbouring clusters when calculating nearby infection pressure. This keeps the calculation faster while still representing local spread.


## 5. Survey simulation

The simulator chooses host IDs according to the selected survey design.

Simple random sampling chooses from the whole host population.

Multistage sampling chooses clusters first and then samples hosts within those clusters.

Targeted sampling gives infected or high-risk hosts and clusters a higher chance of being sampled.

Overlap reuses some host IDs from the previous survey round.


## 6. Analysis

### The survey results are analysed by

### 1. Change Method
   Uses the first and final survey estimates.

### 2. Regression Method
   Uses all survey rounds.

Because the simulator knows the true answer, it can measure error:

error = estimated value - true value


# 13. Troubleshooting

### If Setup.bat says Python is not recognised

### 1. Install Python from python.org.
### 2. Make sure "Add Python to PATH" is ticked.
### 3. Run Setup.bat again.

### If Setup.bat fails while installing packages

### 1. Check internet access.
### 2. Check whether a firewall or proxy is blocking downloads.

### If the application does not open

### 1. Check that Setup.bat has been run.
### 2. Check that the black command window is still open.
### 3. Check whether the command window shows an error.
### 4. Try closing the command window and opening the application again.

If the browser says it cannot reach the page:

### 1. Wait a few more seconds.
### 2. Check the address.
### 3. Dashboard address:

   http://127.0.0.1:8000

### 4. Simulator address:

   http://127.0.0.1:8001

### If maps or shapefiles do not load

### 1. Check that the file path is correct.
### 2. Check that all shapefile sidecar files are present.
### 3. Do not move only the .shp file.
### 4. Try loading the folder that contains the shapefile rather than typing the exact .shp file.

### If the Simulator is slow

### 1. Use fewer scenario comparison runs.
### 2. Start with 1 run for visual checking.
### 3. Increase to 100 or more runs only after the scenario is configured correctly.
### 4. Use a synthetic landscape with fewer clusters while testing settings.

### If the PDF does not generate

### 1. Make sure results have been generated first.
### 2. Check the command window for errors.
### 3. Run Setup.bat again to make sure all report packages are installed.


# 14. Important interpretation notes

The Dashboard analyses the survey record that is supplied to it. If the survey record is targeted toward suspected infected sites, the estimated prevalence may be higher than the true prevalence across the wider host landscape.

The effort benchmarks in the Dashboard are indicative. They are useful reference points, but they do not prove that messy historical survey data followed the ideal assumptions of the sample size formulas.

The Simulator is a simplified model. It is useful for testing survey and analysis behaviour, but it is not a full biological spread model and should not be treated as a forecast.
