# Plant Pest Dashboard and Simulator

## Overview

This project contains two applications that run locally on your computer:

1. **Plant Pest Dashboard** — uses real survey and host data to estimate prevalence trends and create a PDF report.
2. **Plant Pest Simulator** — uses simulated data to test how survey design and epidemic conditions affect estimation reliability.

Both applications open in a web browser, but all processing takes place on your computer.

---

## What the applications do

### Plant Pest Dashboard

Use the Dashboard with real GIS data, including:

- survey polygons;
- host coverage polygons; and
- optional supporting layers, such as SPHN polygons.

The Dashboard can help answer:

- What was the estimated prevalence in each survey year?
- Is prevalence increasing or decreasing?
- How do the Change Method and Regression Method compare?
- Was the observed survey effort close to the indicative required effort?

### Plant Pest Simulator

Use the Simulator to test survey and analysis choices before applying them in practice.

It can explore questions such as:

- What happens when infection is highly clustered?
- What happens when surveys target suspected infected areas?
- What happens when the same hosts are sampled repeatedly?
- How many samples are required under different assumptions?
- Which survey design gives more reliable estimates?
- How well do the Change Method and Regression Method recover the true prevalence change?

The Simulator is a controlled testing tool, not a disease forecast.

---

## Requirements

You need:

- a Windows computer;
- internet access for the first setup;
- permission to install Python if required; and
- the complete project folder.

A recommended location is:

```text
C:\Users\YourName\Documents\plant_pest_dashboard
```

Do not run the applications from inside a ZIP file. Extract the project first.

Keep the Python files, batch files, requirements file, and other project files together.

---

## Shapefile folder setup

The project package does not include DEFRA shapefiles.

Place your own folder containing the required shapefiles in the same directory as the project files. The applications can then find the data automatically.

Example structure:

```text
plant_pest_dashboard/
├── plant_pest_dashboard.py
├── plant_pest_simulator.py
├── plant_pest_backend.py
├── Setup.bat
├── Run Dashboard.bat
├── Run Simulator.bat
├── requirements.txt
└── your_shapefile_folder/
```

The shapefile folder may contain:

- survey polygons;
- host coverage polygons;
- SPHN polygons; and
- other supporting GIS layers.

A shapefile normally includes several files with the same name:

```text
example.shp
example.shx
example.dbf
example.prj
example.cpg
```

Keep all of these files together. Do not move only the `.shp` file.

---

## Installing Python

Python only needs to be installed once.

1. Go to `https://www.python.org/downloads/`.
2. Download Python 3.11 or later.
3. Open the installer.
4. Tick **Add Python to PATH**.
5. Select **Install Now**.
6. Wait for installation to finish.

The **Add Python to PATH** option allows the setup script to find Python automatically.

If Python is already installed, try running `Setup.bat` first.

---

## Downloading from GitHub

1. Open the project page on GitHub.
2. Select **Code**.
3. Select **Download ZIP**.
4. Save and extract the ZIP file.
5. Move the extracted folder to a convenient location.

Do not run `Setup.bat` while the files are still inside the ZIP archive.

---

## First-time setup

Run the setup once after downloading the project.

1. Open the project folder.
2. Double-click `Setup.bat`.
3. A command window will open.
4. The script will create a local Python environment.
5. It will install the packages listed in `requirements.txt`.
6. Wait until setup is complete.

You should then see:

```text
Setup complete.
Use "Run Dashboard.bat" or "Run Simulator.bat".
```

Common causes of setup failure include:

- Python is not installed;
- Python was installed without **Add Python to PATH**;
- internet access is unavailable;
- a firewall or proxy blocks package downloads; or
- the user does not have permission to install software.

---

## Running the Dashboard

1. Open the project folder.
2. Double-click `Run Dashboard.bat`.
3. Keep the command window open.
4. The Dashboard should open in your browser.

Local address:

```text
http://127.0.0.1:8000
```

If the browser does not open automatically, enter the address manually.

To stop the Dashboard, close the browser tab and then close the command window.

---

## Running the Simulator

1. Open the project folder.
2. Double-click `Run Simulator.bat`.
3. Keep the command window open.
4. The Simulator should open in your browser.

Local address:

```text
http://127.0.0.1:8001
```

The Dashboard and Simulator use different addresses and can run at the same time.

---

## Main project files

| File | Purpose |
|---|---|
| `Setup.bat` | Creates the local Python environment and installs packages. |
| `Run Dashboard.bat` | Starts the Dashboard. |
| `Run Simulator.bat` | Starts the Simulator. |
| `requirements.txt` | Lists required Python packages. |
| `plant_pest_dashboard.py` | Main application for real survey data. |
| `plant_pest_simulator.py` | Main simulation application. |
| `plant_pest_backend.py` | Shared simulation and calculation code. |
| `README.md` | User guide. |

Place the user-provided shapefile folder beside these files.

---

# Dashboard guide

## Main sections

The Dashboard has three main sections:

1. **Home**
2. **Data Viewer**
3. **Results**

## Home

Use the Home page to load the required GIS data.

Typical workflow:

1. Open the Dashboard.
2. Check the displayed file paths.
3. Select or confirm the required layers.
4. Press **Load Data**.
5. Open **Data Viewer** or **Results**.

The Dashboard can load:

- survey polygons;
- host coverage polygons; and
- optional SPHN polygons.

If a path is missing or incorrect, that layer may not load.

## Data Viewer

Use the Data Viewer to inspect the loaded GIS layers.

Check:

- whether survey polygons loaded correctly;
- whether host coverage polygons loaded correctly;
- which year is displayed;
- whether survey and host polygons overlap; and
- whether the survey pattern appears broad or targeted.

Use the year control, map layers, and zoom tools before generating results.

## Results

The Results page produces prevalence estimates and report outputs.

Typical workflow:

1. Load the data.
2. Open **Results**.
3. Select **Generate Results**.
4. Review the tables and plots.
5. Adjust settings if required.
6. Select **Results to PDF**.

Outputs include:

- estimated prevalence over time;
- Change Method results;
- Regression Method results;
- yearly survey summaries;
- observed effort compared with indicative required effort; and
- a printable PDF report.

## Results settings

### Survey outcomes

These settings determine which labels are treated as positive, negative, or ignored.

Examples:

- `Confirmed infected` may be positive;
- `No evidence of P. ramorum` may be negative;
- `Awaiting site visit` should usually be ignored.

### Host landscape

These settings determine which polygons form the host population.

If the file contains several species or host categories, select the relevant column and values.

### Host density

Host density converts mapped area into an estimated number of hosts.

For example, at 2,500 hosts per km², an area of 1 km² is treated as containing approximately 2,500 hosts.

### Unmatched survey polygons

Some survey polygons may not overlap the host coverage layer.

If **Include unmatched survey polygons** is enabled, these areas are treated as additional host areas.

### Statistical settings

These affect indicative survey-effort benchmarks. They do not show that historical survey data followed the assumptions of the sample-size methods.

### Regression settings

These control the model used to estimate the prevalence trend.

## How the Dashboard estimates prevalence

The Dashboard:

1. loads the host coverage polygons;
2. estimates host abundance from area and host density;
3. loads survey polygons for each year;
4. identifies overlap between survey and host polygons;
5. classifies outcomes as positive, negative, or ignored;
6. estimates surveyed host area;
7. converts surveyed area into estimated surveyed hosts;
8. converts positive area into estimated infected hosts; and
9. calculates prevalence as:

```text
estimated infected hosts / estimated surveyed hosts
```

This is an area-based approximation and depends on the quality of the GIS data and host-density assumption.

## PDF report

The report includes:

- the overall prevalence trend;
- estimated prevalence change;
- estimated final prevalence;
- estimated annual change;
- key caveats;
- trend plots;
- yearly survey tables; and
- survey-effort summaries.

Targeted surveillance may overestimate prevalence across the wider host population because high-risk locations are more likely to be sampled.

---

# Simulator guide

## Main sections

The Simulator contains:

1. **Host Landscape**
2. **True Prevalence**
3. **Infection Landscape**
4. **Sampling Effort**
5. **Survey Design**
6. **Results**
7. **Scenario Comparison**

## Recommended first run

1. Open the Simulator.
2. In **Host Landscape**, select **Generate synthetic clusters**.
3. Generate the landscape.
4. In **True Prevalence**, keep the default values and generate the curve.
5. In **Infection Landscape**, keep clustering set to **Random**.
6. Generate the infection landscape.
7. In **Sampling Effort**, estimate sample sizes.
8. In **Survey Design**, simulate surveys.
9. In **Results**, fit the two methods.
10. Review the plot and results table.

## Host Landscape

This section creates the host population.

Options:

- **Load host polygons** — uses a real host coverage shapefile;
- **Generate synthetic clusters** — creates a simulated landscape;
- **Host density** — converts area into host count.

Outputs include:

- a host landscape plot;
- a landscape summary; and
- a cluster-size distribution.

Each bubble represents a host cluster. Larger bubbles contain more hosts.

## True Prevalence

This section defines the true prevalence trajectory.

Options include:

- number of survey rounds;
- curve shape;
- initial prevalence; and
- final prevalence.

The true curve is the value the survey methods attempt to recover.

## Infection Landscape

This section assigns infection to the host population while matching the selected prevalence curve.

Clustering options:

- **Random**
- **Low**
- **Medium**
- **High**

Higher clustering concentrates infection into fewer hotspot clusters.

Outputs include:

- an infection landscape plot;
- a round summary; and
- a hotspot table.

## Sampling Effort

This section estimates required sample sizes.

Methods:

- **Change Method** — compares the first and final survey rounds;
- **Regression Method** — uses all survey rounds to estimate a trend.

Important settings include:

- significance level;
- statistical power;
- detectable change;
- target final prevalence; and
- clustering inflation.

Outputs include:

- a sampling-effort table; and
- a required sample-size plot.

## Survey Design

This section simulates field sampling.

Options include:

- simple random sampling;
- multistage sampling;
- targeted sampling;
- temporal overlap; and
- detection sensitivity.

Overlap types:

- **Cross-sectional** — a new sample each round;
- **Rotating panel** — part of the previous sample is reused;
- **Longitudinal** — the same hosts are followed over time.

Outputs include:

- a survey allocation plot;
- round summaries; and
- targeted versus untargeted comparisons.

## Results

The Results section fits both methods to the simulated data.

Outputs include:

- the true prevalence curve;
- survey prevalence estimates;
- the Change Method estimate;
- the Regression Method trend; and
- a summary table.

Use this section to check whether the methods recovered the true trend or systematically overestimated or underestimated it.

## Scenario Comparison

Use this section to compare two simulation settings.

Keep Scenario A as the baseline and change one setting in Scenario B.

Examples:

- random versus targeted sampling;
- random versus highly clustered infection;
- cross-sectional versus longitudinal sampling; and
- no sample-size inflation versus pilot-estimated inflation.

When **runs per scenario** is:

- `1`, the application shows one fitted example;
- greater than `1`, the application repeats the scenario and displays error distributions.

Interpretation:

- values near zero indicate accurate estimates;
- positive values indicate overestimation;
- negative values indicate underestimation; and
- wider distributions indicate greater variability.

Start with one run, then increase to 100 or more after confirming the settings.

---

# Simulator logic

## Host landscape

The landscape consists of host clusters.

Each cluster has:

- a location;
- an area; and
- a host count.

Each host also has a unique ID, allowing it to be resampled in rotating-panel or longitudinal surveys.

## True prevalence

The user defines the true prevalence at each survey round. This is the value the methods attempt to estimate.

## Infection assignment

For each round:

```text
true prevalence × total hosts = target infected hosts
```

Infected hosts are added or removed until the target is reached.

## Infection clustering

Under random infection, hosts have approximately equal infection probability.

Under clustered infection, infection is more likely near:

- existing infected clusters;
- initial hotspots; and
- nearby infected clusters.

The Simulator uses the five nearest neighbouring clusters when calculating local infection pressure.

## Survey simulation

Simple random sampling selects hosts from the full population.

Multistage sampling selects clusters first and then hosts within those clusters.

Targeted sampling increases the probability of selecting infected or high-risk hosts and clusters.

Temporal overlap reuses some host IDs from the previous survey round.

## Analysis

The surveys are analysed using:

1. **Change Method** — uses the first and final survey estimates.
2. **Regression Method** — uses all survey rounds.

Because the Simulator knows the truth, it calculates:

```text
error = estimated value - true value
```

---

# Troubleshooting

## Python is not recognised

1. Install Python from `python.org`.
2. Tick **Add Python to PATH**.
3. Run `Setup.bat` again.

## Package installation fails

Check:

- internet access;
- firewall restrictions;
- proxy settings; and
- installation permissions.

## The application does not open

Check that:

- `Setup.bat` has been run;
- the command window is still open; and
- the command window does not show an error.

Then close the application and try again.

## The browser cannot connect

Wait a few seconds and check the address:

```text
Dashboard: http://127.0.0.1:8000
Simulator: http://127.0.0.1:8001
```

## Shapefiles do not load

Check that:

- the shapefile folder is in the same directory as the project files;
- the displayed path is correct;
- all shapefile components are present; and
- the files have not been renamed or separated.

## The Simulator is slow

- start with one scenario run;
- use fewer synthetic clusters while testing;
- increase to 100 or more runs only after confirming the settings.

## The PDF report does not generate

- generate results first;
- check the command window for errors; and
- rerun `Setup.bat` to confirm that all packages are installed.

---

# Interpretation notes

The Dashboard analyses the survey data supplied to it. If surveillance targets suspected infected locations, estimated prevalence may be higher than prevalence across the wider host population.

Dashboard effort benchmarks are indicative. They do not show that historical survey data followed the assumptions of the sample-size calculations.

The Simulator is a simplified testing model. It is not a biological spread model and should not be used as a forecast.
