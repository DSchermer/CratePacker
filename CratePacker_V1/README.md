# Crate Packing Planner

This project gives teams a point-and-click way to build efficient packing plans without ever touching code. Enter or upload the assemblies you must ship, choose from a catalog of crates or boxes, and the app finds low-cost layouts, generates visuals, and saves everything for record keeping.

## Who this guide is for

- People who receive a ZIP of this folder and need to run the planner locally.
- Folks comfortable opening a terminal, even if they do not write code regularly.
- Mac and Windows users (the steps call out anything that differs).

If you have never used a terminal, pair with someone who has; the rest of the workflow is menu-driven in the browser.

## Before you start

- Works on macOS 12+, Windows 10+, or a modern Linux desktop.
- Needs Python 3.10 or newer and about 1 GB of free disk space the first time you install the dependencies.
- The Streamlit app runs entirely on your machine; no internet connection is required after you download the project.

### Check that Python is installed

macOS / Linux:

1. Open Terminal.
2. Run `python3 --version`.
3. If the version is 3.10 or higher, you are set. Otherwise install the latest Python from https://www.python.org/downloads/.

Windows:

1. Open Command Prompt (Start menu, search for `cmd`).
2. Run `python --version`.
3. If the version is 3.10 or higher, you are set. If the command fails or the version is lower, install Python 3.10+ from https://www.python.org/downloads/ and check the box that adds Python to PATH during setup.

### Download this project

Choose one option:

1. From GitHub: click the green **Code** button, choose **Download ZIP**, unzip it somewhere easy (for example `Documents/CratePlanner`).
2. From a shared drive or email: unzip the folder you were given and place it in a working directory (same example path above).

Throughout the rest of this guide we assume the folder lives at `/Users/<you>/Documents/boxes` (macOS) or `C:\Users\<you>\Documents\boxes` (Windows). Adjust commands if your path is different.

## One-time environment setup

Perform these steps the first time you use the planner on a given machine.

Windows (Command Prompt):

```cmd
cd C:\Users\<you>\Documents\boxes
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- Creating the virtual environment (`.venv`) keeps project libraries separate from your system Python.
- The install takes 1-3 minutes depending on your connection; you will see progress messages from `pip`.
- You only have to install the requirements once unless the project maintainers tell you they changed.

## Launch the planner (every session)

Each time you want to use the app, open a fresh terminal and run the following commands.

macOS / Linux:

```bash
cd /Users/<you>/Documents/boxes
source .venv/bin/activate
streamlit run app.py
```

Windows:

```cmd
cd C:\Users\<you>\Documents\boxes
.venv\Scripts\activate
streamlit run app.py
```

The terminal will show a URL such as `http://localhost:8501`. Hold Control (or Command on Mac) and click the link, or copy and paste it into your browser. Keep the terminal window open while you use the planner; closing it will stop the app. When you are done, press `Ctrl+C` in that window to shut it down.

## Using the app step by step

1. **Pick a catalog**: In the sidebar, choose whether to work with `Crates` or `Boxes`. Each catalog provides dimensions, costs, and packing priorities from the YAML files in the project.
2. **Choose units and buffers**: Set the measurement units (inches) and the buffer you want around each item. The buffer inflates the packing footprint while the app still shows the true base size for reference.
3. **Load your items**:
	- Manually edit the table: click a row to change `label`, `length`, `width`, or `count`.
	- Upload a CSV: use the upload button above the table; see the format section below.
	- Enable `Auto-aggregate` to merge duplicate rows automatically.
4. **Set container options**: Optional filters let you restrict the allowed crate types, override per-crate costs, and adjust priority rules.
5. **Run the pack**: Press **Pack**. The app evaluates permutations of the selected crate types (up to six), fills each bin with the Skyline algorithm, and repacks to ensure the cheapest viable crate is used.
6. **Review the results**: A summary table highlights total cost, cost per item, and utilization. Below that, screenshots show each packed crate with both buffered (outer) and base (inner) rectangles.
7. **Download outputs**: Use the download buttons to grab a ZIP of the session or individual artifacts. Files also save under the `output` folder for later reference.

## CSV format for uploads

The CSV needs four columns with these exact headers (case-sensitive):

```text
label,length,width,count
Widget A,12,6,5
Widget B,10,8,2
```

- Length and width are in inches.
- `count` accepts whole numbers only.
- Blank rows or additional columns are ignored.

## Where outputs are stored

Every packing run creates a timestamped folder under `output/` named `pack_YYYYMMDD_HHMMSS`. Inside each folder you will find:

- `summary.json` with a machine-readable breakdown of crates, items, and costs.
- `crates.csv` (or `boxes.csv`) with the selected container mix.
- One PNG per packed crate showing the layout.
- `session.zip` combining all of the above for easy sharing.

You can safely delete old folders when you no longer need them.

## Resetting or updating the planner

- To wipe all installed packages and start fresh, delete the `.venv` folder and repeat the one-time setup steps.
- If you download a new version of the project, replace the existing folder and reinstall requirements to capture library updates.
- Catalog changes (`crates_catalog.yaml`, `boxes_catalog.yaml`) take effect immediately; you do not need to restart the app.

## Troubleshooting tips

- **Command not found**: If the terminal cannot find `python` or `streamlit`, confirm you activated the virtual environment (`source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows).
- **Port already in use**: If Streamlit reports port 8501 is busy, close any other Streamlit windows or start the app with `streamlit run app.py --server.port 8502`.
- **Dependency errors**: Run `pip install -r requirements.txt` again inside the activated environment.
- **Blank browser page**: Refresh the tab. If that fails, stop the app with `Ctrl+C`, then re-run the launch commands.
- **Need help**: Capture a screenshot of the terminal output and share it with the engineering support contact listed by your team.

You now have everything needed to install, launch, and operate the Crate Packing Planner end to end.
