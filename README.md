# NextGenEE Practicum 3

This folder contains the student version of practicum 3: analyzing how PV
generation changes power flow results.

## Prerequisites

1. Install `uv`: <https://docs.astral.sh/uv/getting-started/installation/>
2. Open this folder in VS Code, PyCharm, or another editor that can run
   Jupyter notebooks.

## Setup

From this folder, install the Python environment:

```bash
uv sync
```

Then select the Python interpreter from `.venv` in your editor.

## Running the Practicum

Open and work through:

```text
practicum_3_power_flow_with_pv.ipynb
```

The notebook is a student exercise. Some code cells contain incomplete lines
marked with `TODO`, such as blank assignments. These are intentional; complete
them before running the later cells that depend on them.

The provided data files are under `data/`, and the plotting helper is under
`plot_utils/`.

## Expected Work

In this practicum, you will:

1. Inspect PV irradiance and example PV generation data.
2. Combine the baseload profile with PV generation.
3. Run batched power flow calculations with PV.
4. Plot load profiles, node voltages, and line loadings.
5. Visualize selected time steps on the grid graph.
