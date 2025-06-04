# NextGenEE Practicum Guideline

## Prerequisites
1. Install PyCharm or VS Code, whichever you prefer.
2. Install `uv` by following the instructions at [uv](https://docs.astral.sh/uv/getting-started/installation/)
3. Install `git` if you haven't already. You can find installation instructions at [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Setup
1. Clone the repository. Use the interactive extensions in your IDE or use the command line:
   ```bash
   git clone git@github.com:sentient-codebot/pgm-arena.git
   ```
2. Navigate to the cloned directory `pgm-arena`. Make sure you have done this step correctly!
3. Open the integrated terminal in your IDE and run the following command to install the required dependencies:
   ```bash
   uv sync
   ```
4. If you are using PyCharm, you may need to set the Python interpreter to the one in the `venv` directory. You can do this by going to `File -> Settings -> Project: pgm-arena -> Python Interpreter` and selecting the interpreter from the `venv` directory.
5. If you are using VS Code, you may need to select the Python interpreter from the `venv` directory. You can do this by pressing `Ctrl + Shift + P`, typing `Python: Select Interpreter`, and selecting the interpreter from the `venv` directory.
6. Now you have the Python environment set up. You can run the code in the `pgm-arena` directory.

## Running the Code
### Practicum 1: Getting Started with PowerGridModel and Power Flow
1. Run `practicum_1_first_step_tutorial.py` to get started with the PowerGridModel and power flow. It contains an official example of `PowerGridModel`.
2. Run `practicum_1_arena_65_power_flow.py` to see the power flow in a realistic power grid.
3. Answer the questions in the `.docx` document.

### Practicum 2: Power Flow with Time Series Data
1. Run `practicum_2_time_series_calculation.ipynb`. 
2. Follow the code and answer questions in the document.

### Practicum 3: 
1. Run `practicum_3_power_flow_with_pv.ipynb.ipynb` to the changes PV panels would bring to the grid.
2. Follow the code and answer questions in the document.