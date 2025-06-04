import power_grid_model as pgm
import numpy as np
import pandas as pd
from power_grid_model import (initialize_array, ComponentType, DatasetType, LoadGenType,
                              CalculationType, CalculationMethod)
from power_grid_model.validation import assert_valid_input_data


def main():
    # This is our grid
    r"""
     -----------------------line_8---------------
    |                                          |
    node_1 ---line_3--- node_2 ----line_5---- node_6
    |                    |                     |
    source_10          sym_load_4           sym_load_7
    """
    # %% Input
    # id are unique across all elements
    node = initialize_array(
        DatasetType.input,
        ComponentType.node,
        3,
    )
    pass  # what is node like here?
    node["id"] = np.array([1, 2, 6])
    node["u_rated"] = np.array([10.5e3, 10.5e3, 10.5e3])
    pass

    # line
    line = initialize_array(
        data_type=DatasetType.input,
        component_type=ComponentType.line,
        shape=3,
    )
    line["id"] = np.array([3, 5, 8])
    line["from_node"] = np.array([1, 2, 1])
    line["to_node"] = np.array([2, 6, 6])
    line["from_status"] = np.array([1, 1, 1])
    line["to_status"] = np.array([1, 1, 1])
    line["r1"] = np.array([0.25, 0.25, 0.25])
    line["x1"] = np.array([0.2, 0.2, 0.2])
    line["c1"] = [10e-6, 10e-6, 10e-6] # NOTE can either list or array
    # line["nonexistent"] = np.array([1, 2, 3]) 
    #   >> will give ValueError: no field of name nonexistent
    line["tan1"] = np.array([0.0, 0.0, 0.0])
    line["i_n"] = [1000, 1000, 1000]
    pass

    # load
    sym_load = initialize_array(
        data_type=DatasetType.input,
        component_type=ComponentType.sym_load,
        shape=2,
    )
    sym_load["id"] = np.array([4, 7])
    sym_load["node"] = [2, 6]
    sym_load["status"] = [1, 1]
    sym_load["type"] = [LoadGenType.const_power, LoadGenType.const_power]
    sym_load["p_specified"] = [20e6, 20e6]  # Watt
    sym_load["q_specified"] = [5e6, 2e6]  # Var

    # source
    source = initialize_array(
        data_type=DatasetType.input,
        component_type=ComponentType.source,
        shape=1,
    )
    source["id"] = [10]
    source["node"] = [1]
    source["status"] = [1]
    source["u_ref"] = [1.0]
    r"""alternative
    source_columns = {
        "id": np.array([10], dtype=attribute_dtype(DatasetType.input, ComponentType.source, "id")),
        "node": np.array([1], dtype=attribute_dtype(DatasetType.input, ComponentType.source, "node")),
        "status": np.array([1], dtype=attribute_dtype(DatasetType.input, ComponentType.source, "status")),
        "u_ref": np.array([1.0], dtype=attribute_dtype(DatasetType.input, ComponentType.source, "u_ref")),
        # We're not creating columns for u_ref_angle, sk, ... Instead, the default values are used. This saves us memory.
    }
    """

    # all
    input_data = {
        ComponentType.node: node,
        ComponentType.line: line,
        ComponentType.sym_load: sym_load,
        ComponentType.source: source,
    }

    # %% Validation
    assert_valid_input_data(
        input_data=input_data,
        calculation_type=CalculationType.power_flow,
    )
    pass

    # %% Construction of model
    model = pgm.PowerGridModel(input_data)

    # %% One-time Power Flow Calculation
    output_data = model.calculate_power_flow(
        symmetric=True,
        error_tolerance=1e-8,
        max_iterations=20,
        calculation_method=CalculationMethod.newton_raphson,
        # output_component_types=[ComponentType.node] # if only some components are needed
        # output_component_types={
        #    ComponentType.node: ["id", "u", "u_angle"],
        # }  # alternative
    )  # dict[ComponentType, np.ndarray]
    pass

    # check results
    print("------node result------")
    print(pd.DataFrame(output_data[ComponentType.node]))
    print("------line result------")
    print(pd.DataFrame(output_data[ComponentType.line]))
    pass

    # %% Update Model
    pass
    # %% Batch Calculation with Time Series
    # logic: starting from a `base`, then each case is an `update` to the `base`.
    pass


if __name__ == "__main__":
    main()