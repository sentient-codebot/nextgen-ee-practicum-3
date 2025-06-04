import json
import pprint

import power_grid_model as pgm
import pandas as pd
from power_grid_model import PowerGridModel, ComponentType, ComponentAttributeFilterOptions
from power_grid_model.utils import json_serialize, json_deserialize
from power_grid_model.validation import assert_valid_input_data


def main():
    with open("data/arena_raw_data_65.json") as fp:
        data = fp.read()
    pprint.pprint(json.loads(data))
    pass

    dataset = json_deserialize(data)
    assert_valid_input_data(dataset, pgm.CalculationType.power_flow)

    model = PowerGridModel(dataset)
    output_data = model.calculate_power_flow(
        calculation_method=pgm.CalculationMethod.newton_raphson,
    )
    pass

    # check results
    print("------node result------")
    print(pd.DataFrame(output_data[ComponentType.node]))
    print("------line result------")
    print(pd.DataFrame(output_data[ComponentType.line]))


if __name__ == "__main__":
    main()