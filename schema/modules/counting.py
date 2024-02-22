from typing import Union
from pydantic import BaseModel


class VehicleCountingParam(BaseModel):
    classes: Union[list, dict]

VEHICLE_COUNTING_ALLOW_CHANGES = []