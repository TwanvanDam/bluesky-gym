from openap import FuelFlow
from bluesky.tools.aero import ft, kts

class FuelModel:
    def __init__(self, aircraft_type: str):
        """Wrapper around the openap FuelFlow model. This wrapper uses SI units (kg, m/s, m) for the input and output of the fuel flow model."""
        self.fuel_flow_model = FuelFlow(aircraft_type)

    def step_fuel_flow(self, mass: float, tas: float, altitude: float) -> float:
        """Calculate the fuel flow in [ kg/s ] based on the current mass [ kg ], true airspeed [ m/s ], and altitude [ m ]."""
        return self.fuel_flow_model.enroute(mass=mass, tas=tas / kts, alt=altitude / ft) # [ kg/s ]