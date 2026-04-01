# Defines the Patient class and helper function for treatment duration.

# imports
import numpy as np
from dataclasses import dataclass

# Median total length of stay for different ESI levels for how long a patient will occupy a bed
# Comes from the MIMIC-IV-ED dataset
MEDIAN_LOS_HOURS = {
    1: 5.1,
    2: 6.5,
    3: 5.7,
    4: 2.9,
    5: 2.2,
}

# Each simulation step represents 5 minutes
STEP_DURATION_HOURS = 5 / 60


def sample_treatment_duration(esi):
    """
    Sample a treatment duration (in steps) for a patient based on their ESI level.

    Use log-normal distribution because treatment times are always positive and have a realistic long right tail, some patients take much longer than average.

    Returns an integer number of steps (minimum of 1).
    """
    # Convert LOS to simulation steps
    mean_steps = MEDIAN_LOS_HOURS[esi] / STEP_DURATION_HOURS

    # Log-normal distribution parameters
    # log_mean is adjusted so that the mean is the mean number of steps
    log_std = 0.5
    log_mean = np.log(mean_steps) - (log_std ** 2) / 2

    duration = int(np.random.lognormal(mean=log_mean, sigma=log_std))

    # Every patient needs to spend at least one step in a bed
    if duration < 1:
        return 1
    return duration


@dataclass
class Patient:
    """
    Represents a single ED patient.

    Fields set on arrival:
        patient_id, esi, arrival_step, treatment_duration

    Fields updated when assigned a bed:
        bed_type, admission_step

    Fields updated each simulation step:
        time_in_bed, discharged, left_without_seen
    """

    patient_id: int
    esi: int            # Acuity level 1-5
    arrival_step: int   # Simulation step when patient arrives
    treatment_duration: int  # Steps needed in bed before discharge

    # Stay None or default until patient assigned bed
    bed_type: str = None        # Type of bed trauma, main, chair
    admission_step: int = None  # Step when assigned a bed
    time_in_bed: int = 0        # Steps elapsed since assigned bed

    discharged: bool = False    # True when discharged
    left_without_seen: bool = False  # True if patient left before being seen

    @property
    def is_admitted(self):
        # Checks if admitted by seeing if they have a bed
        return self.bed_type != None

    @property
    def treatment_complete(self):
        # Checks if it is time to discharge
        return self.time_in_bed >= self.treatment_duration

    def steps_waiting(self, current_step):
        """
        How many steps the patient has been waiting.
        Once admitted, returns the total wait time before they got assigned a bed.
        """
        # Checks if already admitted, and returns how long they waited for a bed
        if self.admission_step != None:
            return self.admission_step - self.arrival_step
        # Otherwise return how long they have been waiting for a bed
        return current_step - self.arrival_step
