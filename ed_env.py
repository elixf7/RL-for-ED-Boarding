# Sets up the Emergency Department Gymnasium environment

# Imports
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from patient import Patient, sample_treatment_duration

# Constants

# Number of beds for each type
BED_CAPACITY = {"trauma": 4, "main": 35, "chair": 15}

# ESI proportions from MIMIC-IV-ED dataset for incoming patients
_ESI_RAW = [0.070, 0.330, 0.540, 0.060, 0.002]  # ESI 1-5

# The dataset does not sum to 1 for some reason (1.002), normalize here for now
total_esi = sum(_ESI_RAW)
ESI_DISTRIBUTION = []
for p in _ESI_RAW:
    ESI_DISTRIBUTION.append(p / total_esi)

# Poisson arrival rates (patients per 5 min step) by shift
ARRIVAL_RATE = {
    "night":   3 / 12,  # 12 AM - 8 AM: 3 patients/hour
    "day":     9 / 12,  # 8 AM - 4 PM: 9 patients/hour
    "evening": 6 / 12,  # 4 PM - 12 AM: 6 patients/hour
}

# Wait penalty for each ESI per step
WAIT_PENALTY = {1: 5.0, 2: 3.0, 3: 1.5, 4: 0.75, 5: 0.25}

# Extra per step penalty for ESI 1/2 waiting beyond guidline time
GUIDELINE_EXTRA_PENALTY = {1: 10.0, 2: 5.0}

# Guidline target treatment times for ESI 1/2 in steps
GUIDELINE_STEPS = {1: 0, 2: 3}

# How many steps a patient waits before leaving
LWBS_THRESHOLD = 48

# Discharge reward by bed fit. Full reward (+10) for an ideal bed; reduced for a fallback.
# ESI 4/5 have no preferred bed type so they always get the full reward.

# Discharge reward
# If discharged from non ideal bed there is a small penalty as well
DISCHARGE_REWARD = 10.0
DISCHARGE_REWARD_WRONG_BED = {
    1: 4.0,   # ESI 1 in main instead of trauma - significant downgrade
    2: 7.0,   # ESI 2 in main instead of trauma - minor downgrade
    3: 6.0,   # ESI 3 in chair instead of main  - moderate downgrade
}
# Penalty for leaving without being seen
LWBS_PENALTY = 20.0

# Length of total episode, 24 hours for now
EPISODE_LENGTH = 288

# Start at 8 AM in the morning (96 steps from midnight)
START_STEP = 96

# Number of action slots for each ESI level
# Each ESI level is guaranteed a number of slots
# within each group the people waiting the longest are listed first
ESI_SLOTS = {1: 3, 2: 6, 3: 7, 4: 3, 5: 1}  # must sum to MAX_QUEUE_VISIBLE

# Number of total ESI slots
MAX_QUEUE_VISIBLE = sum(ESI_SLOTS.values())


class EmergencyDeptEnv(gym.Env):
    """
    A Gymnasium environment simulating an Emergency Department

    At each step, the agent can:
      - Do nothing
      - Admit the patient in slot N of the queue

    The state vector shows up to MAX_QUEUE_VISIBLE patients, each with an
    ESI level, current wait time, and treatment duration.
    """

    def __init__(self):
        super().__init__()

        # Action of 0 is do nothing, add this option
        self.action_space = spaces.Discrete(MAX_QUEUE_VISIBLE + 1)

        # obs_size is patient slots + bed counts + time of day
        obs_size = MAX_QUEUE_VISIBLE * 3 + 5
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Initialize variables by calling _init_state()
        self._init_state()

    def _init_state(self):
        """Set up all simulation variables."""
        # current_step is absolute time from midnight
        self.current_step = START_STEP
        self.episode_step = 0         # steps elapsed in current episode
        self.patient_id_counter = 0   # incremented for each new patient

        # list of Patient objects, ordered oldest first
        self.waiting_queue = []

        # Occupied beds: dict of lists, each holding admitted Patient objects
        self.beds = {"trauma": [], "main": [], "chair": []}

        # Keep track of stats for evaluation
        self.total_discharged = 0
        self.total_lwbs = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}

    def get_slot_patients(self):
        """
        Build the 20 slot patient list.

        Each ESI group gets its reserved number of slots (from ESI_SLOTS), filled by
        longest waiting first. None if no patients in group.

        Slot layout:
          Slots 0-2: 3 ESI 1 patients
          Slots 3-8: 6 ESI 2 patients
          Slots 9-15:  7 ESI 3 patients
          Slots 16-18: 3 ESI 4 patients
          Slot  19: 1 ESI 5 patient
        """
        slots = []
        # Loop through each group of esi slots
        for esi, num_slots in ESI_SLOTS.items():
            # Store patients mathing ESI level in waiting queue
            group = []
            for p in self.waiting_queue:
                if p.esi == esi:
                    group.append(p)

            # Sort by longest wait first
            group.sort(key=lambda p: p.steps_waiting(self.current_step), reverse=True)

            # Fill up reserved slots for the ESI group, None if not enough patients
            for i in range(num_slots):
                if i < len(group):
                    slots.append(group[i])
                else:
                    slots.append(None)

        return slots

    def step(self, action):
        # track overall reward for step
        reward = 0.0

        # 1. PROCESS ACTION

        # Process action if its not "do nothing"
        if action > 0:
            slot_index = action - 1
            # Grab slots of patients for each ESI level
            slots = self.get_slot_patients()
            # Check slot is valid and contains patient (not None)
            if slot_index < len(slots) and slots[slot_index] != None:
                patient = slots[slot_index]
                # Assign a bed to that patient in slot_index
                bed_type = self._assign_bed(patient)
                if bed_type != None:
                    # Move patient from queue to bed and update variables
                    patient.bed_type = bed_type
                    patient.admission_step = self.current_step
                    self.beds[bed_type].append(patient)
                    self.waiting_queue.remove(patient)
                # If bed_type is None no valid bed available and action is ignored

        # 2. ADRESS ADMITTED PATIENTS

        # Loop through each bed type
        for bed_type in self.beds:
            still_in_bed = []
            # Loop through each patient in that bed type
            for patient in self.beds[bed_type]:
                # Increment time in bed
                patient.time_in_bed += 1
                # Discharge if treatment is complete
                if patient.treatment_complete:
                    patient.discharged = True
                    # Logic for patient in non optimal bed, apply penalty
                    # ESI 1/2 belong in trauma, ESI 3 belongs in main (not chair).
                    wrong_bed = (patient.esi in (1, 2) and patient.bed_type != "trauma") or \
                                (patient.esi == 3 and patient.bed_type == "chair")
                    # Apply penalty
                    if wrong_bed:
                        reward += DISCHARGE_REWARD_WRONG_BED.get(patient.esi, DISCHARGE_REWARD)
                    # Apply discharge reward
                    else:
                        reward += DISCHARGE_REWARD
                    self.total_discharged += 1
                # Append patient to still_in_bed if treatment incomplete
                else:
                    still_in_bed.append(patient)
            # Update patients left in beds with still_in_bed
            self.beds[bed_type] = still_in_bed

        # 3. REMOVE PATIENTS WHO HAVE WAITED TOO LONG

        still_waiting = []
        # loop through patients in waiting queue
        for patient in self.waiting_queue:
            # Remove if steps has exceeded LWBS limit and apply penalty
            if patient.steps_waiting(self.current_step) >= LWBS_THRESHOLD:
                patient.left_without_seen = True
                reward -= LWBS_PENALTY
                self.total_lwbs += 1
            else:
                still_waiting.append(patient)
        self.waiting_queue = still_waiting

        # 4. APPLY WAIT PENALTIES
        for patient in self.waiting_queue:
            # Wait penalty weighted by ESI level
            reward -= WAIT_PENALTY[patient.esi]

            # Extra penalty for ESi 1/2 if exceeded guideline wait time
            if patient.esi in GUIDELINE_STEPS:
                wait = patient.steps_waiting(self.current_step)
                if wait > GUIDELINE_STEPS[patient.esi]:
                    reward -= GUIDELINE_EXTRA_PENALTY[patient.esi]

        # Call _generate_arrivals to get new patients in waiting room
        new_patients = self._generate_arrivals()
        self.waiting_queue.extend(new_patients)

        # Increment time 
        self.current_step += 1
        self.episode_step += 1

        # Check if episode is over
        truncated = self.episode_step >= EPISODE_LENGTH
        terminated = False  # no natural end state

        # Get current observable state
        obs = self._get_obs()
        # get info stats for results
        info = {
            "total_discharged": self.total_discharged,
            "total_lwbs": self.total_lwbs,
            "queue_length": len(self.waiting_queue),
        }
        return obs, reward, terminated, truncated, info

    def _assign_bed(self, patient):
        """
        Find the appropriate available bed for a patient based on their ESI.
        Returns the bed type("trauma", "main", "chair") or None if no bed is free.
        """
        # Find number of free beds per type
        free = self._free_beds()

        # Logic for determing which type of bed each ESI can be assigned to
        if patient.esi == 1 or patient.esi == 2:
            if free["trauma"] > 0:
                return "trauma"
            elif free["main"] > 0:
                return "main"
            else:
                return None
        elif patient.esi == 3:
            if free["main"] > 0:
                return "main"
            elif free["chair"] > 0:
                return "chair"
            else:
                return None
        else:  # ESI 4 or 5
            if free["chair"] > 0:
                return "chair"
            elif free["main"] > 0:
                return "main"
            else:
                return None

    def _free_beds(self):
        """Returns a dict of how many beds are free per type."""
        free = {}
        # Loop through each bed type in BED_CAPACITY
        # find available beds and store in "free" dict
        for bed_type in BED_CAPACITY:
            free[bed_type] = BED_CAPACITY[bed_type] - len(self.beds[bed_type])
        return free

    def _generate_arrivals(self):
        """
        Sample new patient arrivals for this timestep from Poisson distribution.
        """
        # Determine arrival rate based on the shift (night, day, evening)
        step_in_day = self.current_step % EPISODE_LENGTH
        if step_in_day < 96:
            rate = ARRIVAL_RATE["night"]
        elif step_in_day < 192:
            rate = ARRIVAL_RATE["day"]
        else:
            rate = ARRIVAL_RATE["evening"]

        # Calculate number of arrivals this step
        num_arrivals = np.random.poisson(rate)

        new_patients = []
        # For each arrival set up a Patient object
        for _ in range(num_arrivals):
            # Choose ESI according to distribution
            esi = np.random.choice([1, 2, 3, 4, 5], p=ESI_DISTRIBUTION)
            patient = Patient(
                patient_id=self.patient_id_counter,
                esi=esi,
                arrival_step=self.current_step,
                treatment_duration=sample_treatment_duration(esi),
            )
            # increment patient id to prevent duplicates
            self.patient_id_counter += 1
            new_patients.append(patient)

        return new_patients

    def _get_obs(self):
        """
        Build the normalized observation vector of length 65.

        First 0-59 are: (20 slots for ESI groups, each with 3 values ESI, wait time, treatment time)
        Next 60-64 are: (trauma beds free, main free, chairs free, sin(time), cos(time))
        """
        # Maximum expected treatment time
        MAX_TREATMENT_STEPS = 360.0

        # Initialize observation vector with zeros
        obs = np.zeros(MAX_QUEUE_VISIBLE * 3 + 5, dtype=np.float32)

        # Fill in paint slots with same ordering as get_slot_patients
        for i, patient in enumerate(self.get_slot_patients()):
            # If patient exists, set up thier observation variables like ESI, wait time, treatment time
            if patient != None:
                base = i * 3
                obs[base] = patient.esi / 5.0
                obs[base + 1] = patient.steps_waiting(self.current_step) / LWBS_THRESHOLD
                obs[base + 2] = patient.treatment_duration / MAX_TREATMENT_STEPS

        # Get free bed counts and normalize by bed capacity
        free = self._free_beds()
        global_start = MAX_QUEUE_VISIBLE * 3
        obs[global_start] = free["trauma"] / BED_CAPACITY["trauma"]
        obs[global_start + 1] = free["main"] / BED_CAPACITY["main"]
        obs[global_start + 2] = free["chair"] / BED_CAPACITY["chair"]

        # Get time of day with sin/cos so that ends and beginnings of the day appear close to the model
        time_fraction = (self.current_step % EPISODE_LENGTH) / EPISODE_LENGTH
        obs[global_start + 3] = np.sin(2 * np.pi * time_fraction)
        obs[global_start + 4] = np.cos(2 * np.pi * time_fraction)

        return obs
