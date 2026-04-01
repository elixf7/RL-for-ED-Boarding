# Baseline agents for EmergencyDeptEnv

class RandomAgent:
    """
    Picks a random action every step.
    """

    def __init__(self, action_space):
        self.action_space = action_space


    def select_action(self):
        return self.action_space.sample()


class FCFSAgent:
    """
    First Come, First Served.
    Always admits the patient who has been waiting the longest, regardless of ESI.

    Because the state slots are grouped by ESI (not arrival order), the oldest
    patient could be in any slot. This agent scans the slots to find them.
    """

    def select_action(self, env):
        # Check if waiting room is empty
        if len(env.waiting_queue) == 0:
            return 0  # nobody waiting

        # Find the oldest patient, waiting_queue is sorted in arrival order so 0 is oldest
        oldest_patient = env.waiting_queue[0]

        # Loop through the environment slots and find patient matching the oldest id
        for i, patient in enumerate(env.get_slot_patients()):
            if patient != None and patient.patient_id == oldest_patient.patient_id:
                # Need to add 1 since 0 is "do nothing"
                return i + 1

        return 0


class AcuityFirstAgent:
    """
    Acuity First.
    Always admits the highest acuity patient available (lowest ESI number).
    Breaks ties by longest wait time.
    """

    def select_action(self, env):
        # Loop through the patients in the environement slots
        # Since get_slot_patients() returns in ESI order, the first non None patient is the highest acuity
        for i, patient in enumerate(env.get_slot_patients()):
            if patient != None:
                return i + 1
        return 0  # nobody waiting
