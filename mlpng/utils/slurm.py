import os


class Slurm:
    """
    A simple handler for some slurm data to be used in the code.
    """

    def __init__(self):
        self.array_index = int(os.getenv("SLURM_ARRAY_TASK_ID", default="-1"))
        self.is_main = self.array_index in {1, -1}
        self.name = os.getenv("SLURM_JOB_NAME", "unknown")
        self.job = int(os.getenv("SLURM_JOB_ID", "0"))
        self.task_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "0"))
        self.array = int(os.getenv("SLURM_ARRAY_JOB_ID", "0"))

        # here we just get the number of cpus, but read in from SLURM if available
        # os.sched_getaffinity(0) gets the number of usable CPUs available, this is different from
        # os.cpu_count() which gets the number of CPUs on the system
        cpus = len(os.sched_getaffinity(0))
        self.n_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", cpus))

    def __repr__(self):
            return f"Slurm({self.name=}, {self.job=}, {self.array=}, {self.array_index=}, {self.task_count=}, {self.is_main=}, {self.n_cpus=})"
