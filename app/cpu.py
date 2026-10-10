import os
from pathlib import Path


def available_cpu_count(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """Respect CPU affinity and Linux container quotas when sizing model threads."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    limits = [max(1, count)]
    try:
        quota, period = (cgroup_root / "cpu.max").read_text().split()
        if quota != "max" and int(quota) > 0 and int(period) > 0:
            limits.append(max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    for directory in (cgroup_root, cgroup_root / "cpu", cgroup_root / "cpu,cpuacct"):
        try:
            quota = int((directory / "cpu.cfs_quota_us").read_text())
            period = int((directory / "cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError):
            pass
    return min(limits)
