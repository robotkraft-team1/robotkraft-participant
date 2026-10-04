import depthai as dai
from datetime import timedelta

device_infos = dai.Device.getAllConnectedDevices()
config = dai.HealthCheckConfig(
    powerSupplyCheckDuration=timedelta(seconds=20)
)

metrics = dai.Device.performHealthCheck(device_infos[0], config)
print(metrics)
