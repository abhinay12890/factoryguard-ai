import requests
import time

url = "http://127.0.0.1:5003/predict"


payload_low={"sensor_2_mean_6":642.6966666667,"sensor_4_mean_6":1408.3833333333,"sensor_7_mean_6":553.7683333333,"sensor_11_mean_6":47.5066666667,"sensor_15_mean_6":8.43315,"sensor_21_mean_6":23.33495,"sensor_3_mean_12":1590.0016666667,"sensor_15_mean_12":8.4326583333,"sensor_17_mean_12":392.6666666667,"sensor_15_ema_12":8.4334302301,"sensor_17_ema_12":392.8826179321,"sensor_21_ema_12":23.3262336787}

payload_medium={"sensor_2_mean_6":643.0033333333,"sensor_4_mean_6":1419.7333333333,"sensor_7_mean_6":552.6266666667,"sensor_11_mean_6":47.8066666667,"sensor_15_mean_6":8.4828,"sensor_21_mean_6":23.1709166667,"sensor_3_mean_12":1594.6716666667,"sensor_15_mean_12":8.4769416667,"sensor_17_mean_12":394.3333333333,"sensor_15_ema_12":8.4799644034,"sensor_17_ema_12":394.3329552484,"sensor_21_ema_12":23.179966116}

payload_high={"sensor_2_mean_6":643.0133333333,"sensor_4_mean_6":1414.595,"sensor_7_mean_6":552.345,"sensor_11_mean_6":47.7433333333,"sensor_15_mean_6":8.4746666667,"sensor_21_mean_6":23.2003333333,"sensor_3_mean_12":1594.7608333333,"sensor_15_mean_12":8.4610916667,"sensor_17_mean_12":394.4166666667,"sensor_15_ema_12":8.4649009988,"sensor_17_ema_12":394.3972136228,"sensor_21_ema_12":23.2184437485}

payload_critical={"sensor_2_mean_6":643.1716666667,"sensor_4_mean_6":1420.2483333333,"sensor_7_mean_6":552.1033333333,"sensor_11_mean_6":47.95,"sensor_15_mean_6":8.4816,"sensor_21_mean_6":23.15875,"sensor_3_mean_12":1596.87,"sensor_15_mean_12":8.47825,"sensor_17_mean_12":394.5833333333,"sensor_15_ema_12":8.4829457657,"sensor_17_ema_12":394.6213670979,"sensor_21_ema_12":23.1630804297}


def test_latency(name,payload):
    print(f"--{name}--")
    start=time.perf_counter()
    response=requests.post(url,json=payload)
    end=time.perf_counter()
    print(response.json())
    print(f"Client Latency: {(end-start)*1000:.2f} ms")

test_latency("LOW",payload_low)
test_latency("MEDIUM",payload_medium)
test_latency("HIGH",payload_high)
test_latency("CRITICAL",payload_critical)