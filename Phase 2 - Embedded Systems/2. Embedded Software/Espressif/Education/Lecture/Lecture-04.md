# Lecture 4 - AI, connectivity, low power, and gateways as measured experiments

**Course:** [Education guide](../Guide.md) | **Previous:** [Native project](Lecture-03.md) | **Next:** [Integrated capstone](Lecture-05.md)

Choose one specialization and give it an input contract, resource budget, and failure test. The official ecosystem supplies components; the engineering task is making their assumptions fit your device.

## AI: count buffers before choosing a model

[ESP-SR](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/index.html) covers speech-oriented components, while [ESP-DL](https://github.com/espressif/esp-dl) provides model deployment/inference facilities. Check target, SDK, operator, and model-format compatibility for the selected version. Current ESP-DL documentation describes ESP-PPQ quantization and `.espdl` deployment; do not assume that renaming an ONNX or TFLite file makes it runnable.

For a teaching voice pipeline, one second of mono audio at 16 kHz with signed 16-bit samples is `16000 × 2 = 32000 bytes`, or 31.25 KiB. A 20 ms block is 320 samples / 640 bytes. Two such input blocks need 1280 bytes before feature buffers, model parameters, activations, task stacks, and driver storage. These are format calculations, not a claim that every ESP-SR model uses this input or buffering scheme.

Now suppose a model has 120000 int8 weights: the raw weights occupy about 117.2 KiB. That is not total RAM. Persistent tensors, peak live activations, allocator alignment, scratch space, and memory placement can dominate. Compare the deployed input preprocessing against the training pipeline and evaluate quantization on representative held-out examples.

Use detection counts and latency, not one successful demo. If a classifier emits 10 decisions/s for one hour, it makes 36000 decisions. Ten false-positive decisions are not necessarily ten false activations after debouncing; report both decision-level error and user-visible activation events. Include silence, noise, unknown inputs, and the actual deployment environment.

## Low power: integrate charge over states

A hypothetical sensor consumes 30 mA for 50 ms, 80 mA for a 20 ms transmit, and 0.02 mA for the remaining 9930 ms of a ten-second cycle:

```text
average current = (30×50 + 80×20 + 0.02×9930) / 10000
                = 0.32986 mA
```

These are assumed values, not measured ESP32 performance. Add retry probability, regulator efficiency, battery limits, and board leakage before estimating life. Measure at the actual supply boundary; a development board USB bridge may remain powered during MCU sleep. Check which wake sources and memory/peripheral state survive the selected mode in the [sleep-mode documentation](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c6/api-reference/system/sleep_modes.html).

## Gateway: preserve meaning across protocols

A gateway can receive a Zigbee attribute or Thread UDP report and expose it to a Linux application. Preserve the original units, source identity, timestamp/clock domain, and validity. Do not timestamp an old cached value as newly sampled merely because the host polled it again.

Start with the [real OpenThread CPU experiment](../../../IoT/OpenThread/labs/CPU-Simulation.md). It proves protocol attachment and traffic in simulation. Then add an application-level stale-data rejection test. Moving the radio to hardware adds RF, power, and coexistence experiments; it does not remove the application contract.

## Choose one falsifiable question

- AI: Does the held-out negative set trigger the indicator more often than the stated target?
- Power: Does measured charge per reporting cycle exceed the budget when retries occur?
- Gateway: Does a stale or duplicate report ever become a fresh observation after reconnect?

A useful result can be “the current design fails.” Record the evidence and change one constraint or implementation choice. Do not turn a vendor benchmark into an unconditional product claim.
