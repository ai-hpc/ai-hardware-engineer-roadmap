# Lecture 3 — SPI transport, GPIOs, and IRQ-driven bring-up

**Course:** [Jetson ESP-Hosted Host Code guide](../Guide.md) · **Phase 2 — Embedded Linux**

**Previous:** [Lecture 02](Lecture-02.md) · **Next:** [Lecture 04 — How Wi-Fi becomes `wlan0`](Lecture-04.md)

---

## 1. The transport file to study

Read:

- `esp_hosted_ng/host/spi/esp_spi.c`

This file is where the host turns:

- SPI bus selection
- chip select
- handshake GPIO
- data-ready GPIO

into a working transport endpoint for the ESP.

That makes it the best file for learning how **transport glue** looks in a real Embedded Linux driver.

## Related Linux kernel concepts

This lecture sits directly on top of:

- [OS Lecture 3 — Interrupts, Exceptions & Bottom Halves](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-03.md)
- [OS Lecture 17 — Linux Device Driver Model & Device Tree](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-17.md)
- [OS Lecture 18 — Character Drivers, Interrupt-Driven I/O & V4L2](../../../../Phase%201%20-%20Foundational%20Knowledge/3.%20Operating%20Systems/Lectures/Lecture-18.md)

Lecture 3 from the OS course gives the important interrupt model: hardware raises an IRQ, the kernel acknowledges it quickly, and real work is deferred safely. That is the right frame for reading `gpio_to_irq(...)`, `request_irq(...)`, and the workqueue-driven SPI path in `esp_spi.c`.

Lecture 17 explains how Linux matches a driver to an already-described device, and Lecture 18 explains event-driven I/O once the device is alive. Together they explain why this code first reuses `spi0.0` from the kernel device model and then turns handshake and data-ready GPIO edges into an interrupt-driven transport instead of polling blindly.

The detail that matters most is that an **IRQ is not “just a GPIO change notification.”** In Linux, an IRQ is the kernel’s promise that when the hardware edge arrives, the driver will get scheduled through the interrupt path and can move the protocol forward with **low latency**.

That is why the validation focused on `/proc/interrupts` counts and not only on pin names. A correctly named GPIO that never produces an IRQ is operationally useless, while a line with rising counters proves that the electrical signal, the GPIO mapping, the IRQ translation, and the kernel handler path are all aligned.

The kernel interfaces to track here are:

- `module_param(...)`
- SPI device lookup and reuse
- `gpio_to_irq(...)`
- `request_irq(...)`
- IRQ-driven work scheduling

---

## 2. Start at the module parameters

Near the top of `esp_spi.c`, look for these:

- `spi_bus_num`
- `spi_chip_select`
- `spi_handshake_gpio`
- `spi_dataready_gpio`
- `spi_mode`

These are exposed as `module_param(...)`.

That tells you:

- transport wiring is now configurable at load time
- the driver can be reused across boards without recompilation

This is a direct improvement over hardcoded board assumptions.

You can see that policy-to-transport bridge right at the top of `spi/esp_spi.c`:

```c
static ushort spi_bus_num = DEFAULT_SPI_BUS_NUM;
static ushort spi_chip_select = DEFAULT_SPI_CHIP_SELECT;
static int spi_handshake_gpio = DEFAULT_HANDSHAKE_PIN;
static int spi_dataready_gpio = DEFAULT_SPI_DATA_READY_PIN;
static uint spi_mode = DEFAULT_SPI_MODE;

module_param(spi_bus_num, ushort, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
module_param(spi_chip_select, ushort, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
module_param(spi_handshake_gpio, int, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
module_param(spi_dataready_gpio, int, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
module_param(spi_mode, uint, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
```

This is where “Jetson pin choices” stop being shell-script text and become transport state consumed by the driver.

---

## 3. Find `spi_dev_init(...)`

`spi_dev_init(...)` is the heart of transport bring-up.

Conceptually, it is responsible for:

- selecting the Linux SPI device
- configuring SPI mode and clock
- setting up the transport context
- requesting and configuring the handshake/data-ready GPIOs
- mapping those GPIOs into IRQs

This is the point where a “Linux board description” becomes an **“active transport endpoint.”**

That is an important Embedded Linux mental shift:

- device tree and sysfs prove **presence**
- transport init proves **usability**

The critical part of `spi_dev_init(...)` is worth reading literally:

```c
esp_info("Config - SPI GPIOs: Handshake[%d] Dataready[%d]\n",
	spi_context.handshake_gpio, spi_context.dataready_gpio);

esp_info("Config - SPI clock[%dMHz] bus[%d] cs[%d] mode[%d]\n",
	spi_context.spi_clk_mhz, esp_board.bus_num,
	esp_board.chip_select, esp_board.mode);

status = gpio_request(spi_context.handshake_gpio, "SPI_HANDSHAKE_PIN");
...
spi_context.handshake_irq = gpio_to_irq(spi_context.handshake_gpio);
status = request_irq(spi_context.handshake_irq, spi_interrupt_handler,
		IRQF_SHARED | IRQF_TRIGGER_RISING,
		"ESP_SPI", spi_context.esp_spi_dev);
...
status = gpio_request(spi_context.dataready_gpio, "SPI_DATA_READY_PIN");
...
spi_context.dataready_irq = gpio_to_irq(spi_context.dataready_gpio);
status = request_irq(spi_context.dataready_irq, spi_data_ready_interrupt_handler,
		IRQF_SHARED | IRQF_TRIGGER_RISING,
		"ESP_SPI_DATA_READY", spi_context.esp_spi_dev);
```

That snippet is the transport bring-up story in one place:

- log the configuration
- request the GPIOs
- convert them to IRQs
- bind the IRQ handlers that make the transport live

---

## 4. Why the IRQs matter more than the GPIO names

The validated Jetson path used:

- `spi_handshake_gpio=471`
- `spi_dataready_gpio=433`

But those numbers alone are not success.

The stronger success signal is:

- the host driver requested them
- they became IRQ sources
- interrupt counters moved when the ESP booted

That is how you should debug **GPIO-based bring-up**:

- name -> request -> IRQ mapping -> actual edge activity

If you stop at “the number looks right,” you have **not really debugged anything**.

---

## 5. Reusing `spi0.0` is a Linux device-model decision

The Jetson fork reuses an existing SPI device such as:

- `spi0.0`

when possible.

That matters because Linux may already have:

- a device tree node
- a bus number
- a chip-select mapping

This is better than always forcing the driver to invent a new SPI device object.

Embedded Linux lesson:

- respect the kernel’s device model when it already describes the hardware correctly

The reuse behavior is explicit:

```c
existing_dev = spi_find_device(esp_board.bus_num, esp_board.chip_select);
if (existing_dev) {
	...
	spi_context.esp_spi_dev = existing_dev;
	esp_info("Reusing existing SPI device spi%u.%u\n",
		esp_board.bus_num, esp_board.chip_select);
} else {
	master = spi_busnum_to_master(esp_board.bus_num);
	...
	spi_context.esp_spi_dev = spi_new_device(master, &esp_board);
}
```

The shell script unbinds `spidev`, but the transport code is written to cooperate with the existing SPI device path.

---

## 6. Runtime clock clamping is a real bring-up policy

The validated Jetson work changed the runtime clock behavior.

The host can receive a request from the ESP boot-up path to move to:

- `26 MHz`

But the validated Jetson flow intentionally keeps the host capped at:

- `10 MHz`

Why this is important:

- **transport stability** beat nominal peak speed during bring-up
- the driver now uses `clockspeed=` as both:
  - initial speed
  - runtime ceiling

That is a very real Embedded Linux engineering pattern:

- make the code reflect the validated board limit
- then widen later only with measurement

The actual clamp logic is concise:

```c
static void adjust_spi_clock(u8 spi_clk_mhz)
{
	u8 target_spi_clk = spi_clk_mhz;

	if (spi_context.spi_clk_cap_mhz && target_spi_clk > spi_context.spi_clk_cap_mhz) {
		esp_info("ESP requested SPI CLK %u MHz, clamping to host limit %u MHz\n",
			 target_spi_clk, spi_context.spi_clk_cap_mhz);
		target_spi_clk = spi_context.spi_clk_cap_mhz;
	}

	if (target_spi_clk != spi_context.spi_clk_mhz) {
		esp_info("ESP Reconfigure SPI CLK to %u MHz\n", target_spi_clk);
		spi_context.spi_clk_mhz = target_spi_clk;
		spi_context.esp_spi_dev->max_speed_hz = target_spi_clk * NUMBER_1M;
	}
}
```

---

## 7. What success looked like on hardware

The key transport-level log lines were:

- `Received ESP boot-up event`
- `Chipset=ESP32-C6 ID=0d detected over SPI`
- `ESP requested SPI CLK 26 MHz, clamping to host limit 10 MHz`

That proves:

- SPI data transfer is alive
- the protocol exchange is alive
- the transport policy is being enforced

This is a much stronger success criterion than:

- module inserted
- no kernel oops
- `/dev/spidev0.0` exists

---

## Lab

Do these:

1. Find where `spi_context.spi_bus_num` and `spi_context.spi_chip_select` get filled.
2. Find where the handshake and data-ready GPIO values enter the transport context.
3. Find the function that handles runtime SPI clock changes.
4. Write down the three strongest transport-level log lines you would want during bring-up.

---

**Previous:** [Lecture 02](Lecture-02.md) · **Next:** [Lecture 04 — How Wi-Fi becomes `wlan0`](Lecture-04.md)
