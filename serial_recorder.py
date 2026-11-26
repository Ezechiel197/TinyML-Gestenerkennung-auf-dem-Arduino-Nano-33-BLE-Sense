#!/usr/bin/env python3
"""

Records motion data from the Arduino Nano 33 BLE Sense (sketch_gyro streaming
"ax,ay,az,gx,gy,gz" per line) and detects gestures automatically: as soon as
|ax|+|ay|+|az| exceeds the threshold, a window starting at that sample is saved.
This is the same trigger the inference sketch (tflite/tflite.ino) uses, so
training windows are aligned like the windows the board classifies later.

Each gesture can yield several windows shifted by a few samples around the
trigger (augmentation), which also makes the model robust to trigger jitter.

Output format is: 6 columns, SAMPLES rows per window,
windows separated by a blank line, one CSV per gesture/label (appended).
A second blank line closes each gesture, so training can split by gesture
instead of scattering the shifted copies of one gesture over train and test.

Requirement:  pip install pyserial
Examples:
  python serial_recorder.py --port COM5 --out data/punch.csv
  python serial_recorder.py --port COM5 --out data/lift.csv --count 20 --per-gesture 5 --jitter 10
"""
import argparse, collections, time
import serial

CHANNELS = 6


def read_samples(ser):
    """Yields (raw line, values) for every valid 6-value line; header and noise lines are skipped."""
    while True:
        line = ser.readline().decode("utf-8", "replace").strip()
        parts = line.split(",")
        if len(parts) != CHANNELS:
            continue
        try:
            values = [float(x) for x in parts]
        except ValueError:
            continue
        yield line, values


def motion(values):
    return abs(values[0]) + abs(values[1]) + abs(values[2])


def window_offsets(per_gesture, jitter):
    """Window starts relative to the trigger sample, spread evenly over [-jitter, +jitter]."""
    if per_gesture <= 1 or jitter == 0:
        return [0]
    return sorted({round(-jitter + 2 * jitter * i / (per_gesture - 1)) for i in range(per_gesture)})


def count_gestures(path):
    try:
        with open(path) as f:
            return sum(1 for block in f.read().split("\n\n\n") if block.strip())
    except FileNotFoundError:
        return 0


def measure_rate(samples, seconds=2.0):
    n, start = 0, time.monotonic()
    for _ in samples:
        n += 1
        if time.monotonic() - start >= seconds:
            return n / (time.monotonic() - start)


def record(samples, f, a, existing):
    offsets = window_offsets(a.per_gesture, a.jitter)
    pre = max(0, -offsets[0])
    seq_len = pre + max(offsets) + a.samples
    history = collections.deque(maxlen=pre)   # samples before the trigger, for negative offsets
    quiet = 0                                 # consecutive samples below threshold, re-arms the trigger
    gestures = windows = 0

    print(f"Fenster-Offsets je Geste: {offsets}")
    print("Ready - gesture start, Strg+C to end.")
    try:
        for line, values in samples:
            armed = quiet >= a.rest and len(history) == pre
            if not (armed and motion(values) >= a.threshold):
                quiet = quiet + 1 if motion(values) < a.threshold else 0
                if pre:
                    history.append(line)
                continue

            seq = list(history) + [line]
            for following, _ in samples:
                seq.append(following)
                if len(seq) == seq_len:
                    break

            for off in offsets:
                f.write("\n".join(seq[pre + off:pre + off + a.samples]) + "\n\n")
            f.write("\n")
            f.flush()
            gestures += 1
            windows += len(offsets)
            print(f"  Gesture {gestures} detected -> {len(offsets)} windows (file: {existing + gestures} gestures)")

            if a.count and gestures >= a.count:
                break
            history.clear()
            quiet = 0
    except KeyboardInterrupt:
        # A gesture interrupted mid-capture is dropped; only complete ones were written.
        print()
    return gestures, windows


def main():
    ap = argparse.ArgumentParser(description="SerialRecorder with automatic gesture recognition (Lab 2 format).")
    ap.add_argument("--port", required=True, help="serial port, e.g., COM5 or /dev/ttyACM0")
    ap.add_argument("--out", required=True, help="target CSV file, e.g., punch.csv (will be appended to)")
    ap.add_argument("--samples", type=int, default=119, help="samples per window (default 119)")
    ap.add_argument("--threshold", type=float, default=1.5,
                    help="trigger threshold |ax|+|ay|+|az| in g, like kTriggerAccelSum in tflite.ino (default 1.5)")
    ap.add_argument("--rest", type=int, default=10,
                    help="rest samples below threshold before the next gesture can trigger (default 10)")
    ap.add_argument("--per-gesture", type=int, default=10, help="windows per detected gesture (default 10)")
    ap.add_argument("--jitter", type=int, default=10,
                    help="max. shift of windows around the trigger time in samples (default 10)")
    ap.add_argument("--count", type=int, default=0, help="exit after this many gestures (0 = until Ctrl+C)")
    ap.add_argument("--baud", type=int, default=9600, help="baud rate (default 9600)")
    a = ap.parse_args()

    existing = count_gestures(a.out)
    with serial.Serial(a.port, a.baud, timeout=1) as ser, open(a.out, "a", newline="") as f:
        time.sleep(2)
        ser.reset_input_buffer()
        samples = read_samples(ser)
        print(f"Connected to {a.port}. Target: {a.out} (already contains {existing} gestures)")
        rate = measure_rate(samples)
        print(f"Sampling rate approx. {rate:.0f} Hz -> a window takes {a.samples / rate:.2f} s")
        gestures, windows = record(samples, f, a, existing)
    print(f"Finished: {gestures} gestures, {windows} windows in {a.out}.")

if __name__ == "__main__":
    main()
