# ROM & Torque Calculation Pipeline

This project computes joint angles (ROM) and teaching-grade joint torques from wearable IMU data. It follows ISB-style axis conventions and is designed for gait analysis and education.

## Processing Flow

IMU Quaternions (pelvis, femur, tibia)
        |
        ├── Preprocessing
        │     - Time sync
        │     - Sensor fusion (quat normalization)
        │     - Static bias removal (gyro offset)
        |
        ├── Yaw Alignment
        │     - Align global heading across segments
        │     - Remove long-term drift
        |
        ├── Drift Correction
        │     - Pelvis–femur yaw sharing (low-pass blend)
        │     - Stride-wise debias (add/rot baseline ≈ 0°)
        │     - High-pass filter (fc ~0.03 Hz) for ultra-slow drift
        |
        ├── Relative Orientations
        │     - Pelvisᵀ · Femur → Hip
        │     - Femurᵀ · Tibia → Knee
        |
        ├── ROM Extraction
        │     - Intrinsic XYZ Euler sequence
        │     - Hip angles = [Flex, Add, Rot]
        │     - Knee angles = [Flex, Add, Rot]
        |
        └── Torque Estimation
              - Anthropometrics (mass, length, COM, inertia)
              - Angular accel (α) + linear accel (a)
              - Newton–Euler inverse dynamics
              - Teaching-grade hip/knee torques

## Axis Conventions (ISB-style)

Segment anatomical axes:
- X = Forward (antero–posterior)
- Y = Left (medio–lateral)
- Z = Up / Long axis (proximal → distal)

Angles reported:
- X (Flexion/Extension): + = flexion (forward raise)
- Y (Adduction/Abduction): + = adduction (toward midline) for both sides
- Z (Internal/External Rotation): + = internal rotation (inward spin)

Side handling:
- Right-side angles flip the Y component so adduction is positive toward midline for both limbs. Z (internal rotation) remains positive without flipping.

## Drift Mitigation

- Gyro Bias Removal → from static windows at trial start/end.
- Yaw Drift Correction → pelvis–femur share low-frequency yaw (remove slow divergence).
- Stridewise Debias → enforce zero-mean adduction/rotation per stride.
- High-pass Filter → optional extra guard (fc = 0.02–0.05 Hz).

## Validation

Synthetic rotations:
- 30° hip flexion → [30, 0, 0]
- 15° hip adduction → [0, 15, 0]
- 20° hip internal rotation → [0, 0, 20]

Synthetic torques:
- Constant angular acceleration → torque ≈ I·α
- Static gravity → torque ≈ m·g·COM offset

Gait checks:
- Neutral stance → ROM ≈ [0, 0, 0]
- Stride averages → mean hip add/rot ≈ 0°
- No visible drift after 60+ seconds of walking

## Notes

- Torque values are teaching-grade and do not include GRF; they illustrate inertial + gravity contributions.
- See `core/math/kinematics.py` (Euler XYZ) and `core/math/drift.py` (filters, yaw sharing) for implementation details.
