# Executable estimation templates

Use actual checked files instead of copying incomplete fragments. Start with the [CPU lab guide](../Camera-Time-Fusion-Lab/Guide.md), including its isolated dependency lock and commands.

| Concept | Executable function | Independent check |
|---|---|---|
| Input/time contract | `contracts.Measurement`, `visible_measurements` | Future arrival cannot affect an earlier output |
| Variable-time linear prediction | `camera_fusion.predict` | Two substeps equal one step for its continuous-noise model |
| Joseph update and gating | `camera_fusion.update` | Positive covariance and known injected outlier |
| Delayed observation replay | `camera_fusion.estimate` | Capture/arrival implementations agree at zero delay |
| Actual camera geometry | `camera_fusion.recover_position` | Noiseless metric recovery and wrong-scale counterexample |
| Nonlinear range update | `estimation_extensions.ekf_range_update` | Hand result and unchanged tangential variance |
| Unscented transform | `estimation_extensions.unscented_square` | Exact Gaussian square moments |
| Orientation error intuition | `estimation_extensions.yaw_error_update` | Angle boundary 179°/−179° |
| Feature elimination | `estimation_extensions.nullspace_project` | Nuisance shift invariance and whitened noise |
| Offline smoothing | `estimation_extensions.rts_scalar` | Two-step RTS analytic result |

Source files: [linear camera/fusion](../Camera-Time-Fusion-Lab/labs/camera_fusion.py), [advanced kernels](../Camera-Time-Fusion-Lab/labs/estimation_extensions.py), [contract](../Camera-Time-Fusion-Lab/labs/contracts.py). Optional NumPy/OpenCV tests live in `labs/optional`; the standard-library contract tests live directly in `labs`.

The orientation and feature-elimination kernels are narrow mathematical exercises. They are not full 3D ESKF/MSCKF implementations. Extend them only after deriving the missing state propagation, clone covariance, noise, and reset operations for your declared conventions.

**Transfer exercise:** replace the synthetic measurement generator with a CSV adapter that validates the same contract. Prove its handling of out-of-order arrivals, duplicate sequence IDs, clock reset, and sensor dropout before adding a real hardware transport.
