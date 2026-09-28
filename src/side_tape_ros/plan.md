# Calibration Jacobian safety update

1. Keep the existing TF-measured jog and return-to-start workflow; switch the camera subscriber to JPEG by default with an explicit raw option, preserving de-rotation.
2. Interlock each jog iteration and each transition with trajectory ownership and monotonic-age Servo status; stop and pause on unsafe input.
3. Gate writes on both measured-fit residuals and Jacobian column norms, with an explicit force-save override; back up the config and add a timestamp/residual YAML comment, not a schema key.
4. Add isolated pure-function tests for gates, file preservation/loadability, and JPEG de-rotation; update package calibration instructions.
5. Run only the requested offline pytest command and `--help`; review the diff independently. Never run the node against hardware or publish Servo commands.
