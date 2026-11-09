# Server profile "mocap31": the RIGHT-arm checkpoint mocap31-10k (step 6000 EMA, 20 train + 3 held-out episodes of the 31 Oct mocap set), start with
#   start_turbovla_server.sh start --profile mocap31
# KEY=value lines only (parsed, not sourced); @TV_HOME@ expands to $TURBOVLA_HOME (default ~/turbovla).
# Training instruction: $T/mocap31/env.sh COMPAL_MOCAP_TASK.  The sha256 is the pin the server verifies BEFORE loading any weights.
CHECKPOINT=@TV_HOME@/mocap31/runs/mocap31-10k/checkpoints/steps_6000_ema_pytorch_model.pt
LANGUAGE=open the front flap of the cardboard box
EXPECT_SHA256=82eab1f745064fbabbdea12f5a8d14f54a5e5195e5f03ac61e713b0e574b3ec6
