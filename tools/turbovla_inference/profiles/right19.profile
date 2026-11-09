# Server profile "right19": the RIGHT-arm checkpoint right19-10k (step 4000 EMA), start with
#   start_turbovla_server.sh start --profile right19
# KEY=value lines only (parsed, not sourced); @TV_HOME@ expands to $TURBOVLA_HOME (default ~/turbovla).
# Training instruction: $T/right19/env.sh COMPAL_MOCAP_TASK.  The sha256 is the pin the server verifies BEFORE loading any weights.
CHECKPOINT=@TV_HOME@/right19/runs/right19-10k/checkpoints/steps_4000_ema_pytorch_model.pt
LANGUAGE=open the front flap of the cardboard box
EXPECT_SHA256=02f163bef316fcc1894ec3b614a8d9cb7640a7d1771c48bc8f23607fa5598068
