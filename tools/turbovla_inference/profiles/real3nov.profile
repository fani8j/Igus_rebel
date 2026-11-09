# Server profile "real3nov": the RIGHT-arm checkpoint real3nov-10k (R1, step 10000 EMA; 12 train + 3 held-out episodes of the 3 Nov robot-playback set
# real_right_3nov_clean_v1: 5 box poses x front/back/right flap), start with
#   start_turbovla_server.sh start --profile real3nov
# KEY=value lines only (parsed, not sourced); @TV_HOME@ expands to $TURBOVLA_HOME (default ~/turbovla).
# Training instruction: $T/real3nov/env.sh COMPAL_MOCAP_TASK (one string for all three flaps).  The sha256 is the pin the server verifies BEFORE loading any weights.
CHECKPOINT=@TV_HOME@/real3nov/real3nov/runs/real3nov-10k/checkpoints/steps_10000_ema_pytorch_model.pt
LANGUAGE=open the next closed flap of the cardboard box
EXPECT_SHA256=53d1f024930295be142cba3aad14550b6f7e1568fead7653b42b24a8b86a83d1
