    <!-- SRDF fragment for the OPT-IN single-arm blade collision link `knife_c_blade` (URDF arg
         blade_collision:=true on igus_rebel_xeg32_wrapper.urdf.xacro, tool_variant knife_c). NOT part of
         igus_rebel2.srdf on purpose: a disable_collisions entry for a link that is not in the model would be
         noise in every default session and would add entries to the default allowed collision matrix, so the
         default MoveIt/executor behaviour stays bit-identical. A launch that builds the URDF with
         blade_collision:=true inserts this fragment before the closing </robot> of igus_rebel2.srdf
         (dual_cut_bringup per_arm_planning.launch.py does).

         Same single pair and same proof as the dual SRDF (rebel_dual_table_moveit_config make_srdf.py,
         DUAL_ONLY_DISABLED_PAIRS): the blade box is rigid with igus_tool0 like the handpiece and starts 1 mm
         inside the handpiece blade seat, so blade-handpiece contact is configuration independent (200000 of
         200000 random arm samples, scan_results/dual_arm_stack_20261014/p0c_blade/logs/blade_pairs.txt;
         test_blade_handpiece_overlap_is_rigid_and_holder_is_clear). The blade-holder pair has 0 hits (the holder
         ends 62 mm behind the seat front), so it stays checked, as does the blade against every other link
         (rare hits only with base_link / motor1 / link1 / link2 / motor2 at extreme joint values), the
         carton, the table and camera pole. -->
    <disable_collisions link1="knife_c_blade" link2="knife_c_handpiece" reason="Adjacent" />
