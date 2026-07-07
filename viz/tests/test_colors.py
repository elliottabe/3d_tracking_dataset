from viz.core import colors
KP = ["Scutellum","WingL_base","WingR_base","Antenna_Base","EyeL","EyeR",
      "WingL_V12","WingL_V13","WingR_V12","WingR_V13","Abd_A4","Abd_tip",
      "T1L_ThxCx","T1L_Tro","T1L_FeTi","T1L_TiTa","T1L_TaT1","T1L_TaT3","T1L_TaTip",
      "T2L_Tro","T2L_FeTi","T2L_TiTa","T2L_TaT1","T2L_TaT3","T2L_TaTip"]

def test_palette_has_core_keys():
    for k in ("fly0","fly1","head","tail","detector","fit","mask","mesh"):
        assert k in colors.PALETTE and len(colors.PALETTE[k]) == 3

def test_keypoint_groups_partition():
    g = colors.keypoint_groups(KP)
    assert KP.index("EyeL") in g["head"] and KP.index("Antenna_Base") in g["head"]
    assert KP.index("Abd_tip") in g["abdomen"]
    assert KP.index("Scutellum") in g["thorax"] and KP.index("WingL_base") in g["thorax"]
    assert KP.index("T1L_TaTip") in g["legs"] and KP.index("T2L_Tro") in g["legs"]

def test_leg_chains_proximal_to_distal():
    chains = colors.leg_chains(KP)
    t1l = chains["T1L"]
    assert [KP[i] for i in t1l] == ["T1L_ThxCx","T1L_Tro","T1L_FeTi","T1L_TiTa","T1L_TaT1","T1L_TaT3","T1L_TaTip"]
    # T2L has no ThxCx segment present -> starts at Tro
    assert [KP[i] for i in chains["T2L"]] == ["T2L_Tro","T2L_FeTi","T2L_TiTa","T2L_TaT1","T2L_TaT3","T2L_TaTip"]
