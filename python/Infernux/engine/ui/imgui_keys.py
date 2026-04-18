"""Shared ImGui key-code constants.

Values match the ``ImGuiKey`` enum in imgui.h (starting at 512 for
named keys).  Import from here instead of hard-coding magic numbers
in individual panel files.

Usage::

    from .imgui_keys import KEY_SPACE, KEY_LEFT_CTRL
    if ctx.is_key_down(KEY_SPACE):
        ...
"""

# ── Navigation ────────────────────────────────────────────────────────
KEY_TAB         = 512
KEY_LEFT_ARROW  = 513
KEY_RIGHT_ARROW = 514
KEY_UP_ARROW    = 515
KEY_DOWN_ARROW  = 516
KEY_PAGE_UP     = 517
KEY_PAGE_DOWN   = 518
KEY_HOME        = 519
KEY_END         = 520
KEY_INSERT      = 521

# ── Editing ───────────────────────────────────────────────────────────
KEY_DELETE      = 522
KEY_BACKSPACE   = 523
KEY_SPACE       = 524
KEY_ENTER       = 525
KEY_ESCAPE      = 526

# ── Modifiers (ImGuiKey_ModXXX bit flags, matches imgui.h) ─────────────
MOD_CTRL = 1 << 12  # 4096 — use with ``ctx.is_key_down(MOD_CTRL)``

# ── Modifiers (named keys) ────────────────────────────────────────────
KEY_LEFT_CTRL   = 527
KEY_LEFT_SHIFT  = 528
KEY_LEFT_ALT    = 529
KEY_LEFT_SUPER  = 530
KEY_RIGHT_CTRL  = 531
KEY_RIGHT_SHIFT = 532
KEY_RIGHT_ALT   = 533
KEY_RIGHT_SUPER = 534
KEY_MENU        = 535

# ── Digits ────────────────────────────────────────────────────────────
KEY_0 = 536;  KEY_1 = 537;  KEY_2 = 538;  KEY_3 = 539;  KEY_4 = 540
KEY_5 = 541;  KEY_6 = 542;  KEY_7 = 543;  KEY_8 = 544;  KEY_9 = 545

# ── Letters ───────────────────────────────────────────────────────────
KEY_A = 546;  KEY_B = 547;  KEY_C = 548;  KEY_D = 549;  KEY_E = 550
KEY_F = 551;  KEY_G = 552;  KEY_H = 553;  KEY_I = 554;  KEY_J = 555
KEY_K = 556;  KEY_L = 557;  KEY_M = 558;  KEY_N = 559;  KEY_O = 560
KEY_P = 561;  KEY_Q = 562;  KEY_R = 563;  KEY_S = 564;  KEY_T = 565
KEY_U = 566;  KEY_V = 567;  KEY_W = 568;  KEY_X = 569;  KEY_Y = 570
KEY_Z = 571

# ── Function keys ─────────────────────────────────────────────────────
KEY_F1  = 572;  KEY_F2  = 573;  KEY_F3  = 574;  KEY_F4  = 575
KEY_F5  = 576;  KEY_F6  = 577;  KEY_F7  = 578;  KEY_F8  = 579
KEY_F9  = 580;  KEY_F10 = 581;  KEY_F11 = 582;  KEY_F12 = 583
