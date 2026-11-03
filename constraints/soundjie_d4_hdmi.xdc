# M7Z100FA20200515 sheet09: AC-coupled differential outputs to HDMI connector.
# XC7Z100 bank35 is HP, VCCO=1.8V. Use LVDS, NOT TMDS_33 or LVDS_25.
# This is a board-specific DVI-video transport, not HDMI electrical certification.
set_property PACKAGE_PIN D15 [get_ports HDMI_D0_P]
set_property PACKAGE_PIN D14 [get_ports HDMI_D0_N]
set_property PACKAGE_PIN G15 [get_ports HDMI_D1_P]
set_property PACKAGE_PIN G14 [get_ports HDMI_D1_N]
set_property PACKAGE_PIN C14 [get_ports HDMI_D2_P]
set_property PACKAGE_PIN C13 [get_ports HDMI_D2_N]
set_property PACKAGE_PIN F15 [get_ports HDMI_CLK_P]
set_property PACKAGE_PIN F14 [get_ports HDMI_CLK_N]
set_property IOSTANDARD LVDS [get_ports {HDMI_D0_P HDMI_D0_N HDMI_D1_P HDMI_D1_N HDMI_D2_P HDMI_D2_N HDMI_CLK_P HDMI_CLK_N}]

# Retained CH24 LCD GPIO E12 is in the SAME HP bank35. The vendor example
# declares LVCMOS15, but M7Z100CORE20211003 sheet07 ties VB35 to VCC1V8.
# Override only this inherited electrical standard; never suppress BIVC-1.
set_property IOSTANDARD LVCMOS18 [get_ports {lcd_gpio_tri_io[1]}]
