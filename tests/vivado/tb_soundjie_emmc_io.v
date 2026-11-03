`timescale 1ns/1ps
module tb_soundjie_emmc_io;
  reg clk=0, cmd_o=0, cmd_t=1, cd_n=0, wp=0;
  reg [3:0] data_o=0, data_t=15;
  reg card_drive=0, card_cmd=0;
  reg [3:0] card_data=0;
  wire clk_fb, pad_clk, cmd_i, ps_cd_n, ps_wp;
  wire [3:0] data_i;
  tri pad_cmd;
  tri [3:0] pad_data;
  assign pad_cmd=card_drive ? card_cmd : 1'bz;
  assign pad_data=card_drive ? card_data : 4'bzzzz;
  soundjie_emmc_io dut(clk,clk_fb,cmd_o,cmd_t,cmd_i,data_o,data_t,data_i,
                       cd_n,wp,ps_cd_n,ps_wp,pad_clk,pad_cmd,pad_data);
  integer v;
  initial begin
    #200; // wait for the Xilinx global simulation startup window
    for(v=0;v<16;v=v+1) begin
      clk=v[0]; cmd_o=v[0]; data_o=v; cmd_t=0; data_t=0;
      #5;
      if(clk_fb !== clk || pad_clk !== ~clk) $fatal(1,"clock phase/feedback");
      if(pad_cmd !== cmd_o || cmd_i !== cmd_o || pad_data !== data_o || data_i !== data_o)
        $fatal(1,"host direction");
      cmd_t=1; data_t=15; #5;
      if(pad_cmd !== 1'bz || pad_data !== 4'bzzzz) $fatal(1,"release bus");
      card_drive=1; card_cmd=~v[0]; card_data=~v; #5;
      if(cmd_i !== card_cmd || data_i !== card_data) $fatal(1,"card direction");
      card_drive=0; cd_n=v[0]; wp=v[1]; #5;
      if(ps_cd_n !== cd_n || ps_wp !== wp) $fatal(1,"status polarity");
    end
    $display("EMMC_IO_SIM_PASS"); $finish;
  end
endmodule
