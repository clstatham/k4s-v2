use std::{
    fs::File,
    io::Read,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::Result;

use crate::k4s::contexts::machine::{MachineContext, MachineState};

slint::slint! {
    import { Button, VerticalBox, HorizontalBox, TextEdit } from "std-widgets.slint";
    import "./src/debugger/assets/BigBlue_Terminal_437TT_Nerd_Font.ttf";

    export component Debugger inherits Window {
        default-font-family: "BigBlue_Terminal_437TT Nerd Font";
        width: 1600px;
        height: 900px;
        background: black;
        title: "K4S Debugger";
        in property <string> registers_text: "";
        in property <string> program_text: "";
        in property <string> error_text: "";
        in property <string> stack_text: "";
        in property <string> output_text: "";
        pure callback cont_program();
        pure callback step_program();
        VerticalBox {
            HorizontalBox {
                Button {
                    text: "Step";
                    width: 150px;
                    height: 20px;
                    clicked => {
                        root.step_program();
                    }
                }
                Button {
                    text: "Continue";
                    width: 150px;
                    height: 20px;
                    clicked => {
                        root.cont_program();
                    }
                }
            }
            Text {
                text: registers_text;
                color: white;
            }
            HorizontalBox {
                HorizontalBox {
                    alignment: start;
                    width: parent.width - 280px;
                    TextEdit {
                        text: program_text;
                        read-only: true;
                        horizontal-alignment: left;
                        height: 400px;
                        width: 100%;
                    }
                }
                HorizontalBox {
                    alignment: end;
                    TextEdit {
                        text: stack_text;
                        read-only: true;
                        horizontal-alignment: right;
                        height: 400px;
                        width: 260px;
                    }
                }

            }
            Text {
                text: error_text;
                color: red;
            }
            Text {
                text: output_text;
                color: white;
            }
        }
    }
}

pub fn debugger_main(bin_path: PathBuf) -> Result<()> {
    let program = {
        let mut file = File::open(bin_path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        buf
    };
    let emulator = MachineContext::new(&program, 0x1000000)?;
    let emulator = Arc::new(Mutex::new(emulator));

    let dbgr = Debugger::new()?;

    let dbgr_handle_on_cont_program = dbgr.as_weak();
    let dbgr_handle_on_step_program = dbgr_handle_on_cont_program.clone();
    let emu_handle_on_view_registers = Arc::downgrade(&emulator);
    let emu_handle_on_step_program = emu_handle_on_view_registers.clone();
    let emu_handle_on_cont_program = emu_handle_on_view_registers;
    dbgr.on_step_program(move || {
        let dbgr = dbgr_handle_on_step_program.unwrap();
        let emu_handle = emu_handle_on_step_program.upgrade().unwrap();
        let mut emu = emu_handle.lock().unwrap();

        match emu.step() {
            Err(_) => {}
            Ok(MachineState::ContDontUpdatePc | MachineState::Continue) => {}
            Ok(MachineState::Halt) => {
                return;
            }
        }
        if let Some(e) = emu.error.as_ref() {
            dbgr.set_error_text(format!("{}", e).into());
        }
        dbgr.set_registers_text(format!("{}", emu.regs).into());
        dbgr.set_program_text(emu.instr_history.join("\n").into());
        let mut stack_text = {
            emu.stack_frame
                .iter()
                .copied()
                .map(|(adr, val)| format!("bp - {:>5}: {:016x}", adr, val))
                .collect::<Vec<String>>()
                .join("\n")
        };
        stack_text.push('\n');
        dbgr.set_stack_text(stack_text.into());
        dbgr.set_output_text(emu.output_history.clone().into());
    });

    dbgr.on_cont_program(move || {
        let emu_handle = emu_handle_on_cont_program.upgrade().unwrap();
        let mut emu = emu_handle.lock().unwrap();
        let dbgr = dbgr_handle_on_cont_program.unwrap();
        loop {
            match emu.step() {
                Err(_) => break,
                Ok(MachineState::ContDontUpdatePc | MachineState::Continue) => {}
                Ok(MachineState::Halt) => {
                    break;
                }
            }
        }
        if let Some(e) = emu.error.as_ref() {
            dbgr.set_error_text(format!("{}", e).into());
        }
        dbgr.set_registers_text(format!("{}", emu.regs).into());
        dbgr.set_program_text(emu.instr_history.join("\n").into());
        let mut stack_text = {
            emu.stack_frame
                .iter()
                .copied()
                .map(|(adr, val)| format!("bp - {:>5}: {:016x}", adr, val))
                .collect::<Vec<String>>()
                .join("\n")
        };
        stack_text.push('\n');
        dbgr.set_stack_text(stack_text.into());
        dbgr.set_output_text(emu.output_history.clone().into());
    });

    dbgr.run()?;

    Ok(())
}
