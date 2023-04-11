use std::{
    fs::File,
    io::Read,
    sync::{Arc, Mutex},
};

use anyhow::Result;

use crate::k4s::contexts::machine::{MachineContext, MachineState};

slint::slint! {
    import { Button, VerticalBox, HorizontalBox } from "std-widgets.slint";
    import "./src/debugger/assets/BigBlue_Terminal_437TT_Nerd_Font.ttf";

    export component Debugger inherits Window {
        default-font-family: "BigBlue_Terminal_437TT Nerd Font";
        width: 1000px;
        height: 600px;
        background: black;
        title: "K4S Debugger";
        in property <string> registers_text: "";
        in property <string> program_text: "";
        in property <string> error_text: "";
        in property <string> stack_text: "";
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
                color: green;
            }
            HorizontalLayout {
                HorizontalBox {
                    alignment: start;
                    Text {
                        text: program_text;
                        color: white;
                        vertical-alignment: top;
                        horizontal-alignment: left;
                        height: 400px;
                    }
                }
                HorizontalBox {
                    alignment: end;
                    Text {
                        text: stack_text;
                        color: yellow;
                        vertical-alignment: top;
                        horizontal-alignment: right;
                        height: 400px;
                    }
                }

            }
            Text {
                text: error_text;
                color: red;
            }
        }
    }
}

pub fn debugger_main() -> Result<()> {
    let program = {
        let mut file = File::open("target/test1.k4s")?;
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
            Err(e) => {
                dbgr.set_error_text(format!("{}", e).into());
            }
            Ok(MachineState::ContDontUpdatePc | MachineState::Continue) => {
                dbgr.set_registers_text(format!("{}", emu.regs).into());
                let scroll_len = emu.instr_history.len().min(30);
                let scroll_begin = emu.instr_history.len() - scroll_len;
                dbgr.set_program_text(
                    emu.instr_history[scroll_begin..scroll_begin + scroll_len]
                        .join("\n")
                        .into(),
                );
                let stack_text = {
                    emu.stack_frame
                        .iter()
                        .copied()
                        .map(|(adr, val)| format!("bp - {:>5}: {:016x}", adr, val))
                        .collect::<Vec<String>>()
                        .join("\n")
                };
                dbgr.set_stack_text(stack_text.into());
            }
            Ok(MachineState::Halt) => {}
        }
    });

    dbgr.on_cont_program(move || {
        let emu_handle = emu_handle_on_cont_program.upgrade().unwrap();
        let mut emu = emu_handle.lock().unwrap();

        loop {
            let dbgr = dbgr_handle_on_cont_program.unwrap();
            match emu.step() {
                Err(e) => {
                    dbgr.set_error_text(format!("{}", e).into());
                    break;
                }
                Ok(MachineState::ContDontUpdatePc | MachineState::Continue) => {
                    dbgr.set_registers_text(format!("{}", emu.regs).into());
                    let scroll_len = emu.instr_history.len().min(30);
                    let scroll_begin = emu.instr_history.len() - scroll_len;
                    dbgr.set_program_text(
                        emu.instr_history[scroll_begin..scroll_begin + scroll_len]
                            .join("\n")
                            .into(),
                    );
                    let stack_text = {
                        emu.stack_frame
                            .iter()
                            .copied()
                            .map(|(adr, val)| format!("bp - {:>5}: {:016x}", adr, val))
                            .collect::<Vec<String>>()
                            .join("\n")
                    };
                    dbgr.set_stack_text(stack_text.into());
                }
                Ok(MachineState::Halt) => break,
            }
        }
    });

    dbgr.run()?;

    Ok(())
}
