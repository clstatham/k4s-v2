pub mod v2;
/*
slint::slint! {
    import { Button, VerticalBox, HorizontalBox, TextEdit, GroupBox, CheckBox } from "std-widgets.slint";
    import "./src/debugger/assets/BigBlue_Terminal_437TT_Nerd_Font.ttf";

    export component Debugger inherits Window {
        default-font-family: "BigBlue_Terminal_437TT Nerd Font";
        width: 1600px;
        height: 900px;
        background: lightgray;
        title: "K4S Debugger";
        in property <string> registers_text: "";
        in property <string> program_text: "";
        in property <string> error_text: "";
        in property <string> stack_text: "";
        in property <string> call_stack_text: "";
        in property <string> output_text: "";
        in-out property <bool> verbose_steps: false;
        pure callback cont_program();
        pure callback step_program();
        public pure function step_program_impl() {
            root.step_program();
        }
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
                CheckBox {
                    text: "Force Update Every Step";
                    toggled => {
                        root.verbose_steps = self.checked;
                    }
                }
            }
            Text {
                text: registers_text;
                color: black;
            }
            HorizontalBox {
                GroupBox {
                    title: "Instruction History";
                    HorizontalBox {
                        TextEdit {
                            text: program_text;
                            read-only: true;
                            horizontal-alignment: left;
                        }

                    }
                }

                VerticalBox {
                    alignment: end;
                    GroupBox {
                        title: "Stack Frame";
                        TextEdit {
                            text: stack_text;
                            read-only: true;
                            horizontal-alignment: right;
                            height: 200px;
                            width: 260px;
                        }
                    }
                    GroupBox {
                        title: "Call Stack";
                        TextEdit {
                            text: call_stack_text;
                            read-only: true;
                            horizontal-alignment: left;
                            height: 200px;
                            width: 260px;
                            wrap: no-wrap;
                        }
                    }

                }

            }
            HorizontalBox {
                GroupBox {
                    title: "Errors";
                    TextEdit {
                        text: error_text;
                        read-only: true;
                    }
                }

                GroupBox {
                    title: "Output";
                    TextEdit {
                        text: output_text;
                        read-only: true;
                    }
                }

            }
        }
    }
}

pub fn debugger_main(bin_paths: Vec<PathBuf>) -> Result<()> {
    let mut programs = vec![];
    for bin_path in bin_paths {
        let mut file = File::open(bin_path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        programs.push(buf);
    }
    let emulator = MachineContext::new(programs, FOUR_GIGS)?;
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

        match emu.step(true) {
            Err(e) => emu.error = Some(e.root_cause().to_string()),
            Ok(MachineState::ContDontUpdatePc | MachineState::Continue) => {}
            Ok(MachineState::Halt) => {}
        }
        if let Some(e) = emu.error.as_ref() {
            dbgr.set_error_text(format!("{}", e).into());
        }
        dbgr.set_registers_text(format!("{}", emu.regs).into());
        dbgr.set_program_text(
            emu.instr_history
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
                .into(),
        );
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
        dbgr.set_call_stack_text(emu.call_stack.join("\n").into());
        dbgr.set_output_text(emu.output_history.clone().into());
        // dbgr.request_redraw();
        dbgr.window().request_redraw();
    });

    dbgr.on_cont_program(move || {
        loop {
            // let dbgr_handle_on_cont_program = dbgr_handle_on_cont_program.clone();
            // slint::invoke_from_event_loop(move || {
            let dbgr = dbgr_handle_on_cont_program.unwrap();
            dbgr.invoke_step_program_impl();

            // })
            // .unwrap();
            let emu_handle = emu_handle_on_cont_program.upgrade().unwrap();
            let emu = emu_handle.lock().unwrap();
            if emu.error.is_some() {
                break;
            }
        }
        // if let Err(e) = emu.update_dbg_info() {
        //     dbgr.set_error_text(format!("{}", e).into());
        // } else if let Some(e) = emu.error.as_ref() {
        //     dbgr.set_error_text(format!("{}", e).into());
        // }
        // dbgr.set_registers_text(format!("{}", emu.regs).into());
        // dbgr.set_program_text(
        //     emu.instr_history
        //         .iter()
        //         .cloned()
        //         .collect::<Vec<_>>()
        //         .join("\n")
        //         .into(),
        // );
        // let mut stack_text = {
        //     emu.stack_frame
        //         .iter()
        //         .copied()
        //         .map(|(adr, val)| format!("bp - {:>5}: {:016x}", adr, val))
        //         .collect::<Vec<String>>()
        //         .join("\n")
        // };
        // stack_text.push('\n');
        // dbgr.set_stack_text(stack_text.into());
        // dbgr.set_call_stack_text(emu.call_stack.join("\n").into());
        // dbgr.set_output_text(emu.output_history.clone().into());
    });

    dbgr.run()?;

    Ok(())
}
 */
