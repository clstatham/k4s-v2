use std::{
    fs::File,
    io::Read,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Error, Result};
use dioxus::prelude::*;
use dioxus_desktop::{Config, LogicalSize, PhysicalSize, WindowBuilder};

use crate::{k4s::contexts::machine::MachineContext, FOUR_GIGS};

pub fn debugger_main(bin_paths: Vec<PathBuf>) -> Result<()> {
    dioxus_desktop::launch_with_props(
        App,
        DebuggerProps { bin_paths },
        Config::new().with_window(
            WindowBuilder::new()
                .with_resizable(false)
                .with_inner_size(LogicalSize::new(1600, 900)),
        ),
    );
    Ok(())
}

struct DebuggerProps {
    bin_paths: Vec<PathBuf>,
}

#[allow(non_snake_case)]
fn App(cx: Scope<DebuggerProps>) -> Element {
    let mut programs = vec![];
    for bin_path in &cx.props.bin_paths {
        let mut file = File::open(bin_path).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
        programs.push(buf);
    }
    let emu = use_ref(cx, || MachineContext::new(programs, FOUR_GIGS).unwrap());
    let style = rsx! {
        style {
            include_str!("assets/style.css")
        }
    };
    let regs_text = { emu.read().regs.to_string() };
    let registers = rsx! {
        textarea {
            class: "debug_textarea regs",
            readonly: true,
            "{regs_text}",
            }
    };
    let instructions_text = {
        emu.read()
            .instr_history
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    };
    let instructions = rsx! {
        textarea {
            class: "debug_textarea instrs",
            readonly: true,
            "{instructions_text}",
        }
    };

    let step_btn = rsx! {
        div {
            class: "controls",
            button {
                onclick: move |_| {
                    emu.write().step(true).unwrap();
                },
                "Step",
            }
            button {
                onclick: move |_| {
                    let mut emu = emu.write();
                    if let Err(e) = emu.run_until_hlt() {
                        log::debug!("{:?}", e);
                    }
                },
                "Continue",
            }
        }
    };

    let stack_text = {
        emu.read()
            .stack_frame
            .iter()
            .copied()
            .map(|(adr, val)| format!("bp - {:>5}: {:016x}", adr, val))
            .collect::<Vec<String>>()
            .join("\n")
    };

    let stack = rsx! {
        textarea {
            class: "debug_textarea stack",
            readonly: true,
            "{stack_text}",
        }
    };

    let call_stack_text = { emu.read().call_stack.join("\n") };
    let calls = rsx! {
        textarea {
            class: "debug_textarea calls",
            readonly: true,
            "{call_stack_text}",
        }
    };

    let errors_text = {
        let lock = emu.read();
        lock.error.as_ref().cloned().unwrap_or(String::new())
    };
    let errors = rsx! {
        textarea {
            class: "debug_textarea errors",
            readonly: true,
            "{errors_text}",
        }
    };

    let output_text = {
        let lock = emu.read();
        lock.output_history.to_owned()
    };
    let output = rsx! {
        textarea {
            class: "debug_textarea output",
            readonly: true,
            "{output_text}",
        }
    };

    cx.render(rsx! {
        style,
        div {
            class: "grid-container",
            step_btn,
            registers,
            instructions,
            stack,
            calls,
            errors,
            output,
        }
    })
}
